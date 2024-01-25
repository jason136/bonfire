use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;
use tokio::task;

use crate::text_generation::token_stream::TokenOutputStream;
use crate::text_generation::utils::{device, hub_load_safetensors, TextGeneratorInner};

use super::utils::TextGenerationArgs;

pub enum Mistral7bModel {
    Mistral7b,
    Mistral7bInstruct,
    Mistral7bInstructV02,
}

pub struct Mistral7b {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Default for Mistral7b {
    fn default() -> Self {
        let args: TextGenerationArgs = TextGenerationArgs::default();
        Self::new(Mistral7bModel::Mistral7b, &args).unwrap()
    }
}

impl Mistral7b {
    /// Creates a new instance of the LLM
    pub fn new(model: Mistral7bModel, args: &TextGenerationArgs) -> anyhow::Result<Self> {
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        println!(
            "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
            args.temperature.unwrap_or(0.),
            args.repeat_penalty,
            args.repeat_last_n
        );

        let start = std::time::Instant::now();
        let api = Api::new()?;

        let repo_id = match model {
            Mistral7bModel::Mistral7b => "mistralai/Mistral-7B-v0.1".to_string(),
            Mistral7bModel::Mistral7bInstruct => "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
            Mistral7bModel::Mistral7bInstructV02 => {
                "mistralai/Mistral-7B-Instruct-v0.2".to_string()
            }
        };

        let repo = api.repo(Repo::with_revision(
            repo_id,
            RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let start = std::time::Instant::now();
        let device = device(false)?;
        let config = Config::config_7b_v0_1(device.is_cuda());

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Model::new(&config, vb)?;

        println!("loaded the model in {:?}", start.elapsed());

        let logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);

        Ok(Self {
            model,
            device: device.clone(),
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
        })
    }

    /// Preloads the model files into the cache
    pub fn cache_model() -> anyhow::Result<()> {
        let start: std::time::Instant = std::time::Instant::now();
        let api = Api::new()?;
        for repo_id in [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ] {
            let repo = api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                "main".to_string(),
            ));
            repo.get("tokenizer.json")?;
            hub_load_safetensors(&repo, "model.safetensors.index.json")?;
        }

        println!("retrieved mistral7b files in {:?}", start.elapsed());
        Ok(())
    }
}

impl TextGeneratorInner for Mistral7b {
    /// Prompts an already loaded LLM and streams output mpsc Sender
    fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()> {
        self.tokenizer.clear();

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find </s> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                let sender_clone = sender.clone();
                task::spawn(async move {
                    // use std::io::Write;
                    // print!("{}", &t);
                    // std::io::stdout().flush().unwrap();
                    sender_clone.send(t).await.unwrap();
                });
            }
        }

        let gen_time = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            task::spawn(async move {
                sender.send(rest).await.unwrap();
            });
        }

        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / gen_time.as_secs_f64(),
        );
        Ok(())
    }
}
