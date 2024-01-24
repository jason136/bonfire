use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixtral::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::Rng;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;
use tokio::task;

use crate::text_generation::token_stream::TokenOutputStream;
use crate::text_generation::utils::{device, hub_load_safetensors, TextGeneratorInner};

#[derive(Debug, Clone)]
pub struct MixtralArgs {
    pub cpu: bool,
    pub tracing: bool,
    pub use_flash_attn: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub quantized: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

pub struct Mixtral {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Default for MixtralArgs {
    fn default() -> Self {
        MixtralArgs {
            cpu: false,
            tracing: false,
            use_flash_attn: false,
            temperature: Some(0.95),
            top_p: Some(0.95),
            seed: rand::thread_rng().gen(),
            quantized: true,
            repeat_penalty: 1.1,
            repeat_last_n: 128,
        }
    }
}

impl Default for Mixtral {
    fn default() -> Self {
        let args = MixtralArgs::default();
        Self::new(&args, false).unwrap()
    }
}

impl Mixtral {
    /// Creates a new instance of the LLM
    pub fn new(args: &MixtralArgs, instruct: bool) -> anyhow::Result<Self> {
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
        let model_id = if instruct {
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string()
        } else {
            "mistralai/Mixtral-8x7B-v0.1".to_string()
        };
        let repo: hf_hub::api::sync::ApiRepo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let start = std::time::Instant::now();
        let config = Config::v0_1_8x7b(args.use_flash_attn);
        let device = device(args.cpu)?;
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
    pub fn preload_model() -> anyhow::Result<()> {
        let start = std::time::Instant::now();
        let api = Api::new()?;
        let regular = api.repo(Repo::with_revision(
            "mistralai/Mixtral-8x7B-v0.1".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        let instruct = api.repo(Repo::with_revision(
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        regular.get("tokenizer.json")?;
        hub_load_safetensors(&regular, "model.safetensors.index.json")?;
        hub_load_safetensors(&instruct, "model.safetensors.index.json")?;

        println!("retrieved mixtral8x7b files in {:?}", start.elapsed());
        Ok(())
    }
}

impl TextGeneratorInner for Mixtral {
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
                    use std::io::Write;
                    print!("{}", &t);
                    std::io::stdout().flush().unwrap();
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
