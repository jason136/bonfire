use std::fs::File;
use std::path::PathBuf;
use std::slice::Iter;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;
use tokio::task;

use crate::text_generation::token_stream::TokenOutputStream;
use crate::text_generation::utils::{device, format_size, TextGeneratorInner};

use super::utils::TextGenerationArgs;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum QuantizedModel {
    Mistral7b,
    Mistral7bInstruct,
    Mistral7bInstructV02,
    Mixtral,
    MixtralInstruct,
    Zephyr7bAlpha,
    Zephyr7bBeta,
    DolphinMixtral,
}

impl QuantizedModel {
    fn tokenizer_repo(&self) -> &'static str {
        match self {
            QuantizedModel::Mixtral => "mistralai/Mixtral-8x7B-v0.1",
            QuantizedModel::MixtralInstruct => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            QuantizedModel::Mistral7b
            | QuantizedModel::Mistral7bInstruct
            | QuantizedModel::Mistral7bInstructV02
            | QuantizedModel::Zephyr7bAlpha
            | QuantizedModel::Zephyr7bBeta
            | QuantizedModel::DolphinMixtral => "mistralai/Mistral-7B-v0.1",
        }
    }

    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = self.tokenizer_repo();
        let api = api.model(repo.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<PathBuf> {
        let (repo, filename) = match self {
            QuantizedModel::Mistral7b => (
                "TheBloke/Mistral-7B-v0.1-GGUF",
                "mistral-7b-v0.1.Q4_K_S.gguf",
            ),
            QuantizedModel::Mistral7bInstruct => (
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            ),
            QuantizedModel::Mistral7bInstructV02 => (
                "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
            ),
            QuantizedModel::Mixtral => (
                "TheBloke/Mixtral-8x7B-v0.1-GGUF",
                "mixtral-8x7b-v0.1.Q4_K_M.gguf",
            ),
            QuantizedModel::MixtralInstruct => (
                "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            ),
            QuantizedModel::Zephyr7bAlpha => (
                "TheBloke/zephyr-7B-alpha-GGUF",
                "zephyr-7b-alpha.Q4_K_M.gguf",
            ),
            QuantizedModel::Zephyr7bBeta => {
                ("TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf")
            },
            QuantizedModel::DolphinMixtral => (
                "TheBloke/dolphin-2.5-mixtral-8x7b-GGUF",
                "dolphin-2.5-mixtral-8x7b.Q4_K_M.gguf",
            ),
        };

        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model(repo.to_string());
        Ok(api.get(filename)?)
    }

    fn iterator() -> Iter<'static, QuantizedModel> {
        static MODELS: [QuantizedModel; 8] = [
            QuantizedModel::Mistral7b,
            QuantizedModel::Mistral7bInstruct,
            QuantizedModel::Mistral7bInstructV02,
            QuantizedModel::Mixtral,
            QuantizedModel::MixtralInstruct,
            QuantizedModel::Zephyr7bAlpha,
            QuantizedModel::Zephyr7bBeta,
            QuantizedModel::DolphinMixtral,
        ];
        MODELS.iter()
    }
}

pub struct Quantized {
    model: ModelWeights,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Quantized {
    /// Creates a new instance of the LLM
    pub fn new(model: QuantizedModel, args: &TextGenerationArgs) -> anyhow::Result<Self> {
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
        let model_path = model.model()?;
        let mut file = File::open(&model_path)?;
        let device = device(true)?;

        let tokenizer: Tokenizer = model.tokenizer()?;

        let model_content =
            gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model_content.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model_content.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        let model = ModelWeights::from_gguf(model_content, &mut file, &device)?;
        println!("model built");

        let logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);

        Ok(Self {
            model,
            device,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty: args.repeat_penalty,
            repeat_last_n: args.repeat_last_n,
        })
    }

    /// Preloads the model files into the cache
    pub fn cache_model() -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        // for model in QuantizedModel::iterator() {
        //     model.model()?;
        //     model.tokenizer()?;
        // }

        let model = QuantizedModel::DolphinMixtral;
        model.model()?;
        model.tokenizer()?;

        println!("retrieved quantized files in {:?}", start.elapsed());
        Ok(())
    }
}

impl TextGeneratorInner for Quantized {
    /// Prompts an already loaded LLM and streams output mpsc Sender
    fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()> {
        self.tokenizer.clear();

        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        let to_sample = sample_len.saturating_sub(1) as usize;
        let tokens = if tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            tokens[tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            tokens
        };

        let mut all_tokens = vec![];

        let mut next_token = {
            let input = Tensor::new(tokens.clone(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            self.logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);
        if let Some(t) = self.tokenizer.next_token(next_token)? {
            let sender_clone = sender.clone();
            task::spawn(async move {
                use std::io::Write;
                print!("{}", &t);
                std::io::stdout().flush().unwrap();
                sender_clone.send(t).await.unwrap();
            });
        }

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find </s> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut generated_tokens = 0usize;
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };
            next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                use std::io::Write;
                std::io::stdout().flush()?;

                let sender_clone = sender.clone();
                task::spawn(async move {
                    sender_clone.send(t).await.unwrap();
                });
            }
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            };
        }

        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            task::spawn(async move {
                sender.send(rest).await.unwrap();
            });
        }

        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / start_gen.elapsed().as_secs_f64(),
        );
        Ok(())
    }
}
