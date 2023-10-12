use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use rand::Rng;

use crate::token_stream::TokenOutputStream;
use crate::utils::device;

enum Model {
    Mistral(Mistral),
    QMistral(QMistral),
}

pub struct TextGenerationArgs {
    pub cpu: bool,
    pub tracing: bool,
    pub use_flash_attn: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub model_id: String,
    pub revision: String,
    pub tokenizer_file: Option<String>,
    pub weight_files: Option<String>,
    pub quantized: bool,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for TextGenerationArgs {
    fn default() -> Self {
        TextGenerationArgs {
            cpu: false,
            tracing: false,
            use_flash_attn: false,
            temperature: Some(0.50),
            top_p: Some(0.95),
            seed: rand::thread_rng().gen(),
            model_id: "lmz/candle-mistral".to_string(),
            revision: "main".to_string(),
            tokenizer_file: None,
            weight_files: None,
            quantized: true,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl Default for TextGeneration {
    fn default() -> Self {
        let args = TextGenerationArgs::default();
        Self::new(args).unwrap()
    }
}

impl TextGeneration {
    pub fn new(args: TextGenerationArgs) -> anyhow::Result<Self> {
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
        let repo = api.repo(Repo::with_revision(
            args.model_id,
            RepoType::Model,
            args.revision,
        ));
        let tokenizer_filename = match args.tokenizer_file {
            Some(file) => std::path::PathBuf::from(file),
            None => repo.get("tokenizer.json")?,
        };
        let filenames = match args.weight_files {
            Some(files) => files
                .split(',')
                .map(std::path::PathBuf::from)
                .collect::<Vec<_>>(),
            None => {
                if args.quantized {
                    vec![repo.get("model-q4k.gguf")?]
                } else {
                    vec![
                        repo.get("pytorch_model-00001-of-00002.safetensors")?,
                        repo.get("pytorch_model-00002-of-00002.safetensors")?,
                    ]
                }
            }
        };
        println!("retrieved the files in {:?}", start.elapsed());
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let start = std::time::Instant::now();
        let config = Config::config_7b_v0_1(args.use_flash_attn);
        let (model, device) = if args.quantized {
            let filename = &filenames[0];
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename)?;
            let model: QMistral = QMistral::new(&config, vb)?;
            (Model::QMistral(model), Device::Cpu)
        } else {
            let device = device(args.cpu)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            let model = Mistral::new(&config, vb)?;
            (Model::Mistral(model), device)
        };
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

    pub fn preload_models(args: TextGenerationArgs) -> anyhow::Result<()> {
        let start: std::time::Instant = std::time::Instant::now();
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            args.model_id,
            RepoType::Model,
            args.revision,
        ));
        if args.tokenizer_file.is_none() {
            repo.get("tokenizer.json")?;
        }
        if args.weight_files.is_none() {
            if args.quantized {
                repo.get("model-q4k.gguf")?;
            } else {
                repo.get("pytorch_model-00001-of-00002.safetensors")?;
                repo.get("pytorch_model-00002-of-00002.safetensors")?;
            }
        }
        println!("retrieved the files in {:?}", start.elapsed());
        Ok(())
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> anyhow::Result<()> {
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
            let logits = match &mut self.model {
                Model::Mistral(m) => m.forward(&input, start_pos)?,
                Model::QMistral(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
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
                print!("{t}");
            }
        }

        let gen_time = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }

        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / gen_time.as_secs_f64(),
        );
        Ok(())
    }
}

// https://github.com/huggingface/candle/blob/main/candle-examples/examples/mistral/main.rs
// https://github.com/huggingface/candle/tree/59ab6d7832600083a1519aa0511e9c7c832ae01c/candle-examples/examples/mistral
// https://docs.rs/hf-hub/0.3.2/hf_hub/api/sync/struct.ApiRepo.html#method.get