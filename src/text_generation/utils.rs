use candle_core::{
    bail,
    utils::{cuda_is_available, metal_is_available},
    Device, Error,
};
use futures::future::join_all;
use parking_lot::Mutex;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::{sync::mpsc::Sender, task};

use super::{
    gguf_quantized::{Quantized, QuantizedModel},
    mistral7b::{Mistral7b, Mistral7bModel},
    mixtral::Mixtral,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextGenerationModel {
    Mistral7b,
    Mistral7bInstruct,
    Mistral7bInstructV02,
    Mixtral,
    MixtralInstruct,
    Mistral7bQuantized,
    Mistral7bInstructQuantized,
    Mistral7bInstructV02Quantized,
    MixtralQuantized,
    MixtralInstructQuantized,
    Zephyr7bAlphaQuantized,
    Zephyr7bBetaQuantized,
}

#[derive(Debug, Clone)]
pub struct TextGenerationArgs {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for TextGenerationArgs {
    fn default() -> Self {
        TextGenerationArgs {
            temperature: Some(0.95),
            top_p: Some(0.95),
            seed: rand::thread_rng().gen(),
            repeat_penalty: 1.1,
            repeat_last_n: 128,
        }
    }
}

pub trait TextGeneratorInner: Send + Sync {
    fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()>;
}

#[derive(Clone)]
pub struct TextGenerator(pub Arc<Mutex<dyn TextGeneratorInner>>);
fn wrap(model: impl TextGeneratorInner + 'static) -> anyhow::Result<TextGenerator> {
    Ok(TextGenerator(Arc::new(Mutex::new(model))))
}

impl TextGenerator {
    pub fn new(model: TextGenerationModel, args: &TextGenerationArgs) -> anyhow::Result<Self> {
        match model {
            TextGenerationModel::Mistral7b => {
                wrap(Mistral7b::new(Mistral7bModel::Mistral7b, args)?)
            }
            TextGenerationModel::Mistral7bInstruct => {
                wrap(Mistral7b::new(Mistral7bModel::Mistral7bInstruct, args)?)
            }
            TextGenerationModel::Mistral7bInstructV02 => {
                wrap(Mistral7b::new(Mistral7bModel::Mistral7bInstructV02, args)?)
            }
            TextGenerationModel::Mixtral => wrap(Mixtral::new(args, false)?),
            TextGenerationModel::MixtralInstruct => wrap(Mixtral::new(args, true)?),
            TextGenerationModel::Mistral7bQuantized => {
                wrap(Quantized::new(QuantizedModel::Mistral7b, args)?)
            }
            TextGenerationModel::Mistral7bInstructQuantized => {
                wrap(Quantized::new(QuantizedModel::Mistral7bInstruct, args)?)
            }
            TextGenerationModel::Mistral7bInstructV02Quantized => {
                wrap(Quantized::new(QuantizedModel::Mistral7bInstructV02, args)?)
            }
            TextGenerationModel::MixtralQuantized => {
                wrap(Quantized::new(QuantizedModel::Mixtral, args)?)
            }
            TextGenerationModel::MixtralInstructQuantized => {
                wrap(Quantized::new(QuantizedModel::MixtralInstruct, args)?)
            }
            TextGenerationModel::Zephyr7bAlphaQuantized => {
                wrap(Quantized::new(QuantizedModel::Zephyr7bAlpha, args)?)
            }
            TextGenerationModel::Zephyr7bBetaQuantized => {
                wrap(Quantized::new(QuantizedModel::Zephyr7bBeta, args)?)
            }
        }
    }

    pub fn default(model: TextGenerationModel) -> anyhow::Result<Self> {
        Self::new(model, &TextGenerationArgs::default())
    }

    pub async fn cache_models() {
        join_all(vec![
            // task::spawn(async {
            //     Mistral7b::cache_model().unwrap();
            // }),
            // task::spawn(async {
            //     Mixtral::cache_model().unwrap();
            // }),
            task::spawn(async {
                Quantized::cache_model().unwrap();
            }),
        ])
        .await;
    }

    pub fn run(
        &mut self,
        prompt: &str,
        sample_len: u32,
        sender: Sender<String>,
    ) -> anyhow::Result<()> {
        self.0.lock().run(prompt, sample_len, sender)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPrompt {
    pub model: TextGenerationModel,
    pub prompt: String,
    pub sample_len: u32,
}

pub fn device(cpu: bool) -> candle_core::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle_core::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file).map_err(Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(Error::wrap))
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

pub fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}
