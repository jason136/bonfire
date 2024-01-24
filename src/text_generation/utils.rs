use candle_core::{
    bail,
    utils::{cuda_is_available, metal_is_available},
    Device, Error,
};
use futures::future::join_all;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::{sync::mpsc::Sender, task};

use super::{
    gguf_quantized::Quantized,
    mistral7b::{Mistral7b, Mistral7bArgs},
    mixtral8x7b::{Mixtral, MixtralArgs},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextGenerationModel {
    Mistral7b,
    Mistral7bQuantized,
    Mixtral,
    MixtralInstruct,
}

#[derive(Debug, Clone)]
pub enum TextGenerationArgs {
    Mistral7b(Mistral7bArgs),
    Mistral7bQuantized(Mistral7bArgs),
    Mixtral(MixtralArgs),
    MixtralInstruct(MixtralArgs),
}

#[derive(Clone)]
pub struct TextGenerator(Arc<Mutex<dyn TextGeneratorInner>>);

impl TextGenerator {
    pub fn new(args: &TextGenerationArgs) -> anyhow::Result<Self> {
        Ok(match args {
            TextGenerationArgs::Mistral7b(args) => {
                let model = Mistral7b::new(args, false)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationArgs::Mistral7bQuantized(args) => {
                let model = Mistral7b::new(args, true)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationArgs::Mixtral(args) => {
                let model = Mixtral::new(args, false)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationArgs::MixtralInstruct(args) => {
                let model = Mixtral::new(args, true)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
        })
    }

    pub async fn preload_models() {
        join_all(vec![
            // task::spawn(async {
            //     Mistral7b::preload_model().unwrap();
            // }),
            // task::spawn(async {
            //     Mixtral::preload_model().unwrap();
            // }),
            task::spawn(async {
                Quantized::preload_models().unwrap();
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

    pub fn model_default(model: TextGenerationModel) -> anyhow::Result<Self> {
        Ok(match model {
            TextGenerationModel::Mistral7b => {
                let args = Mistral7bArgs::default();
                let model = Mistral7b::new(&args, false)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationModel::Mistral7bQuantized => {
                let args = Mistral7bArgs::default();
                let model = Mistral7b::new(&args, true)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationModel::Mixtral => {
                let args = MixtralArgs::default();
                let model = Mixtral::new(&args, false)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
            TextGenerationModel::MixtralInstruct => {
                let args = MixtralArgs::default();
                let model = Mixtral::new(&args, true)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            }
        })
    }
}

pub trait TextGeneratorInner: Send + Sync {
    fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPolledPrompt {
    pub model: TextGenerationModel,
    pub prompt: String,
    pub sample_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStreamedPrompt {
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
