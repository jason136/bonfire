use std::sync::Arc;

use candle_core::{Device, utils::{cuda_is_available, metal_is_available}, Error, bail};
use futures::future::join_all;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use tokio::{sync::mpsc::Sender, task};

use super::{mistral7b::{Mistral7bArgs, Mistral7b}, mixtral8x7b::{Mixtral8x7bArgs, Mixtral8x7b}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextGenerationModel {
    Mistral7b,
    Mixtral8x7b,
}

#[derive(Debug, Clone)]
pub enum TextGenerationArgs {
    Mistral7b(Mistral7bArgs),
    Mixtral8x7b(Mixtral8x7bArgs),
}

#[derive(Clone)]
pub struct TextGenerator(Arc<Mutex<dyn TextGeneratorInner>>);

impl TextGenerator {
    pub fn new(args: &TextGenerationArgs) -> anyhow::Result<Self> {
        Ok(match args {
            TextGenerationArgs::Mistral7b(args) => {
                let model = Mistral7b::new(args)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            },
            TextGenerationArgs::Mixtral8x7b(args) => {
                let model = Mixtral8x7b::new(args)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            },
        })
    }

    pub async fn preload_models() {
        join_all(vec![
            task::spawn(async {
                Mistral7b::preload_model().unwrap();
            }),
            task::spawn(async {
                Mixtral8x7b::preload_model().unwrap();
            }),
        ]).await;
    }

    pub fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()> {
        self.0.lock().run(prompt, sample_len, sender)
    }

    pub fn model_default(model: TextGenerationModel) -> anyhow::Result<Self> {
        Ok(match model {
            TextGenerationModel::Mistral7b => {
                let args = Mistral7bArgs::default();
                let model = Mistral7b::new(&args)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            },
            TextGenerationModel::Mixtral8x7b => {
                let args = Mixtral8x7bArgs::default();
                let model = Mixtral8x7b::new(&args)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            },
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
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(Error::wrap)?;
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
