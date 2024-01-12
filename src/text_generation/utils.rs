use std::sync::Arc;

use futures::future::join_all;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use tokio::{sync::mpsc::Sender, task};

use super::mistral7b::{Mistral7BArgs, Mistral7B};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextGenerationModel {
    Mistral7B,
}

#[derive(Debug, Clone)]
pub enum TextGenerationArgs {
    Mistral7B(Mistral7BArgs),
}

#[derive(Clone)]
pub struct TextGenerator(Arc<Mutex<dyn TextGeneratorInner>>);

impl TextGenerator {
    pub fn new(args: &TextGenerationArgs) -> anyhow::Result<Self> {
        Ok(match args {
            TextGenerationArgs::Mistral7B(args) => {
                let model = Mistral7B::new(args)?;
                TextGenerator(Arc::new(Mutex::new(model)))
            },
        })
    }

    pub async fn preload_models() {
        join_all(vec![
            task::spawn(async {
                Mistral7B::preload_model().unwrap();
            }),
        ]).await;
    }

    pub fn run(&mut self, prompt: &str, sample_len: u32, sender: Sender<String>) -> anyhow::Result<()> {
        self.0.lock().run(prompt, sample_len, sender)
    }

    pub fn model_default(model: TextGenerationModel) -> anyhow::Result<Self> {
        Ok(match model {
            TextGenerationModel::Mistral7B => {
                let args = Mistral7BArgs::default();
                let model = Mistral7B::new(&args)?;
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
