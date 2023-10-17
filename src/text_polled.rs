use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use crossbeam::channel::{bounded, Sender, Receiver};
use futures::lock::Mutex;
use tokio::{
    task,
    time::interval,
};
use uuid::Uuid;

use crate::{error, text_generation::TextGeneration};

type TextPolledMessages = Arc<Mutex<HashMap<Uuid, Option<PolledMessage>>>>;

pub struct TextPolledController {
    pub messages: TextPolledMessages,
}

#[derive(Debug)]
pub struct PolledMessage {
    pub text: String,
    pub generated_at: SystemTime,
}

pub enum PolledMessageState {
    Available(String),
    Generating,
    Missing,
}

impl Default for TextPolledController {
    /// constructs new TextPolledController and spawns cleanup task
    fn default() -> Self {
        let blob_controller = TextPolledController {
            messages: Arc::new(Mutex::new(HashMap::new())),
        };

        TextPolledController::spawn_cleanup(blob_controller.messages.clone());

        blob_controller
    }
}

impl TextPolledController {
    fn spawn_cleanup(inner: TextPolledMessages) {
        task::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;
                TextPolledController::remove_expired_messages(inner.clone()).await;
            }
        });
    }

    async fn remove_expired_messages(messages: TextPolledMessages) {
        messages.lock().await.retain(|_, message| {
            if let Some(message) = message {
                message.generated_at.elapsed().unwrap().as_secs() < 60 * 10
            } else {
                true
            }
        });
    }

    pub async fn prompt(&self, id: Uuid, prompt: String, sample_len: u32) -> error::Result<()> {
        let messages_clone = self.messages.clone();

        let (sync_tx, sync_rx): (Sender<String>, Receiver<String>) = bounded(sample_len as usize);
        
        let handle = task::spawn_blocking(move || {
            TextGeneration::default().run(&prompt, sample_len, sync_tx).unwrap();
            println!("done generating");
        });

        task::spawn(async move {
            messages_clone.lock().await.insert(id, None);
            handle.await.unwrap();
            
            let text = sync_rx.try_iter().collect();

            println!("text: {:?}", text);
            
            let generated_message = PolledMessage {
                text,
                generated_at: SystemTime::now(),
            };

            *messages_clone.lock().await.get_mut(&id).unwrap() = Some(generated_message);
            println!("id: {:?}", id);
        });

        Ok(())
    }

    pub async fn get_message(&self, id: &Uuid) -> PolledMessageState {
        match self.messages.lock().await.get(id) {
            Some(Some(message)) => {
                PolledMessageState::Available(message.text.clone())
            },
            Some(None) => PolledMessageState::Generating,
            None => PolledMessageState::Missing,
        }
    }
}
