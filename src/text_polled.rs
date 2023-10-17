use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use futures::lock::Mutex;
use tokio::{
    sync::mpsc::{channel, Receiver, Sender},
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
        let mut messages_lock = messages.lock().await;

        println!("items before cleaning: {}", messages_lock.len());

        messages_lock.retain(|_, message| {
            if let Some(message) = message {
                message.generated_at.elapsed().unwrap().as_secs() < 60 * 10
            } else {
                true
            }
        });

        println!("items after cleaning: {}", messages_lock.len());
        println!("messages: {:?}", *messages_lock);
    }

    pub async fn prompt(&self, id: Uuid, prompt: String, sample_len: u32) -> error::Result<()> {
        self.messages.lock().await.insert(id, None);

        let messages_clone = self.messages.clone();
        task::spawn(async move {
            let (tx, mut rx): (Sender<String>, Receiver<String>) = channel(sample_len as usize);

            TextGeneration::default()
                .run(&prompt, sample_len, tx)
                .await
                .unwrap();

            let mut text = String::new();
            while let Some(token) = rx.recv().await {
                text.push_str(&token);
            }

            let generated_message = PolledMessage {
                text,
                generated_at: SystemTime::now(),
            };

            println!("messages: {:?}", messages_clone.lock().await);
            println!("id: {:?}", id);
            *messages_clone.lock().await.get_mut(&id).unwrap() = Some(generated_message);
        });

        Ok(())
    }

    pub async fn get_message(&self, id: &Uuid) -> PolledMessageState {
        match self.messages.lock().await.remove(id) {
            Some(Some(message)) => PolledMessageState::Available(message.text.clone()),
            Some(None) => PolledMessageState::Generating,
            None => PolledMessageState::Missing,
        }
    }
}
