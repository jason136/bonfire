use futures::lock::Mutex;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{
    sync::mpsc::{channel, Receiver, Sender},
    task,
    time::interval,
};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use uuid::Uuid;

use crate::{
    error,
    text_generation::utils::{TextGenerator, TextPrompt},
};

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
    fn default() -> Self {
        let polled_controller = TextPolledController {
            messages: Arc::new(Mutex::new(HashMap::new())),
        };

        TextPolledController::spawn_cleanup(polled_controller.messages.clone());

        polled_controller
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

    pub async fn prompt(&self, id: Uuid, prompt: TextPrompt) -> error::Result<()> {
        let (tx, rx): (Sender<String>, Receiver<String>) = channel(prompt.sample_len as usize);
        let handle = task::spawn_blocking(move || {
            TextGenerator::default(prompt.model)
                .unwrap()
                .run(&prompt.prompt, prompt.sample_len, tx)
                .unwrap();
            println!("done generating");
        });

        let messages_clone = self.messages.clone();
        task::spawn(async move {
            messages_clone.lock().await.insert(id, None);
            handle.await.unwrap();

            let text: String = ReceiverStream::new(rx).collect().await;

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
            Some(Some(message)) => PolledMessageState::Available(message.text.clone()),
            Some(None) => PolledMessageState::Generating,
            None => PolledMessageState::Missing,
        }
    }
}
