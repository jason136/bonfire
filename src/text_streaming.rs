use actix_web_lab::sse;
use futures::{future::join_all, lock::Mutex};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    sync::mpsc::{Receiver, Sender},
    task,
    time::interval,
};
use uuid::Uuid;

use crate::{
    error,
    text_generation::{TextGeneration, TextGenerationArgs},
};

type TextStreamingClients = Arc<Mutex<HashMap<Uuid, StreamingClient>>>;

pub struct TextStreamingController {
    pub clients: TextStreamingClients,
}

#[derive(Clone)]
pub struct StreamingClient {
    sse_sender: sse::Sender,
    stream_input: Sender<String>,
    pipe_task: Arc<task::JoinHandle<()>>,
    message_history: Arc<Mutex<Vec<String>>>,
    model_args: TextGenerationArgs,
    model: Arc<Mutex<TextGeneration>>,
}

impl Default for TextStreamingController {
    /// constructs new TextStreamingController and spawns ping task
    fn default() -> Self {
        let streaming_controller = TextStreamingController {
            clients: Arc::new(Mutex::new(HashMap::new())),
        };

        TextStreamingController::spawn_ping(streaming_controller.clients.clone());

        streaming_controller
    }
}

impl TextStreamingController {
    /// pings clients every 10 seconds to see if they are alive and remove them from the client list if not.
    fn spawn_ping(clients: TextStreamingClients) {
        task::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;
                TextStreamingController::remove_stale_clients(clients.clone()).await;
            }
        });
    }

    /// removes all non-responsive clients from client list
    async fn remove_stale_clients(clients: TextStreamingClients) {
        let mut clients_lock = clients.lock().await;

        let futures = clients_lock.iter().map(|(id, client)| async {
            if client
                .sse_sender
                .send(sse::Event::Comment("ping".into()))
                .await
                .is_ok()
            {
                Some(*id)
            } else {
                client.pipe_task.abort();
                None
            }
        });

        let ok_client_ids: Vec<Uuid> = join_all(futures).await.into_iter().flatten().collect();

        clients_lock.retain(|k, _| ok_client_ids.contains(k));
    }
}

impl StreamingClient {
    pub async fn new(model_args: TextGenerationArgs) -> error::Result<Self> {
        let (tx, _) = sse::channel(10);
        tx.send(sse::Data::new("connected")).await?;

        let message_history: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let (stream_input, mut stream_output): (Sender<String>, Receiver<String>) =
            tokio::sync::mpsc::channel(10);

        let message_history_clone = message_history.clone();
        let tx_clone = tx.clone();
        let pipe_task = Arc::new({
            task::spawn(async move {
                while let Some(msg) = stream_output.recv().await {
                    message_history_clone.lock().await.push(msg.clone());
                    tx_clone.send(sse::Data::new(msg)).await.unwrap();
                }
            })
        });

        let model = Arc::new(Mutex::new(TextGeneration::new(&model_args)?));

        Ok(StreamingClient {
            sse_sender: tx.clone(),
            stream_input,
            pipe_task,
            message_history,
            model_args,
            model,
        })
    }

    /// refresh model, needs to be run after every prompt
    pub async fn refresh_model(&mut self) -> error::Result<()> {
        self.model = Arc::new(Mutex::new(TextGeneration::new(&self.model_args)?));
        Ok(())
    }

    /// prompt the underlying model with message history, piping the results to the client
    pub async fn prompt(&mut self, prompt: &str, sample_len: u32) -> error::Result<()> {
        let full_prompt = self.message_history.lock().await.concat() + " " + prompt;

        let sender = self.stream_input.clone();
        self.model
            .lock()
            .await
            .run(&full_prompt, sample_len, sender)
            .await?;

        self.refresh_model().await
    }
}
