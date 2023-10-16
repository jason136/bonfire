use std::{sync::Arc, time::Duration, collections::HashMap};
use actix_web_lab::sse;
use futures::{future::join_all, lock::Mutex};
use actix_web::rt::time::interval;
use tokio::{task, sync::mpsc::{Sender, Receiver}};

use crate::{error, text_generation::{TextGeneration, TextGenerationArgs}};

pub struct StreamingController {
    pub clients: Arc<Mutex<HashMap<String, StreamingClient>>>,
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

impl StreamingController {
    /// constructs new StreamingController and spawns ping loop.
    pub fn create() -> Arc<Self> {
        let streaming_controller = Arc::new(StreamingController {
            clients: Arc::new(Mutex::new(HashMap::new())),
        });

        StreamingController::spawn_ping(streaming_controller.clients.clone());

        streaming_controller
    }

    /// pings clients every 10 seconds to see if they are alive and remove them from the client list if not.
    fn spawn_ping(inner: Arc<Mutex<HashMap<String, StreamingClient>>>) {
        task::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));

            loop {
                interval.tick().await;
                StreamingController::remove_stale_clients(inner.clone()).await;
            }
        });
    }

    /// removes all non-responsive clients from client list
    async fn remove_stale_clients(inner: Arc<Mutex<HashMap<String, StreamingClient>>>) {
        let mut clients = inner.lock().await.clone();

        let futures = clients.iter().map(|(id, client)| async {
            if client.sse_sender.send(sse::Event::Comment("ping".into())).await.is_ok() {
                Some(id.clone())
            } else {
                client.pipe_task.abort();
                None
            }
        });

        let ok_client_ids: Vec<String> = join_all(futures).await.into_iter().flatten().collect();

        clients.retain(|k, _| {
            ok_client_ids.contains(k)
        });

        *inner.lock().await = clients;
    }
}

impl StreamingClient {
    pub async fn new(model_args: TextGenerationArgs) -> error::Result<Self> {
        let (tx, _) = sse::channel(10);
        tx.send(sse::Data::new("connected")).await?;

        let message_history: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));        
        let (stream_input, mut stream_output): (Sender<String>, Receiver<String>) = tokio::sync::mpsc::channel(10);

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
        self.model.lock().await.run(&full_prompt, sample_len, sender).await?;

        self.refresh_model().await
    }
}