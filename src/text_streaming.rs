use axum::response::{sse::Event, Sse};
use futures::lock::Mutex;
use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
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
    sender: Sender<String>,
    // sse_stream: Arc<Sse<ReceiverStream<Result<Event, Infallible>>>>,
    message_history: Arc<Mutex<Vec<String>>>,
    model_args: TextGenerationArgs,
    model: Arc<parking_lot::Mutex<TextGeneration>>,
}

impl Default for TextStreamingController {
    /// constructs new TextStreamingController and spawns ping task
    fn default() -> Self {
        TextStreamingController {
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl StreamingClient {
    pub async fn new(model_args: TextGenerationArgs) -> error::Result<Self> {
        let (tx, rx): (Sender<String>, Receiver<String>) = channel(10);

        let stream = ReceiverStream::new(rx);

        let event_stream = stream.map(|x| {
            let res: Result<Event, Infallible> = Ok(Event::default().data(x));
            res
        });

        Arc::new(
            Sse::new(event_stream).keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(Duration::from_secs(1))
                    .text("keep-alive"),
            ),
        );

        tx.send("hello".to_string()).await?;

        let message_history = Arc::new(Mutex::new(Vec::new()));
        let model = Arc::new(parking_lot::Mutex::new(TextGeneration::new(&model_args)?));

        Ok(StreamingClient {
            sender: tx.clone(),
            // sse_stream,
            message_history,
            model_args,
            model,
        })
    }

    /// refresh model, needs to be run after every prompt
    pub async fn refresh_model(&mut self) -> error::Result<()> {
        self.model = Arc::new(parking_lot::Mutex::new(TextGeneration::new(
            &self.model_args,
        )?));
        Ok(())
    }

    /// prompt the underlying model with message history, piping the results to the client
    pub async fn prompt(&mut self, prompt: &str, sample_len: u32) -> error::Result<()> {
        let full_prompt = self.message_history.lock().await.concat() + " " + prompt;

        self.model
            .lock()
            .run(&full_prompt, sample_len, self.sender.clone())?;

        self.refresh_model().await
    }
}
