use axum::response::{sse::Event, Sse};
use futures::lock::Mutex;
use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use uuid::Uuid;

use crate::{
    error,
    text_generation::utils::{TextGenerationArgs, TextGenerationModel, TextGenerator, TextPrompt},
};

type TextStreamingClients = Arc<Mutex<HashMap<Uuid, StreamingClient>>>;

pub struct TextStreamingController {
    pub clients: TextStreamingClients,
}

#[derive(Clone)]
pub struct StreamingClient {
    model: TextGenerationModel,
    args: TextGenerationArgs,
    sender: Sender<String>,
    // sse_stream: Arc<Sse<ReceiverStream<Result<Event, Infallible>>>>,
    message_history: Arc<Mutex<Vec<String>>>,
}

impl Default for TextStreamingController {
    fn default() -> Self {
        TextStreamingController {
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl StreamingClient {
    pub async fn new(model: TextGenerationModel, args: TextGenerationArgs) -> error::Result<Self> {
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

        Ok(StreamingClient {
            model,
            args,
            sender: tx.clone(),
            // sse_stream,
            message_history,
        })
    }

    /// Prompt the underlying model with message history, piping the results to the client
    pub async fn prompt(&mut self, prompt: TextPrompt) -> error::Result<()> {
        let full_prompt = self.message_history.lock().await.concat() + " " + &prompt.prompt;
        let mut model = TextGenerator::new(self.model.clone(), &self.args)?;

        model.run(&full_prompt, prompt.sample_len, self.sender.clone())?;

        Ok(())
    }
}
