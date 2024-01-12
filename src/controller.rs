use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tokio::task;
use uuid::Uuid;

use crate::{
    error::Result,
    text_polled::PolledMessageState,
    text_streaming::StreamingClient,
    AppState, text_generation::{utils::{TextGenerationArgs, TextPolledPrompt, TextStreamedPrompt}, mistral7b::Mistral7BArgs},
};

pub async fn hello_world(Path(id): Path<i32>) -> Result<Response> {
    Ok((StatusCode::OK, format!("hello world: {id}")).into_response())
}

pub async fn version() -> Result<Response> {
    Ok((StatusCode::OK, env!("CARGO_PKG_VERSION")).into_response())
}

pub async fn new_streaming(State(state): State<AppState>) -> Result<Response> {
    let client = StreamingClient::new(TextGenerationArgs::Mistral7B(Mistral7BArgs::default())).await?;
    let user_id = Uuid::new_v4();
    state
        .text_streaming_controller
        .clients
        .lock()
        .await
        .insert(user_id, client);

    Ok((StatusCode::OK, user_id.to_string()).into_response())
}

pub async fn prompt_streaming(
    Path(id): Path<String>,
    State(state): State<AppState>,
    Json(prompt): Json<TextStreamedPrompt>,
) -> Result<Response> {
    let user_id: Uuid = match id.parse() {
        Ok(id) => id,
        Err(_) => return Ok((StatusCode::BAD_REQUEST, "Invalid Id").into_response()),
    };

    let state_cloned = state.clone();
    if !state
        .text_streaming_controller
        .clients
        .lock()
        .await
        .contains_key(&user_id)
    {
        return Ok((StatusCode::BAD_REQUEST, "Client Not Found").into_response());
    }

    task::spawn(async move {
        let mut clients = state_cloned.text_streaming_controller.clients.lock().await;
        if let Some(client) = clients.get_mut(&user_id) {
            client.prompt(prompt).await.unwrap();
        }
    });

    Ok((StatusCode::OK, "Prompting Begun...").into_response())
}

pub async fn prompt_polled_text(
    State(state): State<AppState>,
    Json(prompt): Json<TextPolledPrompt>,
) -> Result<Response> {
    let id = Uuid::new_v4();

    state
        .text_polled_controller
        .prompt(id, prompt)
        .await
        .unwrap();

    Ok((StatusCode::OK, id.to_string()).into_response())
}

pub async fn poll_text(Path(id): Path<String>, State(state): State<AppState>) -> Result<Response> {
    match id.parse() {
        Ok(message_id) => match state.text_polled_controller.get_message(&message_id).await {
            PolledMessageState::Available(message) => Ok((StatusCode::OK, message).into_response()),
            PolledMessageState::Generating => Ok((StatusCode::OK, "Generating").into_response()),
            PolledMessageState::Missing => {
                Ok((StatusCode::BAD_REQUEST, "Message not found").into_response())
            }
        },
        Err(_) => Ok((StatusCode::BAD_REQUEST, "Invalid Id").into_response()),
    }
}
