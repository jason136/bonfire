use actix_web::{
    get, post,
    web::{Data, Json, Path},
    HttpResponse,
};
use tokio::task;
use uuid::Uuid;

use crate::{
    error::Response,
    text_generation::{TextGenerationArgs, TextGenerationPrompt},
    text_streaming::StreamingClient,
    AppState, text_polled::PolledMessageState,
};

#[get("/hello/{id}")]
pub async fn hello_world(path: Path<i32>) -> Response {
    let id: i32 = path.into_inner();

    Ok(HttpResponse::Ok().body(format!("hello world: {id}")))
}

#[get("/version")]
pub async fn version() -> Response {
    Ok(HttpResponse::Ok().body(env!("CARGO_PKG_VERSION")))
}

#[get("/new_streaming")]
pub async fn new_streaming(state: Data<AppState>) -> Response {
    let client = StreamingClient::new(TextGenerationArgs::default()).await?;
    let user_id = Uuid::new_v4();
    state
        .text_streaming_controller
        .clients
        .lock()
        .await
        .insert(user_id, client);

    Ok(HttpResponse::Ok().body(user_id.to_string()))
}

#[post("/prompt_streaming/{id}")]
pub async fn prompt_streaming(
    id: Path<String>,
    state: Data<AppState>,
    body: Json<TextGenerationPrompt>,
) -> Response {
    let user_id: Uuid = match id.into_inner().parse() {
        Ok(id) => id,
        Err(_) => return Ok(HttpResponse::BadRequest().body("Invalid Id")),
    };

    let state_cloned = state.clone();
    if !state.text_streaming_controller.clients.lock().await.contains_key(&user_id) {
        return Ok(HttpResponse::BadRequest().body("Client Not Found"));
    }

    task::spawn(async move {
        let mut clients = state_cloned.text_streaming_controller.clients.lock().await;
        if let Some(client) = clients
            .get_mut(&user_id) {
                client.prompt(&body.prompt, body.sample_len).await.unwrap();
            }
    });

    Ok(HttpResponse::Ok().body("Prompting Begun..."))
}

#[post("/prompt_blob")]
pub async fn prompt_blob(state: Data<AppState>, body: Json<TextGenerationPrompt>) -> Response {
    let id = Uuid::new_v4();

    task::spawn(async move {
        state
            .text_blob_controller
            .prompt(id, body.prompt.clone(), body.sample_len)
            .await
            .unwrap();
    });

    Ok(HttpResponse::Ok().body(id.to_string()))
}

#[get("/get_blob/{id}")]
pub async fn get_blob(id: Path<String>, state: Data<AppState>) -> Response {
    match id.into_inner().parse() {
        Ok(message_id) => {
            match state.text_blob_controller.get_message(&message_id).await {
                PolledMessageState::Available(message) => Ok(HttpResponse::Ok().body(message)),
                PolledMessageState::Generating => Ok(HttpResponse::Ok().body("Generating")),
                PolledMessageState::Missing => Ok(HttpResponse::BadRequest().body("Message not found")),
            }
        },
        Err(_) => Ok(HttpResponse::BadRequest().body("Invalid Id")),
    }
}