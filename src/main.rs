use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use text_generation::utils::TextGenerator;

mod controller;
mod error;
mod extractors;
mod text_polled;
mod text_streaming;
mod text_generation {
    pub mod mistral7b;
    pub mod mixtral8x7b;
    pub mod token_stream;
    pub mod utils;
}

use crate::{controller::*, text_polled::*, text_streaming::*};

#[derive(Clone)]
pub struct AppState {
    pub text_streaming_controller: Arc<TextStreamingController>,
    pub text_polled_controller: Arc<TextPolledController>,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().expect("Unable to read .env");
    TextGenerator::preload_models().await;

    let app_state = AppState {
        text_streaming_controller: Arc::new(TextStreamingController::default()),
        text_polled_controller: Arc::new(TextPolledController::default()),
    };

    let api_addr = std::env::var("API_ADDRESS").expect("API_ADDRESS must be set");
    let api_port = if let Ok(port) = std::env::var("API_PORT") {
        port
    } else {
        "8080".to_string()
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let app = Router::new()
        .route("/hello/:id", get(hello_world))
        .route("/version", get(version))
        .route("/new_streaming", get(new_streaming))
        .route("/prompt_streaming/:id", post(prompt_streaming))
        .route("/prompt_polled", post(prompt_polled_text))
        .route("/poll_text/:id", get(poll_text))
        .with_state(app_state);

    axum::Server::bind(&format!("{}:{}", api_addr, api_port).parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
