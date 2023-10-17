use std::sync::Arc;

use actix_web::{middleware::Logger, web::Data, App, HttpServer};
use text_polled::TextPolledController;

mod controller;
mod error;
mod extractors;
mod text_polled;
mod text_generation;
mod text_streaming;
mod token_stream;
mod utils;

use crate::{controller::*, text_generation::*, text_streaming::*};

#[derive(Clone)]
pub struct AppState {
    pub text_streaming_controller: Arc<TextStreamingController>,
    pub text_blob_controller: Arc<TextPolledController>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    TextGeneration::preload_models(TextGenerationArgs::default()).unwrap();

    // TextGeneration::default().run("<s> Write a poem about dogs ", 500).unwrap();

    dotenvy::dotenv().unwrap();

    let app_data = AppState {
        text_streaming_controller: Arc::new(TextStreamingController::default()),
        text_blob_controller: Arc::new(TextPolledController::default()),
    };

    let api_addr = std::env::var("API_ADDRESS").expect("API_ADDRESS must be set");
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(app_data.clone()))
            .wrap(Logger::default())
            .service(hello_world)
            .service(version)
            .service(new_streaming)
            .service(prompt_streaming)
            .service(prompt_blob)
            .service(get_blob)
    })
    .bind((api_addr.clone(), 8080))?
    .run()
    .await
}
