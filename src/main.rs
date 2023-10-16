use std::sync::Arc;

use actix_web::{middleware::Logger, App, HttpServer, web::Data};

mod error;
mod extractors;
mod controller;
mod streaming;
mod text_generation;
mod token_stream;
mod utils;

use crate::{controller::*, streaming::*, text_generation::*};

#[derive(Clone)]
struct AppState {
    pub streaming_controller: Arc<StreamingController>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    TextGeneration::preload_models(TextGenerationArgs::default()).unwrap();

    // TextGeneration::default().run("<s> Write a poem about dogs ", 500).unwrap();

    dotenvy::dotenv().unwrap();

    let app_data = AppState {
        streaming_controller: StreamingController::create(),
    };

    let api_addr = std::env::var("API_ADDRESS").expect("API_ADDRESS must be set");
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    HttpServer::new(move || {
        App::new()
            .app_data(Data::new(app_data.clone()))
            .wrap(Logger::default())
            .service(hello_world)
            .service(version)
            .service(mega)
            .service(connect)
            .service(prompt)
    })
    .bind((api_addr.clone(), 8080))?
    .run()
    .await
}