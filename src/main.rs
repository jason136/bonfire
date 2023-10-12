use std::sync::Arc;

use actix_web::{middleware::Logger, App, HttpServer, web::Data};

mod controller;
mod broadcast;
mod text_generation;
mod token_stream;
mod utils;

use crate::{controller::*, broadcast::*, text_generation::*};

#[derive(Clone)]
struct AppState {
    broadcaster: Arc<Broadcaster>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    TextGeneration::preload_models(TextGenerationArgs::default()).unwrap();

    TextGeneration::default().run("<s> Write a poem about dogs ", 500).unwrap();

    dotenvy::dotenv().unwrap();

    let app_data = AppState {
        broadcaster: Broadcaster::create(),
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
    })
    .bind((api_addr.clone(), 8080))?
    .run()
    .await
}
