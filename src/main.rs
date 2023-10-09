use actix_web::{middleware::Logger, App, HttpServer};

mod token_stream;
mod text_generation;
mod controller;

use crate::controller::*;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenvy::dotenv().expect("Error Reading Dotenv");

    let api_addr = std::env::var("API_ADDRESS").expect("API_ADDRESS must be set");
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .service(hello_world)
            .service(version)
            .service(mega)
    })
    .bind((api_addr.clone(), 8080))?
    .run()
    .await
}