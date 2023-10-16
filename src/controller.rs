use actix_web::{get, web::{Path, Data, Json}, HttpResponse, Responder, post};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rand::Rng;

use crate::{streaming::StreamingClient, AppState, error::{Response, self}, text_generation::{TextGenerationArgs, TextGenerationPrompt}, extractors::UserIp};

#[get("/hello/{id}")]
pub async fn hello_world(path: Path<i32>) -> Response {
    let id: i32 = path.into_inner();

    Ok(HttpResponse::Ok().body(format!("hello world v2: {id}")))
}

#[get("/version")]
pub async fn version() -> Response {
    Ok(HttpResponse::Ok().body(env!("CARGO_PKG_VERSION")))
}

#[get("/mb")]
pub async fn mega() -> impl Responder {
    const DESIRED_SIZE: usize = 5_000_000;
    let mut random_bytes = Vec::with_capacity(DESIRED_SIZE);
    let mut rng = rand::thread_rng();

    while random_bytes.len() < DESIRED_SIZE {
        let random_byte: u8 = rng.gen();
        random_bytes.push(random_byte);
    }

    let random_string: String = random_bytes.par_iter().map(|&byte| byte as char).collect();

    HttpResponse::Ok().body(random_string)
}

#[get("/connect")]
async fn connect(user_ip: UserIp, state: Data<AppState>) -> Response {
    let client = StreamingClient::new(TextGenerationArgs::default()).await?;
    state.streaming_controller.clients.lock().await.insert(user_ip.0, client);

    Ok(HttpResponse::Ok().body("Firing up the model..."))
}

#[post("/prompt")]
async fn prompt(
    user_ip: UserIp,
    state: Data<AppState>,
    body: Json<TextGenerationPrompt>,
) -> Response {

    let mut clients = state.streaming_controller.clients.lock().await;
    let client = clients.get_mut(&user_ip.0).ok_or(error::Error::ClientTimedOut)?;
    
    client.prompt(&body.prompt, body.sample_len).await?;
    Ok(HttpResponse::Ok().body("Done Prompting!"))
}