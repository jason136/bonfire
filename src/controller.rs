use actix_web::{
    get,
    web::Path,
    HttpResponse, Responder,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[get("/hello/{id}")]
pub async fn hello_world(path: Path<i32>) -> impl Responder {
    let id: i32 = path.into_inner();

    HttpResponse::Ok().body(format!("hello world v2: {id}"))
}

#[get("/version")]
pub async fn version() -> impl Responder {
    HttpResponse::Ok().body(env!("CARGO_PKG_VERSION"))
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
