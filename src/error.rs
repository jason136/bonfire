use std::fmt::{self, Display, Formatter};

use actix_web::{
    http::{header::ContentType, StatusCode},
    HttpResponse, ResponseError,
};
use thiserror::Error;

// general error type for the application.
#[derive(Error, Debug)]
pub enum Error {
    General(#[from] anyhow::Error),
    Sse(#[from] actix_web_lab::sse::SendError),
    StdIO(#[from] std::io::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::General(e) => write!(f, "General Error: {}", e),
            Error::Sse(e) => write!(f, "Sse Error: {}", e),
            Error::StdIO(e) => write!(f, "StdIO Error: {}", e),
        }
    }
}

// impl err return type for all endpoints
impl ResponseError for Error {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code())
            .insert_header(ContentType::plaintext())
            .body(self.to_string())
    }

    fn status_code(&self) -> StatusCode {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

pub type Result<T> = std::result::Result<T, Error>;
pub type Response = Result<HttpResponse>;
