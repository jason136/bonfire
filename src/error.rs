use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::fmt::{self, Display, Formatter};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    General(#[from] anyhow::Error),
    Send(#[from] tokio::sync::mpsc::error::SendError<String>),
    StdIO(#[from] std::io::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::General(e) => write!(f, "General Error: {}", e),
            Error::Send(e) => write!(f, "Send Error: {}", e),
            Error::StdIO(e) => write!(f, "StdIO Error: {}", e),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()).into_response()
    }
}

pub type Result<T> = std::result::Result<T, Error>;
