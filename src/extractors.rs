use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
};
use uuid::Uuid;

pub struct StreamingClientId(pub Uuid);

#[async_trait]
impl<S> FromRequestParts<S> for StreamingClientId
where
    S: 'static + Send,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let token = match parts.headers.get("Authorization") {
            Some(header) => {
                let string = header.to_str().unwrap();
                let split = string.split(' ').collect::<Vec<&str>>();
                if let Some(token) = split.get(1) {
                    token.to_string()
                } else {
                    return Err((StatusCode::UNAUTHORIZED, "No Auth Token"));
                }
            }
            None => return Err((StatusCode::UNAUTHORIZED, "Invalid Auth Header")),
        };

        if let Ok(id) = token.parse::<Uuid>() {
            Ok(StreamingClientId(id))
        } else {
            Err((StatusCode::UNAUTHORIZED, "Invalid Id Token"))
        }
    }
}
