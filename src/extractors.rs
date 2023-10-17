// use actix_web::{FromRequest, HttpRequest, dev::Payload, error::ErrorUnauthorized};
// use futures::future::{Ready, ready};
// use uuid::Uuid;

// pub struct StreamingClientId(pub Uuid);

// impl FromRequest for StreamingClientId {
//     type Error = actix_web::Error;
//     type Future = Ready<Result<Self, Self::Error>>;

//     fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
//         let token = match req.headers().get("Authorization") {
//             Some(header) => {
//                 let string = header.to_str().unwrap();
//                 let split = string.split(' ').collect::<Vec<&str>>();
//                 if let Some(token) = split.get(1) {
//                     token.to_string()
//                 } else {
//                     return ready(Err(ErrorUnauthorized("No Id Token")));
//                 }
//             }
//             None => return ready(Err(ErrorUnauthorized("Invalid Id Header"))),
//         };

//         if let Ok(id) = token.parse::<Uuid>() {
//             ready(Ok(StreamingClientId(id)))
//         } else {
//             ready(Err(ErrorUnauthorized("Invalid Id Token")))
//         }
//     }
// }
