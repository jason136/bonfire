use actix_web::{FromRequest, HttpRequest, dev::Payload, error::ErrorUnauthorized};
use futures::future::{Ready, ready};

pub struct UserIp(pub String);

impl FromRequest for UserIp {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut Payload) -> Self::Future {
        let info = req.connection_info();
    
        if let Some(ip) = info.realip_remote_addr() {    
            ready(Ok(UserIp(ip.to_string())))
        } else {
            ready(Err(ErrorUnauthorized("IP unable to be resolved")))
        }
    }
}