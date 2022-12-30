use std::ffi::CString;
use std::fmt;

use crate::bindings::{OrtApi, OrtStatus, OrtStatusPtr};

#[derive(Debug)]
pub struct ErrorStatusPtr {
    msg: String,
    status: &'static mut OrtStatus,
}

impl ErrorStatusPtr {
    pub fn new(ptr: OrtStatusPtr, api: &OrtApi) -> Self {
        let cstr_ptr = unsafe { api.GetErrorMessage.unwrap()(ptr) };
        let msg = unsafe { CString::from_raw(cstr_ptr as *mut _) }
            .into_string()
            .unwrap();
        Self {
            status: unsafe { ptr.as_mut().unwrap() },
            msg,
        }
    }

    pub fn into_pointer(self) -> OrtStatusPtr {
        self.status
    }
}

impl fmt::Display for ErrorStatusPtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "non-null status: {:?}", self.msg)
    }
}

impl std::error::Error for ErrorStatusPtr {}

// fn create_error_status<'s>(api: &'s OrtApi, code: u32, msg: &str) -> Result<(), ErrorStatusPtr> {
//     let c_char_ptr = str_to_c_char_ptr(msg);
//     let ptr = unsafe { api.CreateStatus.unwrap()(code, c_char_ptr) };
//     status_to_result(ptr, api)
// }
