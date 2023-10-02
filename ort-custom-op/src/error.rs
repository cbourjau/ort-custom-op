use std::ffi::CStr;
use std::fmt;

use crate::bindings::{OrtApi, OrtStatus};

#[derive(Debug)]
pub struct ErrorStatus {
    msg: String,
    code: u32,
}

impl ErrorStatus {
    /// Consume the given OrtStatusPtr (freeing it).
    pub fn new(ptr: *mut OrtStatus, api: &OrtApi) -> Self {
        let msg = {
            let cstr_ptr = unsafe { api.GetErrorMessage.unwrap()(ptr) };
            unsafe { CStr::from_ptr(cstr_ptr as *mut _) }
                .to_str()
                .unwrap()
                .to_string()
        };
        let code = { unsafe { api.GetErrorCode.unwrap()(ptr) } };

        unsafe { api.ReleaseStatus.unwrap()(ptr) };

        Self { msg, code }
    }
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error {}: {}", self.code, self.msg)
    }
}

impl std::error::Error for ErrorStatus {}
