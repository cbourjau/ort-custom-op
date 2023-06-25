use std::ffi::CStr;
use std::fmt;

use crate::bindings::{OrtApi, OrtStatus, OrtStatusPtr};

#[derive(Debug)]
pub struct ErrorStatus {
    msg: String,
    _code: u32,
}

/// Wraps a status pointer into a result.
///
///A null pointer is mapped to `Ok(())`.
pub fn status_to_result(ptr: OrtStatusPtr, api: &OrtApi) -> Result<(), ErrorStatus> {
    if ptr.is_null() {
        Ok(())
    } else {
        Err(ErrorStatus::new(ptr, api))
    }
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
        let _code = { unsafe { api.GetErrorCode.unwrap()(ptr) } };

        unsafe { api.ReleaseStatus.unwrap()(ptr) };

        Self { msg, _code }
    }
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We must not deallocate the returned string, hence the cstr.

        write!(f, "{:?}", self.msg)
    }
}

impl std::error::Error for ErrorStatus {}
