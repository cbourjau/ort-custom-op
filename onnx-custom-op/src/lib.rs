use std::ffi::CString;
use std::os::raw::c_char;

mod api;
mod bindings;
mod custom_op;
mod error;

pub mod prelude {
    pub use crate::api::{create_custom_op_domain, KernelInfo};
    pub use crate::bindings::{OrtApiBase, OrtCustomOp, OrtSessionOptions, OrtStatus};
    pub use crate::custom_op::{build, CustomOp, Inputs, Outputs};
}
