mod api;
mod bindings;
mod custom_op;
mod error;
mod inputs;
mod outputs;

pub mod prelude {
    pub use crate::api::{create_custom_op_domain, KernelInfo};
    pub use crate::bindings::{OrtApiBase, OrtCustomOp, OrtSessionOptions, OrtStatus};
    pub use crate::custom_op::{build, CustomOp};
    pub use crate::inputs::Inputs;
    pub use crate::outputs::Outputs;
}
