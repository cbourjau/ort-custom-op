use std::ffi::CString;
use std::os::raw::c_char;

mod api;
mod bindings;
mod custom_op;

pub mod prelude {
    pub use crate::api::{create_custom_op_domain, KernelInfo};
    pub use crate::bindings::{OrtApiBase, OrtCustomOp, OrtSessionOptions, OrtStatus};
    pub use crate::custom_op::{build, CustomOp, Inputs, Outputs};
}

trait Inputs {
    fn from_ort(ort_thing: ()) -> Self;
}

trait ReadInput {
    fn from_ort(ort_thing: (), idx: usize) -> Self;
}

impl<T> Inputs for (Vec<T>,)
where
    Vec<T>: ReadInput,
{
    fn from_ort(_ort_thing: ()) -> Self {
        (Vec::<T>::from_ort((), 42),)
    }
}

impl ReadInput for Vec<f32> {
    fn from_ort(_ort_thing: (), _idx: usize) -> Self {
        unimplemented!()
    }
}
