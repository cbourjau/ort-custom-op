use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which simply passes all inputs through
pub struct VariadicIdentity;

impl CustomOp for VariadicIdentity {
    const NAME: &'static str = "VariadicIdentity";

    type OpInputs<'s> = (Vec<ArrayViewD<'s, f32>>,);
    type OpOutputs = (Vec<ArrayD<f32>>,);

    fn kernel_create(_info: &KernelInfo) -> Self {
        VariadicIdentity
    }

    fn kernel_compute(&self, (inputs,): Self::OpInputs<'_>) -> Self::OpOutputs {
        (inputs.into_iter().map(|arr| arr.into_owned()).collect(),)
    }
}
