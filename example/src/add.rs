use std::convert::Infallible;

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct CustomAdd;

impl CustomOp for CustomAdd {
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, f32>);
    type OpOutputs = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(CustomAdd)
    }

    fn kernel_compute(
        &self,
        (array_x, array_y): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::KernelCreateError> {
        Ok((&array_x + &array_y,))
    }
}
