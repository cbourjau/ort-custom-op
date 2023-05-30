use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct CustomAdd;

impl CustomOp for CustomAdd {
    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, f32>);
    type OpOutputs = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Self {
        CustomAdd
    }

    fn kernel_compute(&self, (array_x, array_y): Self::OpInputs<'_>) -> Self::OpOutputs {
        (&array_x + &array_y,)
    }
}
