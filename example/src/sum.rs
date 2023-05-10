use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct CustomSum;

impl CustomOp for CustomSum {
    const NAME: &'static str = "CustomSum";

    type OpInputs<'s> = (Vec<ArrayViewD<'s, f32>>,);
    type OpOutputs = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Self {
        CustomSum
    }

    fn kernel_compute(&self, inputs: Self::OpInputs<'_>) -> Self::OpOutputs {
        (inputs
            .0
            .into_iter()
            .fold(None, |acc, arr| match acc {
                None => Some(arr.into_owned()),
                Some(other) => Some(&arr + other),
            })
            .expect("At least one input."),)
    }
}
