use std::convert::Infallible;

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds a variadic number of inputs
pub struct CustomSum;

impl CustomOp for CustomSum {
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "CustomSum";

    // Require 1 or more inputs
    type OpInputs<'s> = (ArrayViewD<'s, f32>, Vec<ArrayViewD<'s, f32>>);
    type OpOutputs = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(CustomSum)
    }

    fn kernel_compute(
        &self,
        inputs: Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        let (first, rest) = inputs;
        Ok((rest
            .into_iter()
            .fold(first.into_owned(), |acc, arr| acc + arr),))
    }
}
