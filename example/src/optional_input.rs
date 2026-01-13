use std::convert::Infallible;

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct OptionalAdd;

impl CustomOp for OptionalAdd {
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "OptionalAdd";

    type OpInputs<'s> = (ArrayViewD<'s, f64>, Option<ArrayViewD<'s, f64>>);
    type OpOutputs = (ArrayD<f64>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(OptionalAdd)
    }

    fn kernel_compute(
        &self,
        (array_x, optional_y): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        Ok((match optional_y {
            Some(array_y) => &array_x + &array_y,
            None => array_x.to_owned(),
        },))
    }
}
