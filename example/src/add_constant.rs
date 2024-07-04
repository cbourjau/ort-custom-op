// This file reflects the example from the Readme

use std::convert::Infallible;

use anyhow::Error;
use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator that adds a constant to every input of an i64 input
/// tensor.
pub struct AddConstant {
    /// State associated with this operator. Derived from the
    /// respective node's attributes.
    constant: i64,
}

impl CustomOp for AddConstant {
    /// Possible error raised when setting up the kernel during
    /// session creation.
    type KernelCreateError = Error;
    /// Possible error raised during compute call.
    type ComputeError = Infallible;

    /// Name of the operator within its domain
    const NAME: &'static str = "AddConstant";

    /// Input array(s) (as a tuple)
    type OpInputs<'s> = (ArrayViewD<'s, i64>,);

    /// Output array(s) (as a tuple)
    type OpOutputs = (ArrayD<i64>,);

    /// Function called once per node during session creation.
    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        // Read and parse attribute from kernel info object.
        let constant = info.get_attribute_i64("constant")?;
        Ok(AddConstant { constant })
    }

    /// Function called during inference.
    fn kernel_compute(
        &self,
        (array,): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        Ok((&array + self.constant,))
    }
}
