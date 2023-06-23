use anyhow::Error;
use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct FallibleOp;

impl CustomOp for FallibleOp {
    type KernelCreateError = Error;
    type ComputeError = Error;

    const NAME: &'static str = "FallibleOp";

    type OpInputs<'s> = (ArrayViewD<'s, bool>,);
    type OpOutputs = (ArrayD<bool>,);

    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        dbg!("foo");
        let _ = dbg!(info.get_attribute_i64("crash_if_1"))?;
        Ok(FallibleOp)
    }

    fn kernel_compute(
        &self,
        (do_fail,): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::KernelCreateError> {
        do_fail.into_shape(())?;
        unimplemented!()
    }
}
