use anyhow::{bail, Error};
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
        let _ = dbg!(info.get_attribute_i64("required_attr"))?;
        Ok(FallibleOp)
    }

    fn kernel_compute(
        &self,
        (do_fail,): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::KernelCreateError> {
        if (do_fail.mapv(|el| el as u8).sum() == 0) | do_fail.is_empty() {
            return Ok((do_fail.to_owned(),));
        }
        bail!("Non-zero input found");
    }
}
