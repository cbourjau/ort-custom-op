use std::{convert::Infallible, marker::PhantomData, ops::Add};

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct OptionalInput<T> {
    ty: PhantomData<T>,
}

impl<T> CustomOp for OptionalInput<T>
where
    T: 'static + Add<T, Output = T> + Clone,
    for<'a> (ArrayViewD<'a, T>, Option<ArrayViewD<'a, T>>): Inputs<'a>,
    (ArrayD<T>,): Outputs,
{
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "OptionalInput";

    type OpInputs<'s> = (ArrayViewD<'s, T>, Option<ArrayViewD<'s, T>>);
    type OpOutputs = (ArrayD<T>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(OptionalInput { ty: PhantomData })
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
