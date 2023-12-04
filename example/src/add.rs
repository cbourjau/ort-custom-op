use std::{convert::Infallible, marker::PhantomData, ops::Add};

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct CustomAdd<T> {
    ty: PhantomData<T>,
}

impl<'add, T> CustomOp for CustomAdd<T>
where
    T: 'static + Add<T, Output = T> + Clone,
    for<'s> (ArrayViewD<'s, T>, ArrayViewD<'s, T>): Inputs<'s>,
    (ArrayD<T>,): Outputs,
{
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, T>, ArrayViewD<'s, T>);
    type OpOutputs = (ArrayD<T>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(CustomAdd { ty: PhantomData })
    }

    fn kernel_compute(
        &self,
        (array_x, array_y): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        Ok((&array_x + &array_y,))
    }
}
