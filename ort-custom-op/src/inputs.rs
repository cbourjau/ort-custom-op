use crate::api::ElementType;
use crate::bindings::{
    OrtApi, OrtCustomOpInputOutputCharacteristic,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC, OrtKernelContext,
};

use ndarray::{ArrayD, ArrayViewD};

/// Trait which qualifies types to be used as input to the
/// `kernel_compute` function of the custom operator.
pub trait Inputs<'s> {
    const CHARACTERISTICS: &'static [OrtCustomOpInputOutputCharacteristic];

    // TODO: Make this configurable? Why does this even exists?!
    const VARIADIC_MIN_ARITY: usize;

    // Not clear why this exists...
    const VARIADIC_IS_HOMOGENEOUS: bool;

    /// Input types of this kernel.
    ///
    /// Homogeneous variadic inputs have the `ElementType` of their
    /// items. Heterogeneous variadic inputs have `Unknown` element
    /// type.
    const INPUT_TYPES: &'static [ElementType];

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext) -> Self;
}
pub trait Input<'s> {
    const INPUT_TYPE: ElementType;
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: usize) -> Self;
}

trait LastInput<'s> {
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic;

    // TODO: Make this configurable?
    const VARIADIC_MIN_ARITY: usize = 1;
    const VARIADIC_IS_HOMOGENEOUS: bool;
    const INPUT_TYPE: ElementType;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: usize) -> Self;
}

impl<'s, T> LastInput<'s> for T
where
    T: Input<'s>,
{
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic = T::CHARACTERISTIC;
    // Not applicable
    const VARIADIC_IS_HOMOGENEOUS: bool = false;
    const INPUT_TYPE: ElementType = T::INPUT_TYPE;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: usize) -> Self {
        T::from_ort(api, ctx, idx)
    }
}
impl<'s, T> LastInput<'s> for Vec<T>
where
    T: Input<'s>,
{
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
        OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC;
    const VARIADIC_IS_HOMOGENEOUS: bool = true;
    const INPUT_TYPE: ElementType = T::INPUT_TYPE;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, first_idx: usize) -> Self {
        let n_total = ctx.get_input_count(api).expect("Retrieve number of inputs");
        (first_idx..n_total)
            .map(|idx| T::from_ort(api, ctx, idx))
            .collect()
    }
}

macro_rules! impl_input_non_string {
    ($ty:ty, $variant:tt) => {
        impl<'s> Input<'s> for ArrayViewD<'s, $ty> {
            const INPUT_TYPE: ElementType = ElementType::$variant;
            const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
                OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED;

            fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: usize) -> Self {
                ctx.get_input_array::<$ty>(api, idx)
                    .expect("Loading input data of given type failed.")
            }
        }
    };
}

impl_input_non_string!(bool, Bool);
impl_input_non_string!(f32, F32);
impl_input_non_string!(f64, F64);
impl_input_non_string!(i32, I32);
impl_input_non_string!(i64, I64);
impl_input_non_string!(u16, U16);
impl_input_non_string!(u32, U32);
impl_input_non_string!(u64, U64);
impl_input_non_string!(u8, U8);

impl<'s> Input<'s> for ArrayD<String> {
    const INPUT_TYPE: ElementType = ElementType::String;
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
        OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: usize) -> Self {
        ctx.get_input_array_string(api, idx)
            .expect("Loading input data of given type failed.")
    }
}

macro_rules! impl_inputs {
    ($($idx:literal: $param:tt),* | $last_idx:literal: $last_param:tt ) => {
        impl<'s, $($param,)* $last_param> Inputs<'s> for ($($param,)* $last_param, )
        where
            $($param : Input<'s>,)*
            $last_param : LastInput<'s>,
        {
            const CHARACTERISTICS: &'static [OrtCustomOpInputOutputCharacteristic] =
                &[
                    $(<$param as Input>::CHARACTERISTIC,)* $last_param::CHARACTERISTIC
                ];
            const VARIADIC_MIN_ARITY: usize = $last_param::VARIADIC_MIN_ARITY;
            const VARIADIC_IS_HOMOGENEOUS: bool = $last_param::VARIADIC_IS_HOMOGENEOUS;

            const INPUT_TYPES: &'static [ElementType] = &[
                $(<$param as Input>::INPUT_TYPE,)* $last_param::INPUT_TYPE
            ];

            fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext) -> Self {

                (
                    $($param::from_ort(api, ctx, $idx), )* $last_param::from_ort(api, ctx, $last_idx),

                )
            }
        }
    };
}
impl_inputs! {|0: Z}
impl_inputs! {0: A | 1: Z}
impl_inputs! {0: A, 1: B | 2: Z}
impl_inputs! {0: A, 1: B, 2: C | 3: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D | 4: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E | 5: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E, 5: F | 6: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G | 7: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H | 8: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I | 9: Z}
impl_inputs! {0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J | 10: Z}
