use crate::api::ElementType;
use crate::bindings::{
    OrtApi, OrtCustomOpInputOutputCharacteristic,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC, OrtKernelContext,
};

use ndarray::ArrayD;

/// Trait which qualifies types to be used as outputs by the
/// `kernel_compute` function of the custom operator.
pub trait Outputs {
    const CHARACTERISTICS: &'static [OrtCustomOpInputOutputCharacteristic];
    // TODO: Make this configurable? Why does this even exists?!
    const VARIADIC_MIN_ARITY: usize;

    // Not clear why this exists...
    const VARIADIC_IS_HOMOGENEOUS: bool;

    const OUTPUT_TYPES: &'static [ElementType];
    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext);
}

trait Output {
    const OUTPUT_TYPE: ElementType;
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: usize);
}

macro_rules! impl_output_non_string {
    ($ty:ty, $variant:tt) => {
        impl<'s> Output for ArrayD<$ty> {
            const OUTPUT_TYPE: ElementType = ElementType::$variant;
            const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
                OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED;

            fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: usize) {
                let shape = self.shape();
                let shape_i64: Vec<_> = shape.iter().map(|v| *v as i64).collect();
                let val = unsafe { ctx.get_output(api, idx, &shape_i64) }.unwrap();
                let mut arr = val.as_array_mut(api).unwrap();
                arr.assign(&self);
            }
        }
    };
}

impl Output for ArrayD<String> {
    const OUTPUT_TYPE: ElementType = ElementType::String;
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
        OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: usize) {
        ctx.fill_string_tensor(api, idx, self).unwrap();
    }
}

impl_output_non_string!(bool, Bool);
impl_output_non_string!(f32, F32);
impl_output_non_string!(f64, F64);
impl_output_non_string!(i32, I32);
impl_output_non_string!(i64, I64);
impl_output_non_string!(u16, U16);
impl_output_non_string!(u32, U32);
impl_output_non_string!(u64, U64);
impl_output_non_string!(u8, U8);

macro_rules! impl_outputs {
    (($($idx:tt: $param:tt),*) | $last_idx:tt: $last_param:tt ) => {
        impl<$($param,)* $last_param> Outputs for ($($param,)* $last_param, )
        where
            $($param : Output,)*
            $last_param : LastOutput,
        {
            const CHARACTERISTICS: &'static [OrtCustomOpInputOutputCharacteristic] =
                &[
                    $(<$param as Output>::CHARACTERISTIC,)* $last_param::CHARACTERISTIC
                ];
            const VARIADIC_MIN_ARITY: usize = $last_param::VARIADIC_MIN_ARITY;
            const VARIADIC_IS_HOMOGENEOUS: bool = $last_param::VARIADIC_IS_HOMOGENEOUS;

            const OUTPUT_TYPES: &'static [ElementType] = &[
                $(<$param as Output>::OUTPUT_TYPE,)* $last_param::OUTPUT_TYPE
            ];

            fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext,) {
                $(self.$idx.write_to_ort(api, ctx, $idx);)*
                self.$last_idx.write_to_ort(api, ctx, $last_idx);
            }
        }
    };
}

impl_outputs! {()|0: Z}
impl_outputs! {(0: A )| 1: Z}
impl_outputs! {(0: A, 1: B )| 2: Z}
impl_outputs! {(0: A, 1: B, 2: C )| 3: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D )| 4: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E )| 5: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E, 5: F )| 6: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G )| 7: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H )| 8: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I )| 9: Z}
impl_outputs! {(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J )| 10: Z}

trait LastOutput {
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic;

    // TODO: min arity should not be defined here but in the user-impl of CustomOp
    const VARIADIC_MIN_ARITY: usize = 1;
    const VARIADIC_IS_HOMOGENEOUS: bool;
    const OUTPUT_TYPE: ElementType;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: usize);
}

impl<T> LastOutput for T
where
    T: Output,
{
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic = T::CHARACTERISTIC;
    // Not applicable
    const VARIADIC_IS_HOMOGENEOUS: bool = false;
    const OUTPUT_TYPE: ElementType = T::OUTPUT_TYPE;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: usize) {
        self.write_to_ort(api, ctx, idx);
    }
}

impl<T> LastOutput for Vec<T>
where
    T: Output,
{
    const CHARACTERISTIC: OrtCustomOpInputOutputCharacteristic =
        OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC;
    const VARIADIC_IS_HOMOGENEOUS: bool = true;
    const OUTPUT_TYPE: ElementType = T::OUTPUT_TYPE;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, first_idx: usize) {
        for (idx, arr) in self.into_iter().enumerate() {
            arr.write_to_ort(api, ctx, idx + first_idx);
        }
    }
}
