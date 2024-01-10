use crate::bindings::{
    ONNXTensorElementDataType, OrtCustomOpInputOutputCharacteristic,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED,
    OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC,
};
use crate::value::Value;
use anyhow::{bail, Result};
use ndarray::ArrayViewD;

/// Trait defining which types can be used as inputs when implementing [crate::prelude::CustomOp].
///
/// Currently, `Inputs` is implemented for tuples of up to ten
/// elements of of [ArrayViewD] with element types `u8`,
/// `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `bool`, and
/// `&str`. Furthermore, the last element of the tuple may be
/// variadic by being a `Vec` of [ArrayViewD] objects with one of the
/// previously stated element types.
pub trait Inputs<'a>: Sized {
    /// Is the variadic part of the inputs (if any) homogeneous?
    const VARIADIC_IS_HOMOGENEOUS: Option<bool>;
    /// Number of positional (i.e. non-variadic) inputs
    const NUM_POSITIONAL: usize;

    /// Create inputs from `Value` objects
    fn try_from_values(values: Vec<Value<'a>>) -> Result<Self>;

    /// Tensor data type of this input, or `None` if it is not a Tensor
    fn tensor_data_type(index: usize) -> Option<ONNXTensorElementDataType>;

    /// Get the "characteristic" of an input (i.e. if it is
    /// optional). Panics if `index` is out-of-range.
    fn characteristic(index: usize) -> OrtCustomOpInputOutputCharacteristic {
        // Everything is either required or variadic for now
        if index < Self::NUM_POSITIONAL {
            OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_REQUIRED
        } else if index == Self::NUM_POSITIONAL && Self::VARIADIC_IS_HOMOGENEOUS.is_some() {
            OrtCustomOpInputOutputCharacteristic_INPUT_OUTPUT_VARIADIC
        } else {
            panic!("Provided index '{}' is out of range", index)
        }
    }
}

pub trait TryFromValue<'s>: Sized {
    fn try_from_value(value: Value<'s>) -> Result<Self>;
}

/// Get ONNX tensor data type id if possible
trait OnnxTensorDtype {
    fn dtype_id() -> Option<ONNXTensorElementDataType>;
}

/////////////////////
// Implementations //
/////////////////////

impl<'s> TryFromValue<'s> for ArrayViewD<'s, &'s str> {
    fn try_from_value(value: Value<'s>) -> Result<Self> {
        if let Value::TensorStr(arr) = value {
            Ok(arr)
        } else {
            bail!("Expected 'String' tensor, found {:?}", value)
        }
    }
}

macro_rules! impl_try_from {
    ($ty:ty, $variant:path) => {
        impl<'a> TryFromValue<'a> for ArrayViewD<'a, $ty> {
            fn try_from_value(value: Value<'a>) -> Result<Self> {
                if let $variant(arr) = value {
                    Ok(arr)
                } else {
                    bail!("Expected '{}' tensor, found {:?}", stringify!($ty), value)
                }
            }
        }
    };
}

impl_try_from!(bool, Value::TensorBool);
impl_try_from!(u8, Value::TensorU8);
impl_try_from!(u64, Value::TensorU64);
impl_try_from!(u32, Value::TensorU32);
impl_try_from!(u16, Value::TensorU16);
impl_try_from!(i8, Value::TensorI8);
impl_try_from!(i64, Value::TensorI64);
impl_try_from!(i32, Value::TensorI32);
impl_try_from!(i16, Value::TensorI16);
impl_try_from!(f64, Value::TensorF64);
impl_try_from!(f32, Value::TensorF32);

// This could be implemented using the below macro, but then we would
// have to disable some lints.
impl<'s, A> Inputs<'s> for (Vec<A>,)
where
    A: TryFromValue<'s> + OnnxTensorDtype,
{
    const VARIADIC_IS_HOMOGENEOUS: Option<bool> = Some(true);
    const NUM_POSITIONAL: usize = 0;

    fn try_from_values(values: Vec<Value<'s>>) -> Result<Self> {
        let rest = values
            .into_iter()
            .map(|el| TryFromValue::try_from_value(el))
            .collect::<Result<_, _>>()?;

        Ok((rest,))
    }

    fn tensor_data_type(idx: usize) -> Option<ONNXTensorElementDataType> {
        [A::dtype_id()][idx]
    }
}

macro_rules! impl_inputs {
    ($n_min:literal, $is_variadic:literal, $($var_ty:ident)? | $($positional_ty:ident),*) => {
        impl<'s, $($positional_ty,)* $($var_ty)*> Inputs<'s> for ($($positional_ty,)* $(Vec<$var_ty>,)*)
        where
            $($positional_ty: TryFromValue<'s> + OnnxTensorDtype,)*
            $($var_ty: TryFromValue<'s> + OnnxTensorDtype,)*
        {
            const VARIADIC_IS_HOMOGENEOUS: Option<bool> = if $is_variadic {Some(true)} else { None };
            const NUM_POSITIONAL: usize = $n_min;

            fn try_from_values(values: Vec<Value<'s>>) -> Result<Self>
            {
                if $is_variadic {
                    if values.len() < $n_min {
                        bail!("expected at least {} inputs; found {}", $n_min, values.len())
                    }
                } else if values.len() != $n_min {
                    bail!("expected {} inputs; found {}", $n_min, values.len())
                }

                let mut iter = values.into_iter();

                Ok((
                    $(<$positional_ty as TryFromValue>::try_from_value(iter.next().unwrap())?,)*
                        $(iter.map(|el| TryFromValue::try_from_value(el)).collect::<Result<Vec<$var_ty>, _>>()?,)*
                ))
            }
            fn tensor_data_type(idx: usize) -> Option<ONNXTensorElementDataType> {
                [
                    $($positional_ty::dtype_id(),)*
                        $($var_ty::dtype_id())*
                ][idx]
            }
        }
    };
}

// // Positional-only implementations
impl_inputs!(1, false, | A);
impl_inputs!(2, false, | A, B);
impl_inputs!(3, false, | A, B, C);
impl_inputs!(4, false, | A, B, C, D);
impl_inputs!(5, false, | A, B, C, D, E);
impl_inputs!(6, false, | A, B, C, D, E, F);
impl_inputs!(7, false, | A, B, C, D, E, F, G);
impl_inputs!(8, false, | A, B, C, D, E, F, G, H);
impl_inputs!(9, false, | A, B, C, D, E, F, G, H, I);
impl_inputs!(10, false, | A, B, C, D, E, F, G, H, I, J);

// // Variadic implementations; variadic input must be homogeneous, but may be empty
impl_inputs!(1, true, Z | A);
impl_inputs!(2, true, Z | A, B);
impl_inputs!(3, true, Z | A, B, C);
impl_inputs!(4, true, Z | A, B, C, D);
impl_inputs!(5, true, Z | A, B, C, D, E);
impl_inputs!(6, true, Z | A, B, C, D, E, F);
impl_inputs!(7, true, Z | A, B, C, D, E, F, G);
impl_inputs!(8, true, Z | A, B, C, D, E, F, G, H);
impl_inputs!(9, true, Z | A, B, C, D, E, F, G, H, I);
impl_inputs!(10, true, Z | A, B, C, D, E, F, G, H, I, J);

impl<'s> OnnxTensorDtype for ArrayViewD<'s, &'s str> {
    fn dtype_id() -> Option<ONNXTensorElementDataType> {
        Some(crate::bindings::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
    }
}

macro_rules! impl_onnx_tensor_dtype {
    ($ty:ty, $ident:ident) => {
        impl<'s> OnnxTensorDtype for ArrayViewD<'s, $ty> {
            fn dtype_id() -> Option<ONNXTensorElementDataType> {
                Some(crate::bindings::$ident)
            }
        }
    };
}

#[rustfmt::skip] impl_onnx_tensor_dtype!(f32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
#[rustfmt::skip] impl_onnx_tensor_dtype!(f64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
#[rustfmt::skip] impl_onnx_tensor_dtype!(bool, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
#[rustfmt::skip] impl_onnx_tensor_dtype!(u8, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
#[rustfmt::skip] impl_onnx_tensor_dtype!(u16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
#[rustfmt::skip] impl_onnx_tensor_dtype!(u32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
#[rustfmt::skip] impl_onnx_tensor_dtype!(u64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);
#[rustfmt::skip] impl_onnx_tensor_dtype!(i8, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
#[rustfmt::skip] impl_onnx_tensor_dtype!(i16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
#[rustfmt::skip] impl_onnx_tensor_dtype!(i32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
#[rustfmt::skip] impl_onnx_tensor_dtype!(i64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
