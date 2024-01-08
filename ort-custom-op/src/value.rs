use anyhow::{bail, Result};
use ndarray::{Array, ArrayD, ArrayViewD};

#[derive(Debug)]
pub enum Value<'a> {
    TensorBool(ArrayViewD<'a, bool>),
    TensorF32(ArrayViewD<'a, f32>),
    TensorF64(ArrayViewD<'a, f64>),
    TensorI16(ArrayViewD<'a, i16>),
    TensorI32(ArrayViewD<'a, i32>),
    TensorI64(ArrayViewD<'a, i64>),
    TensorI8(ArrayViewD<'a, i8>),
    TensorU16(ArrayViewD<'a, u16>),
    TensorU32(ArrayViewD<'a, u32>),
    TensorU64(ArrayViewD<'a, u64>),
    TensorU8(ArrayViewD<'a, u8>),

    TensorString(TensorString),
}

#[derive(Debug)]
pub struct TensorString {
    pub buf: Vec<u8>,
    pub offsets: Vec<usize>,
    pub shape: Vec<usize>,
}

pub trait Inputs<'a>
where
    Self: 'a,
{
    fn try_from_values(values: &'a [Value]) -> Result<Self>
    where
        Self: Sized;
}

trait TryFromValue<'s>
where
    Self: 's,
{
    fn try_from_value(value: &'s Value) -> Result<Self>
    where
        Self: Sized;
}

/////////////////////
// Implementations //
/////////////////////

impl TensorString {
    fn as_owned_array(&self) -> Result<ArrayD<&str>> {
        // Compute windows with the start and end of each
        // substring and then scan the buffer.
        let very_end = [self.buf.len()];
        let starts = self.offsets.iter();
        let ends = self.offsets.iter().chain(very_end.iter()).skip(1);
        let windows = starts.zip(ends);
        let strings: Vec<_> = windows
            .scan(self.buf.as_slice(), |buf: &mut &[u8], (start, end)| {
                let (this, rest) = buf.split_at(end - start);
                *buf = rest;
                std::str::from_utf8(this).ok()
            })
            .collect();

        Ok(Array::from_vec(strings)
            .into_shape(self.shape.as_slice())
            .expect("Shape information was incorrect."))
    }
}

impl<'s> TryFromValue<'s> for ArrayD<&'s str> {
    fn try_from_value(value: &'s Value) -> Result<Self>
    where
        Self: Sized,
    {
        if let Value::TensorString(tensor_string) = value {
            Ok(tensor_string.as_owned_array()?)
        } else {
            bail!("Expected 'String' tensor, found {:?}", value)
        }
    }
}

macro_rules! impl_try_from {
    ($ty:ty, $variant:path) => {
        impl<'a> TryFromValue<'a> for ArrayViewD<'a, $ty>
        where
            Self: Sized,
        {
            fn try_from_value(value: &'a Value) -> Result<Self> {
                if let $variant(arr) = value {
                    Ok(arr.view())
                } else {
                    bail!("Expected '{}' tensor, found {:?}", stringify!($ty), value)
                }
            }
        }
    };
}

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
    Self: 's,
    A: TryFromValue<'s>,
{
    fn try_from_values(values: &'s [Value]) -> Result<Self>
    where
        Self: Sized,
    {
        let rest = values
            .iter()
            .map(|el| TryFromValue::try_from_value(el))
            .collect::<Result<_, _>>()?;

        Ok((rest,))
    }
}

macro_rules! impl_inputs {
    ($n_min:literal, $is_variadic:literal, $($var_ty:ident)? | $($positional_ty:ident),*) => {
        impl<'s, $($positional_ty,)* $($var_ty)*> Inputs<'s> for ($($positional_ty,)* $(Vec<$var_ty>,)*)
        where
            Self: 's,
            $($positional_ty: TryFromValue<'s>,)*
            $($var_ty: TryFromValue<'s>,)*
        {
            fn try_from_values(values: &'s [Value]) -> Result<Self>
            where
                Self: Sized,
            {
                if $is_variadic {
                    if values.len() < $n_min {
                        bail!("expected at least {} inputs; found {}", $n_min, values.len())
                    }
                } else if values.len() != $n_min {
                    bail!("expected {} inputs; found {}", $n_min, values.len())
                }

                let mut iter = values.iter();

                Ok((
                    $(<$positional_ty as TryFromValue>::try_from_value(iter.next().unwrap())?,)*
                        $(iter.map(|el| TryFromValue::try_from_value(el)).collect::<Result<Vec<$var_ty>, _>>()?,)*
                ))
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
