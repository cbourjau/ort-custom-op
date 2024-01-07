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

impl<'a> TryFrom<&'a Value<'_>> for ArrayD<&'a str> {
    type Error = anyhow::Error;

    fn try_from(value: &'a Value<'_>) -> std::result::Result<Self, Self::Error> {
        if let Value::TensorString(tensor_string) = value {
            Ok(tensor_string.as_owned_array()?)
        } else {
            bail!("Expected 'String' tensor, found {:?}", value)
        }
    }
}

macro_rules! impl_try_from {
    ($ty:ty, $variant:path) => {
        impl<'a, 'b> TryFrom<&'a Value<'_>> for ArrayViewD<'b, $ty>
        where
            'a: 'b,
        {
            type Error = anyhow::Error;

            fn try_from(value: &'a Value<'_>) -> std::result::Result<Self, Self::Error> {
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

pub trait TryIntoInputTuple<TPL> {
    fn try_into_tuple(self) -> Result<TPL>;
}

// This can be implemented using the macro, but would require adding exceptions for warnings
impl<'a, 'b, A> TryIntoInputTuple<(Vec<A>,)> for &'b [Value<'a>]
where
    'a: 'b,
    A: TryFrom<&'b Value<'a>, Error = anyhow::Error>,
{
    fn try_into_tuple(self) -> Result<(Vec<A>,)> {
        let rest = self
            .iter()
            .map(|el| el.try_into().map_err(|e| anyhow::anyhow!("{:?}", e)))
            .collect::<Result<_, _>>()?;

        Ok((rest,))
    }
}

macro_rules! impl_try_into_input_tuple {
    ($n_min:literal, $is_variadic:literal, $($var_ty:ident)? | $($positional_ty:ident),*) => {
        impl<'a, 'b, $($positional_ty,)* $($var_ty)*> TryIntoInputTuple<($($positional_ty,)* $(Vec<$var_ty>,)*)> for &'b[Value<'a>]
        where
            'a: 'b,
            $($positional_ty: TryFrom<&'b Value<'a>, Error = anyhow::Error>,)*
            $($var_ty: TryFrom<&'b Value<'a>, Error = anyhow::Error>,)*
        {
            fn try_into_tuple(self) -> Result<($($positional_ty,)* $(Vec<$var_ty>,)*)> {
                if $is_variadic {
                    if self.len() < $n_min {
                        bail!("expected at least {} inputs; found {}", $n_min, self.len())
                    }
                } else if self.len() != $n_min {
                    bail!("expected {} inputs; found {}", $n_min, self.len())
                }

                let mut iter = self.iter();

                Ok((
                    $(TryInto::<$positional_ty>::try_into(iter.next().unwrap())?,)*
                        $(iter.map(|el| el.try_into()).collect::<Result<Vec<$var_ty>, _>>()?,)*
                ))
            }
        }
    };
}

// Positional-only implementations
impl_try_into_input_tuple!(1, false, | A);
impl_try_into_input_tuple!(2, false, | A, B);
impl_try_into_input_tuple!(3, false, | A, B, C);
impl_try_into_input_tuple!(4, false, | A, B, C, D);
impl_try_into_input_tuple!(5, false, | A, B, C, D, E);
impl_try_into_input_tuple!(6, false, | A, B, C, D, E, F);
impl_try_into_input_tuple!(7, false, | A, B, C, D, E, F, G);
impl_try_into_input_tuple!(8, false, | A, B, C, D, E, F, G, H);
impl_try_into_input_tuple!(9, false, | A, B, C, D, E, F, G, H, I);
impl_try_into_input_tuple!(10, false, | A, B, C, D, E, F, G, H, I, J);

// Variadic implementations; variadic input must be homogeneous, but may be empty
impl_try_into_input_tuple!(1, true, Z | A);
impl_try_into_input_tuple!(2, true, Z | A, B);
impl_try_into_input_tuple!(3, true, Z | A, B, C);
impl_try_into_input_tuple!(4, true, Z | A, B, C, D);
impl_try_into_input_tuple!(5, true, Z | A, B, C, D, E);
impl_try_into_input_tuple!(6, true, Z | A, B, C, D, E, F);
impl_try_into_input_tuple!(7, true, Z | A, B, C, D, E, F, G);
impl_try_into_input_tuple!(8, true, Z | A, B, C, D, E, F, G, H);
impl_try_into_input_tuple!(9, true, Z | A, B, C, D, E, F, G, H, I);
impl_try_into_input_tuple!(10, true, Z | A, B, C, D, E, F, G, H, I, J);

// impl<'a, A> TryIntoInputTuple<(A,)> for Vec<Value<'a>>
// where
//     A: for<'b> TryFrom<&'b Value<'a>, Error = anyhow::Error>,
// {
//     fn try_into_tuple(&self) -> Result<(A,)> {
//         let is_variadic = false;
//         let n_min = 1;

//         if is_variadic {
//             if self.len() < n_min {
//                 bail!("expected at least {} inputs; found {}", n_min, self.len())
//             }
//         } else if self.len() != n_min {
//             bail!("expected {} inputs; found {}", n_min, self.len())
//         }

//         let mut iter = self.iter();

//         Ok((iter.next().unwrap().try_into()?,))
//     }
// }

// impl<'a, A, Z> TryIntoInputTuple<(A, Vec<Z>)> for Vec<Value<'a>>
// where
//     A: for<'b> TryFrom<&'b Value<'a>, Error = anyhow::Error>,
//     Z: for<'b> TryFrom<&'b Value<'a>, Error = anyhow::Error>,
// {
//     fn try_into_tuple(&self) -> Result<(A, Vec<Z>)> {
//         let n_min = 1;

//         if self.len() < n_min {
//             bail!("expected at least {} inputs; found {}", n_min, self.len())
//         }

//         let mut iter = self.iter();

//         Ok((
//             iter.next().unwrap().try_into()?,
//             iter.map(|el| el.try_into()).collect::<Result<_, _>>()?,
//         ))
//     }
// }
