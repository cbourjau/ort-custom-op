use crate::{
    api::ElementType,
    bindings::{OrtApi, OrtValue},
};
use anyhow::Result;
use ndarray::{ArrayView, ArrayViewD};

/// Enum over all currently supported input value types.
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
    TensorStr(ArrayViewD<'a, &'a str>),
}

pub(crate) enum ValueBuffer<Buf, Shape> {
    Tensor { buf: Buf, shape: Shape },
}

pub(crate) enum Buffer<'s> {
    Bool(&'s [bool]),
    F32(&'s [f32]),
    F64(&'s [f64]),
    I16(&'s [i16]),
    I32(&'s [i32]),
    I64(&'s [i64]),
    I8(&'s [i8]),
    U16(&'s [u16]),
    U32(&'s [u32]),
    U64(&'s [u64]),
    U8(&'s [u8]),
    Str(Vec<&'s str>),
}

pub(crate) enum BufferMaybeOwned<'s> {
    Bool(&'s [bool]),
    F32(&'s [f32]),
    F64(&'s [f64]),
    I16(&'s [i16]),
    I32(&'s [i32]),
    I64(&'s [i64]),
    I8(&'s [i8]),
    U16(&'s [u16]),
    U32(&'s [u32]),
    U64(&'s [u64]),
    U8(&'s [u8]),
    String(StringBuffer),
}

/// Object owning the contiguous String buffer.
struct StringBuffer {
    buf: Vec<u8>,
    offsets: Vec<usize>,
}

impl<'s> BufferMaybeOwned<'s> {
    pub fn load_from_ort(
        api: &OrtApi,
        ort_value: &'s mut OrtValue,
        dtype: &ElementType,
    ) -> Result<Self> {
        // tensor data
        Ok(match dtype {
            ElementType::U8 => Self::U8(ort_value.get_data_mut(api)?),
            ElementType::U16 => Self::U16(ort_value.get_data_mut(api)?),
            ElementType::U32 => Self::U32(ort_value.get_data_mut(api)?),
            ElementType::U64 => Self::U64(ort_value.get_data_mut(api)?),
            ElementType::I8 => Self::I8(ort_value.get_data_mut(api)?),
            ElementType::I16 => Self::I16(ort_value.get_data_mut(api)?),
            ElementType::I32 => Self::I32(ort_value.get_data_mut(api)?),
            ElementType::I64 => Self::I64(ort_value.get_data_mut(api)?),
            ElementType::F32 => Self::F32(ort_value.get_data_mut(api)?),
            ElementType::F64 => Self::F64(ort_value.get_data_mut(api)?),
            ElementType::Bool => Self::Bool(ort_value.get_data_mut(api)?),
            ElementType::String => {
                let (buf, offsets) = ort_value.get_string_tensor_single_buf(api)?;
                Self::String(StringBuffer { buf, offsets })
            }
        })
    }

    pub fn view(&self) -> Buffer {
        match self {
            Self::Bool(buf) => Buffer::Bool(buf),
            Self::F32(buf) => Buffer::F32(buf),
            Self::F64(buf) => Buffer::F64(buf),
            Self::I16(buf) => Buffer::I16(buf),
            Self::I32(buf) => Buffer::I32(buf),
            Self::I64(buf) => Buffer::I64(buf),
            Self::I8(buf) => Buffer::I8(buf),
            Self::U16(buf) => Buffer::U16(buf),
            Self::U32(buf) => Buffer::U32(buf),
            Self::U64(buf) => Buffer::U64(buf),
            Self::U8(buf) => Buffer::U8(buf),
            Self::String(string_buf) => Buffer::Str(string_buf.vec_of_strs()),
        }
    }
}

impl StringBuffer {
    fn vec_of_strs(&self) -> Vec<&str> {
        let very_end = [self.buf.len()];
        let starts = self.offsets.iter();
        let ends = self.offsets.iter().chain(very_end.iter()).skip(1);
        let windows = starts.zip(ends);
        windows
            .scan(self.buf.as_slice(), |buf: &mut &[u8], (start, end)| {
                let (this, rest) = buf.split_at(end - start);
                *buf = rest;
                std::str::from_utf8(this).ok()
            })
            .collect()
    }
}

impl<'s> ValueBuffer<BufferMaybeOwned<'s>, Vec<usize>> {
    pub fn normalize_buffers(&'s self) -> ValueBuffer<Buffer<'s>, &'s [usize]> {
        match self {
            Self::Tensor { buf, shape } => ValueBuffer::Tensor {
                shape: shape.as_slice(),
                buf: buf.view(),
            },
        }
    }
}

impl<'s> ValueBuffer<Buffer<'s>, &'s [usize]> {
    pub fn as_value(&'s self) -> Result<Value<'s>> {
        Ok(match self {
            Self::Tensor { shape, buf } => {
                let shape = *shape;
                match buf {
                    Buffer::Bool(buf) => Value::TensorBool(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::F32(buf) => Value::TensorF32(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::F64(buf) => Value::TensorF64(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::I16(buf) => Value::TensorI16(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::I32(buf) => Value::TensorI32(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::I64(buf) => Value::TensorI64(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::I8(buf) => Value::TensorI8(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::U16(buf) => Value::TensorU16(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::U32(buf) => Value::TensorU32(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::U64(buf) => Value::TensorU64(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::U8(buf) => Value::TensorU8(ArrayView::from(buf).into_shape(shape)?),
                    Buffer::Str(buf) => Value::TensorStr(ArrayView::from(buf).into_shape(shape)?),
                }
            }
        })
    }
}
