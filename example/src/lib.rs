use ort_custom_op::prelude::*;

mod add;
mod add_constant;
mod attr_showcase;
mod datetime;
mod fallible_op;
mod sum;
mod variadic_identity;

// Create a constant function table as expected by onnxruntime
const ADD_CONSTANT: OrtCustomOp = build::<add_constant::AddConstant>();

/// Static objects defining the custom operators
const OP_ATTR_SHOWCASE: OrtCustomOp = build::<attr_showcase::AttrShowcase>();
const OP_CUSTOM_ADD_F32: OrtCustomOp = build::<add::CustomAdd<f32>>();
const OP_CUSTOM_ADD_F64: OrtCustomOp = build::<add::CustomAdd<f64>>();
const OP_CUSTOM_SUM: OrtCustomOp = build::<sum::CustomSum>();
const OP_PARSE_DATETIME: OrtCustomOp = build::<datetime::ParseDateTime>();
const OP_VARIADIC_IDENTITY: OrtCustomOp = build::<variadic_identity::VariadicIdentity>();
const OP_FALLIBLE: OrtCustomOp = build::<fallible_op::FallibleOp>();

/// Public function which onnxruntime expects to be in the shared library
#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    create_custom_op_domain(
        options,
        api_base,
        "my.domain",
        &[
            &OP_ATTR_SHOWCASE,
            &OP_CUSTOM_ADD_F32,
            &OP_CUSTOM_ADD_F64,
            &OP_CUSTOM_SUM,
            &OP_PARSE_DATETIME,
            &OP_VARIADIC_IDENTITY,
            &OP_FALLIBLE,
            &ADD_CONSTANT,
        ],
    )
}
