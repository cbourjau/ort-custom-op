use ort_custom_op::prelude::*;

mod add;
mod attr_showcase;
mod datetime;
mod sum;

/// Static objects defining the custom operators
const OP_ATTR_SHOWCASE: OrtCustomOp = build::<attr_showcase::AttrShowcase>();
const OP_CUSTOM_ADD: OrtCustomOp = build::<add::CustomAdd>();
const OP_CUSTOM_SUM: OrtCustomOp = build::<sum::CustomSum>();
const OP_PARSE_DATETIME: OrtCustomOp = build::<datetime::ParseDateTime>();

/// Public function which onnxruntime expects to be in the shared library
#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    let status = create_custom_op_domain(
        options,
        api_base,
        "my.domain",
        &[
            &OP_ATTR_SHOWCASE,
            &OP_CUSTOM_ADD,
            &OP_CUSTOM_SUM,
            &OP_PARSE_DATETIME,
        ],
    );

    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status.into_pointer(),
    }
}
