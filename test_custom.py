import os
from pathlib import Path

import onnxruntime as onnxrt
import numpy as np


ROOT = Path(__file__).parent


def test_rust_dylib():
    onnxrt.set_default_logger_severity(0)
    shared_library = str(ROOT / "rust/custom_op/target/debug/libcustom_op.dylib")
    if not os.path.exists(shared_library):
        raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

    this = os.path.dirname(__file__)
    custom_op_model = os.path.join(
        this, "testdata", "custom_op_library", "custom_op_test.onnx"
    )
    if not os.path.exists(custom_op_model):
        raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

    so1 = onnxrt.SessionOptions()
    so1.register_custom_ops_library(shared_library)

    # Model loading successfully indicates that the custom op node
    # could be resolved successfully
    sess1 = onnxrt.InferenceSession(
        custom_op_model,
        sess_options=so1,
    )
    # Run with input data
    input_name_0 = sess1.get_inputs()[0].name
    input_name_1 = sess1.get_inputs()[1].name
    output_name = sess1.get_outputs()[0].name
    input_0 = np.ones((3, 5)).astype(np.float32)
    input_1 = np.zeros((3, 5)).astype(np.float32)
    res = sess1.run(
        [output_name], {input_name_0: input_0, input_name_1: input_1}
    )
    output_expected = np.ones((3, 5)).astype(np.float32)
    np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)


# def test_register_dylib():
#     onnxrt.set_default_logger_severity(0)
#     shared_library = "custom_op_library_cxx.dylib"
#     if not os.path.exists(shared_library):
#         raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

#     this = os.path.dirname(__file__)
#     custom_op_model = os.path.join(
#         this, "testdata", "custom_op_library", "custom_op_test.onnx"
#     )
#     if not os.path.exists(custom_op_model):
#         raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

#     so1 = onnxrt.SessionOptions()
#     so1.register_custom_ops_library(shared_library)

#     # Model loading successfully indicates that the custom op node
#     # could be resolved successfully
#     sess1 = onnxrt.InferenceSession(
#         custom_op_model,
#         sess_options=so1,
#     )
#     # Run with input data
#     input_name_0 = sess1.get_inputs()[0].name
#     input_name_1 = sess1.get_inputs()[1].name
#     output_name = sess1.get_outputs()[0].name
#     input_0 = np.ones((3, 5)).astype(np.float32)
#     input_1 = np.zeros((3, 5)).astype(np.float32)
#     res = sess1.run(
#         [output_name], {input_name_0: input_0, input_name_1: input_1}
#     )
#     output_expected = np.ones((3, 5)).astype(np.float32)
#     np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

#     # Create an alias of SessionOptions instance
#     # We will use this alias to construct another InferenceSession
#     so2 = so1

#     # Model loading successfully indicates that the custom op node
#     # could be resolved successfully
#     sess2 = onnxrt.InferenceSession(
#         custom_op_model,
#         sess_options=so2,
#     )

#     # Create another SessionOptions instance with the same shared
#     # library referenced
#     so3 = onnxrt.SessionOptions()
#     so3.register_custom_ops_library(shared_library)
#     sess3 = onnxrt.InferenceSession(
#         custom_op_model,
#         sess_options=so3,
#     )
