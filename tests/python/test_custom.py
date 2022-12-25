from pathlib import Path
from platform import platform
import pytest

from onnx import helper, TensorProto
import onnxruntime as onnxrt
import numpy as np

ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def custom_add_model():
    # Using custom operators with the DSL (i.e. `onnx.parse`) for
    # defining ONNX models seems to be unsupported...
    node = helper.make_node("CustomAdd", ["A", "B"], ["C"], domain="my.domain")
    value_infos_input = [
        helper.make_value_info(
            "A", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
        helper.make_value_info(
            "B", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "C", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
    ]
    graph = helper.make_graph(
        [node],
        "graph",
        value_infos_input,
        value_infos_output,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("my.domain", 1)])


@pytest.fixture
def parse_datetime_model():
    # Using custom operators with the DSL (i.e. `onnx.parse`) for
    # defining ONNX models seems to be unsupported...
    node = helper.make_node(
        "ParseDateTime",
        ["A"],
        ["B"],
        domain="my.domain",
        **{"fmt": "%d.%m.%Y %H:%M %P %z"},
    )
    value_infos_input = [
        helper.make_value_info(
            "A", helper.make_tensor_type_proto(TensorProto.STRING, [])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "B", helper.make_tensor_type_proto(TensorProto.INT64, [])
        ),
    ]
    graph = helper.make_graph(
        [node],
        "graph",
        value_infos_input,
        value_infos_output,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("my.domain", 1)])


@pytest.fixture
def shared_lib() -> Path:
    if "macOS" in platform():
        file_name = "libexample.dylib"
    else:
        file_name = "libexample.so"
    path = ROOT / f"target/debug/deps/{file_name}"
    if not path.exists():
        raise FileNotFoundError("Unable to find '{0}'".format(shared_library))
    return path


def setup_session(shared_lib: Path, model) -> onnxrt.InferenceSession:
    onnxrt.set_default_logger_severity(3)
    so = onnxrt.SessionOptions()
    so.register_custom_ops_library(str(shared_lib))

    # Model loading successfully indicates that the custom op node
    # could be resolved successfully
    return onnxrt.InferenceSession(model.SerializeToString(), sess_options=so)


def test_custom_add(shared_lib, custom_add_model):
    sess = setup_session(shared_lib, custom_add_model)
    # Run with input data
    input_name_0 = sess.get_inputs()[0].name
    input_name_1 = sess.get_inputs()[1].name
    output_name = sess.get_outputs()[0].name
    input_0 = np.ones((3, 5)).astype(np.float32)
    input_1 = np.zeros((3, 5)).astype(np.float32)
    res = sess.run([output_name], {input_name_0: input_0, input_name_1: input_1})
    output_expected = np.ones((3, 5)).astype(np.float32)
    np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)


def test_parse_datetime(shared_lib, parse_datetime_model):
    sess = setup_session(shared_lib, parse_datetime_model)
    # Run with input data
    input_feed = {
        sess.get_inputs()[0]
        .name: np.array(["5.8.1994 8:00 am +0000", "5.8.2022 8:00 am +0000"])
        .astype(np.str_),
    }
    output_name = sess.get_outputs()[0].name
    res = sess.run([output_name], input_feed)
    output_expected = np.array([776073600, 1659686400])
    np.testing.assert_equal(output_expected, res[0])
