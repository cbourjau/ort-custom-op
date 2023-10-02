from pathlib import Path
from platform import platform
import pytest

from onnx import helper, TensorProto
from onnx import numpy_helper
import numpy as np
import onnxruntime as onnxrt

ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(params=[TensorProto.FLOAT, TensorProto.DOUBLE])
def onnx_tensor_type(request):
    return request.param


@pytest.fixture
def custom_add_model(onnx_tensor_type):
    # Using custom operators with the DSL (i.e. `onnx.parse`) for
    # defining ONNX models seems to be unsupported...
    node = helper.make_node("CustomAdd", ["A", "B"], ["C"], domain="my.domain")
    value_infos_input = [
        helper.make_value_info(
            "A", helper.make_tensor_type_proto(onnx_tensor_type, [None, None])
        ),
        helper.make_value_info(
            "B", helper.make_tensor_type_proto(onnx_tensor_type, [None, None])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "C", helper.make_tensor_type_proto(onnx_tensor_type, [None, None])
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
            "B", helper.make_tensor_type_proto(TensorProto.DOUBLE, [])
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
def attr_showcase_model():
    u8_tensor = numpy_helper.from_array(np.array([102, 111, 111], dtype=np.uint8))
    node = helper.make_node(
        "AttrShowcase",
        ["IN1", "IN2", "IN3"],
        ["OUT1", "OUT2", "OUT3"],
        domain="my.domain",
        **{
            "float_attr": 3.14,
            "int_attr": 42,
            "string_attr": "bar",
            "floats_attr": [3.14, 3.14],
            "ints_attr": [42, 42],
            "u8_tensor": u8_tensor,
        },
    )
    value_infos_input = [
        helper.make_value_info(
            "IN1", helper.make_tensor_type_proto(TensorProto.FLOAT, [])
        ),
        helper.make_value_info(
            "IN2", helper.make_tensor_type_proto(TensorProto.INT64, [])
        ),
        helper.make_value_info(
            "IN3", helper.make_tensor_type_proto(TensorProto.STRING, [])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "OUT1", helper.make_tensor_type_proto(TensorProto.FLOAT, [])
        ),
        helper.make_value_info(
            "OUT2", helper.make_tensor_type_proto(TensorProto.INT64, [])
        ),
        helper.make_value_info(
            "OUT3", helper.make_tensor_type_proto(TensorProto.STRING, [])
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
def attr_showcase_model_missing_attributes():
    """Create a model which lacks all attributes to test the error case."""
    node = helper.make_node(
        "AttrShowcase",
        ["IN1", "IN2", "IN3"],
        ["OUT1", "OUT2", "OUT3"],
        domain="my.domain",
    )
    value_infos_input = [
        helper.make_value_info(
            "IN1", helper.make_tensor_type_proto(TensorProto.FLOAT, [])
        ),
        helper.make_value_info(
            "IN2", helper.make_tensor_type_proto(TensorProto.INT64, [])
        ),
        helper.make_value_info(
            "IN3", helper.make_tensor_type_proto(TensorProto.STRING, [])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "OUT1", helper.make_tensor_type_proto(TensorProto.FLOAT, [])
        ),
        helper.make_value_info(
            "OUT2", helper.make_tensor_type_proto(TensorProto.INT64, [])
        ),
        helper.make_value_info(
            "OUT3", helper.make_tensor_type_proto(TensorProto.STRING, [])
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
def custom_sum_model():
    # Using custom operators with the DSL (i.e. `onnx.parse`) for
    # defining ONNX models seems to be unsupported...
    node = helper.make_node("CustomSum", ["A", "B", "C"], ["D"], domain="my.domain")
    value_infos_input = [
        helper.make_value_info(
            "A", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
        helper.make_value_info(
            "B", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
        helper.make_value_info(
            "C", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "D", helper.make_tensor_type_proto(TensorProto.FLOAT, [None, None])
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
def variadic_identity_model():
    # Using custom operators with the DSL (i.e. `onnx.parse`) for
    # defining ONNX models seems to be unsupported...
    node = helper.make_node(
        "VariadicIdentity", ["A", "B"], ["C", "D"], domain="my.domain"
    )
    value_infos_input = [
        helper.make_value_info(
            "A", helper.make_tensor_type_proto(TensorProto.FLOAT, [None])
        ),
        helper.make_value_info(
            "B", helper.make_tensor_type_proto(TensorProto.FLOAT, [None])
        ),
    ]
    value_infos_output = [
        helper.make_value_info(
            "C", helper.make_tensor_type_proto(TensorProto.FLOAT, [None])
        ),
        helper.make_value_info(
            "D", helper.make_tensor_type_proto(TensorProto.FLOAT, [None])
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
        raise FileNotFoundError("Unable to find '{0}'".format(path))
    return path


def setup_session(shared_lib: Path, model) -> onnxrt.InferenceSession:
    onnxrt.set_default_logger_severity(3)
    so = onnxrt.SessionOptions()
    so.register_custom_ops_library(str(shared_lib))
    so.log_severity_level = 0

    # Model loading successfully indicates that the custom op node
    # could be resolved successfully
    return onnxrt.InferenceSession(model.SerializeToString(), sess_options=so)


def test_custom_add(shared_lib, custom_add_model, onnx_tensor_type):
    dtype = helper.tensor_dtype_to_np_dtype(onnx_tensor_type)
    sess = setup_session(shared_lib, custom_add_model)
    # Run with input data
    input_name_0 = sess.get_inputs()[0].name
    input_name_1 = sess.get_inputs()[1].name
    output_name = sess.get_outputs()[0].name
    input_0 = np.ones((3, 5)).astype(dtype)
    input_1 = np.zeros((3, 5)).astype(dtype)
    res = sess.run([output_name], {input_name_0: input_0, input_name_1: input_1})
    output_expected = np.ones((3, 5)).astype(dtype)
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


def test_attr_showcase(shared_lib, attr_showcase_model):
    sess = setup_session(shared_lib, attr_showcase_model)
    # Run with input data
    input_feed = {
        "IN1": np.array([0], np.float32),
        "IN2": np.array([0], np.int64),
        "IN3": np.array(["foo"], np.str_),
    }
    a, b, c = sess.run(None, input_feed)
    np.testing.assert_equal(a, np.array([3.14], np.float32))
    np.testing.assert_equal(b, [42])
    np.testing.assert_equal(c, ["foo + bar"])


@pytest.mark.skip(
    reason="Crashes the interpreter but prints a decent error message."
)
def test_attr_showcase_missing_attrs(
    shared_lib,
    attr_showcase_model_missing_attributes,
):
    sess = setup_session(shared_lib, attr_showcase_model_missing_attributes)


def test_custom_sum(shared_lib, custom_sum_model):
    sess = setup_session(shared_lib, custom_sum_model)
    # Run with input data
    input_names = [inp.name for inp in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name
    inputs = {n: np.ones((3, 5)).astype(np.float32) for n in input_names}
    res = sess.run([output_name], inputs)
    output_expected = sum(inputs.values())
    np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)


def test_variadic_identity(shared_lib, variadic_identity_model):
    sess = setup_session(shared_lib, variadic_identity_model)
    # Run with input data
    input_feed = {
        "A": np.array([0], np.float32),
        "B": np.array([0], np.float32),
    }
    c, d = sess.run(None, input_feed)
    a, b = input_feed.values()
    np.testing.assert_equal(a, c)
    np.testing.assert_equal(b, d)
