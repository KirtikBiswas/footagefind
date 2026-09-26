import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from footagefind.runtime import CPU, QNN, OrtModel, probe_node_placement, select_providers

RT = {"providers": [QNN, CPU], "qnn_backend_path": "QnnHtp.dll", "qnn_htp_performance_mode": "burst"}


def test_qnn_selected_first_when_available():
    plan = select_providers(RT, available=[QNN, CPU])
    assert plan.names == [QNN, CPU]
    qnn_opts = dict(plan.providers)[QNN]
    assert qnn_opts["backend_path"] == "QnnHtp.dll" and qnn_opts["htp_performance_mode"] == "burst"
    assert plan.skipped == []


def test_qnn_skipped_on_x86_build():
    plan = select_providers(RT, available=["AzureExecutionProvider", CPU])
    assert plan.names == [CPU] and plan.skipped == [QNN]


def test_falls_back_to_cpu_if_nothing_requested_is_available():
    plan = select_providers({"providers": [QNN]}, available=[CPU])
    assert plan.names == [CPU]


def test_no_silent_cpu_when_fallback_disabled():
    with pytest.raises(RuntimeError, match="disable_cpu_ep_fallback"):
        select_providers({"providers": [QNN, CPU], "disable_cpu_ep_fallback": True}, available=[CPU])


def test_strict_mode_registers_only_the_npu():
    # ORT rejects CPU EP + disable_cpu_ep_fallback together, so CPU must be dropped.
    plan = select_providers({"providers": [QNN, CPU], "disable_cpu_ep_fallback": True}, available=[QNN, CPU])
    assert plan.names == [QNN] and CPU in plan.skipped


def test_env_override(monkeypatch):
    monkeypatch.setenv("FOOTAGEFIND_PROVIDERS", "CPUExecutionProvider")
    plan = select_providers(RT, available=[QNN, CPU])
    assert plan.names == [CPU]


def tiny_model(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).ravel().tolist())
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Relu", ["h"], ["y"])],
                          "tiny", [x], [y], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


def test_ortmodel_reports_actual_provider_and_latency(tmp_path):
    p = tiny_model(tmp_path / "tiny.onnx")
    m = OrtModel(p, RT)                       # QNN requested, not present in this build
    out = m.run({"x": np.array([[1, -2, 3, -4]], np.float32)})[0]
    np.testing.assert_array_equal(out, [[1, 0, 3, 0]])
    d = m.describe()
    assert d["primary_provider"] == CPU and d["runs"] == 1 and d["mean_ms"] is not None
    assert d["skipped_unavailable"] == [QNN] or QNN in d["session_providers"]


def test_ort_itself_rejects_cpu_ep_with_fallback_disabled(tmp_path):
    """Documents the ORT behaviour select_providers() works around."""
    import onnxruntime as ort
    p = tiny_model(tmp_path / "tiny.onnx")
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    with pytest.raises(Exception, match="(?i)conflicting|fallback"):
        ort.InferenceSession(str(p), sess_options=so, providers=[CPU])


def test_strict_mode_on_x86_fails_loudly(tmp_path):
    p = tiny_model(tmp_path / "tiny.onnx")
    with pytest.raises(RuntimeError, match="onnxruntime-qnn"):
        OrtModel(p, {"providers": [QNN, CPU], "disable_cpu_ep_fallback": True})


def test_node_placement_probe(tmp_path):
    p = tiny_model(tmp_path / "tiny.onnx")
    r = probe_node_placement(p, {"providers": [CPU]})
    assert r["nodes_by_provider"].get(CPU, 0) >= 1


def test_missing_model_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        OrtModel(tmp_path / "nope.onnx", RT)


def test_resize_scales_rewritten_to_integer_sizes(tmp_path):
    """Regression: quantizing Resize's float `scales` dropped a channel in w8a16 YOLOv8n."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import onnxruntime as ort
    from quantize_onnx import resize_scales_to_sizes

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    scales = helper.make_node("Constant", [], ["s"], value=helper.make_tensor("sv", TensorProto.FLOAT, [4], [1, 1, 2, 2]))
    rs = helper.make_node("Resize", ["x", "", "s"], ["y"], mode="nearest", name="up")
    m = helper.make_model(helper.make_graph([scales, rs], "g", [x], [y]), opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    feed = {"x": np.random.default_rng(0).random((1, 3, 4, 4), dtype=np.float32)}
    ref = ort.InferenceSession(m.SerializeToString()).run(None, feed)[0]
    assert resize_scales_to_sizes(m) == 1
    node = [n for n in m.graph.node if n.op_type == "Resize"][0]
    assert node.input[2] == "" and node.input[3] == "up_sizes"
    out = ort.InferenceSession(m.SerializeToString()).run(None, feed)[0]
    assert out.shape == (1, 3, 8, 8)
    np.testing.assert_array_equal(out, ref)
