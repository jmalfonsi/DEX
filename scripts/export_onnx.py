"""Export untrained DEX graphs and check ONNX parity. No GPU or paid service."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, helper
import torch
from dex.model import Config, DEX


def export(module, args, path, names, outputs, dynamic):
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=names, output_names=outputs,
                          dynamic_axes=dynamic, opset_version=18, dynamo=False)
    onnx.checker.check_model(str(path))


def quantize_weights_only(source, target):
    # Fixed weight dequantization, no activation scales shared across batch rows.
    # This preserves the fragment/field independence of the floating-point graph.
    graph = onnx.load(str(source))
    initializers, nodes = [], []
    for tensor in graph.graph.initializer:
        array = numpy_helper.to_array(tensor)
        if array.dtype == np.float32 and array.ndim >= 2 and array.size >= 256:
            scale = np.asarray(max(float(np.abs(array).max()) / 127., 1e-12), dtype=np.float32)
            quant = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
            q, sc, zp = tensor.name + '_int8', tensor.name + '_scale', tensor.name + '_zero'
            initializers.extend([numpy_helper.from_array(quant, q), numpy_helper.from_array(scale, sc),
                                 numpy_helper.from_array(np.asarray(0, dtype=np.int8), zp)])
            nodes.append(helper.make_node('DequantizeLinear', [q, sc, zp], [tensor.name]))
        else:
            initializers.append(tensor)
    old_nodes = list(graph.graph.node)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes + old_nodes)
    onnx.save(graph, str(target))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(29)
    cfg = Config()
    model = DEX(cfg).eval()
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    ids = torch.randint(1, cfg.vocab, (4, 128))
    kinds = torch.zeros(4, dtype=torch.long)
    with tempfile.TemporaryDirectory(prefix="dex-export-") as tmp:
        fp32 = Path(tmp) / "encoder.onnx"
        export(model.encoder, (ids, kinds), fp32, ["ids", "kinds"], ["tokens", "summaries"],
               {"ids": {0: "chunks", 1: "length"}, "kinds": {0: "chunks"},
                "tokens": {0: "chunks", 1: "length"}, "summaries": {0: "chunks"}})
        quantize_weights_only(fp32, out/"encoder.int8.onnx")
        onnx.checker.check_model(str(out/"encoder.int8.onnx"))
    with torch.no_grad():
        tokens, summaries = model.encoder(ids, kinds)
    qs = torch.randn(4, cfg.rank)
    bank = torch.nn.functional.normalize(torch.randn(32, cfg.rank), dim=-1)
    valid = ids != 0
    args = (qs, summaries, tokens, valid, bank)
    names = ["questions", "summaries", "tokens", "valid", "bank"]
    dynamic = {"questions": {0: "fields"}, "summaries": {0: "chunks"},
               "tokens": {0: "chunks", 1: "length"}, "valid": {0: "chunks", 1: "length"},
               "bank": {0: "options"}, "logits": {0: "fields", 1: "options"}}
    export(model.core, args, out/"core.fp32.onnx", names, ["logits"], dynamic)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    core = ort.InferenceSession(str(out/"core.fp32.onnx"), sess_options=session_options)
    with torch.no_grad():
        pt = model.core(*args).numpy()
    actual = core.run(None, {name: t.numpy() for name, t in zip(names, args)})[0]
    np.testing.assert_allclose(actual, pt, atol=2e-5, rtol=2e-5)
    encoder = ort.InferenceSession(str(out/"encoder.int8.onnx"), sess_options=session_options)
    enc = encoder.run(None, {"ids": ids.numpy(), "kinds": kinds.numpy()})
    manifest = {"status": "random_weights_untrained_uncalibrated", "seed": 29,
        "config": vars(cfg), "parameters": sum(p.numel() for p in model.parameters()),
        "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "quantization": "weight-only int8 QDQ, float32 activations; not int8 compute",
        "core_max_abs_error": float(np.max(np.abs(actual-pt))),
        "encoder_quantization_max_abs_error": float(np.max(np.abs(enc[0]-tokens.numpy()))),
        "files": {name: (out/name).stat().st_size for name in ("encoder.int8.onnx", "core.fp32.onnx")}}
    (out/"model-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
