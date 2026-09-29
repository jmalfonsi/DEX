"""Measure mechanisms on synthetic vectors. Never a semantic/accuracy benchmark."""
import json
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dex.bounds import softmax
from dex.model import Config, DEX
from dex.refine import ResidualDecision


def cases():
    rng = np.random.default_rng(29)
    for name in ("concentrated", "diffuse", "ambiguous"):
        c, n, r = 128, 10000, 32
        bank = rng.normal(size=(n, r))
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        delta = np.tanh(rng.normal(size=(c, r))) / np.sqrt(r)
        weights = softmax(rng.normal(size=c) * (4 if name == "concentrated" else 1))
        correction = np.tanh(rng.normal(size=n))
        coarse = bank[0] * (5 if name == "concentrated" else (.3 if name == "ambiguous" else 1.5))
        yield name, coarse, weights, bank, delta, correction


def main():
    torch.set_num_threads(4)
    results = {"status": "untrained_synthetic_only", "platform": platform.platform(),
               "torch": torch.__version__, "threads": 4, "numpy": np.__version__, "cases": []}
    for name, coarse, weights, bank, delta, correction in cases():
        reference = softmax(8 * bank @ (coarse + weights @ delta) + correction)
        runner = ResidualDecision(coarse, weights, bank, lambda ids: delta[ids], lambda ids: correction[ids])
        start = time.perf_counter()
        out = runner.run(batch_size=64)
        elapsed = 1000 * (time.perf_counter()-start)
        results["cases"].append({"name": name, "chunks": len(weights), "options": len(bank),
            "rank": len(coarse), "chunks_refined": out.chunks_refined,
            "options_refined": out.options_refined, "rounds": out.rounds,
            "scheduler_ms": elapsed, "selected_p": out.probability,
            "full_p": float(reference[out.value]), "interval": out.probability_interval,
            "same_argmax": out.value == int(np.argmax(reference)),
            "error": abs(out.probability-float(reference[out.value]))})
    cfg = Config()
    torch.manual_seed(29)
    model = DEX(cfg).eval()
    results["parameters"] = sum(p.numel() for p in model.parameters())
    results["fp32_weights_mib"] = results["parameters"]*4/2**20
    results["ideal_int8_weights_mib_excluding_scales"] = results["parameters"]/2**20
    ids = torch.randint(1, cfg.vocab, (4, 128))
    kind = torch.zeros(4, dtype=torch.long)
    with torch.inference_mode():
        model.encoder(ids, kind)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            model.encoder(ids, kind)
            samples.append(1000*(time.perf_counter()-start))
    results["encoder_512_tokens_cpu_fp32_ms"] = {"samples": samples, "median": float(np.median(samples)),
                                               "note": "5 observations; not a reliable p95"}
    outpath = Path(__file__).resolve().parents[1]/"artifacts"/"cpu-mechanisms.json"
    outpath.parent.mkdir(exist_ok=True)
    outpath.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
