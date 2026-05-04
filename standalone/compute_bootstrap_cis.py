"""Compute 95% bootstrap confidence intervals over test samples for each test
and write the result back into the corresponding results.json file.

Bootstrap procedure
-------------------
For each test we form a per-sample-per-timestep nRMSE vector (length = number
of test samples), then percentile-bootstrap-resample with replacement 10,000
times to estimate a 95% CI on the mean. The CI is over **test-sample
variability** (sampling uncertainty across the held-out test set); it does not
capture training-seed variance.

Inputs
------
- Cached prediction tensors at ``.cache/predictions/test_*_predictions.npz``.
  Get them by first running:
      python standalone/evaluate_predictions.py --all
  to download from HuggingFace, or
      python standalone/evaluate_predictions.py --all-entries
  to also include Test 28 + supplementary CFD configs.

Outputs
-------
Each ``results/test_<id>/results.json`` gains the following keys:

    "bootstrap_ci_pertimestep": {
        "mean": float,
        "ci_lower": float, "ci_upper": float,
        "n_samples": int, "n_bootstrap": 10000,
        "method": "percentile", "alpha": 0.05
    }
    "bootstrap_ci_frobenius": { ... same structure, computed when the npz has
                                a saved per_sample array (which is Frobenius) }

Usage
-----
    python standalone/compute_bootstrap_cis.py
    python standalone/compute_bootstrap_cis.py --tests 13 24 26
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / ".cache" / "predictions"
RESULTS = REPO_ROOT / "results"

# (test_id, init_step). init_step=0 marks static problems (no time axis).
HEADLINE_24 = [
    ("01", 10),
    ("02", 10),
    ("03", 10),
    ("04", 10),
    ("05", 10),
    ("06", 10),
    ("07", 10),
    ("08", 10),
    ("09", 10),
    ("10", 10),
    ("11", 10),
    ("13", 10),
    ("16", 10),
    ("17", 10),
    ("19", 10),
    ("20", 10),
    ("21", 0),
    ("22", 0),
    ("23", 0),
    ("24", 0),
    ("25", 0),
    ("26", 5),
    ("27", 5),
    ("29", 5),
]
SUPPLEMENTARY = [
    ("28", 5),
    ("29_M01_Eta01", 5),
    ("29_M10_Eta001", 5),
    ("29_M10_Eta01", 5),
]

# Seed-variance ablation entries: (test_key, init_step, seed).
# Note: For Darcy (Tests 24, 25) init_step=0 (static, no time axis); for time-
# dependent problems (Test 11 Burgers, Test 26 2D Diff-React) it matches the
# headline configuration.
SEED_ABLATION = [
    ("11", 10, 123),
    ("11", 10, 456),
    ("24", 0, 123),
    ("24", 0, 456),
    ("25", 0, 123),
    ("25", 0, 456),
    ("26", 5, 123),
    ("26", 5, 456),
]


def per_sample_pertimestep(pred, target, init_step):
    """Per-sample per-timestep nRMSE matching paper Table 1's metric.

    Returns a 1D array of length B with one nRMSE scalar per sample. For
    static problems (init_step=0, no time axis), this collapses to per-sample
    Frobenius nRMSE.
    """
    if init_step == 0:
        p = pred.astype(np.float32, copy=False)
        t = target.astype(np.float32, copy=False)
        B = p.shape[0]
        num = np.sqrt(((p - t) ** 2).reshape(B, -1).sum(axis=1))
        den = np.sqrt((t**2).reshape(B, -1).sum(axis=1)) + 1e-20
        return num / den

    p = pred[..., init_step:, :].astype(np.float32, copy=False)
    t = target[..., init_step:, :].astype(np.float32, copy=False)
    ndim = p.ndim
    B = p.shape[0]
    T_pred = p.shape[-2]
    nc = p.shape[-1]
    accum = np.zeros(B, dtype=np.float64)
    n_terms = 0
    for ti in range(T_pred):
        for ci in range(nc):
            if ndim == 4:  # 1D autoregressive: [B, X, T, C]
                pp = p[:, :, ti, ci]
                tt = t[:, :, ti, ci]
            elif ndim == 5:  # 2D autoregressive: [B, H, W, T, C]
                pp = p[:, :, :, ti, ci].reshape(B, -1)
                tt = t[:, :, :, ti, ci].reshape(B, -1)
            else:
                raise ValueError(f"unexpected ndim={ndim}")
            err = np.sqrt(((pp - tt) ** 2).sum(axis=-1))
            nrm = np.sqrt((tt**2).sum(axis=-1) + 1e-20)
            accum += err / nrm
            n_terms += 1
    return accum / n_terms


def bootstrap_ci(per_sample, n_boot=10_000, alpha=0.05, seed=42):
    """Vectorized percentile bootstrap CI for the sample mean."""
    rng = np.random.default_rng(seed)
    N = len(per_sample)
    ps = per_sample.astype(np.float64, copy=False)
    if n_boot * N > 20_000_000:
        means = np.empty(n_boot, dtype=np.float64)
        chunk = max(1, 20_000_000 // N)
        for s in range(0, n_boot, chunk):
            e = min(s + chunk, n_boot)
            idx = rng.integers(0, N, size=(e - s, N))
            means[s:e] = ps[idx].mean(axis=1)
    else:
        idx = rng.integers(0, N, size=(n_boot, N))
        means = ps[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def load_test_28_per_sample():
    """Test 28 chunks each store a Frobenius per_sample slice; concatenate."""
    chunks = sorted(CACHE.glob("test_28_predictions_chunk_*.npz"))
    if not chunks:
        return None
    parts = []
    for c in chunks:
        d = np.load(c)
        if "per_sample" in d:
            parts.append(d["per_sample"].astype(np.float64))
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def process_test(test_key, init_step):
    if test_key == "28":
        ps28 = load_test_28_per_sample()
        if ps28 is None:
            return {}
        f_lo, f_hi = bootstrap_ci(ps28)
        return {
            "bootstrap_ci_frobenius": {
                "mean": float(ps28.mean()),
                "ci_lower": f_lo,
                "ci_upper": f_hi,
                "n_samples": int(len(ps28)),
                "n_bootstrap": 10_000,
                "method": "percentile",
                "alpha": 0.05,
                "note": (
                    "Test 28 chunks store per-sample Frobenius nRMSE only; "
                    "per-timestep recompute from full tensors infeasible "
                    "(~26 GB upcast)."
                ),
            }
        }

    npz_path = CACHE / f"test_{test_key}_predictions.npz"
    if not npz_path.exists():
        print(f"  test_{test_key}: cache miss at {npz_path}", file=sys.stderr)
        return {}
    d = np.load(npz_path)
    if "preds" not in d or "targets" not in d:
        print(f"  test_{test_key}: missing preds/targets in npz", file=sys.stderr)
        return {}
    preds = d["preds"]
    targets = d["targets"]
    ps_full = d.get("per_sample")

    # Detect supplementary CFD configs where preds is only a 10-sample slice
    # but per_sample stores the full-test-set Frobenius array. In that case the
    # per-timestep bootstrap from the slice would mislead reviewers (the point
    # estimate displayed in Table 1 is the full-test-set value, not the slice
    # mean). Use only the saved full-N per_sample array for these.
    is_slice_only = ps_full is not None and len(ps_full) > preds.shape[0]

    out = {}

    if not is_slice_only:
        print(
            f"  {test_key}: computing per-sample per-timestep nRMSE (B={preds.shape[0]})...",
            flush=True,
        )
        ps_pt = per_sample_pertimestep(preds, targets, init_step)
        pt_lo, pt_hi = bootstrap_ci(ps_pt)
        out["bootstrap_ci_pertimestep"] = {
            "mean": float(ps_pt.mean()),
            "ci_lower": pt_lo,
            "ci_upper": pt_hi,
            "n_samples": int(len(ps_pt)),
            "n_bootstrap": 10_000,
            "method": "percentile",
            "alpha": 0.05,
        }

    # Frobenius CI from the saved per_sample array. For supplementary configs
    # this is the only statistically valid CI we can offer (full N samples,
    # matches the JSON's nrmse_frobenius exactly).
    if ps_full is not None:
        f_lo, f_hi = bootstrap_ci(ps_full.astype(np.float64))
        out["bootstrap_ci_frobenius"] = {
            "mean": float(ps_full.mean()),
            "ci_lower": f_lo,
            "ci_upper": f_hi,
            "n_samples": int(len(ps_full)),
            "n_bootstrap": 10_000,
            "method": "percentile",
            "alpha": 0.05,
        }
        if is_slice_only:
            out["bootstrap_ci_frobenius"]["note"] = (
                "Computed from full-N per_sample array (Frobenius); the "
                "preds tensor in this npz is a 10-sample visualization slice. "
                "Per-timestep CI not reported for this config because per-"
                "timestep recompute requires raw tensors not hosted at full N."
            )
    return out


def process_seed_ablation_entry(test_key, init_step, seed):
    """Process one seed-ablation entry. Reads the cached ``predictions.npz``
    from ``.cache/predictions/seed_ablation/test_<id>_seed<S>_predictions.npz``
    (downloading from HuggingFace if not cached) and writes CI blocks back to
    ``results/seed_ablation/test_<id>_seed<S>/results.json``.
    """
    npz_path = CACHE / "seed_ablation" / f"test_{test_key}_seed{seed}_predictions.npz"
    if not npz_path.exists():
        # Cache miss → download from HF
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                f"  test_{test_key}_seed{seed}: not cached and huggingface_hub "
                f"is not installed; install with `pip install huggingface_hub`",
                file=sys.stderr,
            )
            return {}
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        remote_name = f"seed_ablation/test_{test_key}_seed{seed}_predictions.npz"
        print(
            f"  test_{test_key}_seed{seed}: downloading {remote_name} from HF...",
            flush=True,
        )
        try:
            hf_hub_download(
                repo_id="pdebench-fno-audit/fno-predictions",
                filename=remote_name,
                repo_type="dataset",
                local_dir=str(CACHE / "seed_ablation_dl"),
            )
            # hf_hub_download places it at <local_dir>/<filename> preserving subdir
            src = CACHE / "seed_ablation_dl" / remote_name
            if src.exists():
                src.replace(npz_path)
        except Exception as e:
            print(
                f"  test_{test_key}_seed{seed}: HF download failed: {e}",
                file=sys.stderr,
            )
            return {}
    if not npz_path.exists():
        return {}
    d = np.load(npz_path)
    if "preds" not in d or "targets" not in d:
        print(
            f"  test_{test_key}_seed{seed}: missing preds/targets in npz",
            file=sys.stderr,
        )
        return {}
    preds = d["preds"]
    targets = d["targets"]
    ps_full = d.get("per_sample")

    print(
        f"  {test_key} seed {seed}: computing per-sample per-timestep nRMSE "
        f"(B={preds.shape[0]})...",
        flush=True,
    )
    ps_pt = per_sample_pertimestep(preds, targets, init_step)
    pt_lo, pt_hi = bootstrap_ci(ps_pt)

    out = {
        "bootstrap_ci_pertimestep": {
            "mean": float(ps_pt.mean()),
            "ci_lower": pt_lo,
            "ci_upper": pt_hi,
            "n_samples": int(len(ps_pt)),
            "n_bootstrap": 10_000,
            "method": "percentile",
            "alpha": 0.05,
        }
    }
    if ps_full is not None and len(ps_full) == len(ps_pt):
        f_lo, f_hi = bootstrap_ci(ps_full.astype(np.float64))
        out["bootstrap_ci_frobenius"] = {
            "mean": float(ps_full.mean()),
            "ci_lower": f_lo,
            "ci_upper": f_hi,
            "n_samples": int(len(ps_full)),
            "n_bootstrap": 10_000,
            "method": "percentile",
            "alpha": 0.05,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        nargs="+",
        default=None,
        help=(
            "Subset of test IDs to process (e.g. --tests 13 24 26). "
            "Default: all 24 headline + 4 supplementary entries."
        ),
    )
    parser.add_argument(
        "--seed-ablation",
        action="store_true",
        help=(
            "Process seed-ablation entries under results/seed_ablation/ "
            "(Tests 11, 24, 25, 26 at seeds 123 and 456). Reads predictions "
            "from .cache/predictions/seed_ablation/."
        ),
    )
    args = parser.parse_args()

    if args.seed_ablation:
        for test_key, init_step, seed in SEED_ABLATION:
            ci = process_seed_ablation_entry(test_key, init_step, seed)
            if not ci:
                continue
            json_path = (
                RESULTS
                / "seed_ablation"
                / f"test_{test_key}_seed{seed}"
                / "results.json"
            )
            if not json_path.exists():
                print(f"  {json_path} not found, skipping", file=sys.stderr)
                continue
            with open(json_path) as f:
                data = json.load(f)
            data.update(ci)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            block = ci.get("bootstrap_ci_pertimestep") or ci.get(
                "bootstrap_ci_frobenius"
            )
            if block is None:
                continue
            metric = "PT" if "bootstrap_ci_pertimestep" in ci else "FRO"
            print(
                f"  {test_key} seed {seed}: {metric} mean={block['mean']:.4e} "
                f"95% CI=[{block['ci_lower']:.4e}, {block['ci_upper']:.4e}] "
                f"(N={block['n_samples']})"
            )
        return

    entries = HEADLINE_24 + SUPPLEMENTARY
    if args.tests:
        wanted = set(args.tests)
        entries = [e for e in entries if e[0] in wanted]
        if len(entries) < len(wanted):
            missing = wanted - {e[0] for e in entries}
            print(f"Unknown test IDs: {sorted(missing)}", file=sys.stderr)

    for test_key, init_step in entries:
        ci = process_test(test_key, init_step)
        if not ci:
            continue
        json_path = RESULTS / f"test_{test_key}" / "results.json"
        if not json_path.exists():
            print(f"  {json_path} not found, skipping", file=sys.stderr)
            continue
        with open(json_path) as f:
            data = json.load(f)
        data.update(ci)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        block = ci.get("bootstrap_ci_pertimestep") or ci.get("bootstrap_ci_frobenius")
        if block is None:
            continue
        metric = "PT" if "bootstrap_ci_pertimestep" in ci else "FRO"
        print(
            f"  {test_key}: {metric} mean={block['mean']:.4e} "
            f"95% CI=[{block['ci_lower']:.4e}, {block['ci_upper']:.4e}] "
            f"(N={block['n_samples']})"
        )


if __name__ == "__main__":
    main()
