# Standalone Verification Scripts

Pure Python/NumPy scripts for verifying our results. **No cloud dependencies.**

## Quick Start

```bash
# CPU-only verification (no GPU, no PyTorch needed):
pip install -r ../requirements-min.txt
python evaluate_predictions.py --catalog        # see available tests
python evaluate_predictions.py --json-only      # table from saved JSONs
python evaluate_predictions.py --test 13        # recompute one test from HF
python evaluate_predictions.py --all            # recompute all 24 tests

# Checkpoint evaluation (needs GPU + PyTorch + PDEBench HDF5):
pip install -r ../requirements.txt
python evaluate.py --checkpoint /path/to/model.pt --data /path/to/data.hdf5 ...
```

## Scripts

| File | Purpose | Needs GPU? | Needs Data? |
|------|---------|:----------:|:-----------:|
| `evaluate_predictions.py` | Recompute nRMSE from HF prediction tensors | No | Downloads from HF |
| `evaluate.py` | Full checkpoint evaluation on PDEBench test set | **Yes** | **Yes** (HDF5) |
| `models.py` | All architectures, metrics, NormalizedMSELoss | No | No |

## What Can I Verify?

### "Do the paper's 21+1+2 results match the saved data?"

```bash
python evaluate_predictions.py --json-only
```

Expected: **21 clear wins, 1 marginal, 2 misses out of 24.**

### "Can I independently recompute the nRMSE values?"

```bash
python evaluate_predictions.py --all
```

Downloads prediction tensors from HuggingFace (~4.8 GB total). Each `.npz`
contains both `preds` (model output) and `targets` (ground truth). The script
computes nRMSE from scratch and runs integrity checks.

### "Where do arXiv v7 and NeurIPS supplement baselines differ?"

```bash
python evaluate_predictions.py --json-only --verify-baselines
```

Shows the 8 rows where the two source versions differ (Advection + Burgers).
Win/loss classifications are unchanged under both.

### "Can I evaluate a checkpoint myself?"

```bash
python evaluate.py --checkpoint ../weights/test_13_best_model.pt \
    --data /path/to/1D_ReacDiff_Nu2.0_Rho1.0.hdf5 \
    --pde 1d --modes 12 --width 32 --init_step 5 --nc 1
```

Requires model weights from [HuggingFace](https://huggingface.co/pdebench-fno-audit/fno-weights) and PDEBench data from [DaRUS](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986).

## Results Directory

```
../results/
├── test_01/ ... test_29/           ← results.json for each of 24 tests
├── test_28_incompressible_ns/      ← detailed Test 28 investigation
│   ├── standard_fno_256/           ← Standard FNO baseline (nRMSE 0.280)
│   ├── dst_psi_v2_512/             ← DST-ψ approach (nRMSE 0.356)
│   ├── vorticity_poisson_512/      ← Vorticity-Poisson FNO (nRMSE 0.262)
│   └── boundary_analysis/          ← Wall-slip diagnostics (14× tangential/normal)
└── ablations/
    ├── A1_normalized_mse/          ← nMSE removal: 2.05× degradation
    ├── A2_local_convolution/       ← Local conv removal: ≤2.11× degradation
    ├── W2_2d_mse_ablation/         ← 2D MSE ablation: 32.1× vs 43.7×
    └── W3_proper_validation/       ← Protocol robustness: <2% change
```
