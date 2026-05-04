# Seed-variance ablation

This directory contains additional training-seed runs for four selected
PDEBench configurations whose paper-reported classification was either
borderline (CI overlaps baseline under bootstrap) or carries an
unusually-large headline factor:

- **Test 11** (1D Burgers ν=1.0, point-estimate miss, bootstrap CI touches baseline)
- **Test 24** (2D Darcy β=10, marginal point-estimate win, bootstrap CI ends at baseline)
- **Test 25** (2D Darcy β=100, robust miss)
- **Test 26** (2D diffusion-reaction, 43.7× headline factor)

The paper's main results all use seed 42. This directory adds runs at
seeds 123 and 456 for each of the four selected tests, so each row has
three independent training seeds in total.

## Why these four

Bootstrap confidence intervals over held-out test samples (Appendix~P
of the paper, `--ci` flag of `evaluate_predictions.py`) only quantify
test-sample variability. They do not measure *training-seed*
variability. The reviewer ask was specifically: **"3-5 training seeds
for Tests 3, 11, 24, 25, plus one large-gain test such as 26 or 20"**.

Test 3's training script was a one-off earlier run not committed to
this repository, so we cannot rerun it under the documented training
recipe. The remaining four selected tests are the borderline rows that
most affect the headline classification.

## Layout

```
seed_ablation/
  test_11_seed123/results.json    # 1D Burgers nu=1.0, seed 123
  test_11_seed456/results.json    # 1D Burgers nu=1.0, seed 456
  test_24_seed123/results.json    # 2D Darcy beta=10, seed 123
  test_24_seed456/results.json    # 2D Darcy beta=10, seed 456
  test_25_seed123/results.json    # 2D Darcy beta=100, seed 123
  test_25_seed456/results.json    # 2D Darcy beta=100, seed 456
  test_26_seed123/results.json    # 2D Diff-React, seed 123
  test_26_seed456/results.json    # 2D Diff-React, seed 456
  README.md                       # this file
```

Each `results.json` records the same fields as the corresponding
seed-42 entry under `results/test_NN/`, plus:

- `seed`: the integer seed used for `torch.manual_seed`,
  `np.random.seed`, and `torch.cuda.manual_seed_all`.
- `bootstrap_ci_pertimestep`, `bootstrap_ci_frobenius`: 95% percentile
  bootstrap CIs computed exactly as in `compute_bootstrap_cis.py` for
  the seed-42 results.

## Reproducing these runs

The training scripts under `training/` accept a `seed` parameter
(default 42). To reproduce a seed-123 run for Test 24:

```bash
modal run training/2d_darcy_flow/fno2d_darcy_batch.py::train --test-key=darcy_beta10.0 --seed=123
```

The script writes to `/results/test_24_seed123/` on the Modal volume
when `seed != 42` (and the original `/results/test_24/` is untouched).

## Bootstrap CI script

`standalone/compute_bootstrap_cis.py` reads these `results.json`
files and recomputes the CI from the saved per-sample arrays in the
companion `predictions.npz` files (hosted on HuggingFace). To run:

```bash
python standalone/compute_bootstrap_cis.py --seed-ablation
```

## Caveats

1. **Per-timestep vs Frobenius.** Tests 24 and 25 (Darcy) are static
   problems with no time axis; per-timestep nRMSE collapses to
   per-sample Frobenius. Test 11 and Test 26 are time-dependent and
   carry separate per-timestep and Frobenius CIs.
2. **Seed scope.** The `seed` controls weight initialization, batch
   shuffling, and the stratified val split (`torch.randperm` line in
   each training script). The held-out test set indices `[N_TRAIN +
   N_VAL : N_TRAIN + N_VAL + N_TEST]` are deterministic and identical
   across seeds.
3. **Single-seed-per-config remains for the other 20 tests.** This
   directory adds 3-seed evidence only for the four borderline rows
   identified above. Multi-seed training of all 24 headline
   configurations is out of scope.
