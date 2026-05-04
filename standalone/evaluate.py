#!/usr/bin/env python3
"""Evaluate a saved model checkpoint on PDEBench test set.

No cloud dependencies. Runs on any machine with a CUDA GPU.

Usage:
    python evaluate.py --checkpoint best_model.pt --data 1D_Burgers_Sols_Nu0.01.hdf5 \
        --pde burgers --modes 12 --width 32 --init_step 5 --nc 1

    python evaluate.py --checkpoint best_model.pt --data 2D_diff-react.hdf5 \
        --pde 2d_react --modes 12 --width 32 --init_step 5 --nc 2 --arch gated

    python evaluate.py --results_dir results/ --verify-baselines
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import (
    FNO1d_AR,
    FNO2d_AR,
    VorticityPoissonFNO,
    compute_nrmse_per_timestep,
)


def load_1d_data(path, nc=1, res_x=1, res_t=1, n_train=9000, n_test=1000):
    """Load 1D PDEBench HDF5 file."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if "tensor" in keys:
            data = np.array(f["tensor"], dtype=np.float32)
        else:
            # Multi-variable (e.g., compressible NS)
            arrays = []
            for k in sorted(
                k for k in keys if k not in ["x-coordinate", "t-coordinate"]
            ):
                arrays.append(np.array(f[k], dtype=np.float32))
            data = np.stack(arrays, axis=-1)

        x_coord = np.array(
            f.get("x-coordinate", np.linspace(0, 1, data.shape[1])), dtype=np.float32
        )

    # Subsample
    if res_x > 1:
        data = data[:, ::res_x]
        x_coord = x_coord[::res_x]
    if res_t > 1:
        data = data[:, :, ::res_t] if data.ndim == 3 else data[:, :, ::res_t, :]

    # Ensure shape [N, X, T, nc]
    if data.ndim == 3:
        data = data[..., np.newaxis]

    N, X, T, C = data.shape
    data_t = torch.from_numpy(data)

    # Normalize
    ch_mean = data_t[:n_train].mean(dim=(0, 1, 2))
    ch_std = data_t[:n_train].std(dim=(0, 1, 2)) + 1e-8
    data_t = (data_t - ch_mean) / ch_std

    grid = torch.from_numpy(x_coord[:, np.newaxis])

    test_data = data_t[n_train : n_train + n_test]
    return test_data, grid, ch_mean, ch_std, T


def load_2d_data(path, nc=2, res=1, res_t=1, n_train=800, n_val=100, n_test=100):
    """Load 2D PDEBench HDF5 file."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if "density" in keys:
            # Compressible CFD: 4 channels
            rho = np.array(f["density"][:, ::res_t, ::res, ::res], dtype=np.float32)
            vx = np.array(f["Vx"][:, ::res_t, ::res, ::res], dtype=np.float32)
            vy = np.array(f["Vy"][:, ::res_t, ::res, ::res], dtype=np.float32)
            prs = np.array(f["pressure"][:, ::res_t, ::res, ::res], dtype=np.float32)
            data = np.stack([np.log(rho + 1e-6), vx, vy, np.log(prs + 1e-6)], axis=-1)
            data = np.transpose(data, (0, 2, 3, 1, 4))  # [N, H, W, T, 4]
            x_c = np.array(f["x-coordinate"][::res], dtype=np.float32)
            y_c = np.array(f["y-coordinate"][::res], dtype=np.float32)
        elif "velocity" in keys:
            # Incompressible NS or time-dependent
            vel = np.array(f["velocity"], dtype=np.float32)
            data = vel
            H = vel.shape[2] if vel.ndim == 5 else vel.shape[1]
            x_c = np.linspace(0, 1, H, dtype=np.float32)
            y_c = x_c.copy()
        else:
            # Generic 2D
            data = np.array(f["tensor"], dtype=np.float32)
            if data.ndim == 4:
                data = data[..., np.newaxis]
            H = data.shape[1]
            x_c = np.linspace(0, 1, H, dtype=np.float32)
            y_c = x_c.copy()

    N = data.shape[0]
    data_t = torch.from_numpy(data).float()

    n_train_actual = min(n_train, N - n_val - n_test)
    ch_mean = data_t[:n_train_actual].mean(dim=(0, 1, 2, 3))
    ch_std = data_t[:n_train_actual].std(dim=(0, 1, 2, 3)) + 1e-8
    data_t = (data_t - ch_mean) / ch_std

    gx, gy = np.meshgrid(x_c, y_c, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1))

    test_data = data_t[n_train_actual + n_val : n_train_actual + n_val + n_test]
    return test_data, grid, ch_mean, ch_std


def evaluate_model(model, test_data, grid, ch_mean, ch_std, init_step, device, dim=1):
    """Run autoregressive evaluation and compute nRMSE."""
    model.eval()
    all_nrmse = []
    batch_size = 16 if dim == 1 else 4

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size].to(device)
            gb = (
                grid.unsqueeze(0)
                .expand(batch.shape[0], *[-1] * (grid.dim()))
                .to(device)
            )

            T = batch.shape[-2]
            nc = batch.shape[-1]
            inp = batch[..., :init_step, :]

            preds = [batch[..., :init_step, :]]

            for t in range(init_step, T):
                if dim == 1:
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], -1)
                    pred = model(inp_flat, gb)
                else:
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                    pred = model(inp_flat, gb)

                preds.append(pred.unsqueeze(-2))
                inp = torch.cat([inp[..., 1:, :], pred.unsqueeze(-2)], dim=-2)

            preds = torch.cat(preds, dim=-2)

            # Denormalize
            pd = preds[..., init_step:, :] * ch_std.to(device) + ch_mean.to(device)
            td = batch[..., init_step:, :] * ch_std.to(device) + ch_mean.to(device)

            # Per-sample nRMSE
            B = pd.shape[0]
            num = ((pd - td) ** 2).reshape(B, -1).sum(1)
            den = (td**2).reshape(B, -1).sum(1) + 1e-20
            nrmse = torch.sqrt(num / den)
            all_nrmse.append(nrmse.cpu())

    per_sample = torch.cat(all_nrmse)
    return {
        "nrmse_mean": float(per_sample.mean()),
        "nrmse_median": float(per_sample.median()),
        "nrmse_std": float(per_sample.std()),
        "nrmse_min": float(per_sample.min()),
        "nrmse_max": float(per_sample.max()),
        "n_samples": len(per_sample),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FNO checkpoint on PDEBench test set"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_model.pt"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to HDF5 data file"
    )
    parser.add_argument(
        "--pde",
        type=str,
        default="1d",
        choices=["1d", "2d_react", "2d_cfd", "2d_swe", "darcy", "vort"],
        help="PDE type",
    )
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--init_step", type=int, default=5)
    parser.add_argument("--nc", type=int, default=1, help="Number of channels")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--arch", type=str, default="base", choices=["base", "gated"])
    parser.add_argument(
        "--res_x", type=int, default=1, help="Spatial subsampling factor"
    )
    parser.add_argument(
        "--res_t", type=int, default=1, help="Temporal subsampling factor"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    if args.pde == "1d":
        model = FNO1d_AR(
            nc=args.nc,
            modes=args.modes,
            width=args.width,
            init_step=args.init_step,
            n_layers=args.n_layers,
        ).to(device)
        test_data, grid, ch_mean, ch_std, T = load_1d_data(
            args.data, nc=args.nc, res_x=args.res_x, res_t=args.res_t
        )
        dim = 1
    elif args.pde in ["2d_react", "2d_cfd", "2d_swe"]:
        model = FNO2d_AR(
            nc=args.nc,
            modes=args.modes,
            width=args.width,
            init_step=args.init_step,
            n_layers=args.n_layers,
        ).to(device)
        test_data, grid, ch_mean, ch_std = load_2d_data(
            args.data, nc=args.nc, res=args.res_x, res_t=args.res_t
        )
        dim = 2
    elif args.pde == "vort":
        model = VorticityPoissonFNO(
            modes=args.modes,
            width=args.width,
            init_step=args.init_step,
            n_layers=args.n_layers,
        ).to(device)
        print("Vorticity-Poisson model — use dedicated eval script for full pipeline")
        return
    else:
        raise ValueError(f"Unknown PDE type: {args.pde}")

    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Test data: {test_data.shape}")

    # Evaluate
    results = evaluate_model(
        model, test_data, grid, ch_mean, ch_std, args.init_step, device, dim
    )
    results["checkpoint"] = args.checkpoint
    results["data"] = args.data
    results["pde"] = args.pde
    results["n_params"] = n_params

    print(f"\nResults:")
    print(f"  nRMSE (mean):   {results['nrmse_mean']:.4e}")
    print(f"  nRMSE (median): {results['nrmse_median']:.4e}")
    print(f"  nRMSE (std):    {results['nrmse_std']:.4e}")
    print(f"  nRMSE (range):  [{results['nrmse_min']:.4e}, {results['nrmse_max']:.4e}]")
    print(f"  Samples:        {results['n_samples']}")

    # Save
    out_path = args.output or args.checkpoint.replace(".pt", "_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
