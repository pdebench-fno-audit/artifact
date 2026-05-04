"""
Evaluate the already-trained FNO on 1D Burgers — load best checkpoint, run eval, generate plots.
"""

import modal
import os
import json

app = modal.App("fno-burgers-eval")
results_volume = modal.Volume.from_name("fno-burgers-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "h5py",
        "numpy",
        "scipy",
        "matplotlib",
        "huggingface_hub",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": results_volume},
    timeout=7200,
    memory=32768,
)
def evaluate():
    import time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from huggingface_hub import hf_hub_download
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    INIT_STEP = 10
    RES_X = 4
    RES_T = 5
    NC = 1
    MODES = 16
    WIDTH = 64
    N_TRAIN = 9000
    N_TEST = 1000
    BATCH = 32
    OUT = "/results/test_2_v2"
    os.makedirs(OUT, exist_ok=True)

    # ── Model ──
    class SpectralConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, modes):
            super().__init__()
            self.modes = modes
            scale = 1 / (in_ch * out_ch)
            self.w = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
            )

        def forward(self, x):
            B = x.shape[0]
            x_ft = torch.fft.rfft(x)
            out_ft = torch.zeros(
                B,
                self.w.shape[1],
                x.size(-1) // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            out_ft[:, :, : self.modes] = torch.einsum(
                "bix,iox->box", x_ft[:, :, : self.modes], self.w
            )
            return torch.fft.irfft(out_ft, n=x.size(-1))

    class FNO1d(nn.Module):
        def __init__(self, nc, modes=16, width=64, init_step=10):
            super().__init__()
            self.fc0 = nn.Linear(init_step * nc + 1, width)
            self.convs = nn.ModuleList(
                [SpectralConv1d(width, width, modes) for _ in range(4)]
            )
            self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
            self.fc1 = nn.Linear(width, 128)
            self.fc2 = nn.Linear(128, nc)

        def forward(self, x, grid):
            x = torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=-1)
            x = self.fc0(x).permute(0, 2, 1)
            for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
                x1, x2 = conv(x), w(x)
                x = F.gelu(x1 + x2) if i < 3 else (x1 + x2)
            x = x.permute(0, 2, 1)
            return self.fc2(F.gelu(self.fc1(x))).unsqueeze(-2)

    class BurgersDS(Dataset):
        def __init__(self, data, grid, init_step=10):
            self.data, self.grid, self.init_step = data, grid, init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i, :, : self.init_step, :], self.data[i], self.grid

    # ── Load checkpoint ──
    results_volume.reload()
    ckpt_path = f"{OUT}/best_model_s42.pt"
    assert os.path.exists(ckpt_path), (
        f"Checkpoint not found at {ckpt_path}. Available: {os.listdir(OUT)}"
    )
    print(f"Loading: {ckpt_path}", flush=True)

    model = FNO1d(NC, MODES, WIDTH, INIT_STEP).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    # ── Download data ──
    print("Downloading dataset...", flush=True)
    hdf5_path = hf_hub_download(
        repo_id="pdebench/Burgers",
        filename="1D_Burgers_Sols_Nu0.1.hdf5",
        repo_type="dataset",
        cache_dir="/tmp/hf_cache",
    )
    with h5py.File(hdf5_path, "r") as f:
        ds = f["tensor"]
        raw_shape = ds.shape
        if len(raw_shape) == 3:
            N, T, X = raw_shape
        else:
            N, T, X, _ = raw_shape
        X_ds = X // RES_X
        T_ds = int(np.ceil(T / RES_T))
        data = np.empty((N, X_ds, T_ds, 1), dtype=np.float32)
        for s in range(0, N, 500):
            e = min(s + 500, N)
            if len(raw_shape) == 3:
                data[s:e, :, :, 0] = np.transpose(ds[s:e, ::RES_T, ::RES_X], (0, 2, 1))
            else:
                data[s:e, :, :, 0] = np.transpose(
                    ds[s:e, ::RES_T, ::RES_X, 0], (0, 2, 1)
                )
        if "x-coordinate" in f:
            grid_np = np.array(f["x-coordinate"], dtype=np.float32)[::RES_X]
        else:
            grid_np = np.linspace(0, 1, X_ds, dtype=np.float32)
    data_t = torch.from_numpy(data)
    grid_t = torch.tensor(grid_np).unsqueeze(-1)
    T_TRAIN = data_t.shape[2]
    print(f"Data: {data_t.shape}, T_TRAIN={T_TRAIN}", flush=True)

    test_loader = DataLoader(
        BurgersDS(data_t[N_TRAIN : N_TRAIN + N_TEST], grid_t, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    grid_dev = grid_t.to(DEVICE)
    del data, data_t

    # ── Evaluate ──
    print("Running evaluation...", flush=True)
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for xx, yy, _ in test_loader:
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)
            pred_full = yy[:, :, :INIT_STEP, :]
            inp = xx.clone()
            for t in range(INIT_STEP, yy.shape[2]):
                inp_flat = inp.reshape(inp.shape[0], inp.shape[1], -1)
                pred = model(inp_flat, grid_dev)
                pred_full = torch.cat([pred_full, pred], dim=2)
                inp = torch.cat([inp[:, :, 1:, :], pred], dim=2)
            all_pred.append(pred_full.cpu())
            all_tgt.append(yy.cpu())
    preds = torch.cat(all_pred, 0)
    targets = torch.cat(all_tgt, 0)

    # nRMSE
    p = preds[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
    tg = targets[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
    nb2, nc2, nx, nt = tg.shape
    err = torch.sqrt(
        torch.mean(
            (p.reshape(nb2, nc2, -1, nt) - tg.reshape(nb2, nc2, -1, nt)) ** 2, dim=2
        )
    )
    nrm = torch.sqrt(torch.mean(tg.reshape(nb2, nc2, -1, nt) ** 2, dim=2))
    final_nrmse = torch.mean(err / nrm).item()
    print(f"\nFinal nRMSE: {final_nrmse:.4e}", flush=True)
    print(f"Published:   4.5e-3", flush=True)
    print(
        f"{'BEAT' if final_nrmse < 0.0045 else 'DID NOT BEAT'} benchmark!", flush=True
    )

    # Per-sample
    per_sample = torch.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        pi = preds[i, :, INIT_STEP:, :].reshape(-1)
        ti = targets[i, :, INIT_STEP:, :].reshape(-1)
        per_sample[i] = torch.norm(pi - ti) / (torch.norm(ti) + 1e-20)
    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi_idx = int(torch.argsort(per_sample)[len(per_sample) // 2])

    # Conservation
    u_int = targets[:, :, :, 0].mean(dim=1)
    u_int_p = preds[:, :, :, 0].mean(dim=1)
    cons_drift = torch.abs(u_int_p[:, -1] - u_int[:, 0]) / (
        torch.abs(u_int[:, 0]) + 1e-20
    )
    mean_cons = cons_drift.mean().item()

    # ── Save ──
    results = {
        "final_nrmse": float(final_nrmse),
        "published": 0.0045,
        "beat": bool(final_nrmse < 0.0045),
        "n_params": n_params,
        "conservation_drift": mean_cons,
        "seed": 42,
        "model": "FNO1d-4L-16M-64W",
        "best_idx": bi,
        "median_idx": mi_idx,
        "worst_idx": wi,
        "best_err": float(per_sample[bi]),
        "median_err": float(per_sample[mi_idx]),
        "worst_err": float(per_sample[wi]),
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        f"{OUT}/predictions.npz",
        preds=preds.numpy(),
        targets=targets.numpy(),
        per_sample=per_sample.numpy(),
        grid=grid_t.numpy(),
    )

    # ── Plots ──
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    grid_np2 = grid_t.squeeze().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for row, (idx, lab) in enumerate([(bi, "Best"), (mi_idx, "Median"), (wi, "Worst")]):
        truth = targets[idx, :, :, 0].numpy()
        pr = preds[idx, :, :, 0].numpy()
        er = np.abs(pr - truth)
        for col, (arr, title, cmap) in enumerate(
            [
                (truth, f"{lab} Truth (err={per_sample[idx]:.4e})", "viridis"),
                (pr, f"{lab} FNO", "viridis"),
                (er, f"{lab} |Error|", "hot"),
            ]
        ):
            kw = dict(aspect="auto", origin="lower")
            if col < 2:
                kw["vmin"], kw["vmax"] = float(truth.min()), float(truth.max())
            im = axes[row, col].imshow(arr, cmap=cmap, **kw)
            axes[row, col].set_title(title)
            plt.colorbar(im, ax=axes[row, col])
        axes[row, 0].set_ylabel("$x$")
    for a in axes[-1]:
        a.set_xlabel("$t$")
    plt.suptitle(
        f"FNO 1D Burgers nu=0.1 (nRMSE={final_nrmse:.4e})", fontsize=16, y=1.01
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/pred_vs_truth.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/pred_vs_truth.pdf", bbox_inches="tight")
    plt.close()

    nt_avail = targets.shape[2]
    snaps = [nt_avail // 4, nt_avail // 2, 3 * nt_avail // 4, nt_avail - 1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, ti in zip(axes, snaps):
        ax.plot(grid_np2, targets[mi_idx, :, ti, 0].numpy(), "k-", lw=2, label="Truth")
        ax.plot(grid_np2, preds[mi_idx, :, ti, 0].numpy(), "r--", lw=2, label="FNO")
        ax.set_title(f"step={ti}")
        ax.set(xlabel="$x$", ylabel="$u$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Median sample (idx={mi_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/line_snapshots.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/line_snapshots.pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        float(final_nrmse), color="red", ls="--", lw=2, label=f"nRMSE={final_nrmse:.4e}"
    )
    ax.axvline(0.0045, color="orange", ls="--", lw=2, label="Published=4.5e-3")
    ax.set(xlabel="Per-sample relL2", ylabel="Count", title="Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_dist.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/nrmse_dist.pdf", bbox_inches="tight")
    plt.close()

    mf = {
        "script": "fno_burgers_eval.py::evaluate",
        "outputs": [
            "results.json",
            "predictions.npz",
            "pred_vs_truth.png",
            "line_snapshots.png",
            "nrmse_dist.png",
        ],
        "nrmse": float(final_nrmse),
        "beat": bool(final_nrmse < 0.0045),
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as f:
        f.write(json.dumps(mf) + "\n")
    results_volume.commit()
    print(
        f"\n{'=' * 60}\nFINAL: nRMSE = {final_nrmse:.4e}\nPublished: 0.004500\nBeat: {final_nrmse < 0.0045}\n{'=' * 60}",
        flush=True,
    )
    return results
