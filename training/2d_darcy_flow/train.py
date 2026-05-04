"""
Wave 4: 2D Darcy Flow — All 5 β variants
==========================================
PDE: -∇(a(x)∇u(x)) = β, x ∈ (0,1)², u=0 on boundary
Input: diffusion coefficient field a(x,y) [128, 128]
Output: steady-state solution u(x,y) [128, 128]
Single-shot prediction (NO autoregressive rollout).

Architecture: 2D FNO with SpectralConv2d + local conv branch.
Protocol: 8000/1000/1000 train/val/test, normalized MSE loss, stratified val.
All baselines from PDEBench arXiv:2210.07182 Table 8.
"""

import modal
import os
import json

app = modal.App("fno-wave4-darcy")
volume = modal.Volume.from_name("fno-wave4-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "h5py",
        "numpy==1.26.4",
        "scipy",
        "matplotlib",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

TEST_CONFIGS = {
    "darcy_beta0.01": {
        "test_id": 21,
        "param": "β=0.01",
        "published_nrmse": 2.5,
        "published_source": "PDEBench Table 8, DarcyFlow β=0.01, nRMSE, FNO",
        "darus_id": "133217",
        "filename": "2D_DarcyFlow_beta0.01_Train.hdf5",
    },
    "darcy_beta0.1": {
        "test_id": 22,
        "param": "β=0.1",
        "published_nrmse": 0.22,
        "published_source": "PDEBench Table 8, DarcyFlow β=0.1, nRMSE, FNO",
        "darus_id": "133218",
        "filename": "2D_DarcyFlow_beta0.1_Train.hdf5",
    },
    "darcy_beta1.0": {
        "test_id": 23,
        "param": "β=1.0",
        "published_nrmse": 0.064,
        "published_source": "PDEBench Table 8, DarcyFlow β=1.0, nRMSE, FNO",
        "darus_id": "133219",
        "filename": "2D_DarcyFlow_beta1.0_Train.hdf5",
    },
    "darcy_beta10.0": {
        "test_id": 24,
        "param": "β=10.0",
        "published_nrmse": 0.012,
        "published_source": "PDEBench Table 8, DarcyFlow β=10.0, nRMSE, FNO",
        "darus_id": "133220",
        "filename": "2D_DarcyFlow_beta10.0_Train.hdf5",
    },
    "darcy_beta100.0": {
        "test_id": 25,
        "param": "β=100.0",
        "published_nrmse": 0.0064,
        "published_source": "PDEBench Table 8, DarcyFlow β=100.0, nRMSE, FNO",
        "darus_id": "133221",
        "filename": "2D_DarcyFlow_beta100.0_Train.hdf5",
    },
}


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": volume},
    timeout=86400,
    memory=32768,
)
def train(test_key: str):
    """Train 2D FNO on Darcy Flow. Single-shot mapping a(x) → u(x)."""
    import time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import subprocess
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = TEST_CONFIGS[test_key]
    test_id = cfg["test_id"]
    published = cfg["published_nrmse"]

    OUT = f"/results/test_{test_id}"
    os.makedirs(OUT, exist_ok=True)

    DEVICE = torch.device("cuda")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Test {test_id}: 2D Darcy Flow {cfg['param']}", flush=True)
    print(f"Published FNO nRMSE: {published}", flush=True)

    # ── Config ──
    N_TRAIN = 8000
    N_VAL = 1000
    N_TEST = 1000
    MODES = 12  # 2D Fourier modes per dimension
    WIDTH = 32  # Channel width (2D needs less than 1D — memory constraint)
    N_LAYERS = 4
    BATCH = 16
    EPOCHS = 500
    LR = 1e-3

    # ═══════════════════════════════════════════════════════════════
    # 2D FNO Architecture
    # ═══════════════════════════════════════════════════════════════
    class SpectralConv2d(nn.Module):
        """2D Fourier layer: FFT → truncate modes → iFFT."""

        def __init__(self, in_ch, out_ch, modes1, modes2):
            super().__init__()
            self.modes1 = modes1
            self.modes2 = modes2
            scale = 1 / (in_ch * out_ch)
            self.w1 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            )
            self.w2 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            )

        def forward(self, x):
            # x: [B, C, H, W]
            B = x.shape[0]
            x_ft = torch.fft.rfft2(x)
            H_ft = x_ft.shape[-2]

            out_ft = torch.zeros(
                B,
                self.w1.shape[1],
                H_ft,
                x.size(-1) // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            # Top-left corner (low freq)
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.w1
            )
            # Bottom-left corner (negative freq in first dim)
            out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.w2
            )
            return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

    class FNO2dBlock(nn.Module):
        """Gated local-global block for 2D."""

        def __init__(self, width, modes1, modes2, local_kernel=5):
            super().__init__()
            self.spectral = SpectralConv2d(width, width, modes1, modes2)
            self.pointwise = nn.Conv2d(width, width, 1)
            self.local_conv = nn.Conv2d(
                width, width, local_kernel, padding=local_kernel // 2
            )
            self.gate = nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            g_out = self.spectral(x) + self.pointwise(x)
            l_out = self.local_conv(x)
            alpha = torch.sigmoid(self.gate)
            return (1 - alpha) * g_out + alpha * l_out

    class FNO2d(nn.Module):
        """2D FNO for Darcy Flow: a(x,y) → u(x,y)."""

        def __init__(self, modes1=12, modes2=12, width=32, n_layers=4):
            super().__init__()
            # Input: a(x,y) [1 ch] + grid coords (x, y) [2 ch] = 3
            self.fc0 = nn.Conv2d(3, width, 1)
            self.blocks = nn.ModuleList(
                [FNO2dBlock(width, modes1, modes2) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 128, 1)
            self.fc2 = nn.Conv2d(128, 1, 1)
            self.n_layers = n_layers

        def forward(self, x, grid):
            # x: [B, H, W, 1], grid: [B, H, W, 2]
            x = torch.cat([x, grid], dim=-1)  # [B, H, W, 3]
            x = x.permute(0, 3, 1, 2)  # [B, 3, H, W]
            x = self.fc0(x)  # [B, W, H, W]
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)  # [B, 1, H, W]
            return x.permute(0, 2, 3, 1)  # [B, H, W, 1]

    class DarcyDS(Dataset):
        def __init__(self, inputs, targets, grid):
            self.inputs = inputs  # [N, H, W, 1]
            self.targets = targets  # [N, H, W, 1]
            self.grid = grid  # [H, W, 2]

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, i):
            return self.inputs[i], self.targets[i], self.grid

    # ═══════════════════════════════════════════════════════════════
    # Download data
    # ═══════════════════════════════════════════════════════════════
    hdf5_path = f"/tmp/{cfg['filename']}"
    if not os.path.exists(hdf5_path) or os.path.getsize(hdf5_path) < 50_000_000:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        url = f"https://darus.uni-stuttgart.de/api/access/datafile/{cfg['darus_id']}"
        print(f"Downloading: {url}", flush=True)
        subprocess.run(
            [
                "aria2c",
                "-x",
                "16",
                "-s",
                "16",
                "--max-connection-per-server=16",
                "--min-split-size=10M",
                "--timeout=600",
                "--max-tries=5",
                "-d",
                "/tmp",
                "-o",
                cfg["filename"],
                url,
            ],
            check=True,
            timeout=3600,
        )
    print(f"Dataset: {os.path.getsize(hdf5_path) / 1e6:.0f} MB", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Load data: nu [N, 128, 128] → inputs, tensor [N, 1, 128, 128] → targets
    # ═══════════════════════════════════════════════════════════════
    with h5py.File(hdf5_path, "r") as f:
        nu = np.array(f["nu"], dtype=np.float32)  # [N, 128, 128] — input
        tensor = np.array(f["tensor"], dtype=np.float32)  # [N, 1, 128, 128] — target
        x_coord = np.array(f["x-coordinate"], dtype=np.float32)
        y_coord = np.array(f["y-coordinate"], dtype=np.float32)

    N, H, W = nu.shape
    print(f"nu (input): {nu.shape}, tensor (target): {tensor.shape}", flush=True)
    print(f"Input range: [{nu.min():.4f}, {nu.max():.4f}]", flush=True)
    print(f"Target range: [{tensor.min():.6f}, {tensor.max():.6f}]", flush=True)

    # Reshape: [N, H, W, 1]
    inputs = torch.from_numpy(nu).unsqueeze(-1)  # [N, 128, 128, 1]
    targets = torch.from_numpy(tensor[:, 0, :, :]).unsqueeze(-1)  # [N, 128, 128, 1]

    # Grid: [H, W, 2]
    gx, gy = np.meshgrid(x_coord, y_coord, indexing="ij")
    grid = torch.from_numpy(
        np.stack([gx, gy], axis=-1).astype(np.float32)
    )  # [128, 128, 2]

    # ── Stratified val split (on input mean) ──
    ic_means = inputs[: N_TRAIN + N_VAL, :, :, 0].mean(dim=(1, 2))
    n_bins = 10
    bin_edges = torch.linspace(ic_means.min() - 1e-6, ic_means.max() + 1e-6, n_bins + 1)
    val_indices, train_indices = [], []
    for b in range(n_bins):
        mask = (ic_means >= bin_edges[b]) & (ic_means < bin_edges[b + 1])
        bin_idx = torch.where(mask)[0]
        if len(bin_idx) == 0:
            continue
        n_val_bin = max(1, round(len(bin_idx) * N_VAL / (N_TRAIN + N_VAL)))
        perm = torch.randperm(len(bin_idx))
        val_indices.extend(bin_idx[perm[:n_val_bin]].tolist())
        train_indices.extend(bin_idx[perm[n_val_bin:]].tolist())

    train_in, train_tgt = inputs[train_indices], targets[train_indices]
    val_in, val_tgt = inputs[val_indices], targets[val_indices]
    test_in = inputs[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    test_tgt = targets[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    print(
        f"Split: train={len(train_indices)}, val={len(val_indices)}, test={test_in.shape[0]}",
        flush=True,
    )

    train_loader = DataLoader(
        DarcyDS(train_in, train_tgt, grid),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DarcyDS(val_in, val_tgt, grid),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        DarcyDS(test_in, test_tgt, grid),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    del inputs, targets, train_in, train_tgt, val_in, val_tgt
    grid_dev = grid.to(DEVICE)

    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════
    def calc_nrmse(preds, targets):
        """Per-sample Frobenius nRMSE (standard for Darcy)."""
        per_sample = torch.sqrt(((preds - targets) ** 2).sum(dim=(1, 2, 3))) / (
            torch.sqrt((targets**2).sum(dim=(1, 2, 3))) + 1e-20
        )
        return per_sample.mean().item(), per_sample

    def do_eval(model, loader):
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for inp, tgt, grd in loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                grd_batch = grd[0:1].expand(inp.shape[0], -1, -1, -1).to(DEVICE)
                pred = model(inp, grd_batch)
                all_pred.append(pred.cpu())
                all_tgt.append(tgt.cpu())
        return torch.cat(all_pred, 0), torch.cat(all_tgt, 0)

    # ═══════════════════════════════════════════════════════════════
    # Build model
    # ═══════════════════════════════════════════════════════════════
    model = FNO2d(MODES, MODES, WIDTH, N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nFNO2d: {n_params:,} params (modes={MODES}, width={WIDTH}, layers={N_LAYERS})",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ═══════════════════════════════════════════════════════════════
    # Checkpoint-resume
    # ═══════════════════════════════════════════════════════════════
    CKPT_PATH = f"{OUT}/resume_checkpoint.pt"
    train_losses, val_log = [], []
    best_val_nrmse = float("inf")
    start_epoch = 1

    volume.reload()
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        train_losses = ckpt["train_losses"]
        val_log = ckpt["val_log"]
        best_val_nrmse = ckpt["best_val_nrmse"]
        print(f"Resumed at epoch {start_epoch}", flush=True)
    else:
        print("Starting fresh training", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Training loop — normalized MSE loss
    # ═══════════════════════════════════════════════════════════════
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for inp, tgt, grd in train_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            grd_batch = grd[0:1].expand(inp.shape[0], -1, -1, -1).to(DEVICE)
            pred = model(inp, grd_batch)
            B = pred.shape[0]
            pred_flat = pred.reshape(B, -1)
            tgt_flat = tgt.reshape(B, -1)
            per_sample_mse = ((pred_flat - tgt_flat) ** 2).mean(dim=1)
            per_sample_rms2 = (tgt_flat**2).mean(dim=1) + 1e-8
            loss = (per_sample_mse / per_sample_rms2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
        scheduler.step()
        train_losses.append(ep_loss / nb)

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"  Ep {epoch:4d}/{EPOCHS} | loss={ep_loss / nb:.4e} | lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
                flush=True,
            )

        if epoch % 50 == 0 or epoch == EPOCHS:
            val_preds, val_targets = do_eval(model, val_loader)
            val_nrmse, _ = calc_nrmse(val_preds, val_targets)
            val_log.append({"epoch": epoch, "val_nrmse": val_nrmse})
            print(
                f"  Ep {epoch:4d}/{EPOCHS} | val nRMSE={val_nrmse:.4e} | {time.time() - t0:.0f}s",
                flush=True,
            )
            if val_nrmse < best_val_nrmse:
                best_val_nrmse = val_nrmse
                torch.save(model.state_dict(), f"{OUT}/best_model.pt")
                print(f"    -> New best: {best_val_nrmse:.4e}", flush=True)

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_log": val_log,
                    "best_val_nrmse": best_val_nrmse,
                },
                CKPT_PATH,
            )
            volume.commit()

    dt = time.time() - t0
    print(f"\nTraining done: best val={best_val_nrmse:.4e}, time={dt:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Final test evaluation
    # ═══════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    test_preds, test_targets = do_eval(model, test_loader)
    test_nrmse, per_sample = calc_nrmse(test_preds, test_targets)

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"TEST: nRMSE={test_nrmse:.4e} | Published={published} | Beat={test_nrmse < published}",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi = int(torch.argsort(per_sample)[len(per_sample) // 2])

    # Data leak checks
    no_nan = not (torch.isnan(test_preds).any() or torch.isinf(test_preds).any())
    pred_differ = not torch.allclose(test_preds, test_targets, atol=1e-6)
    near_perfect = (per_sample < 1e-6).float().mean().item()

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    results = {
        "test_id": test_id,
        "pde": "2D Darcy Flow",
        "parameter": cfg["param"],
        "nrmse_pertimestep": float(
            test_nrmse
        ),  # For Darcy, this IS the Frobenius nRMSE
        "nrmse_frobenius": float(test_nrmse),
        "published_fno": published,
        "published_source": cfg["published_source"],
        "beat_pertimestep": bool(test_nrmse < published),
        "beat_frobenius": bool(test_nrmse < published),
        "best_val_nrmse": float(best_val_nrmse),
        "n_params": n_params,
        "training_time_s": dt,
        "split": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "data_leak_checks": {
            "predictions_differ": bool(pred_differ),
            "no_nan_inf": bool(no_nan),
            "near_perfect_fraction": float(near_perfect),
        },
        "per_sample_stats": {
            "best_idx": bi,
            "best_err": float(per_sample[bi]),
            "median_idx": mi,
            "median_err": float(per_sample[mi]),
            "worst_idx": wi,
            "worst_err": float(per_sample[wi]),
        },
        "architecture": {
            "type": "fno2d",
            "modes": MODES,
            "width": WIDTH,
            "n_layers": N_LAYERS,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "loss": "normalized_mse",
            "prediction_type": "single_shot",
        },
    }
    with open(f"{OUT}/results.json", "w") as fout:
        json.dump(results, fout, indent=2)

    np.savez(
        f"{OUT}/training_histories.npz",
        train_loss=np.array(train_losses),
        val_epochs=np.array([d["epoch"] for d in val_log]),
        val_nrmse=np.array([d["val_nrmse"] for d in val_log]),
    )
    np.savez(
        f"{OUT}/predictions.npz",
        preds=test_preds.numpy(),
        targets=test_targets.numpy(),
        per_sample=per_sample.numpy(),
    )

    # ═══════════════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════════════
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    # Training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.semilogy(train_losses, alpha=0.8, color="steelblue")
    a1.set(xlabel="Epoch", ylabel="Loss", title=f"Training Loss — Test {test_id}")
    a1.grid(True, alpha=0.3)
    ep_v = [d["epoch"] for d in val_log]
    nv = [d["val_nrmse"] for d in val_log]
    a2.semilogy(ep_v, nv, "o-", ms=4, color="steelblue", label="Val nRMSE")
    a2.axhline(
        published, color="red", ls="--", lw=2, label=f"Published FNO {published}"
    )
    a2.set(xlabel="Epoch", ylabel="nRMSE", title=f"Val nRMSE — Test {test_id}")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2D pred vs truth (best, median, worst)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for row, (idx, lab) in enumerate([(bi, "Best"), (mi, "Median"), (wi, "Worst")]):
        inp_2d = test_in[idx, :, :, 0].numpy()
        tgt_2d = test_targets[idx, :, :, 0].numpy()
        prd_2d = test_preds[idx, :, :, 0].numpy()
        err_2d = np.abs(prd_2d - tgt_2d)

        for col, (arr, title, cmap) in enumerate(
            [
                (inp_2d, f"{lab}: Input a(x,y)", "viridis"),
                (tgt_2d, f"{lab}: Truth u(x,y) (err={per_sample[idx]:.3e})", "RdBu_r"),
                (prd_2d, f"{lab}: FNO Pred", "RdBu_r"),
                (err_2d, f"{lab}: |Error|", "hot"),
            ]
        ):
            kw = dict(origin="lower", extent=[0, 1, 0, 1])
            if col in (1, 2):
                kw["vmin"], kw["vmax"] = float(tgt_2d.min()), float(tgt_2d.max())
            im = axes[row, col].imshow(arr.T, cmap=cmap, **kw)
            axes[row, col].set_title(title, fontsize=10)
            plt.colorbar(im, ax=axes[row, col])
    plt.suptitle(
        f"Test {test_id}: 2D Darcy Flow {cfg['param']} — nRMSE={test_nrmse:.4e}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/pred_vs_truth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        float(test_nrmse), color="red", ls="--", lw=2, label=f"nRMSE={test_nrmse:.4e}"
    )
    ax.axvline(published, color="orange", ls="--", lw=2, label=f"Published={published}")
    ax.set(
        xlabel="Per-sample relL2",
        ylabel="Count",
        title=f"Error Distribution — Test {test_id}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Manifest
    mf = {
        "script": f"fno2d_darcy_batch.py::train({test_key})",
        "test_id": test_id,
        "nrmse": float(test_nrmse),
        "beat": bool(test_nrmse < published),
        "split": "8000/1000/1000",
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as fout:
        fout.write(json.dumps(mf) + "\n")

    volume.commit()
    print(f"\nFINAL — Test {test_id}: Darcy {cfg['param']}")
    print(
        f"  nRMSE: {test_nrmse:.4e} | Published: {published} | Beat: {test_nrmse < published}",
        flush=True,
    )
    return results
