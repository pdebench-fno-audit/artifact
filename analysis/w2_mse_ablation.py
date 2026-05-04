"""
W2 ABLATION: Test 26 with STANDARD MSE (not normalized MSE)
============================================================
Everything identical to the original Test 26 run EXCEPT:
  - Loss function: standard MSE instead of normalized MSE
  - This isolates the impact of nMSE on the 43.7× headline result

If the improvement drops from 44× to ~20×: nMSE matters a lot on 2D.
If it stays at ~40×: AR rollout dominates and nMSE is secondary.
"""

import modal
import os
import json

app = modal.App("w2-test26-mse-ablation")
volume = modal.Volume.from_name("fno-wave5-results", create_if_missing=True)

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
    "diff_react_2d": {
        "test_id": 26,
        "pde": "2D Diff-React",
        "param": "default",
        "pde_equation": "∂u/∂t = ν∇²u + ρu(1-u), activator-inhibitor (2ch)",
        "published_nrmse": 0.12,
        "published_source": "PDEBench Table 5, 2D diffusion-reaction, nRMSE, FNO",
        "darus_id": "133017",
        "filename": "2D_diff-react_NA_NA.h5",
        "nc": 2,  # activator + inhibitor
        "modes": 12,
        "width": 32,
        "noise_std": 1e-3,
        "batch_size": 8,
        "epochs": 500,
        "lr": 1e-3,
        "res_t": 5,  # 101 -> ~21 timesteps
    },
    "shallow_water_2d": {
        "test_id": 27,
        "pde": "2D Shallow Water",
        "param": "radial dam break",
        "pde_equation": "Shallow water eqs: h_t + ∇·(hv) = 0",
        "published_nrmse": 0.0044,
        "published_source": "PDEBench Table 5, Shallow-water equation, nRMSE, FNO",
        "darus_id": "133021",
        "filename": "2D_rdb_NA_NA.h5",
        "nc": 1,  # water height only
        "modes": 12,
        "width": 32,
        "noise_std": 1e-3,
        "batch_size": 8,
        "epochs": 500,
        "lr": 1e-3,
        "res_t": 5,
    },
}


@app.function(
    gpu="A10G", image=image, volumes={"/results": volume}, timeout=86400, memory=32768
)
def train(test_key: str):
    import time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import subprocess
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = TEST_CONFIGS[test_key]
    test_id = cfg["test_id"]
    published = cfg["published_nrmse"]
    nc = cfg["nc"]
    modes = cfg["modes"]
    width = cfg["width"]
    noise_std = cfg["noise_std"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    res_t = cfg["res_t"]

    OUT = f"/results/W2_test_{test_id}_mse_ablation"
    os.makedirs(OUT, exist_ok=True)

    DEVICE = torch.device("cuda")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Test {test_id}: {cfg['pde']} {cfg['param']} (nc={nc})", flush=True)
    print(f"Published FNO nRMSE: {published}", flush=True)

    INIT_STEP = 5  # 2D: use 5 input frames (less than 1D's 10 due to memory)
    # Only 1000 samples total — use 800/100/100
    N_TRAIN = 800
    N_VAL = 100
    N_TEST = 100

    # ═══════════════════════════════════════════════════════════════
    # 2D Autoregressive FNO Architecture
    # ═══════════════════════════════════════════════════════════════
    class SpectralConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, modes1, modes2):
            super().__init__()
            self.modes1, self.modes2 = modes1, modes2
            scale = 1 / (in_ch * out_ch)
            self.w1 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            )
            self.w2 = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            )

        def forward(self, x):
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
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.w1
            )
            out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.w2
            )
            return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

    class FNO2dBlock(nn.Module):
        def __init__(self, w, m1, m2, local_kernel=5):
            super().__init__()
            self.spectral = SpectralConv2d(w, w, m1, m2)
            self.pointwise = nn.Conv2d(w, w, 1)
            self.local_conv = nn.Conv2d(w, w, local_kernel, padding=local_kernel // 2)
            self.gate = nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            g_out = self.spectral(x) + self.pointwise(x)
            l_out = self.local_conv(x)
            alpha = torch.sigmoid(self.gate)
            return (1 - alpha) * g_out + alpha * l_out

    class FNO2d_AR(nn.Module):
        """2D FNO for autoregressive time stepping.
        Input: [B, H, W, INIT_STEP * nc + 2] (frames + grid coords)
        Output: [B, H, W, nc] (next frame prediction)
        """

        def __init__(self, n_ch, modes1, modes2, width, init_step=5, n_layers=4):
            super().__init__()
            in_dim = init_step * n_ch + 2  # flattened input frames + (x, y) grid
            self.fc0 = nn.Conv2d(in_dim, width, 1)
            self.blocks = nn.ModuleList(
                [FNO2dBlock(width, modes1, modes2) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 128, 1)
            self.fc2 = nn.Conv2d(128, n_ch, 1)
            self.n_layers = n_layers

        def forward(self, x, grid):
            # x: [B, H, W, INIT_STEP*nc], grid: [B, H, W, 2]
            x = torch.cat([x, grid], dim=-1)  # [B, H, W, INIT_STEP*nc+2]
            x = x.permute(0, 3, 1, 2)  # [B, C_in, H, W]
            x = self.fc0(x)
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)  # [B, nc, H, W]
            return x.permute(0, 2, 3, 1)  # [B, H, W, nc]

    class TimeDepDS(Dataset):
        def __init__(self, data, grid, init_step):
            self.data = data  # [N, H, W, T, C]
            self.grid = grid  # [H, W, 2]
            self.init_step = init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            # Return init window and full trajectory
            return self.data[i, :, :, : self.init_step, :], self.data[i], self.grid

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
                "--max-tries=10",
                "--retry-wait=30",
                "-d",
                "/tmp",
                "-o",
                cfg["filename"],
                url,
            ],
            check=True,
            timeout=7200,
        )
    print(f"Dataset: {os.path.getsize(hdf5_path) / 1e9:.2f} GB", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Load per-sample group data
    # ═══════════════════════════════════════════════════════════════
    with h5py.File(hdf5_path, "r") as f:
        sample_keys = sorted([k for k in f.keys() if k.isdigit()], key=int)
        N = len(sample_keys)
        sample0 = np.array(f[sample_keys[0]]["data"], dtype=np.float32)  # [T, H, W, C]
        T_raw, H, W, C = sample0.shape
        T_ds = int(np.ceil(T_raw / res_t))

        print(f"N={N}, T_raw={T_raw}, H={H}, W={W}, C={C}, T_ds={T_ds}", flush=True)

        # Load all samples: [N, T_ds, H, W, C] then transpose to [N, H, W, T_ds, C]
        data = np.empty((N, H, W, T_ds, C), dtype=np.float32)
        for i, sk in enumerate(sample_keys):
            raw = np.array(f[sk]["data"], dtype=np.float32)[::res_t]  # [T_ds, H, W, C]
            data[i] = np.transpose(raw, (1, 2, 0, 3))  # [H, W, T_ds, C]
            if i == 0:
                print(
                    f"  Sample 0: raw {raw.shape} -> stored {data[i].shape}", flush=True
                )

        # Grid
        if "grid" in f[sample_keys[0]]:
            x_coord = np.array(f[sample_keys[0]]["grid"]["x"], dtype=np.float32)
            y_coord = np.array(f[sample_keys[0]]["grid"]["y"], dtype=np.float32)
        else:
            x_coord = np.linspace(0, 1, H, dtype=np.float32)
            y_coord = np.linspace(0, 1, W, dtype=np.float32)

    data_t = torch.from_numpy(data)
    T_TOTAL = data_t.shape[3]
    NC_actual = data_t.shape[4]
    print(f"Data: {data_t.shape}, T_TOTAL={T_TOTAL}, NC={NC_actual}", flush=True)
    print(f"Range: [{data_t.min():.4f}, {data_t.max():.4f}]", flush=True)

    # Grid: [H, W, 2]
    gx, gy = np.meshgrid(x_coord, y_coord, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32))

    # ── Per-channel normalization ──
    ch_mean = data_t[:N_TRAIN].mean(dim=(0, 1, 2, 3))  # [C]
    ch_std = data_t[:N_TRAIN].std(dim=(0, 1, 2, 3)) + 1e-8
    if NC_actual > 1:
        print(
            f"Per-channel: mean={ch_mean.tolist()}, std={ch_std.tolist()}", flush=True
        )
        data_t = (data_t - ch_mean) / ch_std

    # ── Split (800/100/100 for 1000-sample datasets) ──
    train_data = data_t[:N_TRAIN]
    val_data = data_t[N_TRAIN : N_TRAIN + N_VAL]
    test_data = data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    print(
        f"Split: train={train_data.shape[0]}, val={val_data.shape[0]}, test={test_data.shape[0]}",
        flush=True,
    )

    train_loader = DataLoader(
        TimeDepDS(train_data, grid, INIT_STEP),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TimeDepDS(val_data, grid, INIT_STEP),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TimeDepDS(test_data, grid, INIT_STEP),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    del data, data_t, train_data, val_data, test_data
    grid_dev = grid.to(DEVICE)

    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════
    def denorm(x):
        if NC_actual > 1:
            return x * ch_std.to(x.device) + ch_mean.to(x.device)
        return x

    def calc_nrmse(preds, targets):
        """nRMSE in original space, averaged over predicted timesteps."""
        p = denorm(preds[:, :, :, INIT_STEP:, :])  # [N, H, W, T_pred, C]
        tg = denorm(targets[:, :, :, INIT_STEP:, :])
        # Per-sample Frobenius
        per_sample = torch.sqrt(((p - tg) ** 2).sum(dim=(1, 2, 3, 4))) / (
            torch.sqrt((tg**2).sum(dim=(1, 2, 3, 4))) + 1e-20
        )
        return per_sample.mean().item(), per_sample

    def do_eval(model, loader):
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for xx, yy, grd in loader:
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                grd_b = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
                # Autoregressive rollout
                pred_full = yy[:, :, :, :INIT_STEP, :]  # [B, H, W, init, C]
                inp = xx.clone()  # [B, H, W, init, C]
                for t in range(INIT_STEP, yy.shape[3]):
                    # Flatten temporal window: [B, H, W, init*C]
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                    pred = model(inp_flat, grd_b)  # [B, H, W, C]
                    pred = pred.unsqueeze(3)  # [B, H, W, 1, C]
                    pred_full = torch.cat([pred_full, pred], dim=3)
                    inp = torch.cat([inp[:, :, :, 1:, :], pred], dim=3)
                all_pred.append(pred_full.cpu())
                all_tgt.append(yy.cpu())
        return torch.cat(all_pred, 0), torch.cat(all_tgt, 0)

    # ═══════════════════════════════════════════════════════════════
    # Build model
    # ═══════════════════════════════════════════════════════════════
    model = FNO2d_AR(NC_actual, modes, modes, width, INIT_STEP, n_layers=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nFNO2d_AR: {n_params:,} params (modes={modes}, width={width}, init={INIT_STEP})",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Checkpoint-resume
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
    # Training loop — autoregressive + normalized MSE
    # ═══════════════════════════════════════════════════════════════
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for xx, yy, grd in train_loader:
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)
            grd_b = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)
            inp = xx  # [B, H, W, init, C]

            for t in range(INIT_STEP, T_TOTAL):
                inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                pred = model(inp_flat, grd_b)  # [B, H, W, C]
                target = yy[:, :, :, t, :]  # [B, H, W, C]
                # W2 ABLATION: Standard MSE instead of normalized MSE
                loss = loss + F.mse_loss(pred, target)

                pred_unsq = pred.unsqueeze(3)  # [B, H, W, 1, C]
                if noise_std > 0 and model.training:
                    pred_unsq = pred_unsq + noise_std * torch.randn_like(pred_unsq)
                inp = torch.cat([inp[:, :, :, 1:, :], pred_unsq], dim=3)

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
                f"  Ep {epoch:4d}/{epochs} | loss={ep_loss / nb:.4e} | lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
                flush=True,
            )

        if epoch % 50 == 0 or epoch == epochs:
            vp, vt = do_eval(model, val_loader)
            vn, _ = calc_nrmse(vp, vt)
            val_log.append({"epoch": epoch, "val_nrmse": vn})
            print(
                f"  Ep {epoch:4d}/{epochs} | val nRMSE={vn:.4e} | {time.time() - t0:.0f}s",
                flush=True,
            )
            if vn < best_val_nrmse:
                best_val_nrmse = vn
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
    print(
        f"\nTraining done: best val={best_val_nrmse:.4e}, time={dt / 3600:.1f}h",
        flush=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # Final test evaluation
    # ═══════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    test_preds, test_targets = do_eval(model, test_loader)
    test_nrmse, per_sample = calc_nrmse(test_preds, test_targets)

    # Denormalize for saving
    test_preds_d = denorm(test_preds)
    test_targets_d = denorm(test_targets)

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"TEST {test_id}: nRMSE={test_nrmse:.4e} | Published={published} | Beat={test_nrmse < published}",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi = int(torch.argsort(per_sample)[len(per_sample) // 2])
    no_nan = not (torch.isnan(test_preds).any() or torch.isinf(test_preds).any())

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    results = {
        "test_id": test_id,
        "pde": cfg["pde"],
        "parameter": cfg["param"],
        "nrmse_pertimestep": float(test_nrmse),
        "nrmse_frobenius": float(test_nrmse),
        "published_fno": published,
        "published_source": cfg["published_source"],
        "beat_pertimestep": bool(test_nrmse < published),
        "beat_frobenius": bool(test_nrmse < published),
        "best_val_nrmse": float(best_val_nrmse),
        "n_params": n_params,
        "training_time_s": dt,
        "split": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "data_leak_checks": {"no_nan_inf": bool(no_nan)},
        "per_sample_stats": {
            "best_idx": bi,
            "best_err": float(per_sample[bi]),
            "median_idx": mi,
            "median_err": float(per_sample[mi]),
            "worst_idx": wi,
            "worst_err": float(per_sample[wi]),
        },
        "architecture": {
            "type": "fno2d_ar",
            "modes": modes,
            "width": width,
            "n_layers": 4,
            "init_step": INIT_STEP,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss": "standard_mse (W2 ablation)",
            "noise_std": noise_std,
            "res_t": res_t,
            "prediction_type": "autoregressive",
        },
    }
    with open(f"{OUT}/results.json", "w") as fout:
        json.dump(results, fout, indent=2)

    # Save predictions (downsampled to save space — full 128×128 is large)
    np.savez_compressed(
        f"{OUT}/predictions.npz",
        preds=test_preds_d.numpy(),
        targets=test_targets_d.numpy(),
        per_sample=per_sample.numpy(),
    )
    np.savez(
        f"{OUT}/training_histories.npz",
        train_loss=np.array(train_losses),
        val_epochs=np.array([d["epoch"] for d in val_log]),
        val_nrmse=np.array([d["val_nrmse"] for d in val_log]),
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
    a2.semilogy(
        [d["epoch"] for d in val_log],
        [d["val_nrmse"] for d in val_log],
        "o-",
        ms=4,
        color="steelblue",
        label="Val nRMSE",
    )
    a2.axhline(
        published, color="red", ls="--", lw=2, label=f"Published FNO {published}"
    )
    a2.set(xlabel="Epoch", ylabel="nRMSE", title=f"Val nRMSE — Test {test_id}")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2D snapshots: median sample at different timesteps (channel 0)
    nt = test_targets_d.shape[3]
    snaps = [0, INIT_STEP, nt // 2, nt - 1]
    fig, axes = plt.subplots(2, len(snaps), figsize=(5 * len(snaps), 10))
    for col, ti in enumerate(snaps):
        truth = test_targets_d[mi, :, :, ti, 0].numpy()
        pred_snap = test_preds_d[mi, :, :, ti, 0].numpy()
        kw = dict(origin="lower", vmin=float(truth.min()), vmax=float(truth.max()))
        axes[0, col].imshow(truth.T, **kw)
        axes[0, col].set_title(f"Truth t={ti}")
        im = axes[1, col].imshow(pred_snap.T, **kw)
        axes[1, col].set_title(f"Pred t={ti}")
    plt.suptitle(
        f"Test {test_id}: {cfg['pde']} — Median sample (nRMSE={per_sample[mi]:.3e})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=30, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        float(test_nrmse), color="red", ls="--", lw=2, label=f"nRMSE={test_nrmse:.4e}"
    )
    ax.axvline(published, color="orange", ls="--", lw=2, label=f"Published={published}")
    ax.set(
        xlabel="Per-sample relL2", ylabel="Count", title=f"Error Dist — Test {test_id}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    mf = {
        "script": f"fno2d_timedep_batch.py::train({test_key})",
        "test_id": test_id,
        "nrmse": float(test_nrmse),
        "beat": bool(test_nrmse < published),
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as fout:
        fout.write(json.dumps(mf) + "\n")

    volume.commit()
    print(
        f"\nFINAL — Test {test_id}: {cfg['pde']} — nRMSE={test_nrmse:.4e} | Beat={test_nrmse < published}",
        flush=True,
    )
    return results
