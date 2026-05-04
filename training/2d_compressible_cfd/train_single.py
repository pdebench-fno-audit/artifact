"""
Test 29: 2D Compressible CFD (M=0.1, η=ζ=0.01, Rand periodic)
================================================================
4 channels: density, Vx, Vy, pressure
128×128×21 timesteps, 10000 samples, periodic BCs
Log-space for density and pressure channels
Published FNO nRMSE: 1.7e-1 (PDEBench Table 11)
"""

import modal
import os
import json

app = modal.App("fno-wave6-2dcfd")
volume = modal.Volume.from_name("fno-wave6-results", create_if_missing=True)

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


@app.function(
    gpu="A100", image=image, volumes={"/results": volume}, timeout=86400, memory=32768
)
def train():
    import time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import subprocess
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "/results/test_29"
    os.makedirs(OUT, exist_ok=True)
    DEVICE = torch.device("cuda")
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Test 29: 2D Compressible CFD (M=0.1, η=ζ=0.01, periodic)", flush=True)

    PUBLISHED = 0.17
    NC = 4  # density, Vx, Vy, pressure
    MODES = 12
    WIDTH = 32
    BATCH = 8
    EPOCHS = 500
    LR = 1e-3
    INIT_STEP = 5
    NOISE_STD = 5e-3
    N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000
    LOG_EPS = 1e-6

    # ═══════════════════════════════════════════════════════════════
    # 2D FNO AR Architecture (same as Wave 5)
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
        def __init__(self, w, m1, m2, lk=5):
            super().__init__()
            self.spectral = SpectralConv2d(w, w, m1, m2)
            self.pw = nn.Conv2d(w, w, 1)
            self.lc = nn.Conv2d(w, w, lk, padding=lk // 2)
            self.gate = nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            g = self.spectral(x) + self.pw(x)
            l = self.lc(x)
            a = torch.sigmoid(self.gate)
            return (1 - a) * g + a * l

    class FNO2d_AR(nn.Module):
        def __init__(self, nc, modes, width, init_step=5, n_layers=4):
            super().__init__()
            self.fc0 = nn.Conv2d(init_step * nc + 2, width, 1)
            self.blocks = nn.ModuleList(
                [FNO2dBlock(width, modes, modes) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 128, 1)
            self.fc2 = nn.Conv2d(128, nc, 1)
            self.n_layers = n_layers

        def forward(self, x, grid):
            x = torch.cat([x, grid], dim=-1).permute(0, 3, 1, 2)
            x = self.fc0(x)
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
            return self.fc2(F.gelu(self.fc1(x))).permute(0, 2, 3, 1)

    class DS(Dataset):
        def __init__(self, data, grid, init_step):
            self.data, self.grid, self.init_step = data, grid, init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i, :, :, : self.init_step, :], self.data[i], self.grid

    # ═══════════════════════════════════════════════════════════════
    # Download (55GB — will take a while)
    # ═══════════════════════════════════════════════════════════════
    hdf5_path = "/tmp/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    if not os.path.exists(hdf5_path) or os.path.getsize(hdf5_path) < 1_000_000_000:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        print("Downloading 55GB dataset...", flush=True)
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
                os.path.basename(hdf5_path),
                "https://darus.uni-stuttgart.de/api/access/datafile/164687",
            ],
            check=True,
            timeout=7200,
        )
    print(f"Dataset: {os.path.getsize(hdf5_path) / 1e9:.1f} GB", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Load: separate variables [N, T, X, Y] → [N, X, Y, T, C]
    # Log-space for density (ch0) and pressure (ch3)
    # ═══════════════════════════════════════════════════════════════
    with h5py.File(hdf5_path, "r") as f:
        print(f"Keys: {list(f.keys())}", flush=True)
        rho = np.array(f["density"], dtype=np.float32)  # [N, T, X, Y]
        vx = np.array(f["Vx"], dtype=np.float32)
        vy = np.array(f["Vy"], dtype=np.float32)
        prs = np.array(f["pressure"], dtype=np.float32)
        x_c = np.array(f["x-coordinate"], dtype=np.float32)
        y_c = np.array(f["y-coordinate"], dtype=np.float32)

    N, T, H, W = rho.shape
    print(f"Raw: N={N}, T={T}, H={H}, W={W}", flush=True)
    print(
        f"ρ:[{rho.min():.2f},{rho.max():.2f}] vx:[{vx.min():.2f},{vx.max():.2f}] vy:[{vy.min():.2f},{vy.max():.2f}] p:[{prs.min():.2f},{prs.max():.2f}]",
        flush=True,
    )

    # Log-space for density and pressure
    log_rho = np.log(rho + LOG_EPS)
    log_prs = np.log(prs + LOG_EPS)
    print(
        f"log(ρ):[{log_rho.min():.2f},{log_rho.max():.2f}] log(p):[{log_prs.min():.2f},{log_prs.max():.2f}]",
        flush=True,
    )

    # Stack: [N, T, X, Y, 4] → [N, X, Y, T, 4]
    raw = np.stack([log_rho, vx, vy, log_prs], axis=-1)  # [N, T, X, Y, 4]
    data = np.transpose(raw, (0, 2, 3, 1, 4))  # [N, X, Y, T, 4]
    del rho, vx, vy, prs, log_rho, log_prs, raw

    data_t = torch.from_numpy(data)
    T_TOTAL = data_t.shape[3]
    NC_actual = data_t.shape[4]
    print(f"Data: {data_t.shape}, T_TOTAL={T_TOTAL}, NC={NC_actual}", flush=True)

    gx, gy = np.meshgrid(x_c, y_c, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32))

    # Per-channel normalization (on log-space data)
    ch_mean = data_t[:N_TRAIN].mean(dim=(0, 1, 2, 3))
    ch_std = data_t[:N_TRAIN].std(dim=(0, 1, 2, 3)) + 1e-8
    print(f"Per-ch mean: {ch_mean.tolist()}", flush=True)
    print(f"Per-ch std:  {ch_std.tolist()}", flush=True)
    data_t = (data_t - ch_mean) / ch_std

    # Split
    train_data = data_t[:N_TRAIN]
    val_data = data_t[N_TRAIN : N_TRAIN + N_VAL]
    test_data = data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    print(
        f"Split: {train_data.shape[0]}/{val_data.shape[0]}/{test_data.shape[0]}",
        flush=True,
    )

    train_loader = DataLoader(
        DS(train_data, grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DS(val_data, grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        DS(test_data, grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    del data, data_t, train_data, val_data, test_data
    grid_dev = grid.to(DEVICE)

    # ═══════════════════════════════════════════════════════════════
    # Denormalize: undo norm → exp for density/pressure
    # ═══════════════════════════════════════════════════════════════
    def denorm(x):
        out = x * ch_std.to(x.device) + ch_mean.to(x.device)
        orig = out.clone()
        orig[..., 0] = torch.exp(out[..., 0])  # density
        orig[..., 3] = torch.exp(out[..., 3])  # pressure
        return orig

    def calc_nrmse(preds, targets):
        p = denorm(preds[:, :, :, INIT_STEP:, :])
        tg = denorm(targets[:, :, :, INIT_STEP:, :])
        ps = torch.sqrt(((p - tg) ** 2).sum(dim=(1, 2, 3, 4))) / (
            torch.sqrt((tg**2).sum(dim=(1, 2, 3, 4))) + 1e-20
        )
        return ps.mean().item(), ps

    def do_eval(model, loader):
        model.eval()
        ap, at = [], []
        with torch.no_grad():
            for xx, yy, grd in loader:
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                gb = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
                pf = yy[:, :, :, :INIT_STEP, :]
                inp = xx.clone()
                for t in range(INIT_STEP, yy.shape[3]):
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                    pred = model(inp_flat, gb).unsqueeze(3)
                    pf = torch.cat([pf, pred], dim=3)
                    inp = torch.cat([inp[:, :, :, 1:, :], pred], dim=3)
                ap.append(pf.cpu())
                at.append(yy.cpu())
        return torch.cat(ap, 0), torch.cat(at, 0)

    # Build model
    model = FNO2d_AR(NC_actual, MODES, WIDTH, INIT_STEP, 4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nFNO2d_AR: {n_params:,} params (4ch, modes={MODES}, w={WIDTH})", flush=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Checkpoint-resume
    CKPT = f"{OUT}/resume_checkpoint.pt"
    train_losses, val_log = [], []
    best_val = float("inf")
    start_epoch = 1
    volume.reload()
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        train_losses = ck["train_losses"]
        val_log = ck["val_log"]
        best_val = ck["best_val_nrmse"]
        print(f"Resumed at epoch {start_epoch}", flush=True)
    else:
        print("Starting fresh", flush=True)

    # NOTE: torch.compile removed — complex64 FFT weights cause infinite recompilation
    # A100 hardware alone provides sufficient speedup

    # Training
    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        el, nb = 0.0, 0
        for xx, yy, grd in train_loader:
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)
            gb = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)
            inp = xx
            for t in range(INIT_STEP, T_TOTAL):
                inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                pred = model(inp_flat, gb)
                target = yy[:, :, :, t, :]
                B = pred.shape[0]
                pf = pred.reshape(B, -1)
                tf = target.reshape(B, -1)
                pm = ((pf - tf) ** 2).mean(dim=1)
                pr2 = (tf**2).mean(dim=1) + 1e-8
                loss = loss + (pm / pr2).mean()
                pu = pred.unsqueeze(3)
                if NOISE_STD > 0:
                    pu = pu + NOISE_STD * torch.randn_like(pu)
                inp = torch.cat([inp[:, :, :, 1:, :], pu], dim=3)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            el += loss.item()
            nb += 1
        scheduler.step()
        train_losses.append(el / nb)

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"  Ep {epoch:4d}/{EPOCHS} | loss={el / nb:.4e} | lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
                flush=True,
            )

        if epoch % 50 == 0 or epoch == EPOCHS:
            vp, vt = do_eval(model, val_loader)
            vn, _ = calc_nrmse(vp, vt)
            val_log.append({"epoch": epoch, "val_nrmse": vn})
            print(
                f"  Ep {epoch:4d}/{EPOCHS} | val nRMSE={vn:.4e} | {time.time() - t0:.0f}s",
                flush=True,
            )
            if vn < best_val:
                best_val = vn
                torch.save(model.state_dict(), f"{OUT}/best_model.pt")
                print(f"    -> New best: {best_val:.4e}", flush=True)

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_log": val_log,
                    "best_val_nrmse": best_val,
                },
                CKPT,
            )
            volume.commit()

    dt = time.time() - t0

    # Final test
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    tp, tt = do_eval(model, test_loader)
    test_nrmse, per_sample = calc_nrmse(tp, tt)
    tp_d = denorm(tp)
    tt_d = denorm(tt)

    # Physical bounds
    neg_rho = (tp_d[:, :, :, :, 0] < 0).sum().item()
    neg_prs = (tp_d[:, :, :, :, 3] < 0).sum().item()

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"TEST 29: nRMSE={test_nrmse:.4e} | Published={PUBLISHED} | Beat={test_nrmse < PUBLISHED}",
        flush=True,
    )
    print(
        f"  Neg density: {neg_rho} | Neg pressure: {neg_prs} (should be 0 — log-space)",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi = int(torch.argsort(per_sample)[len(per_sample) // 2])

    results = {
        "test_id": 29,
        "pde": "2D Comp NS",
        "parameter": "M=0.1, η=ζ=0.01, Rand, periodic",
        "nrmse_pertimestep": float(test_nrmse),
        "nrmse_frobenius": float(test_nrmse),
        "published_fno": PUBLISHED,
        "published_source": "PDEBench Table 11, M=0.1, η=ζ=0.01, Rand periodic, nRMSE, FNO",
        "beat_pertimestep": bool(test_nrmse < PUBLISHED),
        "best_val_nrmse": float(best_val),
        "n_params": n_params,
        "training_time_s": dt,
        "split": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "physical_bounds": {
            "neg_density": neg_rho,
            "neg_pressure": neg_prs,
            "positivity_guaranteed": neg_rho == 0 and neg_prs == 0,
            "method": "log-space prediction",
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
            "type": "fno2d_ar_logspace",
            "modes": MODES,
            "width": WIDTH,
            "n_layers": 4,
            "nc": NC,
            "init_step": INIT_STEP,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "noise_std": NOISE_STD,
            "loss": "normalized_mse",
            "log_channels": "density(ch0), pressure(ch3)",
        },
    }
    with open(f"{OUT}/results.json", "w") as fo:
        json.dump(results, fo, indent=2)
    np.savez_compressed(
        f"{OUT}/predictions.npz",
        preds=tp_d.numpy(),
        targets=tt_d.numpy(),
        per_sample=per_sample.numpy(),
    )
    np.savez(
        f"{OUT}/training_histories.npz",
        train_loss=np.array(train_losses),
        val_epochs=np.array([d["epoch"] for d in val_log]),
        val_nrmse=np.array([d["val_nrmse"] for d in val_log]),
    )

    # Plots
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.semilogy(train_losses, alpha=0.8, color="steelblue")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss — Test 29")
    a1.grid(True, alpha=0.3)
    a2.semilogy(
        [d["epoch"] for d in val_log],
        [d["val_nrmse"] for d in val_log],
        "o-",
        ms=4,
        color="steelblue",
        label="Val nRMSE",
    )
    a2.axhline(PUBLISHED, color="red", ls="--", lw=2, label=f"Published {PUBLISHED}")
    a2.set(xlabel="Epoch", ylabel="nRMSE", title="Val nRMSE — Test 29")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        float(test_nrmse), color="red", ls="--", lw=2, label=f"nRMSE={test_nrmse:.4e}"
    )
    ax.axvline(PUBLISHED, color="orange", ls="--", lw=2, label=f"Published={PUBLISHED}")
    ax.set(xlabel="Per-sample relL2", ylabel="Count", title="Error Dist — Test 29")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    mf = {
        "script": "fno2d_compcfd.py",
        "test_id": 29,
        "nrmse": float(test_nrmse),
        "beat": bool(test_nrmse < PUBLISHED),
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as fo:
        fo.write(json.dumps(mf) + "\n")
    volume.commit()
    print(
        f"\nFINAL — Test 29: nRMSE={test_nrmse:.4e} | Beat={test_nrmse < PUBLISHED}",
        flush=True,
    )
    return results
