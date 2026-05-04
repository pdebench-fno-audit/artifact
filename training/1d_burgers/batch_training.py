"""
Wave 2 Batch: FNO for PDEBench 1D Burgers (ν=0.01, ν=1.0) + Diff-React (ν=2.0, ρ=1.0)
========================================================================================
Parameterized deploy+spawn pattern. One app, 3 independent GPU tasks.

PROTOCOL: 8000/1000/1000 train/val/test split.
  - Validation (indices 8000-8999): model selection (best checkpoint)
  - Test (indices 9000-9999): final reported nRMSE (evaluated ONCE)

Published baselines verified from PDEBench arXiv:2210.07182 Tables 7 & 9.
"""

import modal
import os
import json

app = modal.App("fno-wave2-batch")
volume = modal.Volume.from_name("fno-wave2-results", create_if_missing=True)

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
        "huggingface_hub",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# ═══════════════════════════════════════════════════════════════════
# Test configurations
# ═══════════════════════════════════════════════════════════════════
TEST_CONFIGS = {
    "burgers_nu0.01": {
        "test_id": 10,
        "pde": "1D Burgers",
        "param": "ν=0.01",
        "published_nrmse": 0.0078,  # Table 7: 7.8e-3
        "published_source": "PDEBench arXiv:2210.07182 Table 7, FNO, Burgers nu=0.01",
        "darus_id": "281363",
        "filename": "1D_Burgers_Sols_Nu0.01.hdf5",
        "data_source": "darus",  # Not on HuggingFace
        "architecture": "enhanced",  # 32 modes, 96 width, H1 loss
        "nc": 1,
        "modes": 32,
        "width": 96,
        "h1_weight": 0.05,
        "noise_std": 1e-3,
        "batch_size": 32,
        "epochs": 500,
        "lr": 1e-3,
        "res_x": 4,  # 1024 -> 256
        "res_t": 5,  # 200 -> 40
    },
    "burgers_nu1.0": {
        "test_id": 11,
        "pde": "1D Burgers",
        "param": "ν=1.0",
        "published_nrmse": 0.004,  # Table 7: 4.0e-3
        "published_source": "PDEBench arXiv:2210.07182 Table 7, FNO, Burgers nu=1.0",
        "darus_id": "281365",
        "filename": "1D_Burgers_Sols_Nu1.0.hdf5",
        "data_source": "darus",
        "architecture": "enhanced",
        "nc": 1,
        "modes": 32,
        "width": 96,
        "h1_weight": 0.0,
        "noise_std": 1e-3,  # keep noise for rollout robustness
        "batch_size": 32,
        "epochs": 500,
        "lr": 5e-4,
        "res_x": 4,
        "res_t": 5,
        "use_normalized_loss": True,  # v3: same fix that turned Test 13 from miss to 2× beat
        "use_stratified_val": True,  # v3: representative val split
    },
    "reacdiff_nu2.0_rho1.0": {
        "test_id": 13,
        "pde": "1D Diff-React",
        "param": "ν=2.0, ρ=1.0",
        "published_nrmse": 0.0007,  # Table 9: 7.0e-4
        "published_source": "PDEBench arXiv:2210.07182 Table 9, FNO, ReacDiff nu=2.0 rho=1.0",
        "darus_id": "133185",
        "filename": "ReacDiff_Nu2.0_Rho1.0.hdf5",
        "data_source": "darus",
        "architecture": "enhanced",
        "nc": 1,
        "modes": 32,
        "width": 96,
        "h1_weight": 0.0,
        "noise_std": 0.0,
        "batch_size": 32,
        "epochs": 500,
        "lr": 5e-4,
        "res_x": 4,
        "res_t": 5,
        "use_normalized_loss": True,  # v3: loss = MSE / (target_rms² + eps) — aligns with nRMSE metric
        "use_stratified_val": True,  # v3: stratified val split — ensures rare ICs represented in val
    },
}


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": volume},
    timeout=86400,
    memory=32768,
)
def train(test_key: str, seed: int = 42):
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
    h1_weight = cfg["h1_weight"]
    noise_std = cfg["noise_std"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    res_x = cfg["res_x"]
    res_t = cfg["res_t"]
    use_normalized_loss = cfg.get("use_normalized_loss", False)
    use_stratified_val = cfg.get("use_stratified_val", False)

    SEED = int(seed)
    if SEED != 42:
        OUT = f"/results/test_{test_id}_seed{SEED}"
    else:
        OUT = f"/results/test_{test_id}"
    os.makedirs(OUT, exist_ok=True)

    DEVICE = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Test {test_id}: {cfg['pde']} {cfg['param']} [SEED={SEED}]", flush=True)
    print(f"Published FNO nRMSE: {published} ({cfg['published_source']})", flush=True)
    print(
        f"Architecture: {cfg['architecture']} (modes={modes}, width={width})",
        flush=True,
    )
    if use_normalized_loss:
        print("Using NORMALIZED MSE loss (aligned with nRMSE metric)", flush=True)
    if use_stratified_val:
        print(
            "Using STRATIFIED val split (rare ICs proportionally represented)",
            flush=True,
        )

    # ── PROTOCOL: 8000/1000/1000 ──
    INIT_STEP = 10
    N_TRAIN = 8000
    N_VAL = 1000
    N_TEST = 1000

    # ═══════════════════════════════════════════════════════════════
    # Model definitions
    # ═══════════════════════════════════════════════════════════════
    class SpectralConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, n_modes):
            super().__init__()
            self.modes = n_modes
            scale = 1 / (in_ch * out_ch)
            self.w = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, n_modes, dtype=torch.cfloat)
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

    class FNOBlock(nn.Module):
        """Gated local-global block (used by enhanced arch only)."""

        def __init__(self, w, m, local_kernel=7):
            super().__init__()
            self.spectral = SpectralConv1d(w, w, m)
            self.pointwise = nn.Conv1d(w, w, 1)
            self.local_conv = nn.Conv1d(w, w, local_kernel, padding=local_kernel // 2)
            self.gate = nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            g_out = self.spectral(x) + self.pointwise(x)
            l_out = self.local_conv(x)
            alpha = torch.sigmoid(self.gate)
            return (1 - alpha) * g_out + alpha * l_out

    class FNO1d_Base(nn.Module):
        """Base FNO: 4 spectral layers, no local conv."""

        def __init__(self, n_ch, n_modes=16, n_width=64, init_step=10):
            super().__init__()
            self.fc0 = nn.Linear(init_step * n_ch + 1, n_width)
            self.convs = nn.ModuleList(
                [SpectralConv1d(n_width, n_width, n_modes) for _ in range(4)]
            )
            self.ws = nn.ModuleList([nn.Conv1d(n_width, n_width, 1) for _ in range(4)])
            self.fc1 = nn.Linear(n_width, 128)
            self.fc2 = nn.Linear(128, n_ch)

        def forward(self, x, grid):
            x = torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=-1)
            x = self.fc0(x).permute(0, 2, 1)
            for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
                x1, x2 = conv(x), w(x)
                x = F.gelu(x1 + x2) if i < 3 else (x1 + x2)
            x = x.permute(0, 2, 1)
            return self.fc2(F.gelu(self.fc1(x))).unsqueeze(-2)

    class FNO1d_Enhanced(nn.Module):
        """Enhanced FNO: gated local-global blocks, deeper projection."""

        def __init__(self, n_ch, n_modes=32, n_width=96, init_step=10, n_layers=4):
            super().__init__()
            self.fc0 = nn.Linear(init_step * n_ch + 1, n_width)
            self.blocks = nn.ModuleList(
                [FNOBlock(n_width, n_modes) for _ in range(n_layers)]
            )
            self.fc1 = nn.Linear(n_width, 128)
            self.fc2 = nn.Linear(128, n_ch)
            self.n_layers = n_layers

        def forward(self, x, grid):
            x = torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=-1)
            x = self.fc0(x).permute(0, 2, 1)
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i < self.n_layers - 1:
                    x = F.gelu(x)
            x = x.permute(0, 2, 1)
            return self.fc2(F.gelu(self.fc1(x))).unsqueeze(-2)

    class PDEDS(Dataset):
        def __init__(self, data, grid, init_step=10):
            self.data, self.grid, self.init_step = data, grid, init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i, :, : self.init_step, :], self.data[i], self.grid

    # ═══════════════════════════════════════════════════════════════
    # Download data
    # ═══════════════════════════════════════════════════════════════
    hdf5_path = f"/tmp/{cfg['filename']}"
    if not os.path.exists(hdf5_path) or os.path.getsize(hdf5_path) < 100_000_000:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        if cfg["data_source"] == "darus":
            url = (
                f"https://darus.uni-stuttgart.de/api/access/datafile/{cfg['darus_id']}"
            )
            print(f"Downloading from DaRUS: {url}", flush=True)
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
        else:
            from huggingface_hub import hf_hub_download

            hdf5_path = hf_hub_download(
                repo_id=cfg.get("hf_repo", "pdebench/Burgers"),
                filename=cfg["filename"],
                repo_type="dataset",
                cache_dir="/tmp/hf_cache",
            )

        print(f"Downloaded: {os.path.getsize(hdf5_path) / 1e9:.2f} GB", flush=True)
    else:
        print(f"Cached: {os.path.getsize(hdf5_path) / 1e9:.2f} GB", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Load & preprocess
    # ═══════════════════════════════════════════════════════════════
    with h5py.File(hdf5_path, "r") as f:
        print(f"Keys: {list(f.keys())}", flush=True)
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}", flush=True)

        if "tensor" in f:
            ds = f["tensor"]
        else:
            ds = None
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset) and len(f[k].shape) >= 3:
                    ds = f[k]
                    break

        raw_shape = ds.shape
        print(f"Raw shape: {raw_shape}", flush=True)

        if len(raw_shape) == 3:
            N, T, X = raw_shape
        else:
            N, T, X, _ = raw_shape

        X_ds = X // res_x
        T_ds = int(np.ceil(T / res_t))
        print(f"N={N}, T={T}, X={X} -> X_ds={X_ds}, T_ds={T_ds}", flush=True)

        data = np.empty((N, X_ds, T_ds, nc), dtype=np.float32)
        for s in range(0, N, 500):
            e = min(s + 500, N)
            if len(raw_shape) == 3:
                data[s:e, :, :, 0] = np.transpose(ds[s:e, ::res_t, ::res_x], (0, 2, 1))
            else:
                chunk = ds[s:e, ::res_t, ::res_x]
                if chunk.shape[-1] == nc:
                    data[s:e] = np.transpose(chunk, (0, 2, 1, 3))
                else:
                    data[s:e, :, :, 0] = np.transpose(chunk[..., 0], (0, 2, 1))

        if "x-coordinate" in f:
            grid_np = np.array(f["x-coordinate"], dtype=np.float32)[::res_x]
        elif "x" in f:
            grid_np = np.array(f["x"], dtype=np.float32)[::res_x]
        else:
            grid_np = np.linspace(0, 1, X_ds, dtype=np.float32)

    data_t = torch.from_numpy(data)
    grid_t = torch.tensor(grid_np).unsqueeze(-1)
    T_TOTAL = data_t.shape[2]
    print(f"Data: {data_t.shape}, T_TOTAL={T_TOTAL}", flush=True)
    print(f"Range: [{data_t.min():.4f}, {data_t.max():.4f}]", flush=True)

    # ── 8000/1000/1000 split ──
    if use_stratified_val:
        # Stratified val: ensure rare ICs are proportionally represented in val
        # Compute IC means for the first 9000 samples (train+val pool)
        pool_data = data_t[: N_TRAIN + N_VAL]
        ic_means = pool_data[:, :, 0, 0].mean(dim=1)  # [9000]
        # Sort into bins and sample proportionally
        n_bins = 10
        bin_edges = torch.linspace(
            ic_means.min() - 1e-6, ic_means.max() + 1e-6, n_bins + 1
        )
        val_indices = []
        train_indices = []
        for b in range(n_bins):
            mask = (ic_means >= bin_edges[b]) & (ic_means < bin_edges[b + 1])
            bin_idx = torch.where(mask)[0]
            if len(bin_idx) == 0:
                continue
            # Take proportional share for val (1000/9000 ≈ 11.1%)
            n_val_bin = max(1, round(len(bin_idx) * N_VAL / (N_TRAIN + N_VAL)))
            # Shuffle within bin for randomness
            perm = torch.randperm(len(bin_idx))
            val_indices.extend(bin_idx[perm[:n_val_bin]].tolist())
            train_indices.extend(bin_idx[perm[n_val_bin:]].tolist())
        train_data = pool_data[train_indices]
        val_data = pool_data[val_indices]
        test_data = data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
        # Report val IC distribution
        val_ic_means = val_data[:, :, 0, 0].mean(dim=1)
        n_rare_val = (val_ic_means < 0.35).sum().item()
        print(
            f"Stratified val: {len(val_indices)} val, {len(train_indices)} train",
            flush=True,
        )
        print(
            f"  Val rare ICs (mean<0.35): {n_rare_val}/{len(val_indices)}", flush=True
        )
    else:
        train_data = data_t[:N_TRAIN]
        val_data = data_t[N_TRAIN : N_TRAIN + N_VAL]
        test_data = data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    print(
        f"Split: train={train_data.shape[0]}, val={val_data.shape[0]}, test={test_data.shape[0]}",
        flush=True,
    )

    train_loader = DataLoader(
        PDEDS(train_data, grid_t, INIT_STEP),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PDEDS(val_data, grid_t, INIT_STEP),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        PDEDS(test_data, grid_t, INIT_STEP),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    del data, data_t, train_data, val_data, test_data

    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════
    grid_dev = grid_t.to(DEVICE)

    def calc_nrmse_pertimestep(preds, targets, init_step=INIT_STEP):
        """Per-timestep RMSE/RMS, averaged over (N, C, T)."""
        p = preds[:, :, init_step:, :].permute(0, 3, 1, 2)
        tg = targets[:, :, init_step:, :].permute(0, 3, 1, 2)
        err = torch.sqrt(torch.mean((p - tg) ** 2, dim=2))
        nrm = torch.sqrt(torch.mean(tg**2, dim=2)) + 1e-20
        return torch.mean(err / nrm).item()

    def calc_nrmse_frobenius(preds, targets, init_step=INIT_STEP):
        """Per-sample Frobenius norm ratio."""
        p = preds[:, :, init_step:, :]
        tg = targets[:, :, init_step:, :]
        per_sample = torch.sqrt(((p - tg) ** 2).sum(dim=(1, 2, 3))) / (
            torch.sqrt((tg**2).sum(dim=(1, 2, 3))) + 1e-20
        )
        return per_sample.mean().item()

    def do_eval(model, loader):
        model.eval()
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for xx, yy, _ in loader:
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
        return torch.cat(all_pred, 0), torch.cat(all_tgt, 0)

    def h1_loss_fn(pred, target):
        """Sobolev H1: penalizes spatial gradient errors."""
        return F.mse_loss(
            pred[:, 1:, :] - pred[:, :-1, :], target[:, 1:, :] - target[:, :-1, :]
        )

    # ═══════════════════════════════════════════════════════════════
    # Build model
    # ═══════════════════════════════════════════════════════════════
    if cfg["architecture"] == "enhanced":
        model = FNO1d_Enhanced(nc, modes, width, INIT_STEP, n_layers=4).to(DEVICE)
    else:
        model = FNO1d_Base(nc, modes, width, INIT_STEP).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel ({cfg['architecture']}): {n_params:,} params", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    mse_fn = nn.MSELoss(reduction="mean")

    # ═══════════════════════════════════════════════════════════════
    # Checkpoint-resume
    # ═══════════════════════════════════════════════════════════════
    CKPT_PATH = f"{OUT}/resume_checkpoint.pt"
    train_losses, val_log = [], []
    best_val_nrmse = float("inf")
    start_epoch = 1

    volume.reload()
    if os.path.exists(CKPT_PATH):
        print("Resuming from checkpoint...", flush=True)
        ckpt = torch.load(CKPT_PATH, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        train_losses = ckpt["train_losses"]
        val_log = ckpt["val_log"]
        best_val_nrmse = ckpt["best_val_nrmse"]
        print(
            f"Resumed at epoch {start_epoch}, best val nRMSE={best_val_nrmse:.4e}",
            flush=True,
        )
    else:
        print("Starting fresh training", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for xx, yy, _ in train_loader:
            xx, yy = xx.to(DEVICE), yy.to(DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)
            inp = xx
            for t in range(INIT_STEP, T_TOTAL):
                inp_flat = inp.reshape(inp.shape[0], inp.shape[1], -1)
                pred = model(inp_flat, grid_dev)
                target = yy[:, :, t : t + 1, :]
                B = pred.size(0)
                if use_normalized_loss:
                    # Normalized MSE: MSE / (target_rms² + eps)
                    # Aligns training loss with nRMSE evaluation metric
                    # Low-magnitude targets get proportionally more gradient
                    target_flat = target.reshape(B, -1)
                    pred_flat = pred.reshape(B, -1)
                    per_sample_mse = ((pred_flat - target_flat) ** 2).mean(dim=1)  # [B]
                    per_sample_rms2 = (target_flat**2).mean(dim=1) + 1e-8  # [B]
                    loss = loss + (per_sample_mse / per_sample_rms2).mean()
                else:
                    loss = loss + mse_fn(pred.reshape(B, -1), target.reshape(B, -1))
                if h1_weight > 0:
                    loss = loss + h1_weight * h1_loss_fn(
                        pred[:, :, 0, :], target[:, :, 0, :]
                    )
                if noise_std > 0 and model.training:
                    inp = torch.cat(
                        [inp[:, :, 1:, :], pred + noise_std * torch.randn_like(pred)],
                        dim=-2,
                    )
                else:
                    inp = torch.cat([inp[:, :, 1:, :], pred], dim=-2)

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
                f"  Ep {epoch:4d}/{epochs} | loss={ep_loss / nb:.4e} | "
                f"lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
                flush=True,
            )

        # ── Validation for model selection (NOT test set!) ──
        if epoch % 50 == 0 or epoch == epochs:
            val_preds, val_targets = do_eval(model, val_loader)
            val_nrmse = calc_nrmse_pertimestep(val_preds, val_targets)
            val_log.append({"epoch": epoch, "val_nrmse": val_nrmse})
            print(
                f"  Ep {epoch:4d}/{epochs} | val nRMSE={val_nrmse:.4e} | {time.time() - t0:.0f}s",
                flush=True,
            )
            if val_nrmse < best_val_nrmse:
                best_val_nrmse = val_nrmse
                torch.save(model.state_dict(), f"{OUT}/best_model.pt")
                print(f"    -> New best val: {best_val_nrmse:.4e}", flush=True)

        # ── Checkpoint every 10 epochs ──
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
        f"\nTraining done: best val nRMSE={best_val_nrmse:.4e}, time={dt:.0f}s",
        flush=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # FINAL evaluation on TEST SET (ONCE — this is the reported number)
    # ═══════════════════════════════════════════════════════════════
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    test_preds, test_targets = do_eval(model, test_loader)
    test_nrmse_pt = calc_nrmse_pertimestep(test_preds, test_targets)
    test_nrmse_frob = calc_nrmse_frobenius(test_preds, test_targets)

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"TEST SET EVALUATION (indices 9000-9999, N={test_preds.shape[0]})", flush=True
    )
    print(f"  nRMSE (per-timestep): {test_nrmse_pt:.4e}", flush=True)
    print(f"  nRMSE (Frobenius):    {test_nrmse_frob:.4e}", flush=True)
    print(f"  Published FNO:        {published}", flush=True)
    print(f"  Beat (per-timestep):  {test_nrmse_pt < published}", flush=True)
    print(f"  Beat (Frobenius):     {test_nrmse_frob < published}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Per-sample errors (Frobenius)
    per_sample = torch.zeros(test_preds.shape[0])
    for i in range(test_preds.shape[0]):
        pi = test_preds[i, :, INIT_STEP:, :].reshape(-1)
        ti = test_targets[i, :, INIT_STEP:, :].reshape(-1)
        per_sample[i] = torch.norm(pi - ti) / (torch.norm(ti) + 1e-20)
    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi_idx = int(torch.argsort(per_sample)[len(per_sample) // 2])

    # Data leak checks
    ic_match = torch.allclose(
        test_preds[:, :, :INIT_STEP], test_targets[:, :, :INIT_STEP], atol=1e-10
    )
    pred_differ = not torch.allclose(
        test_preds[:, :, INIT_STEP:], test_targets[:, :, INIT_STEP:], atol=1e-6
    )
    no_nan = not (torch.isnan(test_preds).any() or torch.isinf(test_preds).any())
    near_perfect = (per_sample < 1e-6).float().mean().item()

    print(f"\nData leak checks:", flush=True)
    print(f"  IC preserved exactly: {ic_match}", flush=True)
    print(f"  Predictions differ from truth: {pred_differ}", flush=True)
    print(f"  No NaN/Inf: {no_nan}", flush=True)
    print(f"  Near-perfect samples (<1e-6): {near_perfect * 100:.1f}%", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    results = {
        "test_id": test_id,
        "pde": cfg["pde"],
        "parameter": cfg["param"],
        "nrmse_pertimestep": float(test_nrmse_pt),
        "nrmse_frobenius": float(test_nrmse_frob),
        "metric_note": "Both metrics computed; per-timestep is primary for PDEBench comparison",
        "published_fno": published,
        "published_source": cfg["published_source"],
        "beat_pertimestep": bool(test_nrmse_pt < published),
        "beat_frobenius": bool(test_nrmse_frob < published),
        "best_val_nrmse": float(best_val_nrmse),
        "n_params": n_params,
        "training_time_s": dt,
        "split": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
        "data_leak_checks": {
            "ic_preserved": ic_match,
            "predictions_differ": pred_differ,
            "no_nan_inf": no_nan,
            "near_perfect_fraction": near_perfect,
        },
        "per_sample_stats": {
            "best_idx": bi,
            "best_err": float(per_sample[bi]),
            "median_idx": mi_idx,
            "median_err": float(per_sample[mi_idx]),
            "worst_idx": wi,
            "worst_err": float(per_sample[wi]),
        },
        "architecture": {
            "type": cfg["architecture"],
            "modes": modes,
            "width": width,
            "n_layers": 4,
            "h1_weight": h1_weight,
            "noise_std": noise_std,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "res_x": res_x,
            "res_t": res_t,
            "init_step": INIT_STEP,
        },
    }
    with open(f"{OUT}/results.json", "w") as fout:
        json.dump(results, fout, indent=2)
    with open(f"{OUT}/hyperparameter_log.json", "w") as fout:
        json.dump(
            [
                {
                    "seed": SEED,
                    "modes": modes,
                    "width": width,
                    "lr": lr,
                    "epochs": epochs,
                    "n_params": n_params,
                    "best_val_nrmse": float(best_val_nrmse),
                    "test_nrmse_pt": float(test_nrmse_pt),
                    "time_s": dt,
                }
            ],
            fout,
            indent=2,
        )

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
        grid=grid_t.numpy(),
    )

    # ═══════════════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════════════
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    grid_np2 = grid_t.squeeze().numpy()

    # 1. Training + validation curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.semilogy(train_losses, alpha=0.8, color="steelblue")
    a1.set(xlabel="Epoch", ylabel="Loss", title=f"Training Loss — Test {test_id}")
    a1.grid(True, alpha=0.3)
    ep = [d["epoch"] for d in val_log]
    nv = [d["val_nrmse"] for d in val_log]
    a2.semilogy(ep, nv, "o-", ms=4, color="steelblue", label="Val nRMSE")
    a2.axhline(
        published, color="red", ls="--", lw=2, label=f"Published FNO {published}"
    )
    a2.set(xlabel="Epoch", ylabel="nRMSE", title=f"Val nRMSE — Test {test_id}")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/training_curves.pdf", bbox_inches="tight")
    plt.close()

    # 2. Best/Median/Worst pred vs truth
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for row, (idx, lab) in enumerate([(bi, "Best"), (mi_idx, "Median"), (wi, "Worst")]):
        truth = test_targets[idx, :, :, 0].numpy()
        pr = test_preds[idx, :, :, 0].numpy()
        er = np.abs(pr - truth)
        for col, (arr, title, cmap) in enumerate(
            [
                (truth, f"{lab} Truth (err={per_sample[idx]:.4e})", "viridis"),
                (pr, f"{lab} FNO Pred", "viridis"),
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
        a.set_xlabel("$t$ (step)")
    plt.suptitle(
        f"Test {test_id}: {cfg['pde']} {cfg['param']} — nRMSE={test_nrmse_pt:.4e}",
        fontsize=16,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/pred_vs_truth.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/pred_vs_truth.pdf", bbox_inches="tight")
    plt.close()

    # 3. Line snapshots (median sample)
    nt = test_targets.shape[2]
    snaps = [INIT_STEP, nt // 3, 2 * nt // 3, nt - 1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, ti in zip(axes, snaps):
        ax.plot(
            grid_np2, test_targets[mi_idx, :, ti, 0].numpy(), "k-", lw=2, label="Truth"
        )
        ax.plot(
            grid_np2, test_preds[mi_idx, :, ti, 0].numpy(), "r--", lw=2, label="FNO"
        )
        ax.set_title(f"step={ti}")
        ax.set(xlabel="$x$", ylabel="$u$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Median sample (idx={mi_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/line_snapshots.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/line_snapshots.pdf", bbox_inches="tight")
    plt.close()

    # 4. Error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        float(test_nrmse_frob),
        color="red",
        ls="--",
        lw=2,
        label=f"nRMSE(Frob)={test_nrmse_frob:.4e}",
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
    plt.savefig(f"{OUT}/nrmse_dist.pdf", bbox_inches="tight")
    plt.close()

    # 5. nRMSE vs rollout step (error over time)
    nrmse_per_step = []
    for t in range(INIT_STEP, nt):
        p = test_preds[:, :, t : t + 1, :].permute(0, 3, 1, 2)
        tg = test_targets[:, :, t : t + 1, :].permute(0, 3, 1, 2)
        err = torch.sqrt(torch.mean((p - tg) ** 2, dim=2))
        nrm = torch.sqrt(torch.mean(tg**2, dim=2)) + 1e-20
        nrmse_per_step.append(torch.mean(err / nrm).item())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(INIT_STEP, nt), nrmse_per_step, "b-", lw=2)
    ax.set(
        xlabel="Rollout step",
        ylabel="nRMSE",
        title=f"Error vs Rollout Step — Test {test_id}",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_vs_step.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/nrmse_vs_step.pdf", bbox_inches="tight")
    plt.close()

    np.savez(
        f"{OUT}/nrmse_per_step.npz",
        steps=np.arange(INIT_STEP, nt),
        nrmse=np.array(nrmse_per_step),
    )

    # Manifest
    mf = {
        "script": f"fno_wave2_batch.py::train({test_key})",
        "test_id": test_id,
        "outputs": [
            "results.json",
            "hyperparameter_log.json",
            "training_histories.npz",
            "predictions.npz",
            "nrmse_per_step.npz",
            "best_model.pt",
            "training_curves.png",
            "training_curves.pdf",
            "pred_vs_truth.png",
            "pred_vs_truth.pdf",
            "line_snapshots.png",
            "line_snapshots.pdf",
            "nrmse_dist.png",
            "nrmse_dist.pdf",
            "nrmse_vs_step.png",
            "nrmse_vs_step.pdf",
        ],
        "nrmse_pertimestep": float(test_nrmse_pt),
        "nrmse_frobenius": float(test_nrmse_frob),
        "beat": bool(test_nrmse_pt < published),
        "split": "8000/1000/1000",
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as fout:
        fout.write(json.dumps(mf) + "\n")

    volume.commit()
    print(f"\n{'=' * 60}")
    print(f"FINAL — Test {test_id}: {cfg['pde']} {cfg['param']}")
    print(f"  nRMSE (per-timestep): {test_nrmse_pt:.4e}")
    print(f"  nRMSE (Frobenius):    {test_nrmse_frob:.4e}")
    print(f"  Published FNO:        {published}")
    print(f"  Beat:                 {test_nrmse_pt < published}")
    print(f"  Split:                8000/1000/1000")
    print(f"{'=' * 60}", flush=True)
    return results
