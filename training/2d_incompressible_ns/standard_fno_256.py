"""Test 28A: Standard FNO2d at 256×256 — exact published config + matching eval window."""

import modal, os, json

app = modal.App("t28-std-256")
data_vol = modal.Volume.from_name("ns-incom-data")
results_vol = modal.Volume.from_name("fno-wave6-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "h5py", "numpy==1.26.4", "matplotlib")
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": data_vol, "/results": results_vol},
    timeout=86400,
    memory=65536,
)
def train():
    import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    OUT = "/results/test_28_std_256"
    os.makedirs(OUT, exist_ok=True)
    DEVICE = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Test 28A: Standard FNO2d — 256×256, published eval window", flush=True)

    # EXACT published config
    MODES = 12
    WIDTH = 20
    N_LAYERS = 4
    INIT_STEP = (
        2  # 2 downsampled steps = 10 native steps (matching published initial_step=10)
    )
    BATCH = 5
    EPOCHS = 500
    LR = 1e-3
    N_TRAIN, N_VAL, N_TEST = 800, 100, 100
    PUBLISHED = 0.15

    # ── Standard FNO2d (periodic FFT, NO padding — matching published) ──
    class SpectralConv2d(nn.Module):
        def __init__(self, ic, oc, m1, m2):
            super().__init__()
            self.m1, self.m2 = m1, m2
            s = 1 / (ic * oc)
            self.w1 = nn.Parameter(s * torch.randn(ic, oc, m1, m2, dtype=torch.cfloat))
            self.w2 = nn.Parameter(s * torch.randn(ic, oc, m1, m2, dtype=torch.cfloat))

        def forward(self, x):
            B = x.shape[0]
            xf = torch.fft.rfft2(x)
            Hf = xf.shape[-2]
            o = torch.zeros(
                B,
                self.w1.shape[1],
                Hf,
                x.size(-1) // 2 + 1,
                device=x.device,
                dtype=torch.cfloat,
            )
            o[:, :, : self.m1, : self.m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, : self.m1, : self.m2], self.w1
            )
            o[:, :, -self.m1 :, : self.m2] = torch.einsum(
                "bixy,ioxy->boxy", xf[:, :, -self.m1 :, : self.m2], self.w2
            )
            return torch.fft.irfft2(o, s=(x.size(-2), x.size(-1)))

    class FNO2d(nn.Module):
        """Exact published architecture: 4 spectral + pointwise layers, NO padding."""

        def __init__(self):
            super().__init__()
            self.fc0 = nn.Linear(INIT_STEP * 2 + 2, WIDTH)  # vel frames + grid
            self.convs = nn.ModuleList(
                [SpectralConv2d(WIDTH, WIDTH, MODES, MODES) for _ in range(N_LAYERS)]
            )
            self.ws = nn.ModuleList(
                [nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(N_LAYERS)]
            )
            self.fc1 = nn.Linear(WIDTH, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x, grid):
            x = torch.cat([x, grid], dim=-1)
            x = self.fc0(x).permute(0, 3, 1, 2)
            for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
                x1 = conv(x)
                x2 = w(x)
                x = F.gelu(x1 + x2) if i < N_LAYERS - 1 else (x1 + x2)
            x = x.permute(0, 2, 3, 1)
            return self.fc2(F.gelu(self.fc1(x)))

    class DS(Dataset):
        def __init__(self, data, grid, init_step):
            self.data, self.grid, self.init_step = data, grid, init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i, :, :, : self.init_step, :], self.data[i], self.grid

    # ── Load 256×256 preprocessed cache ──
    CACHE = "/data/preprocessed_256x256_t101.npz"
    data_vol.reload()
    if not os.path.exists(CACHE):
        return {"error": "No 256×256 cache — run preprocess_256.py first"}
    t0 = time.time()
    cached = np.load(CACHE)
    data = cached["velocity"]  # [N, 256, 256, T, 2]
    print(f"Loaded {data.shape} in {time.time() - t0:.1f}s", flush=True)

    N, H, W, T_TOTAL, C = data.shape
    data_t = torch.from_numpy(data)
    x = np.linspace(0, 1, H, dtype=np.float32)
    y = np.linspace(0, 1, W, dtype=np.float32)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32))

    # Per-channel normalization
    ch_mean = data_t[:N_TRAIN].mean(dim=(0, 1, 2, 3))
    ch_std = data_t[:N_TRAIN].std(dim=(0, 1, 2, 3)) + 1e-8
    data_t = (data_t - ch_mean) / ch_std
    print(f"Data: {data_t.shape}, T_TOTAL={T_TOTAL}, INIT_STEP={INIT_STEP}", flush=True)

    train_loader = DataLoader(
        DS(data_t[:N_TRAIN], grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        DS(data_t[N_TRAIN : N_TRAIN + N_VAL], grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
    )
    test_loader = DataLoader(
        DS(data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST], grid, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
    )
    del data, data_t
    grid_dev = grid.to(DEVICE)

    def denorm(x):
        return x * ch_std.to(x.device) + ch_mean.to(x.device)

    def calc_nrmse(p, t):
        pd = denorm(p[:, :, :, INIT_STEP:, :])
        td = denorm(t[:, :, :, INIT_STEP:, :])
        ps = torch.sqrt(((pd - td) ** 2).sum(dim=(1, 2, 3, 4))) / (
            torch.sqrt((td**2).sum(dim=(1, 2, 3, 4))) + 1e-20
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

    model = FNO2d().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Standard FNO2d: {n_params:,} params (w={WIDTH}, m={MODES})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Published StepLR: halve every 100 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()  # Published uses standard MSE

    # Checkpoint-resume
    CKPT = f"{OUT}/checkpoint.pt"
    train_losses = []
    val_log = []
    best_val = float("inf")
    start_ep = 1
    results_vol.reload()
    if os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, weights_only=False, map_location=DEVICE)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_ep = ck["epoch"] + 1
            train_losses = ck["train_losses"]
            val_log = ck["val_log"]
            best_val = ck["best_val"]
            print(f"Resumed at epoch {start_ep}", flush=True)
        except:
            print("Checkpoint incompatible, starting fresh", flush=True)
    else:
        print("Starting fresh", flush=True)

    # Training (published-style: accumulate loss over all AR steps, then backprop)
    t_start = time.time()
    for epoch in range(start_ep, EPOCHS + 1):
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
                loss += loss_fn(pred.reshape(B, -1), target.reshape(B, -1))
                inp = torch.cat([inp[:, :, :, 1:, :], pred.unsqueeze(3)], dim=3)
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
                f"  Ep {epoch:4d}/{EPOCHS} | loss={el / nb:.4e} | lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t_start:.0f}s",
                flush=True,
            )

        if epoch % 50 == 0 or epoch == EPOCHS:
            vp, vt = do_eval(model, val_loader)
            vn, _ = calc_nrmse(vp, vt)
            val_log.append({"epoch": epoch, "val_nrmse": vn})
            print(f"  Ep {epoch:4d} | val nRMSE={vn:.4e}", flush=True)
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
                    "best_val": best_val,
                },
                CKPT,
            )
            results_vol.commit()

    dt = time.time() - t_start
    model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    tp, tt = do_eval(model, test_loader)
    test_nrmse, ps = calc_nrmse(tp, tt)

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"Test 28A STANDARD FNO 256×256: nRMSE={test_nrmse:.4e} vs ~{PUBLISHED}",
        flush=True,
    )
    print(f"Beat: {test_nrmse < PUBLISHED}", flush=True)
    print(f"{'=' * 60}", flush=True)

    results = {
        "test_id": "28_std_256",
        "nrmse": float(test_nrmse),
        "published": PUBLISHED,
        "beat": bool(test_nrmse < PUBLISHED),
        "n_params": n_params,
        "time_s": dt,
        "arch": "Standard FNO2d (periodic FFT, no padding, MSE, StepLR)",
        "resolution": "256x256",
        "temporal": "first 101 native steps / RES_T=5 → 21 steps",
        "config": "width=20, modes=12, 4 layers, MSE, StepLR(100,0.5), wd=1e-4",
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    return results
