"""Test 28 v2: DST-ψ-FNO at 512×512 with THREE improvements from physics critique:

1. 2D DST spectral convolutions with full [ic, oc, modes_x, modes_y] weight tensors
   → 24× more spectral parameters, enabling joint (kx, ky) mode coupling
   → Critical for vortex dynamics where diagonal mode interactions matter

2. Forcing field f(x,y) as additional input channels
   → Each sample has a unique random forcing the model never saw before
   → The forcing field is already in the preprocessed data

3. Pushforward (curriculum) training
   → Start with short rollouts, gradually increase
   → Phase 1: unroll 1 step (epochs 1-100)
   → Phase 2: unroll 5 steps (epochs 101-250)
   → Phase 3: unroll 10 steps (epochs 251-400)
   → Phase 4: unroll all steps (epochs 401-500)
"""

import modal, os, json

app = modal.App("t28-dst-v2")
data_vol = modal.Volume.from_name("ns-incom-data")
results_vol = modal.Volume.from_name("fno-wave6-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch==2.4.1 h5py numpy==1.26.4 matplotlib")
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
    from torch.utils.checkpoint import checkpoint as ckpt_fn
    import glob

    OUT = "/results/test_28_dst_v2"
    os.makedirs(OUT, exist_ok=True)
    DEVICE = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    print("=" * 60, flush=True)
    print("Test 28 v2: DST-ψ-FNO 512×512", flush=True)
    print("  FIX 1: 2D DST (full kx-ky coupling, 24× more spectral params)", flush=True)
    print("  FIX 2: Forcing f(x,y) as input", flush=True)
    print("  FIX 3: Pushforward curriculum training", flush=True)
    print("=" * 60, flush=True)

    MODES = 12
    WIDTH = 20
    N_LAYERS = 4
    INIT_STEP = 2
    BATCH = 2
    LR = 1e-3
    TOTAL_EPOCHS = 500
    N_TRAIN, N_VAL, N_TEST = 800, 100, 100
    PUBLISHED = 0.15

    # Full rollout from epoch 1 — no curriculum
    # Curriculum caused 1-step overfitting. Full rollout is what worked
    # on all previous DST runs and matches published PDEBench approach.
    UNROLL_STEPS = 19  # T_TOTAL - INIT_STEP = 21 - 2 = 19

    def get_unroll_steps(epoch):
        return UNROLL_STEPS

    # ═══════════════════════════════════════════════════════════════
    # DST-I transforms (same proven implementation)
    # ═══════════════════════════════════════════════════════════════
    def dst1_1d(x, dim=-1):
        N = x.shape[dim]
        zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
        x_ext = torch.cat([zeros, x, zeros, -x.flip(dims=[dim])], dim=dim)
        x_ft = torch.fft.fft(x_ext, dim=dim)
        return -torch.narrow(x_ft.imag, dim, 1, N)

    def idst1_1d(x_st, dim=-1):
        N = x_st.shape[dim]
        return dst1_1d(x_st, dim=dim) / (2 * (N + 1))

    def dst1_2d(x):
        """2D DST-I: apply 1D DST along both spatial dims."""
        return dst1_1d(dst1_1d(x, dim=-1), dim=-2)

    def idst1_2d(x):
        """Inverse 2D DST-I."""
        return idst1_1d(idst1_1d(x, dim=-1), dim=-2)

    # ═══════════════════════════════════════════════════════════════
    # FIX 1: 2D DST spectral convolution with full mode coupling
    # Weight shape: [ic, oc, modes_x, modes_y] — NOT separable
    # This gives 20×20×12×12 = 57,600 params per layer (vs 9,600 before)
    # ═══════════════════════════════════════════════════════════════
    class SpectralConvDST2d(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            scale = 1.0 / (ic * oc)
            self.w = nn.Parameter(scale * torch.randn(ic, oc, mx, my))

        def forward(self, x):
            # x: [B, C, H, W]
            x_st = dst1_2d(x)  # 2D DST: [B, C, H, W]
            # Truncate to modes, apply weight, zero-pad back
            x_trunc = x_st[:, :, : self.mx, : self.my]  # [B, ic, mx, my]
            out_trunc = torch.einsum("bihw,iohw->bohw", x_trunc, self.w)
            out = torch.zeros_like(x_st)
            out[:, :, : self.mx, : self.my] = out_trunc
            return idst1_2d(out)

    class DSTBlock2d(nn.Module):
        def __init__(self, w, mx, my):
            super().__init__()
            self.spec = SpectralConvDST2d(w, w, mx, my)
            self.pw = nn.Conv2d(w, w, 1)
            self.norm = nn.InstanceNorm2d(w)

        def forward(self, x):
            xn = self.norm(x)
            return x + F.gelu(self.spec(xn) + self.pw(xn))

    class DSTStreamFNO2d_v2(nn.Module):
        """DST-ψ-FNO with 2D spectral conv and forcing input."""

        def __init__(self, modes, width, init_step, n_layers, dx, has_forcing=True):
            super().__init__()
            # Input: init_step*2 (velocity) + 2 (grid) + 2 (forcing) = init_step*2 + 4
            in_ch = init_step * 2 + 2 + (2 if has_forcing else 0)
            self.fc0 = nn.Conv2d(in_ch, width, 1)
            self.blocks = nn.ModuleList(
                [DSTBlock2d(width, modes, modes) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 128, 1)
            self.fc2 = nn.Conv2d(128, 1, 1)
            self.output_scale = 2.0 * dx
            self.has_forcing = has_forcing

        def forward(self, x, grid, forcing=None):
            # x: [B, H, W, init_step*2], grid: [B, H, W, 2], forcing: [B, H, W, 2]
            parts = [x, grid]
            if self.has_forcing and forcing is not None:
                parts.append(forcing)
            inp = torch.cat(parts, dim=-1).permute(0, 3, 1, 2)  # [B, C, H, W]
            h = self.fc0(inp)
            for block in self.blocks:
                h = block(h)
            return self.fc2(F.gelu(self.fc1(h))) * self.output_scale

    # ═══════════════════════════════════════════════════════════════
    # Filtered spectral derivative (proven from previous run)
    # ═══════════════════════════════════════════════════════════════
    K_CUT = 32

    def _build_spectral_filter(N, k_cut, device):
        k = torch.fft.rfftfreq(N, d=1.0 / N).to(device)
        filt = torch.exp(-((k / k_cut).clamp(min=0) ** 2 - 1).clamp(min=0) * 4)
        filt[k <= k_cut] = 1.0
        return filt

    def psi_to_velocity(psi, H, W):
        """Filtered spectral derivative."""
        dx_phys = 1.0 / W
        psi_fft_y = torch.fft.rfft(psi, dim=-1)
        ky = torch.fft.rfftfreq(W, d=dx_phys).to(psi.device) * 2 * np.pi
        filt_y = _build_spectral_filter(W, K_CUT, psi.device).reshape(1, 1, 1, -1)
        vx = torch.fft.irfft(
            1j * ky.reshape(1, 1, 1, -1) * filt_y * psi_fft_y, n=W, dim=-1
        )
        psi_fft_x = torch.fft.rfft(psi, dim=-2)
        kx = torch.fft.rfftfreq(H, d=dx_phys).to(psi.device) * 2 * np.pi
        filt_x = _build_spectral_filter(H, K_CUT, psi.device).reshape(1, 1, -1, 1)
        vy = -torch.fft.irfft(
            1j * kx.reshape(1, 1, -1, 1) * filt_x * psi_fft_x, n=H, dim=-2
        )
        return vx, vy

    # ═══════════════════════════════════════════════════════════════
    # Dataset with forcing
    # ═══════════════════════════════════════════════════════════════
    class DS(Dataset):
        def __init__(self, vel, forcing, grid, init_step):
            self.vel = vel  # [N, H, W, T, 2]
            self.forcing = forcing  # [N, H, W, 2]
            self.grid = grid  # [H, W, 2]
            self.init_step = init_step

        def __len__(self):
            return self.vel.shape[0]

        def __getitem__(self, i):
            return (
                self.vel[i, :, :, : self.init_step, :],  # input vel
                self.vel[i],  # full trajectory
                self.forcing[i],  # per-sample forcing
                self.grid,
            )

    # ── Load data ──
    data_vol.reload()
    batch_files = sorted(glob.glob("/data/preprocessed_512_batch_*.npz"))
    print(f"Loading {len(batch_files)} batch files...", flush=True)
    t0 = time.time()
    all_vel, all_force = [], []
    for bf in batch_files:
        d = np.load(bf)
        all_vel.append(d["velocity"])
        all_force.append(d["forcing"])
    vel = np.concatenate(all_vel, axis=0)
    force = np.concatenate(all_force, axis=0)
    del all_vel, all_force
    N, H, W, T_TOTAL, C = vel.shape
    dx = 1.0 / H
    print(
        f"Velocity: {vel.shape}, Forcing: {force.shape} in {time.time() - t0:.0f}s",
        flush=True,
    )

    # Normalize velocity (per-channel, global stats from training set)
    data_t = torch.from_numpy(vel)
    del vel
    ch_mean = data_t[:N_TRAIN].mean(dim=(0, 1, 2, 3))
    ch_std = data_t[:N_TRAIN].std(dim=(0, 1, 2, 3)) + 1e-8
    data_t = (data_t - ch_mean) / ch_std

    # Normalize forcing similarly
    force_t = torch.from_numpy(force)
    del force
    f_mean = force_t[:N_TRAIN].mean(dim=(0, 1, 2))
    f_std = force_t[:N_TRAIN].std(dim=(0, 1, 2)) + 1e-8
    force_t = (force_t - f_mean) / f_std

    # Grid
    x = np.linspace(0, 1, H, dtype=np.float32)
    y = np.linspace(0, 1, W, dtype=np.float32)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32))

    train_ds = DS(data_t[:N_TRAIN], force_t[:N_TRAIN], grid, INIT_STEP)
    val_ds = DS(
        data_t[N_TRAIN : N_TRAIN + N_VAL],
        force_t[N_TRAIN : N_TRAIN + N_VAL],
        grid,
        INIT_STEP,
    )
    test_ds = DS(
        data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST],
        force_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST],
        grid,
        INIT_STEP,
    )
    del data_t, force_t

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

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
            for xx, yy, ff, grd in loader:
                xx, yy, ff = xx.to(DEVICE), yy.to(DEVICE), ff.to(DEVICE)
                gb = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
                fb = ff  # [B, H, W, 2]
                pf = yy[:, :, :, :INIT_STEP, :]
                inp = xx.clone()
                for t_step in range(INIT_STEP, yy.shape[3]):
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                    psi = model(inp_flat, gb, fb)
                    vx_p, vy_p = psi_to_velocity(psi[:, 0:1], H, W)
                    vel_pred = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1).unsqueeze(
                        3
                    )
                    pf = torch.cat([pf, vel_pred], dim=3)
                    inp = torch.cat([inp[:, :, :, 1:, :], vel_pred], dim=3)
                ap.append(pf.cpu())
                at.append(yy.cpu())
        return torch.cat(ap, 0), torch.cat(at, 0)

    # ── Create model ──
    model = DSTStreamFNO2d_v2(
        MODES, WIDTH, INIT_STEP, N_LAYERS, dx, has_forcing=True
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} (was ~42K, now should be ~300K+)", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5
    )
    loss_fn = nn.MSELoss()

    train_losses = []
    val_log = []
    best_val = float("inf")
    start_epoch = 1

    # ── Resume from checkpoint if available ──
    results_vol.reload()
    ckpt_path = f"{OUT}/checkpoint.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        train_losses = ck.get("train_losses", [])
        val_log = ck.get("val_log", [])
        print(f"Resumed from epoch {ck['epoch']}, best_val={best_val:.4e}", flush=True)
    else:
        print("Starting fresh (no checkpoint found)", flush=True)

    t_start = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        unroll = get_unroll_steps(epoch)
        el, nb = 0.0, 0

        for xx, yy, ff, grd in train_loader:
            xx, yy, ff = xx.to(DEVICE), yy.to(DEVICE), ff.to(DEVICE)
            gb = grd[0:1].expand(xx.shape[0], -1, -1, -1).to(DEVICE)
            fb = ff
            loss = torch.tensor(0.0, device=DEVICE)
            inp = xx

            # Pushforward: randomly select start within the trajectory
            max_start = T_TOTAL - INIT_STEP - unroll
            if max_start > 0:
                start_offset = torch.randint(0, max_start, (1,)).item()
            else:
                start_offset = 0

            # Re-slice input for this random start
            inp = yy[:, :, :, start_offset : start_offset + INIT_STEP, :]

            for t_step in range(unroll):
                abs_step = start_offset + INIT_STEP + t_step
                if abs_step >= T_TOTAL:
                    break
                inp_flat = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], -1)
                psi = ckpt_fn(model, inp_flat, gb, fb, use_reentrant=False)
                vx_p, vy_p = psi_to_velocity(psi[:, 0:1], H, W)
                vel_pred = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)
                target = yy[:, :, :, abs_step, :]
                B = vel_pred.shape[0]
                loss += loss_fn(vel_pred.reshape(B, -1), target.reshape(B, -1))
                inp = torch.cat([inp[:, :, :, 1:, :], vel_pred.unsqueeze(3)], dim=3)

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            el += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(el / max(nb, 1))

        if epoch <= 5 or epoch % 10 == 0:
            elapsed = time.time() - t_start
            print(
                f"  Ep {epoch:4d}/{TOTAL_EPOCHS} | loss={el / max(nb, 1):.4e} | "
                f"unroll={unroll} | lr={scheduler.get_last_lr()[0]:.1e} | {elapsed:.0f}s",
                flush=True,
            )

        # Validation every 25 epochs or at phase transitions
        phase_transitions = [100, 250, 400]
        if epoch % 25 == 0 or epoch in phase_transitions or epoch == TOTAL_EPOCHS:
            vp, vt = do_eval(model, val_loader)
            vn, _ = calc_nrmse(vp, vt)
            val_log.append({"epoch": epoch, "val_nrmse": vn, "unroll": unroll})
            print(
                f"  Ep {epoch:4d} | val nRMSE={vn:.4e} (published={PUBLISHED})",
                flush=True,
            )
            if vn < best_val:
                best_val = vn
                torch.save(model.state_dict(), f"{OUT}/best_model.pt")
                print(
                    f"    -> New best: {best_val:.4e} ({PUBLISHED / best_val:.2f}× to go)",
                    flush=True,
                )

        # Checkpoint every 50 epochs
        if epoch % 50 == 0:
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
                f"{OUT}/checkpoint.pt",
            )
            results_vol.commit()

    dt = time.time() - t_start

    # ── Final eval on test set ──
    if os.path.exists(f"{OUT}/best_model.pt"):
        model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    tp, tt = do_eval(model, test_loader)
    test_nrmse, per_sample = calc_nrmse(tp, tt)

    print(f"\n{'=' * 60}")
    print(f"DST-ψ-FNO v2: 2D DST + Forcing + Pushforward")
    print(f"  Best val nRMSE:  {best_val:.4e}")
    print(f"  Test nRMSE:      {test_nrmse:.4e}")
    print(f"  Published:       ~{PUBLISHED}")
    print(f"  Beat:            {test_nrmse < PUBLISHED}")
    print(f"  Params:          {n_params:,}")
    print(f"  Total time:      {dt:.0f}s ({dt / 3600:.1f}h)")
    print(f"  Rollout:         full {UNROLL_STEPS} steps from epoch 1")
    print(f"{'=' * 60}", flush=True)

    results = {
        "test_id": "28_dst_v2",
        "nrmse_test": float(test_nrmse),
        "nrmse_best_val": float(best_val),
        "published": PUBLISHED,
        "beat": bool(test_nrmse < PUBLISHED),
        "n_params": n_params,
        "training_time_s": dt,
        "improvements": [
            "2D DST spectral conv (full kx-ky coupling)",
            "Forcing f(x,y) as input",
            "Full rollout from epoch 1 (no curriculum)",
            "Filtered spectral derivative (K_CUT=32)",
        ],
        "rollout": f"full {UNROLL_STEPS} steps from epoch 1",
        "val_log": val_log,
        "per_sample_stats": {
            "best": float(per_sample.min()),
            "median": float(per_sample.median()),
            "worst": float(per_sample.max()),
        },
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions for analysis
    np.savez(
        f"{OUT}/predictions.npz",
        pred=tp.numpy(),
        true=tt.numpy(),
        per_sample=per_sample.numpy(),
    )
    results_vol.commit()
    return results
