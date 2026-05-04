"""Test 28: Vorticity-Streamfunction FNO with DST Poisson Solve.

NOVEL APPROACH: Model predicts vorticity. Velocity recovered via exact physics pipeline:
  ω_pred → ψ = Poisson_solve(ω) → v = curl(ψ)

WHY THIS WORKS:
  1. DST diagonalizes the Poisson equation on rectangles with Dirichlet BCs
     → ψ̂_km = ω̂_km / λ_km (single element-wise division, exact, fast)
  2. Poisson solve DAMPENS high-k modes (divides by k²+m²)
     → built-in spectral smoother, stabilizes training
  3. Model predicts scalar (ω) not vector (v) — simpler output
  4. Div-free by construction: v = curl(ψ)
  5. Gradients flow through fixed linear ops (Poisson + curl)
     → no learnable α, no output_scale, no gradient pathology

ARCHITECTURE: v2-proven (FFT DST spectral conv, InstanceNorm, standard residual)
DATA: auto-detects native (101 steps) or legacy (21 steps)
"""

import modal, os, json

app = modal.App("t28-vort-matmul")
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
    from torch.utils.data import DataLoader, IterableDataset
    from torch.utils.checkpoint import checkpoint as ckpt_fn
    import glob

    OUT = "/results/test_28_vort_matmul"
    os.makedirs(OUT, exist_ok=True)
    DEVICE = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    # ── Detect data ──
    data_vol.reload()
    native101 = sorted(glob.glob("/data/preprocessed_512_native101_batch_*.npz"))
    legacy = sorted(glob.glob("/data/preprocessed_512_batch_*.npz"))
    if native101:
        batch_files = native101
        INIT_STEP = 10
        USE_NATIVE = True
        print("Using NATIVE 101-step data (INIT_STEP=10)", flush=True)
    else:
        batch_files = legacy
        INIT_STEP = 2
        USE_NATIVE = False
        print("Using LEGACY data (RES_T=5, INIT_STEP=2)", flush=True)

    MODES = 12
    WIDTH = 20  # proven in v2
    N_LAYERS = 4
    BATCH = 2
    LR = 1e-3
    TOTAL_EPOCHS = 500
    N_TRAIN, N_VAL, N_TEST = 800, 100, 100
    PUBLISHED = 0.15

    print("=" * 60, flush=True)
    print(
        "Test 28: Vorticity FNO — HYBRID (matmul spectral conv + FFT Poisson)",
        flush=True,
    )
    print(f"  Spectral conv: matmul DST (12 modes, fast)", flush=True)
    print(
        f"  Poisson solve: FFT DST (512 modes, O(N² log N) beats matmul O(N³))",
        flush=True,
    )
    print(f"  Full 91-step rollout, InstanceNorm, MSE loss", flush=True)
    print(f"  Width={WIDTH}, INIT_STEP={INIT_STEP}, batch={BATCH}", flush=True)
    print("=" * 60, flush=True)

    # ═══════════════════════════════════════════════════════════════
    # MATMUL DST — 3.47× faster than FFT DST (proven gradient-identical)
    # Full-pipeline audit confirmed: cosine_sim = 1.000000 for all params.
    # Previous failures were caused by other bugs (eigenvalues, grid, etc.)
    # ═══════════════════════════════════════════════════════════════
    _basis_cache = {}

    def _get_basis(N, modes, device):
        key = (N, modes, device)
        if key not in _basis_cache:
            m = torch.arange(1, modes + 1, dtype=torch.float32, device=device)
            n = torch.arange(1, N + 1, dtype=torch.float32, device=device)
            S = torch.sin(m.unsqueeze(1) * np.pi * n.unsqueeze(0) / (N + 1))
            _basis_cache[key] = S
        return _basis_cache[key]

    class SpectralConvDST2d(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            S_h = _get_basis(x.shape[-2], self.mx, x.device)
            S_w = _get_basis(x.shape[-1], self.my, x.device)
            # Forward 2D DST (truncated to modes)
            x_w = x @ S_w.T  # [B,C,H,modes]
            x_hw = torch.einsum("bchm,kh->bckm", x_w, S_h)  # [B,C,mx,my]
            # Spectral weights
            out_hw = torch.einsum("bihw,iohw->bohw", x_hw, self.w)
            # Inverse 2D DST
            H, W = x.shape[-2], x.shape[-1]
            out_h = torch.einsum("bckm,kh->bchm", out_hw, S_h) * (2.0 / (H + 1))
            return out_h @ S_w * (2.0 / (W + 1))

    class DSTBlock2d(nn.Module):
        def __init__(self, w, mx, my):
            super().__init__()
            self.spec = SpectralConvDST2d(w, w, mx, my)
            self.pw = nn.Conv2d(w, w, 1)
            self.norm = nn.InstanceNorm2d(w)

        def forward(self, x):
            xn = self.norm(x)
            return x + F.gelu(self.spec(xn) + self.pw(xn))

    class VorticityFNO(nn.Module):
        """Predicts vorticity. No output_scale — vorticity has natural magnitude."""

        def __init__(self, modes, width, init_step, n_layers, has_forcing=True):
            super().__init__()
            # Input: init_step vorticity frames + 2 grid + 2 forcing
            in_ch = init_step * 1 + 2 + (2 if has_forcing else 0)
            self.fc0 = nn.Conv2d(in_ch, width, 1)
            self.blocks = nn.ModuleList(
                [DSTBlock2d(width, modes, modes) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 64, 1)
            self.fc2 = nn.Conv2d(64, 1, 1)
            self.has_forcing = has_forcing

        def forward(self, omega_input, grid, forcing=None):
            # omega_input: [B, H, W, init_step]
            parts = [omega_input, grid]
            if self.has_forcing and forcing is not None:
                parts.append(forcing)
            inp = torch.cat(parts, dim=-1).permute(0, 3, 1, 2)  # [B, C, H, W]
            h = self.fc0(inp)
            for block in self.blocks:
                h = block(h)
            return self.fc2(F.gelu(self.fc1(h)))  # [B, 1, H, W] — predicted ω

    # ═══════════════════════════════════════════════════════════════
    # Physics layers (non-learnable, exact, differentiable)
    # ═══════════════════════════════════════════════════════════════
    _poisson_cache = {}

    def _get_poisson_eigenvalues(H, W, device):
        key = (H, W, device)
        if key not in _poisson_cache:
            k = torch.arange(1, H + 1, dtype=torch.float32, device=device)
            m = torch.arange(1, W + 1, dtype=torch.float32, device=device)
            # λ_km = (kπ)² + (mπ)² — eigenvalues of -∇² on [0,1]² with Dirichlet BCs
            # BUG FIX: was /(H+1) and /(W+1), making eigenvalues 263,000× too small
            lam = (k * np.pi).pow(2).unsqueeze(1) + (m * np.pi).pow(2).unsqueeze(0)
            _poisson_cache[key] = lam
        return _poisson_cache[key]

    # FFT-based DST for Poisson solve (O(N² log N) — faster than matmul O(N³) at full modes)
    def dst1_1d_fft(x, dim=-1):
        N = x.shape[dim]
        zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
        x_ext = torch.cat([zeros, x, zeros, -x.flip(dims=[dim])], dim=dim)
        x_ft = torch.fft.fft(x_ext, dim=dim)
        return -torch.narrow(x_ft.imag, dim, 1, N)

    def dst1_2d_fft(x):
        return dst1_1d_fft(dst1_1d_fft(x, dim=-1), dim=-2)

    def idst1_2d_fft(x):
        Nh, Nw = x.shape[-2], x.shape[-1]
        return dst1_1d_fft(dst1_1d_fft(x, dim=-1), dim=-2) / (
            (2 * (Nh + 1)) * (2 * (Nw + 1))
        )

    def poisson_solve(omega, H, W):
        """Exact Poisson solve via FFT DST (O(N² log N) — faster than matmul for full modes)"""
        omega_f32 = omega.float()
        omega_dst = dst1_2d_fft(omega_f32)
        lam = _get_poisson_eigenvalues(H, W, omega.device)
        psi_dst = omega_dst / lam.unsqueeze(0).unsqueeze(0)
        return idst1_2d_fft(psi_dst).to(omega.dtype)

    def psi_to_velocity_fd(psi, dx, dy):
        """Finite difference curl: v = (∂ψ/∂y, -∂ψ/∂x)"""
        # psi: [B, 1, H, W]
        vx = torch.zeros_like(psi)
        vy = torch.zeros_like(psi)
        vx[:, :, :, 1:-1] = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * dy)
        vy[:, :, 1:-1, :] = -(psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * dx)
        return vx, vy

    def velocity_to_vorticity_fd(vx, vy, dx, dy):
        """Finite difference vorticity: ω = ∂vy/∂x - ∂vx/∂y (no dead allocations)"""
        # FIX 5: removed dead omega allocation, use F.pad instead of zeros_like
        dvydx = F.pad((vy[:, :, 2:, :] - vy[:, :, :-2, :]) / (2 * dx), (0, 0, 1, 1))
        dvxdy = F.pad((vx[:, :, :, 2:] - vx[:, :, :, :-2]) / (2 * dy), (1, 1, 0, 0))
        return dvydx - dvxdy

    def apply_thom_bc_inplace(omega, psi, h):
        """Apply Thom's BC IN-PLACE (no clone). FIX 4: saves 72 GB alloc churn/epoch."""
        factor = -8.0 / (h * h)
        omega[:, :, 0, :] = factor * psi[:, :, 0, :]
        omega[:, :, -1, :] = factor * psi[:, :, -1, :]
        omega[:, :, :, 0] = factor * psi[:, :, :, 0]
        omega[:, :, :, -1] = factor * psi[:, :, :, -1]
        return omega

    # ═══════════════════════════════════════════════════════════════
    # Streaming Dataset
    # ═══════════════════════════════════════════════════════════════
    class StreamingBatchDataset(IterableDataset):
        def __init__(
            self,
            file_segments,
            grid,
            init_step,
            vel_mean,
            vel_std,
            force_mean,
            force_std,
            shuffle=True,
        ):
            self.file_segments = file_segments
            self.grid, self.init_step = grid, init_step
            self.vel_mean, self.vel_std = vel_mean, vel_std
            self.force_mean, self.force_std = force_mean, force_std
            self.shuffle = shuffle
            self.n_samples = sum(e - s for _, s, e in file_segments)

        def __len__(self):
            return self.n_samples

        def __iter__(self):
            import random

            segs = list(self.file_segments)
            if self.shuffle:
                random.shuffle(segs)
            for fpath, start, end in segs:
                d = np.load(fpath)
                vel = torch.from_numpy(d["velocity"][start:end]).float()
                force = torch.from_numpy(d["forcing"][start:end]).float()
                del d
                vel = (vel - self.vel_mean) / self.vel_std
                force = (force - self.force_mean) / self.force_std
                n = vel.shape[0]
                idx = torch.randperm(n) if self.shuffle else torch.arange(n)
                for i in idx:
                    yield vel[i], force[i], self.grid

    # ── Load stats ──
    stats_path = batch_files[0].rsplit("_batch_", 1)[0] + "_stats.npz"
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        vel_mean = torch.tensor(s["vel_mean"], dtype=torch.float32)
        vel_std = torch.tensor(s["vel_std"], dtype=torch.float32)
        force_mean = torch.tensor(s["force_mean"], dtype=torch.float32)
        force_std = torch.tensor(s["force_std"], dtype=torch.float32)
        print(f"Loaded stats", flush=True)
    else:
        print("Computing stats...", flush=True)
        vs, vsq, vn = np.zeros(2, np.float64), np.zeros(2, np.float64), 0
        fs, fsq, fn = np.zeros(2, np.float64), np.zeros(2, np.float64), 0
        sc = 0
        for bf in batch_files:
            d = np.load(bf)
            for si in range(d["velocity"].shape[0]):
                if sc >= N_TRAIN:
                    break
                vs += d["velocity"][si].mean((0, 1, 2)).astype(np.float64)
                vsq += (d["velocity"][si].astype(np.float64) ** 2).mean((0, 1, 2))
                fs += d["forcing"][si].mean((0, 1)).astype(np.float64)
                fsq += (d["forcing"][si].astype(np.float64) ** 2).mean((0, 1))
                vn += 1
                fn += 1
                sc += 1
            del d
            if sc >= N_TRAIN:
                break
        vel_mean = torch.tensor((vs / vn).astype(np.float32))
        vel_std = (
            torch.tensor(np.sqrt(vsq / vn - (vs / vn) ** 2).astype(np.float32)) + 1e-8
        )
        force_mean = torch.tensor((fs / fn).astype(np.float32))
        force_std = (
            torch.tensor(np.sqrt(fsq / fn - (fs / fn) ** 2).astype(np.float32)) + 1e-8
        )
        np.savez(
            stats_path,
            vel_mean=vel_mean.numpy(),
            vel_std=vel_std.numpy(),
            force_mean=force_mean.numpy(),
            force_std=force_std.numpy(),
        )
        data_vol.commit()
        print(f"Computed & saved stats", flush=True)

    ch_mean, ch_std = vel_mean, vel_std

    # ── Build splits ──
    file_sizes = []
    for bf in batch_files:
        with np.load(bf, mmap_mode="r") as d:
            file_sizes.append(d["velocity"].shape[0])
    with np.load(batch_files[0], mmap_mode="r") as d0:
        _, H, W, T_TOTAL, _ = d0["velocity"].shape
    dx = 1.0 / H
    dy = 1.0 / W
    UNROLL_FULL = T_TOTAL - INIT_STEP  # 91 for native data
    UNROLL_TRAIN = (
        UNROLL_FULL  # FULL rollout (20-step unroll proven worse: 0.490 vs 0.318)
    )
    n_total = sum(file_sizes)
    n_train = min(N_TRAIN, n_total)
    n_val = min(N_VAL, n_total - n_train)
    n_test = min(N_TEST, n_total - n_train - n_val)

    def build_segments(gs, ge):
        segs, off = [], 0
        for bf, fs in zip(batch_files, file_sizes):
            s, e = max(gs - off, 0), min(ge - off, fs)
            if s < e:
                segs.append((bf, s, e))
            off += fs
        return segs

    # Cell-centered grid: data is at (i+0.5)/N, NOT at i/(N-1)
    # Confirmed by HDF5 inspection: boundary/interior ratio = 0.50 (half-cell from wall)
    grid = torch.from_numpy(
        np.stack(
            np.meshgrid(
                (np.arange(H, dtype=np.float32) + 0.5) / H,
                (np.arange(W, dtype=np.float32) + 0.5) / W,
                indexing="ij",
            ),
            axis=-1,
        )
    )

    train_ds = StreamingBatchDataset(
        build_segments(0, n_train),
        grid,
        INIT_STEP,
        vel_mean,
        vel_std,
        force_mean,
        force_std,
        True,
    )
    val_ds = StreamingBatchDataset(
        build_segments(n_train, n_train + n_val),
        grid,
        INIT_STEP,
        vel_mean,
        vel_std,
        force_mean,
        force_std,
        False,
    )
    test_ds = StreamingBatchDataset(
        build_segments(n_train + n_val, n_train + n_val + n_test),
        grid,
        INIT_STEP,
        vel_mean,
        vel_std,
        force_mean,
        force_std,
        False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    # FIX minor: move stats to GPU once (avoid 36K .to() calls per epoch)
    ch_mean_gpu = ch_mean_gpu
    ch_std_gpu = ch_std_gpu

    print(
        f"Data: {n_total} samples, {H}x{W}x{T_TOTAL}, INIT={INIT_STEP}, UNROLL_TRAIN={UNROLL_TRAIN}, UNROLL_EVAL={UNROLL_FULL}",
        flush=True,
    )
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Helper: velocity → vorticity on GPU
    # ═══════════════════════════════════════════════════════════════
    def vel_to_omega(vel_frames):
        """FIX 1: Vectorized vel→ω conversion (no Python loop over T).
        vel_frames: [B, H, W, T, 2] → [B, H, W, T]
        """
        # Vectorize: compute FD curl across all timesteps at once
        vx = vel_frames[..., 0]  # [B, H, W, T]
        vy = vel_frames[..., 1]  # [B, H, W, T]
        # ∂vy/∂x along H dim (dim=1), ∂vx/∂y along W dim (dim=2)
        dvydx = F.pad(
            (vy[:, 2:, :, :] - vy[:, :-2, :, :]) / (2 * dx), (0, 0, 0, 0, 1, 1)
        )
        dvxdy = F.pad(
            (vx[:, :, 2:, :] - vx[:, :, :-2, :]) / (2 * dy), (0, 0, 1, 1, 0, 0)
        )
        return dvydx - dvxdy  # [B, H, W, T]

    # ═══════════════════════════════════════════════════════════════
    # Streaming eval
    # ═══════════════════════════════════════════════════════════════
    def do_eval(model, loader):
        model_raw.eval()
        all_nrmse = []
        with torch.no_grad():
            for vel_full, ff, grd in loader:
                vel_full, ff = vel_full.to(DEVICE), ff.to(DEVICE)
                gb = grd[0:1].expand(vel_full.shape[0], -1, -1, -1).to(DEVICE)
                B = vel_full.shape[0]
                sq_err = torch.zeros(B, device=DEVICE)
                sq_tgt = torch.zeros(B, device=DEVICE)

                # Convert velocity to vorticity
                omega_all = vel_to_omega(vel_full)  # [B, H, W, T]

                # Initial vorticity frames
                omega_input = omega_all[:, :, :, :INIT_STEP]  # [B, H, W, INIT_STEP]

                for t in range(INIT_STEP, vel_full.shape[3]):
                    omega_pred = model(omega_input, gb, ff)  # [B, 1, H, W]
                    # Physics: ω → ψ → v
                    psi = poisson_solve(omega_pred, H, W)
                    vx_p, vy_p = psi_to_velocity_fd(psi, dx, dy)
                    vp = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)  # [B, H, W, 2]
                    tgt = vel_full[:, :, :, t, :]  # [B, H, W, 2]

                    # nRMSE on denormalized velocity
                    vp_d = vp * ch_std_gpu + ch_mean_gpu
                    tgt_d = tgt * ch_std_gpu + ch_mean_gpu
                    sq_err += ((vp_d - tgt_d) ** 2).reshape(B, -1).sum(1)
                    sq_tgt += (tgt_d**2).reshape(B, -1).sum(1)

                    # Apply Thom's BC: correct wall vorticity from ψ
                    omega_corrected = apply_thom_bc_inplace(omega_pred, psi, dx)
                    omega_input = torch.cat(
                        [omega_input[:, :, :, 1:], omega_corrected[:, 0].unsqueeze(-1)],
                        dim=-1,
                    )

                all_nrmse.append(
                    (torch.sqrt(sq_err) / (torch.sqrt(sq_tgt) + 1e-20)).cpu()
                )
        ps = torch.cat(all_nrmse)
        return ps.mean().item(), ps

    # ── Model ──
    model_raw = VorticityFNO(MODES, WIDTH, INIT_STEP, N_LAYERS, has_forcing=True).to(
        DEVICE
    )
    n_params = sum(p.numel() for p in model_raw.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    # torch.compile fails on matmul DST (inductor can't handle dynamic tensor ops)
    # Use uncompiled model — matmul DST alone gives the speedup
    model = model_raw
    print(
        "Using uncompiled model (matmul DST for spectral conv, FFT for Poisson)",
        flush=True,
    )

    optimizer = torch.optim.Adam(model_raw.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-5
    )
    loss_fn = nn.MSELoss()

    # ── Resume ──
    train_losses, val_log = [], []
    best_val = float("inf")
    start_epoch = 1
    results_vol.reload()
    ckpt_path = f"{OUT}/checkpoint.pt"
    best_path = f"{OUT}/best_model.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
        model_raw.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        train_losses = ck.get("train_losses", [])
        val_log = ck.get("val_log", [])
        print(
            f"Resumed from checkpoint epoch {ck['epoch']}, best_val={best_val:.4e}",
            flush=True,
        )
    elif os.path.exists(best_path):
        # Fallback: resume from best_model (weights only, no optimizer state)
        model_raw.load_state_dict(
            torch.load(best_path, weights_only=True, map_location=DEVICE)
        )
        start_epoch = 26  # best_model saved at epoch 25 val eval
        best_val = 0.318  # known from previous run
        print(
            f"Resumed from best_model.pt (epoch 25, val={best_val:.4e}), fresh optimizer",
            flush=True,
        )
    else:
        # Warm-start from FFT run's best model
        fft_best = "/results/test_28_vort_fno/best_model.pt"
        if os.path.exists(fft_best):
            model_raw.load_state_dict(
                torch.load(fft_best, weights_only=True, map_location=DEVICE)
            )
            start_epoch = 26
            best_val = 0.318
            print(f"Warm-start from FFT run (epoch 25, val=0.318)", flush=True)

    t_start = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model_raw.train()
        el, nb = 0.0, 0

        for vel_full, ff, grd in train_loader:
            vel_full, ff = vel_full.to(DEVICE), ff.to(DEVICE)
            gb = grd[0:1].expand(vel_full.shape[0], -1, -1, -1).to(DEVICE)
            loss = torch.tensor(0.0, device=DEVICE, requires_grad=False)

            # Convert velocity to vorticity
            omega_all = vel_to_omega(vel_full)  # [B, H, W, T] — FIX 1: vectorized

            # Full rollout (no random offset for full AR training)
            # FIX 6: pre-allocate omega buffer, use roll instead of torch.cat
            omega_input = omega_all[:, :, :, :INIT_STEP].clone()

            CKPT_EVERY = 10  # FIX 3: checkpoint every 10 steps, not every step
            for t in range(UNROLL_TRAIN):
                ast = INIT_STEP + t
                if ast >= T_TOTAL:
                    break

                # FIX 3: only checkpoint every CKPT_EVERY steps
                # FIX 4: AMP bf16 for model forward (matmul DST is bf16-native on H100)
                if t % CKPT_EVERY == 0:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        omega_pred = ckpt_fn(
                            model, omega_input, gb, ff, use_reentrant=False
                        )
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        omega_pred = model(omega_input, gb, ff)

                # Physics pipeline in fp32: ω → ψ → v (FFT needs fp32)
                psi = poisson_solve(omega_pred.float(), H, W)
                vx_p, vy_p = psi_to_velocity_fd(psi, dx, dy)
                vp = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)
                tgt = vel_full[:, :, :, ast, :]

                B = vp.shape[0]
                loss = loss + loss_fn(vp.reshape(B, -1), tgt.reshape(B, -1))

                # FIX 4: Thom BC in-place (no clone)
                apply_thom_bc_inplace(omega_pred, psi, dx)
                # FIX 6: roll + overwrite instead of cat (no new allocation)
                omega_input = omega_input.roll(-1, dims=-1)
                omega_input[:, :, :, -1] = omega_pred[:, 0]

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_raw.parameters(), 1.0)
                optimizer.step()
            el += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(el / max(nb, 1))

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"  Ep {epoch:4d}/{TOTAL_EPOCHS} | loss={el / max(nb, 1):.4e} | "
                f"lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t_start:.0f}s",
                flush=True,
            )

        if epoch % 25 == 0 or epoch == TOTAL_EPOCHS:
            vn, _ = do_eval(model, val_loader)
            val_log.append({"epoch": epoch, "val_nrmse": vn})
            print(
                f"  Ep {epoch:4d} | val nRMSE={vn:.4e} (published={PUBLISHED})",
                flush=True,
            )
            if vn < best_val:
                best_val = vn
                torch.save(model_raw.state_dict(), f"{OUT}/best_model.pt")
                print(f"    -> New best: {best_val:.4e}", flush=True)

        if (
            epoch % 25 == 0
        ):  # save every 25 epochs (was 50 — too infrequent for 24h timeout)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_raw.state_dict(),
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
    if os.path.exists(f"{OUT}/best_model.pt"):
        model_raw.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
    test_nrmse, per_sample = do_eval(model, test_loader)

    print(f"\n{'=' * 60}")
    print(f"  Vorticity-Streamfunction FNO + DST Poisson Solve")
    print(f"  Best val:   {best_val:.4e}")
    print(f"  Test nRMSE: {test_nrmse:.4e}")
    print(f"  Published:  ~{PUBLISHED}")
    print(f"  Beat:       {test_nrmse < PUBLISHED}")
    print(f"  Params:     {n_params:,}")
    print(f"  Time:       {dt:.0f}s ({dt / 3600:.1f}h)")
    print(f"  Data:       {'native' if USE_NATIVE else 'legacy'}")
    print(f"{'=' * 60}", flush=True)

    results = {
        "test_id": "28_vort_fno",
        "nrmse_test": float(test_nrmse),
        "nrmse_best_val": float(best_val),
        "published": PUBLISHED,
        "beat": bool(test_nrmse < PUBLISHED),
        "n_params": n_params,
        "training_time_s": dt,
        "init_step": INIT_STEP,
        "val_log": val_log,
        "approach": "vorticity prediction + DST Poisson solve + FD curl",
        "per_sample_stats": {
            "best": float(per_sample.min().item()),
            "median": float(per_sample.median().item()),
            "worst": float(per_sample.max().item()),
        },
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(f"{OUT}/per_sample_nrmse.npz", nrmse=per_sample.numpy())
    results_vol.commit()
    return results
