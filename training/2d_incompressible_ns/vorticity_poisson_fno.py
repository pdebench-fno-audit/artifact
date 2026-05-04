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

app = modal.App("t28-vort-fno")
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

    OUT = "/results/test_28_vort_fno"
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
    print("Test 28: Vorticity-Streamfunction FNO + DST Poisson Solve", flush=True)
    print(f"  Model predicts vorticity, physics recovers velocity", flush=True)
    print(f"  FFT DST spectral conv (proven), InstanceNorm, MSE loss", flush=True)
    print(f"  Width={WIDTH}, INIT_STEP={INIT_STEP}, batch={BATCH}", flush=True)
    print("=" * 60, flush=True)

    # ═══════════════════════════════════════════════════════════════
    # FFT-based DST (proven in v2 — WORKS)
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
        return dst1_1d(dst1_1d(x, dim=-1), dim=-2)

    def idst1_2d(x):
        N_h, N_w = x.shape[-2], x.shape[-1]
        return dst1_1d(dst1_1d(x, dim=-1), dim=-2) / ((2 * (N_h + 1)) * (2 * (N_w + 1)))

    # ═══════════════════════════════════════════════════════════════
    # Spectral convolution (FFT DST, same as v2)
    # ═══════════════════════════════════════════════════════════════
    class SpectralConvDST2d(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            x_f32 = x.float() if x.dtype != torch.float32 else x
            x_st = dst1_2d(x_f32)
            x_trunc = x_st[:, :, : self.mx, : self.my]
            out_trunc = torch.einsum("bihw,iohw->bohw", x_trunc, self.w.float())
            out = torch.zeros_like(x_st)
            out[:, :, : self.mx, : self.my] = out_trunc
            return idst1_2d(out).to(x.dtype)

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

    def poisson_solve(omega, H, W):
        """Exact Poisson solve: ∇²ψ = -ω with ψ=0 on boundary.
        DST diagonalizes this: ψ̂_km = ω̂_km / λ_km
        """
        omega_f32 = omega.float()
        omega_dst = dst1_2d(omega_f32)  # [B, 1, H, W]
        lam = _get_poisson_eigenvalues(H, W, omega.device)  # [H, W]
        psi_dst = omega_dst / lam.unsqueeze(0).unsqueeze(0)  # element-wise division
        psi = idst1_2d(psi_dst)
        return psi.to(omega.dtype)

    def psi_to_velocity_fd(psi, dx, dy):
        """Finite difference curl: v = (∂ψ/∂y, -∂ψ/∂x)"""
        # psi: [B, 1, H, W]
        vx = torch.zeros_like(psi)
        vy = torch.zeros_like(psi)
        vx[:, :, :, 1:-1] = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * dy)
        vy[:, :, 1:-1, :] = -(psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * dx)
        return vx, vy

    def velocity_to_vorticity_fd(vx, vy, dx, dy):
        """Finite difference vorticity: ω = ∂vy/∂x - ∂vx/∂y"""
        # vx, vy: [B, 1, H, W]
        omega = torch.zeros_like(vx)
        dvydx = torch.zeros_like(vy)
        dvxdy = torch.zeros_like(vx)
        dvydx[:, :, 1:-1, :] = (vy[:, :, 2:, :] - vy[:, :, :-2, :]) / (2 * dx)
        dvxdy[:, :, :, 1:-1] = (vx[:, :, :, 2:] - vx[:, :, :, :-2]) / (2 * dy)
        omega = dvydx - dvxdy
        return omega

    def apply_thom_bc(omega, psi, h):
        """Apply Thom's vorticity BC: ω_wall = -2ψ₁/h².

        For cell-centered grid: first cell center is h/2 from wall, ψ_wall=0.
        One-sided difference: ∂ψ/∂n ≈ (ψ₀ - 0)/(h/2) = 2ψ₀/h
        Thom: ω_wall ≈ -∂²ψ/∂n² ≈ -2(ψ₀ - 0)/((h/2)·h) = -4ψ₀/h² (cell-centered)

        We override the wall-adjacent cells of ω with this physics-based value.
        omega: [B, 1, H, W], psi: [B, 1, H, W]
        """
        h2 = h * h
        # For cell-centered: distance from wall to first cell = h/2
        # Thom formula adapted: ω = -2ψ_first / (h/2)^2 = -8ψ_first / h^2
        # But standard Thom on node-centered: ω = -2ψ₁/h²
        # Use the cell-centered version: ω = -2ψ₀/(h/2)² = -8ψ₀/h²
        factor = -8.0 / h2
        omega_corrected = omega.clone()
        omega_corrected[:, :, 0, :] = factor * psi[:, :, 0, :]  # top wall
        omega_corrected[:, :, -1, :] = factor * psi[:, :, -1, :]  # bottom wall
        omega_corrected[:, :, :, 0] = factor * psi[:, :, :, 0]  # left wall
        omega_corrected[:, :, :, -1] = factor * psi[:, :, :, -1]  # right wall
        return omega_corrected

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

    print(
        f"Data: {n_total} samples, {H}x{W}x{T_TOTAL}, INIT={INIT_STEP}, UNROLL_TRAIN={UNROLL_TRAIN}, UNROLL_EVAL={UNROLL_FULL}",
        flush=True,
    )
    print(f"Splits: train={n_train}, val={n_val}, test={n_test}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Helper: velocity → vorticity on GPU
    # ═══════════════════════════════════════════════════════════════
    def vel_to_omega(vel_frames):
        """Convert velocity frames [B, H, W, T, 2] to vorticity [B, H, W, T].
        Uses finite differences on normalized velocity.
        """
        vx = vel_frames[..., 0:1].permute(0, 4, 3, 1, 2)  # [B, 1, T, H, W]... no
        # vel_frames: [B, H, W, T, 2]
        B, Hv, Wv, T, _ = vel_frames.shape
        omega_frames = []
        for t in range(T):
            vx_t = vel_frames[:, :, :, t, 0:1].permute(0, 3, 1, 2)  # [B, 1, H, W]
            vy_t = vel_frames[:, :, :, t, 1:2].permute(0, 3, 1, 2)  # [B, 1, H, W]
            omega_t = velocity_to_vorticity_fd(vx_t, vy_t, dx, dy)  # [B, 1, H, W]
            omega_frames.append(omega_t[:, 0])  # [B, H, W]
        return torch.stack(omega_frames, dim=-1)  # [B, H, W, T]

    # ═══════════════════════════════════════════════════════════════
    # Streaming eval
    # ═══════════════════════════════════════════════════════════════
    def do_eval(model, loader):
        model.eval()
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
                    vp_d = vp * ch_std.to(DEVICE) + ch_mean.to(DEVICE)
                    tgt_d = tgt * ch_std.to(DEVICE) + ch_mean.to(DEVICE)
                    sq_err += ((vp_d - tgt_d) ** 2).reshape(B, -1).sum(1)
                    sq_tgt += (tgt_d**2).reshape(B, -1).sum(1)

                    # Apply Thom's BC: correct wall vorticity from ψ
                    omega_corrected = apply_thom_bc(omega_pred, psi, dx)
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
    model = VorticityFNO(MODES, WIDTH, INIT_STEP, N_LAYERS, has_forcing=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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
        model.load_state_dict(ck["model"])
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
        model.load_state_dict(
            torch.load(best_path, weights_only=True, map_location=DEVICE)
        )
        start_epoch = 26  # best_model saved at epoch 25 val eval
        best_val = 0.318  # known from previous run
        print(
            f"Resumed from best_model.pt (epoch 25, val={best_val:.4e}), fresh optimizer",
            flush=True,
        )

    t_start = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        el, nb = 0.0, 0

        for vel_full, ff, grd in train_loader:
            vel_full, ff = vel_full.to(DEVICE), ff.to(DEVICE)
            gb = grd[0:1].expand(vel_full.shape[0], -1, -1, -1).to(DEVICE)
            loss = torch.tensor(0.0, device=DEVICE)

            # Convert velocity to vorticity
            omega_all = vel_to_omega(vel_full)  # [B, H, W, T]

            # Random start offset: pick a random window of UNROLL_TRAIN steps
            max_start = T_TOTAL - INIT_STEP - UNROLL_TRAIN
            start_offset = (
                torch.randint(0, max(max_start, 1), (1,)).item() if max_start > 0 else 0
            )
            omega_input = omega_all[:, :, :, start_offset : start_offset + INIT_STEP]

            for t in range(UNROLL_TRAIN):
                ast = start_offset + INIT_STEP + t
                if ast >= T_TOTAL:
                    break

                omega_pred = ckpt_fn(model, omega_input, gb, ff, use_reentrant=False)

                # Physics pipeline: ω → ψ → v
                psi = poisson_solve(omega_pred, H, W)
                vx_p, vy_p = psi_to_velocity_fd(psi, dx, dy)
                vp = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)  # [B, H, W, 2]
                tgt = vel_full[:, :, :, ast, :]

                B = vp.shape[0]
                loss = loss + loss_fn(vp.reshape(B, -1), tgt.reshape(B, -1))

                # Apply Thom's BC: correct wall vorticity from ψ before feedback
                omega_corrected = apply_thom_bc(omega_pred, psi, dx)
                omega_input = torch.cat(
                    [omega_input[:, :, :, 1:], omega_corrected[:, 0].unsqueeze(-1)],
                    dim=-1,
                )

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
                torch.save(model.state_dict(), f"{OUT}/best_model.pt")
                print(f"    -> New best: {best_val:.4e}", flush=True)

        if (
            epoch % 25 == 0
        ):  # save every 25 epochs (was 50 — too infrequent for 24h timeout)
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
    if os.path.exists(f"{OUT}/best_model.pt"):
        model.load_state_dict(torch.load(f"{OUT}/best_model.pt", weights_only=True))
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
