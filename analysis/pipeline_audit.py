"""Vorticity FNO Pipeline Audit — eval-only, no training.

Runs on the existing vorticity FNO checkpoint to diagnose why val nRMSE = 20.67.

Step A — Sanity checks (no training):
  1. Zero predictor: nRMSE should be ~1
  2. Ground-truth projection: ω* → ψ* → v_recon — best possible nRMSE of the pipeline
  3. Train/eval parity: model.train() vs model.eval() nRMSE comparison
  4. Scale logging: per-step |v_pred|_RMS / |v_true|_RMS, correlation, |ω|_RMS, |ψ|_RMS

Step B — Multi-horizon diagnostic:
  nRMSE at H ∈ {1, 5, 10, 25, 50, 91}
  Per-step nRMSE curve
"""

import modal, os, json

app = modal.App("t28-vort-audit")
data_vol = modal.Volume.from_name("ns-incom-data")
results_vol = modal.Volume.from_name("fno-wave6-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch==2.4.1 h5py numpy==1.26.4")
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/data": data_vol, "/results": results_vol},
    timeout=3600,
    memory=65536,
)
def audit():
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import glob

    DEVICE = torch.device("cuda")
    torch.manual_seed(42)

    print("=" * 70, flush=True)
    print("VORTICITY FNO PIPELINE AUDIT", flush=True)
    print("=" * 70, flush=True)

    # ── DST functions (same as vort_fno.py) ──
    def dst1_1d(x, dim=-1):
        N = x.shape[dim]
        zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
        x_ext = torch.cat([zeros, x, zeros, -x.flip(dims=[dim])], dim=dim)
        x_ft = torch.fft.fft(x_ext, dim=dim)
        return -torch.narrow(x_ft.imag, dim, 1, N)

    def dst1_2d(x):
        return dst1_1d(dst1_1d(x, dim=-1), dim=-2)

    def idst1_2d(x):
        Nh, Nw = x.shape[-2], x.shape[-1]
        return dst1_1d(dst1_1d(x, dim=-1), dim=-2) / ((2 * (Nh + 1)) * (2 * (Nw + 1)))

    _poisson_cache = {}

    def _get_poisson_eigenvalues(H, W, device):
        key = (H, W, device)
        if key not in _poisson_cache:
            k = torch.arange(1, H + 1, dtype=torch.float32, device=device)
            m = torch.arange(1, W + 1, dtype=torch.float32, device=device)
            # λ_km = (kπ)² + (mπ)² — eigenvalues of -∇² on [0,1]² with Dirichlet BCs
            # NOTE: the /(N+1) factor belongs to the DST normalization, NOT the eigenvalues!
            # Previous bug had /(H+1) and /(W+1), making eigenvalues 263,000× too small.
            _poisson_cache[key] = (k * np.pi).pow(2).unsqueeze(1) + (m * np.pi).pow(
                2
            ).unsqueeze(0)
        return _poisson_cache[key]

    def poisson_solve(omega, H, W):
        omega_f32 = omega.float()
        omega_dst = dst1_2d(omega_f32)
        lam = _get_poisson_eigenvalues(H, W, omega.device)
        psi_dst = omega_dst / lam.unsqueeze(0).unsqueeze(0)
        return idst1_2d(psi_dst).to(omega.dtype)

    def psi_to_velocity_fd(psi, dx, dy):
        vx = torch.zeros_like(psi)
        vy = torch.zeros_like(psi)
        vx[:, :, :, 1:-1] = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * dy)
        vy[:, :, 1:-1, :] = -(psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * dx)
        return vx, vy

    def velocity_to_vorticity_fd(vx, vy, dx, dy):
        dvydx = torch.zeros_like(vy)
        dvxdy = torch.zeros_like(vx)
        dvydx[:, :, 1:-1, :] = (vy[:, :, 2:, :] - vy[:, :, :-2, :]) / (2 * dx)
        dvxdy[:, :, :, 1:-1] = (vx[:, :, :, 2:] - vx[:, :, :, :-2]) / (2 * dy)
        return dvydx - dvxdy

    # ── Model architecture (must match vort_fno.py exactly) ──
    MODES, WIDTH, N_LAYERS, INIT_STEP = 12, 20, 4, 10

    class SpectralConvDST2d(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            x_f32 = x.float()
            x_st = dst1_2d(x_f32)
            x_t = x_st[:, :, : self.mx, : self.my]
            o = torch.einsum("bihw,iohw->bohw", x_t, self.w.float())
            out = torch.zeros_like(x_st)
            out[:, :, : self.mx, : self.my] = o
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
        def __init__(self, modes, width, init_step, n_layers, has_forcing=True):
            super().__init__()
            in_ch = init_step * 1 + 2 + (2 if has_forcing else 0)
            self.fc0 = nn.Conv2d(in_ch, width, 1)
            self.blocks = nn.ModuleList(
                [DSTBlock2d(width, modes, modes) for _ in range(n_layers)]
            )
            self.fc1 = nn.Conv2d(width, 64, 1)
            self.fc2 = nn.Conv2d(64, 1, 1)
            self.has_forcing = has_forcing

        def forward(self, omega_input, grid, forcing=None):
            parts = [omega_input, grid]
            if self.has_forcing and forcing is not None:
                parts.append(forcing)
            inp = torch.cat(parts, dim=-1).permute(0, 3, 1, 2)
            h = self.fc0(inp)
            for block in self.blocks:
                h = block(h)
            return self.fc2(F.gelu(self.fc1(h)))

    # ── Load data (first batch of native data for audit) ──
    data_vol.reload()
    results_vol.reload()
    native101 = sorted(glob.glob("/data/preprocessed_512_native101_batch_*.npz"))
    d = np.load(native101[0])
    vel_raw = torch.from_numpy(d["velocity"][:8]).float()  # 8 samples for audit
    force_raw = torch.from_numpy(d["forcing"][:8]).float()
    del d

    B, H, W, T, C = vel_raw.shape
    dx, dy = 1.0 / H, 1.0 / W
    print(f"Audit data: {vel_raw.shape}", flush=True)

    # Load stats
    stats_path = native101[0].rsplit("_batch_", 1)[0] + "_stats.npz"
    s = np.load(stats_path)
    vel_mean = torch.tensor(s["vel_mean"], dtype=torch.float32)
    vel_std = torch.tensor(s["vel_std"], dtype=torch.float32)
    force_mean = torch.tensor(s["force_mean"], dtype=torch.float32)
    force_std = torch.tensor(s["force_std"], dtype=torch.float32)

    # Normalize
    vel_norm = (vel_raw - vel_mean) / vel_std
    force_norm = (force_raw - force_mean) / force_std

    grid = torch.from_numpy(
        np.stack(
            np.meshgrid(
                np.linspace(0, 1, H, dtype=np.float32),
                np.linspace(0, 1, W, dtype=np.float32),
                indexing="ij",
            ),
            axis=-1,
        )
    )

    # Helper: compute vorticity from velocity
    def vel_to_omega_batch(vel_frames):
        """vel_frames: [B, H, W, T, 2] -> [B, H, W, T]"""
        omegas = []
        for t in range(vel_frames.shape[3]):
            vx_t = vel_frames[:, :, :, t, 0:1].permute(0, 3, 1, 2)
            vy_t = vel_frames[:, :, :, t, 1:2].permute(0, 3, 1, 2)
            omegas.append(velocity_to_vorticity_fd(vx_t, vy_t, dx, dy)[:, 0])
        return torch.stack(omegas, dim=-1)

    def compute_nrmse(pred, true):
        """pred, true: [B, H, W, 2]"""
        p_d = pred * vel_std + vel_mean
        t_d = true * vel_std + vel_mean
        num = ((p_d - t_d) ** 2).reshape(pred.shape[0], -1).sum(1)
        den = (t_d**2).reshape(true.shape[0], -1).sum(1) + 1e-20
        return (torch.sqrt(num) / torch.sqrt(den)).mean().item()

    # ══════════════════════════════════════════════════════════════
    # TEST 1: ZERO PREDICTOR
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 1: ZERO PREDICTOR CHECK")
    print(f"{'=' * 70}")
    zero_pred = torch.zeros_like(vel_norm[:, :, :, INIT_STEP, :])
    zero_nrmse = compute_nrmse(zero_pred, vel_norm[:, :, :, INIT_STEP, :])
    print(f"  nRMSE(zero, target) at step {INIT_STEP}: {zero_nrmse:.4f}")
    print(f"  Expected: ~1.0")
    print(
        f"  Status: {'PASS' if 0.8 < zero_nrmse < 1.2 else 'FAIL — metric/normalization issue!'}"
    )

    # Multi-step zero predictor
    zero_nrmse_total = []
    for t in range(INIT_STEP, T):
        z = torch.zeros_like(vel_norm[:, :, :, t, :])
        zero_nrmse_total.append(compute_nrmse(z, vel_norm[:, :, :, t, :]))
    print(
        f"  Mean zero nRMSE over all {T - INIT_STEP} steps: {np.mean(zero_nrmse_total):.4f}"
    )

    # ══════════════════════════════════════════════════════════════
    # TEST 2: GROUND-TRUTH PROJECTION
    # ω* → ψ* → v_recon. Best possible nRMSE of the pipeline.
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 2: GROUND-TRUTH PROJECTION (ω* → ψ → v)")
    print(f"{'=' * 70}")

    proj_nrmse_per_step = []
    vel_norm_gpu = vel_norm.to(DEVICE)

    for t in range(INIT_STEP, min(T, INIT_STEP + 20)):  # first 20 steps
        vx_true = vel_norm_gpu[:, :, :, t, 0:1].permute(0, 3, 1, 2)
        vy_true = vel_norm_gpu[:, :, :, t, 1:2].permute(0, 3, 1, 2)
        omega_true = velocity_to_vorticity_fd(vx_true, vy_true, dx, dy)
        psi_recon = poisson_solve(omega_true, H, W)
        vx_recon, vy_recon = psi_to_velocity_fd(psi_recon, dx, dy)
        v_recon = torch.stack([vx_recon[:, 0], vy_recon[:, 0]], dim=-1)
        target = vel_norm_gpu[:, :, :, t, :]
        nrmse = compute_nrmse(v_recon.cpu(), target.cpu())
        proj_nrmse_per_step.append(nrmse)

    print(f"  Step-by-step projection nRMSE:")
    for i, n in enumerate(proj_nrmse_per_step[:10]):
        print(f"    t={INIT_STEP + i}: {n:.6f}")
    print(
        f"  Mean projection nRMSE (first 20 steps): {np.mean(proj_nrmse_per_step):.6f}"
    )
    print(f"  Max projection nRMSE: {np.max(proj_nrmse_per_step):.6f}")

    if np.mean(proj_nrmse_per_step) > 0.3:
        print(f"  STATUS: FAIL — pipeline representation itself is a major bottleneck!")
    elif np.mean(proj_nrmse_per_step) > 0.1:
        print(f"  STATUS: WARNING — boundary/staggering mismatch, may be usable")
    else:
        print(f"  STATUS: PASS — projection is accurate")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: TRAIN/EVAL PARITY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 3: TRAIN/EVAL PARITY (InstanceNorm behavior)")
    print(f"{'=' * 70}")

    # Load model checkpoint
    ckpt_path = "/results/test_28_vort_fno/checkpoint.pt"
    best_path = "/results/test_28_vort_fno/best_model.pt"
    model = VorticityFNO(MODES, WIDTH, INIT_STEP, N_LAYERS, has_forcing=True).to(DEVICE)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        print(f"  Loaded checkpoint from epoch {ck.get('epoch', '?')}", flush=True)
    elif os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, weights_only=True, map_location=DEVICE)
        )
        print(f"  Loaded best model", flush=True)
    else:
        print(f"  WARNING: No checkpoint found! Using random weights.", flush=True)

    vel_gpu = vel_norm.to(DEVICE)
    force_gpu = force_norm.to(DEVICE)
    gb = grid.unsqueeze(0).expand(B, -1, -1, -1).to(DEVICE)

    omega_all = vel_to_omega_batch(vel_gpu)

    # Eval mode (standard validation)
    model.eval()
    with torch.no_grad():
        omega_input = omega_all[:, :, :, :INIT_STEP]
        omega_pred_eval = model(omega_input, gb, force_gpu)
        psi_eval = poisson_solve(omega_pred_eval, H, W)
        vx_eval, vy_eval = psi_to_velocity_fd(psi_eval, dx, dy)
        v_eval = torch.stack([vx_eval[:, 0], vy_eval[:, 0]], dim=-1)
        nrmse_eval = compute_nrmse(v_eval.cpu(), vel_norm[:, :, :, INIT_STEP, :])

    # Train mode (same data, no_grad)
    model.train()
    with torch.no_grad():
        omega_input = omega_all[:, :, :, :INIT_STEP]
        omega_pred_train = model(omega_input, gb, force_gpu)
        psi_train = poisson_solve(omega_pred_train, H, W)
        vx_train, vy_train = psi_to_velocity_fd(psi_train, dx, dy)
        v_train = torch.stack([vx_train[:, 0], vy_train[:, 0]], dim=-1)
        nrmse_train = compute_nrmse(v_train.cpu(), vel_norm[:, :, :, INIT_STEP, :])

    print(f"  1-step nRMSE (model.eval()): {nrmse_eval:.6f}")
    print(f"  1-step nRMSE (model.train()): {nrmse_train:.6f}")
    print(f"  Difference: {abs(nrmse_eval - nrmse_train):.6f}")
    if abs(nrmse_eval - nrmse_train) > 0.1:
        print(f"  STATUS: FAIL — InstanceNorm eval/train mismatch!")
    else:
        print(f"  STATUS: PASS")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: SCALE LOGGING (per-step)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 4: PER-STEP SCALE DIAGNOSTICS")
    print(f"{'=' * 70}")

    model.eval()
    with torch.no_grad():
        omega_input = omega_all[:, :, :, :INIT_STEP].clone()
        print(
            f"  {'Step':>4} | {'|v_pred|_RMS':>12} | {'|v_true|_RMS':>12} | {'ratio':>8} | "
            f"{'corr':>6} | {'|ω_pred|_RMS':>13} | {'|ψ|_RMS':>10} | {'1-step nRMSE':>13}"
        )
        print("  " + "-" * 100)

        for t in range(INIT_STEP, min(T, INIT_STEP + 30)):
            omega_pred = model(omega_input, gb, force_gpu)
            psi = poisson_solve(omega_pred, H, W)
            vx_p, vy_p = psi_to_velocity_fd(psi, dx, dy)
            v_pred = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)
            v_true = vel_gpu[:, :, :, t, :]

            # Denormalize for scale comparison
            vp_d = v_pred * vel_std.to(DEVICE) + vel_mean.to(DEVICE)
            vt_d = v_true * vel_std.to(DEVICE) + vel_mean.to(DEVICE)

            rms_pred = vp_d.pow(2).mean().sqrt().item()
            rms_true = vt_d.pow(2).mean().sqrt().item()
            ratio = rms_pred / (rms_true + 1e-20)

            # Correlation
            vp_flat = vp_d.reshape(-1)
            vt_flat = vt_d.reshape(-1)
            corr = torch.corrcoef(torch.stack([vp_flat, vt_flat]))[0, 1].item()

            omega_rms = omega_pred.pow(2).mean().sqrt().item()
            psi_rms = psi.pow(2).mean().sqrt().item()
            step_nrmse = compute_nrmse(v_pred.cpu(), v_true.cpu())

            print(
                f"  {t - INIT_STEP:4d} | {rms_pred:12.4e} | {rms_true:12.4e} | {ratio:8.4f} | "
                f"{corr:6.3f} | {omega_rms:13.4e} | {psi_rms:10.4e} | {step_nrmse:13.6f}"
            )

            omega_input = torch.cat(
                [omega_input[:, :, :, 1:], omega_pred[:, 0].unsqueeze(-1)], dim=-1
            )

    # ══════════════════════════════════════════════════════════════
    # TEST 5: MULTI-HORIZON nRMSE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 5: MULTI-HORIZON nRMSE")
    print(f"{'=' * 70}")

    model.eval()
    horizons = [1, 5, 10, 25, 50, 91]
    with torch.no_grad():
        omega_input = omega_all[:, :, :, :INIT_STEP].clone()
        cum_sq_err = torch.zeros(B, device=DEVICE)
        cum_sq_tgt = torch.zeros(B, device=DEVICE)

        for t_idx, t in enumerate(range(INIT_STEP, T)):
            omega_pred = model(omega_input, gb, force_gpu)
            psi = poisson_solve(omega_pred, H, W)
            vx_p, vy_p = psi_to_velocity_fd(psi, dx, dy)
            vp = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)
            tgt = vel_gpu[:, :, :, t, :]

            vp_d = vp * vel_std.to(DEVICE) + vel_mean.to(DEVICE)
            tgt_d = tgt * vel_std.to(DEVICE) + vel_mean.to(DEVICE)
            cum_sq_err += ((vp_d - tgt_d) ** 2).reshape(B, -1).sum(1)
            cum_sq_tgt += (tgt_d**2).reshape(B, -1).sum(1)

            step_h = t_idx + 1
            if step_h in horizons:
                nrmse_h = (
                    (torch.sqrt(cum_sq_err) / (torch.sqrt(cum_sq_tgt) + 1e-20))
                    .mean()
                    .item()
                )
                print(f"  H={step_h:3d}: nRMSE = {nrmse_h:.6f}")

            omega_input = torch.cat(
                [omega_input[:, :, :, 1:], omega_pred[:, 0].unsqueeze(-1)], dim=-1
            )

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Zero predictor nRMSE:     {zero_nrmse:.4f} (expect ~1.0)")
    print(f"  Projection nRMSE (mean):  {np.mean(proj_nrmse_per_step):.6f}")
    print(f"  1-step eval nRMSE:        {nrmse_eval:.6f}")
    print(f"  1-step train nRMSE:       {nrmse_train:.6f}")
    print(f"  Train/eval difference:    {abs(nrmse_eval - nrmse_train):.6f}")

    results = {
        "zero_predictor_nrmse": float(zero_nrmse),
        "projection_nrmse_mean": float(np.mean(proj_nrmse_per_step)),
        "projection_nrmse_per_step": [float(x) for x in proj_nrmse_per_step],
        "one_step_eval_nrmse": float(nrmse_eval),
        "one_step_train_nrmse": float(nrmse_train),
    }
    out = "/results/test_28_vort_fno/audit_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"\nSaved to {out}", flush=True)
    return results
