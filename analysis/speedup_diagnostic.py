"""Two speedup diagnostics:
1. MATMUL DST: Full-pipeline gradient audit (find the integration bug)
2. TORCH.COMPILE: Benchmark FFT DST with compilation

Both run on the SAME model checkpoint and data. Results tell us which speedup is viable.
"""

import modal, os

app = modal.App("speedup-diagnostic")
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
def diagnose():
    import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

    DEVICE = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # Disable TF32 for precise comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    H = W = 512
    MODES = 12
    WIDTH = 20
    N_LAYERS = 4
    INIT_STEP = 10
    dx = 1.0 / H

    print("=" * 70)
    print("SPEEDUP DIAGNOSTICS")
    print("  1. Matmul DST full-pipeline gradient audit")
    print("  2. torch.compile benchmark")
    print(f"  TF32 disabled for precision")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # FFT DST implementation (proven, from t28_vort_fno.py)
    # ═══════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════
    # Matmul DST implementation (faster, suspected integration bug)
    # Using ORTHONORMAL basis per professor's suggestion
    # ═══════════════════════════════════════════════════════════════
    def _build_basis(N, modes, device, dtype=torch.float32):
        m = torch.arange(1, modes + 1, dtype=torch.float32, device=device)
        n = torch.arange(1, N + 1, dtype=torch.float32, device=device)
        S = torch.sin(m.unsqueeze(1) * np.pi * n.unsqueeze(0) / (N + 1))
        return S.to(dtype)

    # Full N-mode basis for Poisson solve
    S_h_full = _build_basis(H, H, DEVICE)  # [H, H]
    S_w_full = _build_basis(W, W, DEVICE)  # [W, W]
    # Truncated basis for spectral conv
    S_h_trunc = _build_basis(H, MODES, DEVICE)  # [MODES, H]
    S_w_trunc = _build_basis(W, MODES, DEVICE)  # [MODES, W]

    # ═══════════════════════════════════════════════════════════════
    # Spectral conv: FFT version
    # ═══════════════════════════════════════════════════════════════
    class SpectralConv_FFT(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            x_st = dst1_2d_fft(x)
            x_t = x_st[:, :, : self.mx, : self.my]
            o = torch.einsum("bihw,iohw->bohw", x_t, self.w)
            out = torch.zeros_like(x_st)
            out[:, :, : self.mx, : self.my] = o
            return idst1_2d_fft(out)

    # ═══════════════════════════════════════════════════════════════
    # Spectral conv: Matmul version
    # ═══════════════════════════════════════════════════════════════
    class SpectralConv_Mat(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            # Forward 2D DST (truncated to modes)
            x_w = x @ S_w_trunc.T  # [B,C,H,modes]
            x_hw = torch.einsum("bchm,kh->bckm", x_w, S_h_trunc)  # [B,C,mx,my]
            # Weight
            out_hw = torch.einsum("bihw,iohw->bohw", x_hw, self.w)
            # Inverse 2D DST
            out_h = torch.einsum("bckm,kh->bchm", out_hw, S_h_trunc) * (2.0 / (H + 1))
            return out_h @ S_w_trunc * (2.0 / (W + 1))

    # ═══════════════════════════════════════════════════════════════
    # Full model (VorticityFNO)
    # ═══════════════════════════════════════════════════════════════
    class DSTBlock(nn.Module):
        def __init__(self, w, mx, my, SpectralConvClass):
            super().__init__()
            self.spec = SpectralConvClass(w, w, mx, my)
            self.pw = nn.Conv2d(w, w, 1)
            self.norm = nn.InstanceNorm2d(w)

        def forward(self, x):
            xn = self.norm(x)
            return x + F.gelu(self.spec(xn) + self.pw(xn))

    class VortFNO(nn.Module):
        def __init__(self, SpectralConvClass):
            super().__init__()
            in_ch = INIT_STEP + 4  # vort + grid + forcing
            self.fc0 = nn.Conv2d(in_ch, WIDTH, 1)
            self.blocks = nn.ModuleList(
                [
                    DSTBlock(WIDTH, MODES, MODES, SpectralConvClass)
                    for _ in range(N_LAYERS)
                ]
            )
            self.fc1 = nn.Conv2d(WIDTH, 64, 1)
            self.fc2 = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            # x: [B, in_ch, H, W]
            h = self.fc0(x)
            for block in self.blocks:
                h = block(h)
            return self.fc2(F.gelu(self.fc1(h)))

    # ═══════════════════════════════════════════════════════════════
    # Poisson solve (both versions)
    # ═══════════════════════════════════════════════════════════════
    _poisson_eig = None

    def get_poisson_eig():
        nonlocal _poisson_eig
        if _poisson_eig is None:
            k = torch.arange(1, H + 1, dtype=torch.float32, device=DEVICE)
            m = torch.arange(1, W + 1, dtype=torch.float32, device=DEVICE)
            _poisson_eig = (k * np.pi).pow(2).unsqueeze(1) + (m * np.pi).pow(
                2
            ).unsqueeze(0)
        return _poisson_eig

    def poisson_fft(omega):
        omega_dst = dst1_2d_fft(omega)
        return idst1_2d_fft(omega_dst / get_poisson_eig())

    def poisson_mat(omega):
        # Full-mode matmul DST for Poisson (not truncated)
        o_w = omega @ S_w_full.T  # [B,1,H,W] @ [W,W] -> [B,1,H,W]
        o_hw = torch.einsum("bchw,kh->bckw", o_w, S_h_full)  # [B,1,H,W]
        psi_hw = o_hw / get_poisson_eig()
        psi_h = torch.einsum("bckw,kh->bchw", psi_hw, S_h_full) * (2.0 / (H + 1))
        return psi_h @ S_w_full * (2.0 / (W + 1))

    def psi_to_vel(psi):
        vx = torch.zeros_like(psi)
        vy = torch.zeros_like(psi)
        vx[:, :, :, 1:-1] = (psi[:, :, :, 2:] - psi[:, :, :, :-2]) / (2 * dx)
        vy[:, :, 1:-1, :] = -(psi[:, :, 2:, :] - psi[:, :, :-2, :]) / (2 * dx)
        return vx, vy

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Full-pipeline gradient comparison
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 1: FULL-PIPELINE MATMUL vs FFT GRADIENT AUDIT")
    print(f"{'=' * 70}")

    # Create both models with IDENTICAL weights
    model_fft = VortFNO(SpectralConv_FFT).to(DEVICE)
    model_mat = VortFNO(SpectralConv_Mat).to(DEVICE)
    with torch.no_grad():
        for p_fft, p_mat in zip(model_fft.parameters(), model_mat.parameters()):
            p_mat.copy_(p_fft)

    # Random input (simulating one AR step)
    B = 2
    x = torch.randn(B, INIT_STEP + 4, H, W, device=DEVICE)
    target_vel = torch.randn(B, 2, H, W, device=DEVICE)

    # Full pipeline: model -> Poisson -> velocity -> loss
    model_fft.train()
    model_mat.train()

    # FFT path
    omega_fft = model_fft(x)
    psi_fft = poisson_fft(omega_fft)
    vx_fft, vy_fft = psi_to_vel(psi_fft)
    vel_fft = torch.cat([vx_fft, vy_fft], dim=1)
    loss_fft = F.mse_loss(vel_fft, target_vel)
    loss_fft.backward()

    # Matmul path
    omega_mat = model_mat(x.clone())
    psi_mat = poisson_mat(omega_mat)
    vx_mat, vy_mat = psi_to_vel(psi_mat)
    vel_mat = torch.cat([vx_mat, vy_mat], dim=1)
    loss_mat = F.mse_loss(vel_mat, target_vel)
    loss_mat.backward()

    # Compare at every stage
    print(f"\n--- Forward pass comparison ---")
    print(f"  omega:    max_diff = {(omega_fft - omega_mat).abs().max():.2e}")
    print(f"  psi:      max_diff = {(psi_fft - psi_mat).abs().max():.2e}")
    print(f"  vel:      max_diff = {(vel_fft - vel_mat).abs().max():.2e}")
    print(
        f"  loss:     FFT={loss_fft.item():.6e}  Mat={loss_mat.item():.6e}  diff={abs(loss_fft.item() - loss_mat.item()):.2e}"
    )

    print(f"\n--- Per-parameter-group gradient comparison ---")
    param_groups = {}
    for (n_fft, p_fft), (n_mat, p_mat) in zip(
        model_fft.named_parameters(), model_mat.named_parameters()
    ):
        g_fft = p_fft.grad
        g_mat = p_mat.grad
        if g_fft is None or g_mat is None:
            continue
        abs_diff = (g_fft - g_mat).abs().max().item()
        rel_diff = ((g_fft - g_mat) / (g_fft.abs() + 1e-10)).abs().max().item()
        cos_sim = F.cosine_similarity(
            g_fft.flatten().unsqueeze(0), g_mat.flatten().unsqueeze(0)
        ).item()

        # Group by component
        group = n_fft.split(".")[0]
        if "spec" in n_fft:
            group = "spectral_weights"
        elif "pw" in n_fft:
            group = "pointwise_conv"
        elif "norm" in n_fft:
            group = "instancenorm"
        elif "fc" in n_fft:
            group = n_fft.split(".")[0]

        if group not in param_groups:
            param_groups[group] = []
        param_groups[group].append((n_fft, abs_diff, rel_diff, cos_sim))

    for group, params in sorted(param_groups.items()):
        max_abs = max(p[1] for p in params)
        max_rel = max(p[2] for p in params)
        min_cos = min(p[3] for p in params)
        status = "✓" if min_cos > 0.99 and max_rel < 0.1 else "✗ MISMATCH"
        print(
            f"  {group:20s} | abs_diff={max_abs:.2e} | rel_diff={max_rel:.2e} | cos_sim={min_cos:.6f} | {status}"
        )

        if min_cos < 0.99:
            for name, ad, rd, cs in params:
                print(f"    {name}: abs={ad:.2e} rel={rd:.2e} cos={cs:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: torch.compile benchmark
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 2: TORCH.COMPILE BENCHMARK")
    print(f"{'=' * 70}")

    # Reset model
    model_fft.zero_grad()
    model_fft.eval()

    # Benchmark without compile
    x_bench = torch.randn(2, INIT_STEP + 4, H, W, device=DEVICE)
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model_fft(x_bench)
    torch.cuda.synchronize()

    t0 = time.time()
    N_ITER = 10
    for _ in range(N_ITER):
        with torch.no_grad():
            omega = model_fft(x_bench)
            psi = poisson_fft(omega)
            vx, vy = psi_to_vel(psi)
    torch.cuda.synchronize()
    time_no_compile = (time.time() - t0) / N_ITER

    # Compile
    print("  Compiling model + Poisson pipeline...", flush=True)
    try:
        model_compiled = torch.compile(model_fft, mode="reduce-overhead")

        # Warmup compiled
        for _ in range(3):
            with torch.no_grad():
                _ = model_compiled(x_bench)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(N_ITER):
            with torch.no_grad():
                omega = model_compiled(x_bench)
                psi = poisson_fft(omega)
                vx, vy = psi_to_vel(psi)
        torch.cuda.synchronize()
        time_compiled = (time.time() - t0) / N_ITER

        print(f"  Without compile: {time_no_compile * 1000:.1f} ms/step")
        print(f"  With compile:    {time_compiled * 1000:.1f} ms/step")
        print(f"  Speedup:         {time_no_compile / time_compiled:.2f}×")
        print(f"  Per-epoch estimate (91 steps × 400 batches):")
        print(f"    Without: {time_no_compile * 91 * 400 / 60:.1f} min")
        print(f"    With:    {time_compiled * 91 * 400 / 60:.1f} min")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
        time_compiled = None

    # Also benchmark matmul forward (no backward)
    print(f"\n  --- Matmul DST forward benchmark ---")
    model_mat.eval()
    for _ in range(3):
        with torch.no_grad():
            _ = model_mat(x_bench)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N_ITER):
        with torch.no_grad():
            omega = model_mat(x_bench)
            psi = poisson_mat(omega)
            vx, vy = psi_to_vel(psi)
    torch.cuda.synchronize()
    time_matmul = (time.time() - t0) / N_ITER

    print(f"  FFT DST:     {time_no_compile * 1000:.1f} ms/step")
    print(f"  Matmul DST:  {time_matmul * 1000:.1f} ms/step")
    print(f"  Speedup:     {time_no_compile / time_matmul:.2f}×")
    print(f"  Per-epoch (matmul): {time_matmul * 91 * 400 / 60:.1f} min")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  FFT DST (baseline):  {time_no_compile * 1000:.1f} ms/step → {time_no_compile * 91 * 400 / 60:.1f} min/epoch"
    )
    if time_compiled:
        print(
            f"  torch.compile:       {time_compiled * 1000:.1f} ms/step → {time_compiled * 91 * 400 / 60:.1f} min/epoch ({time_no_compile / time_compiled:.2f}×)"
        )
    print(
        f"  Matmul DST:          {time_matmul * 1000:.1f} ms/step → {time_matmul * 91 * 400 / 60:.1f} min/epoch ({time_no_compile / time_matmul:.2f}×)"
    )

    return {
        "fft_ms": time_no_compile * 1000,
        "compile_ms": time_compiled * 1000 if time_compiled else None,
        "matmul_ms": time_matmul * 1000,
    }
