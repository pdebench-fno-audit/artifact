"""Boundary decomposition diagnostic (professor's request).

Computes nRMSE in bands by wall distance for v2's best model:
  - Wall pixels (0-1 px from boundary)
  - Near-wall (1-4 px)
  - Transition (4-16 px)
  - Buffer (16-64 px)
  - Interior (64+ px)

Also splits normal vs tangential velocity error at walls.
This reveals whether the DST-psi model fails primarily through:
  A. Tangential wall slip (psi=0 != no-slip) → the BC mismatch
  B. Interior spectral error → representational capacity
"""

import modal, os, json

app = modal.App("t28-boundary-analysis")
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
    timeout=7200,
    memory=65536,
)
def analyze():
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import glob, time, json

    DEVICE = torch.device("cuda")
    torch.manual_seed(42)

    print("=" * 60, flush=True)
    print("Boundary Decomposition Analysis", flush=True)
    print("=" * 60, flush=True)

    # ── Load v2 model architecture (must match exactly) ──
    MODES, WIDTH, N_LAYERS, INIT_STEP = 12, 20, 4, 2

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

    K_CUT = 32

    def _build_spectral_filter(N, k_cut, device):
        k = torch.fft.rfftfreq(N, d=1.0 / N).to(device)
        filt = torch.exp(-((k / k_cut).clamp(min=0) ** 2 - 1).clamp(min=0) * 4)
        filt[k <= k_cut] = 1.0
        return filt

    def psi_to_velocity(psi, H, W):
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

    class SpectralConvDST2d(nn.Module):
        def __init__(self, ic, oc, mx, my):
            super().__init__()
            self.mx, self.my = mx, my
            self.w = nn.Parameter(torch.randn(ic, oc, mx, my) / (ic * oc))

        def forward(self, x):
            x_st = dst1_2d(x)
            x_t = x_st[:, :, : self.mx, : self.my]
            o = torch.einsum("bihw,iohw->bohw", x_t, self.w)
            out = torch.zeros_like(x_st)
            out[:, :, : self.mx, : self.my] = o
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
        def __init__(self, modes, width, init_step, n_layers, dx, has_forcing=True):
            super().__init__()
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
            parts = [x, grid]
            if self.has_forcing and forcing is not None:
                parts.append(forcing)
            inp = torch.cat(parts, dim=-1).permute(0, 3, 1, 2)
            h = self.fc0(inp)
            for block in self.blocks:
                h = block(h)
            return self.fc2(F.gelu(self.fc1(h))) * self.output_scale

    # ── Load data ──
    data_vol.reload()
    results_vol.reload()
    batch_files = sorted(glob.glob("/data/preprocessed_512_batch_*.npz"))
    print(f"Loading {len(batch_files)} batch files...", flush=True)
    all_vel, all_force = [], []
    for bf in batch_files:
        d = np.load(bf)
        all_vel.append(d["velocity"])
        all_force.append(d["forcing"])
    vel = np.concatenate(all_vel, 0)
    force = np.concatenate(all_force, 0)
    N_total, H, W, T_TOTAL, C = vel.shape
    dx = 1.0 / H
    print(f"Data: {vel.shape}", flush=True)

    data_t = torch.from_numpy(vel).float()
    ch_mean = data_t[:800].mean(dim=(0, 1, 2, 3))
    ch_std = data_t[:800].std(dim=(0, 1, 2, 3)) + 1e-8
    data_t = (data_t - ch_mean) / ch_std
    force_t = torch.from_numpy(force).float()
    f_mean = force_t[:800].mean(dim=(0, 1, 2))
    f_std = force_t[:800].std(dim=(0, 1, 2)) + 1e-8
    force_t = (force_t - f_mean) / f_std

    x_grid = np.linspace(0, 1, H, dtype=np.float32)
    y_grid = np.linspace(0, 1, W, dtype=np.float32)
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1))

    # ── Load v2 best model ──
    model = DSTStreamFNO2d_v2(
        MODES, WIDTH, INIT_STEP, N_LAYERS, dx, has_forcing=True
    ).to(DEVICE)
    best_path = "/results/test_28_dst_v2/best_model.pt"
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, weights_only=True, map_location=DEVICE)
        )
        print(f"Loaded v2 best model", flush=True)
    else:
        print("ERROR: v2 best model not found!", flush=True)
        return

    # ── Build wall-distance mask ──
    # Distance from each pixel to nearest wall (in pixels)
    dist_x = torch.minimum(
        torch.arange(H, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32).flip(0),
    )
    dist_y = torch.minimum(
        torch.arange(W, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32).flip(0),
    )
    wall_dist = torch.minimum(
        dist_x.unsqueeze(1).expand(H, W), dist_y.unsqueeze(0).expand(H, W)
    )  # [H, W]

    bands = [
        ("wall (0-1px)", 0, 1),
        ("near-wall (1-4px)", 1, 4),
        ("transition (4-16px)", 4, 16),
        ("buffer (16-64px)", 16, 64),
        ("interior (64+px)", 64, 999),
    ]

    # ── Normal vs tangential at walls ──
    # Top wall (y=0): normal = +y, tangential = x
    # Bottom wall (y=511): normal = -y, tangential = x
    # Left wall (x=0): normal = +x, tangential = y
    # Right wall (x=511): normal = -x, tangential = y
    # For pixels at distance 0: they're ON the wall

    # ── Run evaluation with spatial decomposition ──
    model.eval()
    N_TEST_START = 800 + 100  # test set starts at sample 900
    N_TEST_END = N_TEST_START + 100
    test_vel = data_t[N_TEST_START:N_TEST_END]
    test_force = force_t[N_TEST_START:N_TEST_END]

    # Accumulate per-band squared errors and norms
    band_sq_err = {name: 0.0 for name, _, _ in bands}
    band_sq_tgt = {name: 0.0 for name, _, _ in bands}
    band_n_pixels = {name: 0 for name, _, _ in bands}

    # Normal vs tangential at wall pixels (dist=0)
    wall_normal_sq_err = 0.0
    wall_tangent_sq_err = 0.0
    wall_normal_sq_tgt = 0.0
    wall_tangent_sq_tgt = 0.0

    print("Running boundary decomposition on test set...", flush=True)
    BATCH_EVAL = 4
    n_test = test_vel.shape[0]
    t_start = time.time()

    with torch.no_grad():
        for bi in range(0, n_test, BATCH_EVAL):
            be = min(bi + BATCH_EVAL, n_test)
            yy = test_vel[bi:be].to(DEVICE)
            ff = test_force[bi:be].to(DEVICE)
            gb = grid.unsqueeze(0).expand(be - bi, -1, -1, -1).to(DEVICE)
            B = yy.shape[0]

            inp = yy[:, :, :, :INIT_STEP, :]
            for t_step in range(INIT_STEP, T_TOTAL):
                inp_flat = inp.reshape(B, H, W, -1)
                psi = model(inp_flat, gb, ff)
                vx_p, vy_p = psi_to_velocity(psi[:, 0:1], H, W)
                vel_pred = torch.stack([vx_p[:, 0], vy_p[:, 0]], dim=-1)  # [B, H, W, 2]
                target = yy[:, :, :, t_step, :]

                # Denormalize
                vp_d = vel_pred * ch_std.to(DEVICE) + ch_mean.to(DEVICE)
                tgt_d = target * ch_std.to(DEVICE) + ch_mean.to(DEVICE)
                err_sq = (vp_d - tgt_d) ** 2  # [B, H, W, 2]

                # Per-band accumulation
                wd = wall_dist.to(DEVICE)
                for name, lo, hi in bands:
                    mask = (wd >= lo) & (wd < hi)  # [H, W]
                    if mask.sum() == 0:
                        continue
                    band_sq_err[name] += err_sq[:, mask, :].sum().item()
                    band_sq_tgt[name] += (tgt_d[:, mask, :] ** 2).sum().item()
                    band_n_pixels[name] += (
                        mask.sum().item() * B * 2
                    )  # 2 velocity components

                # Normal/tangential at wall (dist < 1)
                # Top row (i=0): normal=vx (x-component), tangential=vy
                # Bottom row (i=511): normal=vx, tangential=vy
                # Left col (j=0): normal=vy (y-component), tangential=vx
                # Right col (j=511): normal=vy, tangential=vx

                # Simplified: wall pixels are where wall_dist < 1
                wall_mask = wd < 1  # [H, W]
                if wall_mask.sum() > 0:
                    # For simplicity, compute velocity magnitude components
                    # Top/bottom walls: rows 0 and 511
                    for row in [0, H - 1]:
                        # Normal to top/bottom = y-direction = vy (index 1)
                        wall_normal_sq_err += err_sq[:, row, :, 1].sum().item()
                        wall_tangent_sq_err += err_sq[:, row, :, 0].sum().item()
                        wall_normal_sq_tgt += (tgt_d[:, row, :, 1] ** 2).sum().item()
                        wall_tangent_sq_tgt += (tgt_d[:, row, :, 0] ** 2).sum().item()
                    # Left/right walls: cols 0 and 511
                    for col in [0, W - 1]:
                        wall_normal_sq_err += err_sq[:, :, col, 0].sum().item()
                        wall_tangent_sq_err += err_sq[:, :, col, 1].sum().item()
                        wall_normal_sq_tgt += (tgt_d[:, :, col, 0] ** 2).sum().item()
                        wall_tangent_sq_tgt += (tgt_d[:, :, col, 1] ** 2).sum().item()

                inp = torch.cat([inp[:, :, :, 1:, :], vel_pred.unsqueeze(3)], dim=3)

            if (bi // BATCH_EVAL) % 5 == 0:
                print(
                    f"  Batch {bi}/{n_test}, {time.time() - t_start:.0f}s", flush=True
                )

    # ── Compute nRMSE per band ──
    print(f"\n{'=' * 60}")
    print("BOUNDARY DECOMPOSITION RESULTS (v2 best model, test set)")
    print(f"{'=' * 60}")
    print(f"\n{'Band':<25} | {'nRMSE':>10} | {'% of total err':>14} | {'pixels':>8}")
    print("-" * 65)
    total_err = sum(band_sq_err.values())
    for name, lo, hi in bands:
        se = band_sq_err[name]
        st = band_sq_tgt[name]
        nrmse = np.sqrt(se) / (np.sqrt(st) + 1e-20)
        pct = 100 * se / (total_err + 1e-20)
        npx = band_n_pixels[name]
        print(f"{name:<25} | {nrmse:>10.4f} | {pct:>13.1f}% | {npx:>8}")

    print(f"\n{'=' * 60}")
    print("WALL VELOCITY DECOMPOSITION (normal vs tangential)")
    print(f"{'=' * 60}")
    n_nrmse = np.sqrt(wall_normal_sq_err) / (np.sqrt(wall_normal_sq_tgt) + 1e-20)
    t_nrmse = np.sqrt(wall_tangent_sq_err) / (np.sqrt(wall_tangent_sq_tgt) + 1e-20)
    print(f"Normal velocity nRMSE:     {n_nrmse:.4f}")
    print(f"Tangential velocity nRMSE: {t_nrmse:.4f}")
    print(f"Ratio (tangent/normal):    {t_nrmse / (n_nrmse + 1e-20):.2f}")
    print(f"\nIf tangential >> normal: psi=0 enforces impermeability but NOT no-slip")
    print(f"If normal >> tangential: the issue is something else")

    # ── Also check: what's the actual velocity at the walls? ──
    # If Dirichlet BCs are correct, velocity should be ~0 at walls
    print(f"\n{'=' * 60}")
    print("TARGET VELOCITY AT WALLS (should be ~0 for no-slip)")
    print(f"{'=' * 60}")
    test_vel_raw = (test_vel * ch_std + ch_mean).numpy()
    wall_vel_top = np.abs(test_vel_raw[:, 0, :, :, :]).mean()
    wall_vel_bot = np.abs(test_vel_raw[:, -1, :, :, :]).mean()
    wall_vel_left = np.abs(test_vel_raw[:, :, 0, :, :]).mean()
    wall_vel_right = np.abs(test_vel_raw[:, :, -1, :, :]).mean()
    interior_vel = np.abs(test_vel_raw[:, 64:-64, 64:-64, :, :]).mean()
    print(f"Top wall mean |v|:    {wall_vel_top:.6f}")
    print(f"Bottom wall mean |v|: {wall_vel_bot:.6f}")
    print(f"Left wall mean |v|:   {wall_vel_left:.6f}")
    print(f"Right wall mean |v|:  {wall_vel_right:.6f}")
    print(f"Interior mean |v|:    {interior_vel:.6f}")
    print(
        f"Wall/Interior ratio:  {(wall_vel_top + wall_vel_bot + wall_vel_left + wall_vel_right) / (4 * interior_vel):.4f}"
    )

    # Save results
    results = {
        "bands": {
            name: {
                "nrmse": float(
                    np.sqrt(band_sq_err[name]) / (np.sqrt(band_sq_tgt[name]) + 1e-20)
                ),
                "pct_error": float(100 * band_sq_err[name] / (total_err + 1e-20)),
                "n_pixels": band_n_pixels[name],
            }
            for name, _, _ in bands
        },
        "wall_decomposition": {
            "normal_nrmse": float(n_nrmse),
            "tangential_nrmse": float(t_nrmse),
            "ratio_tangent_over_normal": float(t_nrmse / (n_nrmse + 1e-20)),
        },
        "wall_velocity": {
            "top": float(wall_vel_top),
            "bottom": float(wall_vel_bot),
            "left": float(wall_vel_left),
            "right": float(wall_vel_right),
            "interior": float(interior_vel),
        },
    }

    out_path = "/results/test_28_dst_v2/boundary_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"\nSaved to {out_path}", flush=True)
    return results
