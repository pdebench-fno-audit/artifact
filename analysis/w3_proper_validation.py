"""
W3: Wave 1 Advection re-run with PROPER VALIDATION splits
===========================================================
Re-runs Advection beta=1.0 with 8000/1000/1000 train/val/test split
instead of 9000/1000 train/test with test-set model selection.

This addresses reviewer concern W3: Wave 1 uses the same evaluation
protocol the paper criticizes PDEBench for.
"""

import modal
import os
import json

app = modal.App("w3-advection-proper-val")
volume = modal.Volume.from_name("fno-wave5-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "h5py",
        "numpy",
        "scipy",
        "matplotlib",
        "huggingface_hub",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/results": volume},
    timeout=14400,  # 4 hours
    memory=32768,
)
def train_all():
    import time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from huggingface_hub import hf_hub_download
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Config ──
    INIT_STEP = 10
    RES_X = 4  # 1024 -> 256
    RES_T = 5  # 200 -> 40
    # W3: Proper 8000/1000/1000 split with validation-based model selection
    N_TRAIN, N_VAL, N_TEST = 8000, 1000, 1000
    BATCH = 50
    EPOCHS = 500
    LR = 1e-3
    MODES, WIDTH = 16, 64
    NC = 1
    OUT = "/results/W3_advection_proper_val"
    os.makedirs(OUT, exist_ok=True)

    # ── Model ──
    class SpectralConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, modes):
            super().__init__()
            self.modes = modes
            scale = 1 / (in_ch * out_ch)
            self.w = nn.Parameter(
                scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
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

    class FNO1d(nn.Module):
        def __init__(self, nc, modes=16, width=64, init_step=10):
            super().__init__()
            self.fc0 = nn.Linear(init_step * nc + 1, width)
            self.convs = nn.ModuleList(
                [SpectralConv1d(width, width, modes) for _ in range(4)]
            )
            self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
            self.fc1 = nn.Linear(width, 128)
            self.fc2 = nn.Linear(128, nc)

        def forward(self, x, grid):
            # x: [B, X, init_step*nc], grid: [X, 1]
            x = torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=-1)
            x = self.fc0(x).permute(0, 2, 1)  # [B, W, X]
            for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
                x1, x2 = conv(x), w(x)
                x = F.gelu(x1 + x2) if i < 3 else (x1 + x2)
            x = x.permute(0, 2, 1)  # [B, X, W]
            return self.fc2(F.gelu(self.fc1(x))).unsqueeze(-2)  # [B, X, 1, nc]

    class LpLoss:
        def __init__(self, p=2):
            self.p = p

        def __call__(self, x, y, eps=1e-20):
            B = x.size(0)
            d = torch.norm(x.view(B, -1) - y.view(B, -1), self.p, 1)
            n = eps + torch.norm(y.view(B, -1), self.p, 1)
            return torch.mean(d / n)

    class AdvDS(Dataset):
        def __init__(self, data, grid, init_step=10):
            self.data, self.grid, self.init_step = data, grid, init_step

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i, :, : self.init_step, :], self.data[i], self.grid

    # ── Download + Load ──
    print("Downloading dataset...", flush=True)
    hdf5_path = hf_hub_download(
        repo_id="pdebench/Advection",
        filename="1D_Advection_Sols_beta1.0.hdf5",
        repo_type="dataset",
        cache_dir="/tmp/hf_cache",
    )
    with h5py.File(hdf5_path, "r") as f:
        ds = f["tensor"]
        N, T, X = ds.shape
        X_ds = X // RES_X
        T_ds = int(np.ceil(T / RES_T))
        data = np.empty((N, X_ds, T_ds, 1), dtype=np.float32)
        for s in range(0, N, 500):
            e = min(s + 500, N)
            chunk = ds[s:e, ::RES_T, ::RES_X]
            data[s:e, :, :, 0] = np.transpose(chunk, (0, 2, 1))
        grid_np = np.array(f["x-coordinate"], dtype=np.float32)[::RES_X]

    data_t = torch.from_numpy(data)
    grid_t = torch.tensor(grid_np).unsqueeze(-1)
    T_TRAIN = data_t.shape[2]
    print(f"Data: {data_t.shape}, T_TRAIN={T_TRAIN}", flush=True)

    train_loader = DataLoader(
        AdvDS(data_t[:N_TRAIN], grid_t, INIT_STEP),
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    # W3: Separate val (for model selection) and test (evaluated ONCE at end)
    val_loader = DataLoader(
        AdvDS(data_t[N_TRAIN : N_TRAIN + N_VAL], grid_t, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        AdvDS(data_t[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST], grid_t, INIT_STEP),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    del data, data_t

    # ── Ensemble of 3 models ──
    seeds = [7]  # seeds 42 (0.00475) and 123 (0.00470) already complete on volume
    all_models = []
    all_train_losses = []
    all_test_logs = []
    hp_log = []

    for model_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FNO1d(NC, MODES, WIDTH, INIT_STEP).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n{'=' * 60}\nModel {model_idx + 1}/3 (seed={seed}): {n_params:,} params\n{'=' * 60}",
            flush=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )
        loss_fn = nn.MSELoss(reduction="mean")
        loss_fn_eval = LpLoss()
        grid_dev = grid_t.to(DEVICE)

        train_losses, test_log = [], []
        best_nrmse = float("inf")
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_loss, nb = 0.0, 0
            for xx, yy, _ in train_loader:
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                loss = torch.tensor(0.0, device=DEVICE)
                inp = xx
                for t in range(INIT_STEP, T_TRAIN):
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], -1)
                    pred = model(inp_flat, grid_dev)
                    target = yy[:, :, t : t + 1, :]
                    B = pred.size(0)
                    loss = loss + loss_fn(pred.reshape(B, -1), target.reshape(B, -1))
                    inp = torch.cat([inp[:, :, 1:, :], pred], dim=-2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                nb += 1
            scheduler.step()
            avg_loss = ep_loss / nb
            train_losses.append(avg_loss)

            if epoch <= 5 or epoch % 10 == 0:
                print(
                    f"  Ep {epoch:4d}/{EPOCHS} | loss={avg_loss:.4e} | lr={scheduler.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s",
                    flush=True,
                )

            if epoch % 50 == 0 or epoch == EPOCHS:
                model.eval()
                all_pred, all_tgt = [], []
                with torch.no_grad():
                    for xx, yy, _ in val_loader:  # W3: select on val, not test
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
                preds = torch.cat(all_pred, 0)
                targets = torch.cat(all_tgt, 0)
                p = preds[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
                tg = targets[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
                nb2, nc2, nx, nt = tg.shape
                err = torch.sqrt(
                    torch.mean(
                        (p.reshape(nb2, nc2, -1, nt) - tg.reshape(nb2, nc2, -1, nt))
                        ** 2,
                        dim=2,
                    )
                )
                nrm = torch.sqrt(torch.mean(tg.reshape(nb2, nc2, -1, nt) ** 2, dim=2))
                nrmse = torch.mean(err / nrm).item()
                test_log.append({"epoch": epoch, "nrmse": nrmse})
                print(
                    f"  Ep {epoch:4d}/{EPOCHS} | nRMSE={nrmse:.4e} | {time.time() - t0:.0f}s",
                    flush=True,
                )
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
                    torch.save(model.state_dict(), f"{OUT}/best_model_s{seed}.pt")
                    print(f"    -> New best: {best_nrmse:.4e}", flush=True)

        dt = time.time() - t0
        print(
            f"  Model {model_idx + 1} done: best nRMSE={best_nrmse:.4e}, time={dt:.0f}s",
            flush=True,
        )
        model.load_state_dict(
            torch.load(f"{OUT}/best_model_s{seed}.pt", weights_only=True)
        )
        all_models.append(model)
        all_train_losses.append(train_losses)
        all_test_logs.append(test_log)
        hp_log.append(
            {
                "seed": seed,
                "modes": MODES,
                "width": WIDTH,
                "n_layers": 4,
                "lr": LR,
                "epochs": EPOCHS,
                "best_nrmse": best_nrmse,
                "time_s": dt,
                "n_params": n_params,
            }
        )

    # ── Ensemble eval ──
    print(f"\n{'=' * 60}\nEnsemble Evaluation\n{'=' * 60}", flush=True)
    grid_dev = grid_t.to(DEVICE)
    ensemble_preds_list = []
    for mi, model in enumerate(all_models):
        model.eval()
        all_pred = []
        with torch.no_grad():
            for xx, yy, _ in test_loader:
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                pred_full = yy[:, :, :INIT_STEP, :]
                inp = xx.clone()
                for t in range(INIT_STEP, yy.shape[2]):
                    inp_flat = inp.reshape(inp.shape[0], inp.shape[1], -1)
                    pred = model(inp_flat, grid_dev)
                    pred_full = torch.cat([pred_full, pred], dim=2)
                    inp = torch.cat([inp[:, :, 1:, :], pred], dim=2)
                all_pred.append(pred_full.cpu())
        ensemble_preds_list.append(torch.cat(all_pred, 0))

    # Reload targets
    all_tgt = []
    with torch.no_grad():
        for _, yy, _ in test_loader:
            all_tgt.append(yy)
    targets = torch.cat(all_tgt, 0)

    # Ensemble average
    ens_pred = torch.stack(ensemble_preds_list).mean(0)

    def calc_nrmse(preds, targets):
        p = preds[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
        tg = targets[:, :, INIT_STEP:, :].permute(0, 3, 1, 2)
        nb2, nc2, nx, nt = tg.shape
        err = torch.sqrt(
            torch.mean(
                (p.reshape(nb2, nc2, -1, nt) - tg.reshape(nb2, nc2, -1, nt)) ** 2, dim=2
            )
        )
        nrm = torch.sqrt(torch.mean(tg.reshape(nb2, nc2, -1, nt) ** 2, dim=2))
        return torch.mean(err / nrm).item()

    for mi, seed in enumerate(seeds):
        n = calc_nrmse(ensemble_preds_list[mi], targets)
        print(f"  Model seed={seed}: nRMSE = {n:.4e}", flush=True)

    ens_nrmse = calc_nrmse(ens_pred, targets)
    print(f"\n  ENSEMBLE: nRMSE = {ens_nrmse:.4e}", flush=True)
    print(f"  Published: 5.9e-3 = 0.005900", flush=True)
    print(
        f"  {'BEAT' if ens_nrmse < 0.0059 else 'DID NOT BEAT'} benchmark!", flush=True
    )

    # Per-sample errors
    per_sample = torch.zeros(ens_pred.shape[0])
    for i in range(ens_pred.shape[0]):
        pi = ens_pred[i, :, INIT_STEP:, :].reshape(-1)
        ti = targets[i, :, INIT_STEP:, :].reshape(-1)
        per_sample[i] = torch.norm(pi - ti) / (torch.norm(ti) + 1e-20)
    bi = int(torch.argmin(per_sample))
    wi = int(torch.argmax(per_sample))
    mi_idx = int(torch.argsort(per_sample)[len(per_sample) // 2])

    # Conservation
    u_int = targets[:, :, :, 0].mean(dim=1)
    u_int_p = ens_pred[:, :, :, 0].mean(dim=1)
    cons_drift = torch.abs(u_int_p[:, -1] - u_int[:, 0]) / (
        torch.abs(u_int[:, 0]) + 1e-20
    )
    mean_cons = cons_drift.mean().item()
    print(f"  Conservation drift: {mean_cons:.4e}", flush=True)

    # ── Save ──
    results = {
        "ensemble_nrmse": ens_nrmse,
        "published": 0.0059,
        "beat": bool(ens_nrmse < 0.0059),
        "individual": {str(s): hp_log[i]["best_nrmse"] for i, s in enumerate(seeds)},
        "n_params": n_params,
        "conservation_drift": mean_cons,
        "best_idx": bi,
        "median_idx": mi_idx,
        "worst_idx": wi,
        "best_err": float(per_sample[bi]),
        "median_err": float(per_sample[mi_idx]),
        "worst_err": float(per_sample[wi]),
    }
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{OUT}/hyperparameter_log.json", "w") as f:
        json.dump(hp_log, f, indent=2)

    np.savez(
        f"{OUT}/training_histories.npz",
        **{f"tl_s{s}": np.array(all_train_losses[i]) for i, s in enumerate(seeds)},
        **{
            f"tn_ep_s{s}": np.array([d["epoch"] for d in all_test_logs[i]])
            for i, s in enumerate(seeds)
        },
        **{
            f"tn_val_s{s}": np.array([d["nrmse"] for d in all_test_logs[i]])
            for i, s in enumerate(seeds)
        },
    )
    np.savez(
        f"{OUT}/predictions.npz",
        preds=ens_pred.numpy(),
        targets=targets.numpy(),
        per_sample=per_sample.numpy(),
        grid=grid_t.numpy(),
    )
    torch.save(
        {
            "states": [m.state_dict() for m in all_models],
            "seeds": seeds,
            "ens_nrmse": ens_nrmse,
        },
        f"{OUT}/ensemble_ckpt.pt",
    )

    # ── Plots ──
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    grid_np2 = grid_t.squeeze().numpy()

    # Plot 1: Loss curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, s in enumerate(seeds):
        a1.semilogy(all_train_losses[i], alpha=0.7, label=f"seed={s}")
    a1.set(xlabel="Epoch", ylabel="MSE Loss", title="Training Loss")
    a1.legend()
    a1.grid(True, alpha=0.3)
    for i, s in enumerate(seeds):
        ep = [d["epoch"] for d in all_test_logs[i]]
        nv = [d["nrmse"] for d in all_test_logs[i]]
        a2.semilogy(ep, nv, "o-", ms=4, label=f"seed={s}")
    a2.axhline(0.0059, color="r", ls="--", lw=2, label="Published 5.9e-3")
    a2.set(xlabel="Epoch", ylabel="nRMSE", title="Test nRMSE")
    a2.legend()
    a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/training_curves.pdf", bbox_inches="tight")
    plt.close()

    # Plot 2: Pred vs Truth heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for row, (idx, lab) in enumerate([(bi, "Best"), (mi_idx, "Median"), (wi, "Worst")]):
        truth = targets[idx, :, :, 0].numpy()
        pr = ens_pred[idx, :, :, 0].numpy()
        er = np.abs(pr - truth)
        for col, (arr, title, cmap) in enumerate(
            [
                (truth, f"{lab} Truth (err={per_sample[idx]:.4e})", "viridis"),
                (pr, f"{lab} FNO Pred", "viridis"),
                (er, f"{lab} |Error|", "hot"),
            ]
        ):
            kw = dict(aspect="auto", origin="lower", extent=[0, 2, 0, 1])
            if col < 2:
                kw["vmin"], kw["vmax"] = float(truth.min()), float(truth.max())
            im = axes[row, col].imshow(arr, cmap=cmap, **kw)
            axes[row, col].set_title(title)
            plt.colorbar(im, ax=axes[row, col])
        axes[row, 0].set_ylabel("$x$")
    for a in axes[-1]:
        a.set_xlabel("$t$")
    plt.suptitle(
        f"FNO Ensemble — 1D Advection (nRMSE={ens_nrmse:.4e})", fontsize=16, y=1.01
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/pred_vs_truth.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/pred_vs_truth.pdf", bbox_inches="tight")
    plt.close()

    # Plot 3: Line snapshots
    nt_avail = targets.shape[2]
    snaps = [nt_avail // 4, nt_avail // 2, 3 * nt_avail // 4, nt_avail - 1]
    t_vals = np.linspace(0, 2, nt_avail)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, ti in zip(axes, snaps):
        ax.plot(grid_np2, targets[mi_idx, :, ti, 0].numpy(), "k-", lw=2, label="Truth")
        ax.plot(grid_np2, ens_pred[mi_idx, :, ti, 0].numpy(), "r--", lw=2, label="FNO")
        ax.set_title(f"$t={t_vals[ti]:.2f}$")
        ax.set(xlabel="$x$", ylabel="$u$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Median sample (idx={mi_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/line_snapshots.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/line_snapshots.pdf", bbox_inches="tight")
    plt.close()

    # Plot 4: nRMSE distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        per_sample.numpy(), bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(ens_nrmse, color="red", ls="--", lw=2, label=f"Ensemble={ens_nrmse:.4e}")
    ax.axvline(0.0059, color="orange", ls="--", lw=2, label="Published=5.9e-3")
    ax.set(xlabel="Per-sample relL2", ylabel="Count", title="Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/nrmse_dist.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{OUT}/nrmse_dist.pdf", bbox_inches="tight")
    plt.close()

    # Manifest
    mf = {
        "script": "fno_advection_modal.py::train_all",
        "outputs": [
            "results.json",
            "hyperparameter_log.json",
            "training_histories.npz",
            "predictions.npz",
            "ensemble_ckpt.pt",
            "training_curves.png",
            "pred_vs_truth.png",
            "line_snapshots.png",
            "nrmse_dist.png",
        ],
        "ensemble_nrmse": ens_nrmse,
        "beat": bool(ens_nrmse < 0.0059),
    }
    with open(f"{OUT}/_script_manifest.jsonl", "w") as f:
        f.write(json.dumps(mf) + "\n")

    volume.commit()
    print(f"\n{'=' * 60}\nFINAL: Ensemble nRMSE = {ens_nrmse:.4e}")
    print(f"Published: 0.005900\nBeat: {ens_nrmse < 0.0059}\n{'=' * 60}", flush=True)
    return results
