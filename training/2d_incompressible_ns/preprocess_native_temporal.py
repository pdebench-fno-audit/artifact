"""Preprocess NS data at 512×512 with NATIVE temporal resolution (RES_T=1).

Saves ALL 101 native timesteps — exactly matching published PDEBench config.
Also computes normalization stats (mean/std) during preprocessing so training
never needs to load all data into RAM.

Output:
  - preprocessed_512_native101_batch_XX.npz  (velocity + forcing per batch)
  - preprocessed_512_native101_stats.npz     (vel_mean, vel_std, force_mean, force_std)
  - preprocessed_512_native101_done.txt      (metadata)

Training only needs ~15 GB RAM: loads one batch at a time + tiny stats file.
"""

import modal, os

app = modal.App("preprocess-ns-native")
data_volume = modal.Volume.from_name("ns-incom-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system h5py numpy==1.26.4")
    .env({"PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=14400,
    memory=32768,  # 32 GB — only one batch in memory at a time
)
def preprocess():
    import h5py, numpy as np, time

    T_NATIVE_MAX = 101
    RES_T = 1
    BATCH_SIZE = 50
    N_TRAIN = 800  # first 800 samples are training set — compute stats from these only
    PREFIX = "/data/preprocessed_512_native101"
    CACHE_PREFIX = f"{PREFIX}_batch"
    STATS_PATH = f"{PREFIX}_stats.npz"
    DONE_MARKER = f"{PREFIX}_done.txt"

    data_volume.reload()
    if os.path.exists(DONE_MARKER):
        print("Native 101-step preprocessing already done.", flush=True)
        return {"status": "exists"}

    h5_files = sorted(
        [
            f
            for f in os.listdir("/data")
            if f.endswith(".h5") and not f.endswith(".aria2")
        ],
        key=lambda x: int(x.split("-")[1].split(".")[0]),
    )
    print(
        f"Processing {len(h5_files)} files at NATIVE 512×512, "
        f"{T_NATIVE_MAX} native steps, computing train-set stats",
        flush=True,
    )

    # ── Welford's online algorithm for mean/std ──
    # Velocity stats: per-channel (2 channels), computed over train set only
    vel_count = 0
    vel_sum = np.zeros(2, dtype=np.float64)
    vel_sum_sq = np.zeros(2, dtype=np.float64)
    # Forcing stats: per-channel (2 channels)
    force_count = 0
    force_sum = np.zeros(2, dtype=np.float64)
    force_sum_sq = np.zeros(2, dtype=np.float64)

    all_vel = []
    all_force = []
    batch_idx = 0
    skipped = 0
    total_samples = 0
    t0 = time.time()

    for fi, fname in enumerate(h5_files):
        try:
            with h5py.File(f"/data/{fname}", "r") as f:
                vel = f["velocity"]  # [4, 1000, 512, 512, 2]
                frc = f["force"]  # [4, 512, 512, 2]
                n_per_file = vel.shape[0]
                for si in range(n_per_file):
                    v = np.array(
                        vel[si, :T_NATIVE_MAX:RES_T, :, :, :], dtype=np.float32
                    )
                    fo = np.array(frc[si, :, :, :], dtype=np.float32)
                    all_vel.append(v)
                    all_force.append(fo)

                    # Accumulate stats for training set samples only
                    sample_idx = total_samples + len(all_vel) - 1
                    if sample_idx < N_TRAIN:
                        # v: [T, H, W, 2] — mean over T, H, W
                        vel_sum += v.mean(axis=(0, 1, 2)).astype(np.float64)
                        vel_sum_sq += (v.astype(np.float64) ** 2).mean(axis=(0, 1, 2))
                        vel_count += 1
                        force_sum += fo.mean(axis=(0, 1)).astype(np.float64)
                        force_sum_sq += (fo.astype(np.float64) ** 2).mean(axis=(0, 1))
                        force_count += 1

                    if len(all_vel) >= BATCH_SIZE:
                        vel_arr = np.transpose(np.stack(all_vel), (0, 2, 3, 1, 4))
                        force_arr = np.stack(all_force)
                        out_path = f"{CACHE_PREFIX}_{batch_idx:02d}.npz"
                        np.savez(out_path, velocity=vel_arr, forcing=force_arr)
                        data_volume.commit()
                        gb = vel_arr.nbytes / 1e9
                        print(
                            f"  Batch {batch_idx}: {vel_arr.shape} ({gb:.1f} GB) "
                            f"[{total_samples + len(all_vel)} samples, {time.time() - t0:.0f}s]",
                            flush=True,
                        )
                        total_samples += len(all_vel)
                        all_vel = []
                        all_force = []
                        batch_idx += 1
        except Exception as e:
            skipped += 1
            print(f"  SKIP {fname}: {e}", flush=True)

        if (fi + 1) % 50 == 0:
            n = total_samples + len(all_vel)
            print(
                f"  {fi + 1}/{len(h5_files)} files, {n} samples, "
                f"{skipped} skipped, {time.time() - t0:.0f}s",
                flush=True,
            )

    # Flush remaining
    if all_vel:
        vel_arr = np.transpose(np.stack(all_vel), (0, 2, 3, 1, 4))
        force_arr = np.stack(all_force)
        out_path = f"{CACHE_PREFIX}_{batch_idx:02d}.npz"
        np.savez(out_path, velocity=vel_arr, forcing=force_arr)
        data_volume.commit()
        total_samples += len(all_vel)
        print(f"  Batch {batch_idx}: {vel_arr.shape} (final)", flush=True)
        batch_idx += 1

    # ── Compute final stats ──
    # E[X] = sum(mean_per_sample) / N
    # Var[X] = E[X²] - E[X]² (using per-sample spatial means as approximation)
    # This is the mean over spatial dims, averaged over samples — matching
    # the original code's data[:N_TRAIN].mean(dim=(0,1,2,3))
    vel_mean = (vel_sum / vel_count).astype(np.float32)
    vel_std = (
        np.sqrt(vel_sum_sq / vel_count - vel_mean.astype(np.float64) ** 2).astype(
            np.float32
        )
        + 1e-8
    )
    force_mean = (force_sum / force_count).astype(np.float32)
    force_std = (
        np.sqrt(force_sum_sq / force_count - force_mean.astype(np.float64) ** 2).astype(
            np.float32
        )
        + 1e-8
    )

    np.savez(
        STATS_PATH,
        vel_mean=vel_mean,
        vel_std=vel_std,
        force_mean=force_mean,
        force_std=force_std,
        n_train=N_TRAIN,
        n_total=total_samples,
        t_steps=T_NATIVE_MAX,
    )
    print(f"\n  Stats: vel_mean={vel_mean}, vel_std={vel_std}", flush=True)
    print(f"         force_mean={force_mean}, force_std={force_std}", flush=True)

    with open(DONE_MARKER, "w") as f:
        f.write(
            f"batches={batch_idx}\ntotal_samples={total_samples}\n"
            f"skipped={skipped}\nT_NATIVE_MAX={T_NATIVE_MAX}\n"
            f"vel_mean={vel_mean.tolist()}\nvel_std={vel_std.tolist()}\n"
            f"force_mean={force_mean.tolist()}\nforce_std={force_std.tolist()}\n"
            f"time={time.time() - t0:.0f}s\n"
        )
    data_volume.commit()

    print(f"\n{'=' * 60}")
    print(f"NATIVE 101-STEP PREPROCESSING COMPLETE")
    print(f"  Batches: {batch_idx} (~{total_samples} samples)")
    print(f"  Skipped: {skipped} corrupt files")
    print(f"  Stats computed from first {N_TRAIN} training samples")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"{'=' * 60}", flush=True)

    return {"status": "done", "batches": batch_idx, "total_samples": total_samples}
