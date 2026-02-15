from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Fixed-policy baselines
# -------------------------
t30 = np.array([
    1.561, 0.873, 0.882, 0.870, 0.874, 0.886, 0.903, 1.252, 1.273, 1.277, 1.275,
    1.269, 1.273, 1.269, 1.267, 1.269, 1.266, 1.269, 1.267, 1.259, 1.260, 1.263,
    1.263, 1.269, 1.267, 1.290, 1.262, 1.261, 1.256, 1.256, 1.264, 1.257, 1.168
], dtype=float)

t15 = np.array([
    1.149, 0.455, 0.464, 0.463, 0.445, 0.454, 0.448, 0.441, 0.448, 0.440, 0.444,
    0.440, 0.465, 0.623, 0.634, 0.636, 0.637, 0.635, 0.634, 0.630, 0.632, 0.634,
    0.631, 0.633, 0.634, 0.635, 0.631, 0.634, 0.631, 0.632, 0.635, 0.639, 0.590
], dtype=float)

# -------------------------
# Adaptive (AIMD) run
# (time per batch + how many frames were processed in that batch)
# -------------------------
t_adapt = np.array([
    1.585, 0.626, 0.673, 0.690, 0.708, 0.754, 0.779, 0.807, 0.825, 0.855, 0.886,
    0.879, 0.883, 0.906, 1.248, 0.889, 0.930, 0.974, 0.679, 0.717, 0.759, 0.801,
    0.846, 0.892, 0.930, 0.974, 0.675, 0.719, 0.765, 0.807, 0.851, 0.887, 0.922
], dtype=float)

keep_adapt = np.array([
    30, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    30, 30, 30, 30, 21, 22, 23, 16, 17, 18, 19,
    20, 21, 22, 23, 16, 17, 18, 19, 20, 21, 22
], dtype=int)

# For batch_000032 you had (kept 22/28). If you want exact, set:
# total_adapt[-1] = 28. We'll keep totals at 30 except last.
total_adapt = np.full_like(keep_adapt, 30, dtype=int)
total_adapt[-1] = 28

# -------------------------
# Parameters
# -------------------------
BATCH_INTERVAL_S = 1.0
out_dir = Path("plots_new")
out_dir.mkdir(exist_ok=True)

k30 = np.arange(len(t30))
k15 = np.arange(len(t15))
kad = np.arange(len(t_adapt))

# -------------------------
# Helpers
# -------------------------
def simulate_lag(proc_times, interval_s=1.0):
    lag = np.zeros_like(proc_times)
    cur = 0.0
    for i, s in enumerate(proc_times):
        cur = max(0.0, cur + float(s) - float(interval_s))
        lag[i] = cur
    return lag

def drift_from_lag(lag):
    d = np.zeros_like(lag)
    d[0] = lag[0]
    d[1:] = lag[1:] - lag[:-1]
    return d

def moving_avg(x, w=5):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

# -------------------------
# Derived series
# -------------------------
lag30 = simulate_lag(t30, BATCH_INTERVAL_S)
lag15 = simulate_lag(t15, BATCH_INTERVAL_S)
lagad = simulate_lag(t_adapt, BATCH_INTERVAL_S)

drift30 = drift_from_lag(lag30)
drift15 = drift_from_lag(lag15)
driftad = drift_from_lag(lagad)

rtf30 = t30 / BATCH_INTERVAL_S
rtf15 = t15 / BATCH_INTERVAL_S
rtfad = t_adapt / BATCH_INTERVAL_S

fps30 = 30 / t30
fps15 = 15 / t15
fpsad = keep_adapt / t_adapt  # effective processed FPS for adaptive

# -------------------------
# 1) Processing time vs batch index
# -------------------------
plt.figure()
plt.plot(k30, t30, marker="o", label="Fixed: 30 frames")
plt.plot(k15, t15, marker="o", label="Fixed: 15 frames")
plt.plot(kad, t_adapt, marker="o", label="Adaptive (AIMD)")
plt.xlabel("Batch index")
plt.ylabel("Processing time (s)")
plt.title("Processing time per batch")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "1_processing_time.png", dpi=200)

# -------------------------
# 2) RTF vs batch index
# -------------------------
plt.figure()
plt.plot(k30, rtf30, marker="o", label="Fixed: 30 frames")
plt.plot(k15, rtf15, marker="o", label="Fixed: 15 frames")
plt.plot(kad, rtfad, marker="o", label="Adaptive (AIMD)")
plt.axhline(1.0, linestyle="--", label="RTF=1 (real-time boundary)")
plt.xlabel("Batch index")
plt.ylabel("RTF (processing_time / 1s)")
plt.title("Real-Time Factor (RTF)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "2_rtf.png", dpi=200)

# -------------------------
# 3) Simulated lag accumulation
# -------------------------
plt.figure()
plt.plot(k30, lag30, marker="o", label="Fixed: 30 frames")
plt.plot(k15, lag15, marker="o", label="Fixed: 15 frames")
plt.plot(kad, lagad, marker="o", label="Adaptive (AIMD)")
plt.xlabel("Batch index")
plt.ylabel("Lag (s)")
plt.title("Simulated lag accumulation (queue model)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "3_simulated_lag.png", dpi=200)

# -------------------------
# 4) Drift per batch
# -------------------------
plt.figure()
plt.plot(k30, drift30, marker="o", label="Fixed: 30 frames")
plt.plot(k15, drift15, marker="o", label="Fixed: 15 frames")
plt.plot(kad, driftad, marker="o", label="Adaptive (AIMD)")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Batch index")
plt.ylabel("Lag drift (s per batch)")
plt.title("Lag drift per batch")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "4_lag_drift.png", dpi=200)

# -------------------------
# 5) Throughput FPS vs batch index
# (Fixed: 30/t, 15/t; Adaptive: kept/t)
# -------------------------
plt.figure()
plt.plot(k30, fps30, marker="o", label=f"Fixed 30: mean {np.mean(fps30):.1f} FPS")
plt.plot(k15, fps15, marker="o", label=f"Fixed 15: mean {np.mean(fps15):.1f} FPS")
plt.plot(kad, fpsad, marker="o", label=f"Adaptive: mean {np.mean(fpsad):.1f} FPS")
plt.xlabel("Batch index")
plt.ylabel("Throughput (processed frames / second)")
plt.title("Inference throughput (FPS)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "5_throughput_fps.png", dpi=200)

# ==========================================================
# EXTRA PLOTS (highly recommended)
# ==========================================================

# 6) Adaptive keep_n over time (control signal)
plt.figure()
plt.plot(kad, keep_adapt, marker="o")
plt.xlabel("Batch index")
plt.ylabel("Frames processed in batch (keep_n)")
plt.title("Adaptive controller output (keep_n)")
plt.tight_layout()
plt.savefig(out_dir / "6_adaptive_keep_n.png", dpi=200)

# 7) Processing time vs keep_n scatter (should look ~linear)
plt.figure()
plt.plot(keep_adapt, t_adapt, "o")
plt.xlabel("Frames processed in batch (keep_n)")
plt.ylabel("Processing time (s)")
plt.title("Adaptive run: processing time vs keep_n")
plt.tight_layout()
plt.savefig(out_dir / "7_time_vs_keep_scatter.png", dpi=200)

# 8) How close to the 1s budget the adaptive run stays (error to target)
target = 0.95
err = t_adapt - target
plt.figure()
plt.plot(kad, err, marker="o")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Batch index")
plt.ylabel("t - target (s)")
plt.title("Adaptive run: budget tracking error (target=0.95s)")
plt.tight_layout()
plt.savefig(out_dir / "8_budget_tracking_error.png", dpi=200)

# 9) Smoothed processing time (helps show regime changes)
w = 5
t30_ma = moving_avg(t30, w)
t15_ma = moving_avg(t15, w)
tad_ma = moving_avg(t_adapt, w)
plt.figure()
plt.plot(np.arange(len(t30_ma)), t30_ma, label=f"Fixed 30 (MA{w})")
plt.plot(np.arange(len(t15_ma)), t15_ma, label=f"Fixed 15 (MA{w})")
plt.plot(np.arange(len(tad_ma)), tad_ma, label=f"Adaptive (MA{w})")
plt.xlabel("Batch index (moving-average)")
plt.ylabel("Processing time (s)")
plt.title("Smoothed processing time (moving average)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "9_smoothed_processing_time.png", dpi=200)

plt.show()

print("Saved plots to:", out_dir.resolve())