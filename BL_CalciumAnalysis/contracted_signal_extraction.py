"""Contracted calcium signal extraction with bleaching correction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass(frozen=True)
class ContractConfig:
    """Configuration for the contracted preprocessing pipeline."""

    z_threshold: float = 2.0
    spike_baseline_window_seconds: float = 5.0
    spike_expand_seconds: float = 2.5
    mask_expand_min_frames: int = 0
    lowess_window_frames: int = 100
    lowess_it: int = 3
    fps: float = 1.0
    use_motion_corrected: bool = True
    f0_window_seconds: float = 50.0
    f0_max_fraction: float = 0.4
    f0_activity_fraction: float = 0.3
    f0_low_percentile: float = 10.0
    f0_high_percentile: float = 10.0


@dataclass(frozen=True)
class ContractAnalysisOutputs:
    analysis_dir: Path
    traces_csv: Path
    dff_csv: Path
    traces_plot: Path
    dff_plot: Path
    bleaching_baseline_csv: Path
    bleaching_fit_json: Path
    corrected_traces_csv: Path
    slow_baseline_csv: Path
    env_report_json: Path
    qc_pdf: Path
    spike_debug_csv: Path
    sliding_f0_csv: Path
    sliding_pct_csv: Path
    sliding_dff_csv: Path
    roi_peaks_csv: Path
    roi_peak_counts_csv: Path
    roi1_f0_debug_csv: Path
    roi1_f0_debug_plot: Path
    roi1_peaks_csv: Path


@dataclass(frozen=True)
class SpikeDetectionResult:
    mask: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    zscore: np.ndarray
    window_frames: int
    expand_frames: int


def validate_environment(env_file: Path | None = None) -> dict[str, Any]:
    """Validate required numerical capabilities before any data modification."""
    import numpy as np_module
    import scipy as scipy_module
    import statsmodels as statsmodels_module
    from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_fn

    info: dict[str, Any] = {
        "numpy": np_module.__version__,
        "scipy": scipy_module.__version__,
        "statsmodels": statsmodels_module.__version__,
    }

    if "it" not in lowess_fn.__code__.co_varnames:
        raise RuntimeError("LOWESS implementation does not support robust iteration.")

    if env_file and env_file.exists():
        info["environment_file"] = str(env_file)
        info["environment_hash_sha256"] = hashlib.sha256(env_file.read_bytes()).hexdigest()

    return info


def _ensure_movie_3d(movie: np.ndarray, label: str) -> np.ndarray:
    movie = np.asarray(movie)
    if movie.ndim == 4:
        if movie.shape[-1] in {1, 3}:
            movie = movie.mean(axis=-1)
        else:
            raise ValueError(
                f"{label} movie has unsupported channel dimension {movie.shape[-1]}."
            )
    elif movie.ndim == 3 and movie.shape[-1] in {1, 3} and movie.shape[-1] <= 4:
        movie = movie.mean(axis=-1)
    if movie.ndim != 3:
        raise ValueError(f"{label} movie must be 3D (T, Y, X), got shape {movie.shape}.")
    return movie


def _load_roi_labels(roi_path: Path, movie_shape: tuple[int, int, int]) -> np.ndarray:
    roi_data = tiff.imread(roi_path)
    if roi_data.ndim == 2:
        roi_data = np.broadcast_to(roi_data, movie_shape)
    elif roi_data.ndim == 3:
        t_roi, y_roi, x_roi = roi_data.shape
        t_mov, y_mov, x_mov = movie_shape
        if (y_roi != y_mov) or (x_roi != x_mov):
            raise ValueError("ROI (Y,X) does NOT match movie size.")
        if t_roi != t_mov:
            roi_data = np.broadcast_to(roi_data[0], movie_shape)
    else:
        raise ValueError("ROI mask must be 2D (Y, X) or 3D (T, Y, X).")
    return roi_data


def _base_stem_from_raw(raw_path: Path) -> str:
    stem = raw_path.stem
    if stem.endswith("_raw"):
        return stem[: -len("_raw")]
    return stem


def extract_static_traces(
    movie: np.ndarray,
    static_labels: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Compute mean intensity traces for each static ROI."""
    roi_ids = np.unique(static_labels)
    roi_ids = roi_ids[roi_ids != 0]
    traces: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for rid in roi_ids:
        mask = static_labels == rid
        counts[int(rid)] = int(mask.sum())
        traces[int(rid)] = movie[:, mask].mean(axis=1)
    if not traces:
        raise ValueError("No ROI IDs found in ROI labels.")
    return traces, counts


def detect_rough_spikes(
    trace: np.ndarray,
    fps: float,
    z_threshold: float = 2.0,
    baseline_window_s: float = 5.0,
    expand_window_s: float = 2.5,
    min_expand_frames: int = 0,
) -> SpikeDetectionResult:
    """Return permissive spike frames using local robust stats."""
    trace = np.asarray(trace, dtype=float)
    n = trace.shape[0]
    if n == 0 or fps <= 0:
        empty = np.zeros_like(trace, dtype=bool)
        return SpikeDetectionResult(
            mask=empty,
            median=np.zeros_like(trace, dtype=float),
            scale=np.ones_like(trace, dtype=float),
            zscore=np.zeros_like(trace, dtype=float),
            window_frames=0,
            expand_frames=0,
        )

    # Rolling robust baseline (median) and scale (MAD) to handle drift/steps.
    win_frames = max(3, int(round(baseline_window_s * fps)))
    if win_frames % 2 == 0:
        win_frames += 1
    min_periods = max(1, win_frames // 2)

    ser = pd.Series(trace)
    rolling_med = ser.rolling(win_frames, center=True, min_periods=min_periods).median()
    rolling_mad = ser.rolling(win_frames, center=True, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=False
    )

    med = rolling_med.to_numpy()
    mad = rolling_mad.to_numpy()
    scale = 1.4826 * mad  # MAD to std
    scale = np.where(~np.isfinite(scale) | (scale < 1e-9), 1e-9, scale)

    zscores = (trace - med) / scale
    spikes = (zscores > z_threshold) & np.isfinite(zscores)

    expand_frames = int(np.ceil(expand_window_s * fps))
    expand_frames = max(expand_frames, int(min_expand_frames))
    if expand_frames > 0:
        kernel = np.ones(2 * expand_frames + 1, dtype=int)
        spikes = np.convolve(spikes.astype(int), kernel, mode="same") > 0
    return SpikeDetectionResult(
        mask=spikes,
        median=med,
        scale=scale,
        zscore=zscores,
        window_frames=win_frames,
        expand_frames=expand_frames,
    )


def _bi_exponential(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, c: float) -> np.ndarray:
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def fit_bleaching_baseline(
    t: np.ndarray,
    trace: np.ndarray,
    silent_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a bi-exponential bleaching model on silent frames."""
    t = np.asarray(t, dtype=float)
    trace = np.asarray(trace, dtype=float)
    silent_mask = np.asarray(silent_mask, dtype=bool)
    if t.ndim != 1 or trace.ndim != 1:
        raise ValueError("t and trace must be 1D arrays.")
    if t.shape[0] != trace.shape[0]:
        raise ValueError("t and trace length mismatch.")

    valid = np.isfinite(trace) & np.isfinite(t)
    fit_mask = silent_mask & valid
    if fit_mask.sum() < 5:
        raise ValueError("Not enough silent frames for bleaching fit.")

    t_fit = t[fit_mask]
    y_fit = trace[fit_mask]
    t_span = float(np.max(t_fit) - np.min(t_fit))
    if t_span <= 0:
        raise ValueError("Cannot fit bleaching baseline with zero time span.")

    y_min = float(np.min(y_fit))
    y_max = float(np.max(y_fit))
    amp = max(y_max - y_min, np.finfo(float).eps)
    tau_fast = max(t_span / 5.0, 1e-3)
    tau_slow = max(t_span / 1.5, 1e-3)
    p0 = [amp * 0.6, tau_fast, amp * 0.4, tau_slow, y_min]
    bounds = ([0.0, 1e-6, 0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf, np.inf, np.inf])

    params, _ = curve_fit(
        _bi_exponential,
        t_fit,
        y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    a1, tau1, a2, tau2, c = params
    baseline = _bi_exponential(t, a1, tau1, a2, tau2, c)
    baseline = np.maximum(baseline, np.finfo(float).eps)
    return baseline, {
        "a1": float(a1),
        "tau1": float(tau1),
        "a2": float(a2),
        "tau2": float(tau2),
        "c": float(c),
    }


def lowess_baseline(
    trace: np.ndarray,
    window_frames: int = 100,
    robust_it: int = 3,
) -> np.ndarray:
    """Compute a robust LOWESS baseline for a single trace."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    trace = np.asarray(trace, dtype=float)
    n = trace.shape[0]
    if n < 2:
        return trace.copy()
    frac = min(1.0, max(window_frames / float(n), 0.01))
    x = np.arange(n, dtype=float)
    finite = np.isfinite(trace)
    if finite.all():
        return lowess(trace, x, frac=frac, it=robust_it, return_sorted=False)
    x_fit = x[finite]
    y_fit = trace[finite]
    if x_fit.size < 2:
        return np.full_like(trace, np.nanmean(trace))
    baseline_fit = lowess(y_fit, x_fit, frac=frac, it=robust_it, return_sorted=False)
    return np.interp(x, x_fit, baseline_fit)


def compute_dff_from_baseline(trace: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    denom = np.nanmedian(baseline)
    if not np.isfinite(denom) or denom <= 0:
        denom = np.finfo(float).eps
    return (trace - baseline) / denom


def compute_sliding_f0_adaptive(
    trace: np.ndarray,
    fps: float,
    target_window_s: float = 50.0,
    max_fraction: float = 0.4,
    activity_fraction: float = 0.3,
    low_percentile: float = 10.0,
    high_percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute adaptive sliding F0 using a time-based window and activity-aware percentile."""
    trace = np.asarray(trace, dtype=float)
    n = trace.shape[0]
    if n == 0 or fps <= 0:
        return np.zeros_like(trace), np.zeros_like(trace), 0

    target_frames = int(round(target_window_s * fps))
    max_frames_cap = int(np.floor(n * max_fraction))
    if max_frames_cap >= 3:
        target_frames = min(target_frames, max_frames_cap)
    win_frames = max(5, target_frames)
    if win_frames % 2 == 0:
        win_frames += 1
    half = win_frames // 2

    f0 = np.zeros(n, dtype=float)
    pct_used = np.zeros(n, dtype=float)
    eps = np.finfo(float).eps

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = trace[start:end]
        med = np.nanmedian(window)
        mad = np.nanmedian(np.abs(window - med))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(window)
        scale = scale if np.isfinite(scale) and scale > 0 else eps
        high_thresh = med + 0.5 * scale
        activity = np.mean(window > high_thresh) if window.size else 0.0
        frac = np.clip(activity / max(activity_fraction, eps), 0.0, 1.0)
        pct = low_percentile + (high_percentile - low_percentile) * frac
        f0[i] = np.nanpercentile(window, pct)
        pct_used[i] = pct

    f0 = np.where(np.isfinite(f0) & (f0 > 0), f0, eps)
    return f0, pct_used, win_frames


def _plot_traces(traces: dict[int, np.ndarray], out_path: Path, title: str) -> Path:
    plt.figure(figsize=(6, 6))
    offset = 0.0
    all_vals = np.concatenate([np.asarray(tr) for tr in traces.values()])
    spacing = float(np.nanmax(all_vals)) * 0.05 if all_vals.size else 1.0
    for rid in sorted(traces.keys()):
        plt.plot(traces[rid] + offset, label=f"ROI {rid}")
        offset += spacing
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Mean Intensity (offset)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _plot_dff_traces(dff_traces: dict[int, np.ndarray], out_path: Path, title: str) -> Path:
    plt.figure(figsize=(12, 6))
    for rid in sorted(dff_traces.keys()):
        plt.plot(dff_traces[rid], label=f"ROI {rid}")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("dF/F")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_qc_pdf(
    out_path: Path,
    roi_ids: list[int],
    raw_traces: dict[int, np.ndarray],
    sliding_f0: dict[int, np.ndarray],
    sliding_pct: dict[int, np.ndarray],
    sliding_dff: dict[int, np.ndarray],
    roi_peak_thresholds: dict[int, float],
    roi_peaks_by_roi: dict[int, list[tuple[int, float]]],
) -> Path:
    """Save a multi-page PDF with sliding-F0 panels only (raw+F0, percentile, dF/F+peaks)."""
    with PdfPages(out_path) as pdf:
        for rid in roi_ids:
            raw = np.asarray(raw_traces[rid])
            f0 = np.asarray(sliding_f0[rid])
            pct = np.asarray(sliding_pct[rid])
            dff = np.asarray(sliding_dff[rid])
            thresh = roi_peak_thresholds.get(rid, 0.0)
            peaks_idx = [p[0] for p in roi_peaks_by_roi.get(rid, [])]
            peaks_vals = [p[1] for p in roi_peaks_by_roi.get(rid, [])]
            frames = np.arange(raw.shape[0])

            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

            axes[0].plot(frames, raw, color="black", linewidth=1.0, label="Trace")
            axes[0].plot(frames, f0, color="darkorange", linewidth=1.0, label="Adaptive F0")
            axes[0].set_ylabel("Intensity")
            axes[0].set_title(f"ROI {rid}: Raw and Adaptive F0")
            axes[0].legend(loc="upper right", fontsize=8)

            axes[1].plot(frames, pct, color="steelblue", linewidth=0.9)
            axes[1].set_ylabel("Percentile used")

            axes[2].plot(frames, dff, color="purple", linewidth=0.9, label="dF/F (sliding F0)")
            axes[2].axhline(thresh, color="darkorange", linestyle="--", linewidth=1.0, label="Peak threshold (z=2)")
            if peaks_idx:
                axes[2].scatter(peaks_idx, peaks_vals, color="tomato", s=12, zorder=3, label="Peaks")
            axes[2].set_ylabel("dF/F")
            axes[2].set_xlabel("Frame")
            axes[2].legend(loc="upper right", fontsize=8)

            for ax in axes:
                ax.grid(alpha=0.2, linewidth=0.5)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return out_path


def process_contract_analysis(
    manifest_path: Path,
    roi_path: Path,
    output_dir: Path | None = None,
    config: ContractConfig | None = None,
) -> ContractAnalysisOutputs:
    """Run the contracted bleaching-corrected pipeline and write outputs."""
    cfg = config or ContractConfig()
    repo_root = Path(__file__).resolve().parents[1]
    env_info = validate_environment(env_file=repo_root / "env" / "environment.yml")

    payload = json.loads(manifest_path.read_text())
    paths = payload.get("paths", {})
    raw_tiff = Path(paths["raw_tiff"]) if paths.get("raw_tiff") else None
    mc_tiff = Path(paths["motion_corrected_tiff"]) if paths.get("motion_corrected_tiff") else None
    if cfg.use_motion_corrected:
        movie_path = mc_tiff
    else:
        movie_path = raw_tiff

    if movie_path is None:
        raise FileNotFoundError("Manifest missing movie path for requested source.")
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie TIFF not found: {movie_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {roi_path}")

    movie = _ensure_movie_3d(tiff.imread(movie_path), "Movie").astype(float)
    roi_data = _load_roi_labels(roi_path, movie.shape)
    static_labels = roi_data.max(axis=0)

    traces, counts = extract_static_traces(movie, static_labels)
    roi_ids = sorted(traces.keys())
    trace_stack = np.stack([traces[rid] for rid in roi_ids], axis=0)
    weights = np.array([counts[rid] for rid in roi_ids], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("ROI pixel counts sum to zero.")
    global_trace = np.average(trace_stack, axis=0, weights=weights)

    # Spike masks kept for diagnostics; bleaching correction is disabled.
    roi_spike_masks = {
        rid: detect_rough_spikes(
            traces[rid],
            fps=cfg.fps,
            z_threshold=cfg.z_threshold,
            baseline_window_s=cfg.spike_baseline_window_seconds,
            expand_window_s=cfg.spike_expand_seconds,
            min_expand_frames=cfg.mask_expand_min_frames,
        )
        for rid in roi_ids
    }
    spikes_union = np.logical_or.reduce(
        np.stack([roi_spike_masks[rid].mask for rid in roi_ids], axis=0)
    )

    # Bleaching disabled: use raw traces directly.
    bleaching_baseline = np.ones_like(global_trace, dtype=float)
    fit_params: dict[str, Any] = {"bleaching_correction": "disabled"}
    corrected_traces = traces
    slow_baselines = {
        rid: lowess_baseline(
            corrected_traces[rid],
            window_frames=cfg.lowess_window_frames,
            robust_it=cfg.lowess_it,
        )
        for rid in corrected_traces.keys()
    }
    dff_traces = {
        rid: compute_dff_from_baseline(corrected_traces[rid], slow_baselines[rid])
        for rid in corrected_traces.keys()
    }

    # Sliding F0 + dF/F + peak detection (per ROI, no bleaching baseline).
    sliding_f0: dict[int, np.ndarray] = {}
    sliding_pct: dict[int, np.ndarray] = {}
    sliding_dff: dict[int, np.ndarray] = {}
    roi_peak_thresholds: dict[int, float] = {}
    roi_peaks_by_roi: dict[int, list[tuple[int, float]]] = {}
    peak_rows: list[dict[str, float | int]] = []
    peak_counts_rows: list[dict[str, float | int]] = []
    win_frames_used: int = 0
    for rid in roi_ids:
        f0, pct_used, win_frames = compute_sliding_f0_adaptive(
            traces[rid],
            fps=cfg.fps,
            target_window_s=cfg.f0_window_seconds,
            max_fraction=cfg.f0_max_fraction,
            activity_fraction=cfg.f0_activity_fraction,
            low_percentile=cfg.f0_low_percentile,
            high_percentile=cfg.f0_high_percentile,
        )
        win_frames_used = win_frames or win_frames_used
        dff_sliding = (traces[rid] - f0) / f0
        mean = float(np.nanmean(dff_sliding))
        std = float(np.nanstd(dff_sliding))
        thresh = mean + 2.0 * std if np.isfinite(std) and std > 0 else mean
        peaks_idx, props = find_peaks(dff_sliding, height=thresh)
        sliding_f0[rid] = f0
        sliding_pct[rid] = pct_used
        sliding_dff[rid] = dff_sliding
        roi_peak_thresholds[rid] = thresh
        roi_peaks_by_roi[rid] = [(int(idx), float(dff_sliding[idx])) for idx in peaks_idx]
        for idx in peaks_idx:
            val = dff_sliding[idx]
            peak_rows.append(
                {
                    "roi": rid,
                    "frame": int(idx),
                    "time_seconds": float(idx) / float(cfg.fps),
                    "dff": float(val),
                    "dff_zscore": (val - mean) / (std + 1e-9),
                    "threshold": thresh,
                }
            )
        peak_counts_rows.append(
            {
                "roi": rid,
                "peak_count": int(len(peaks_idx)),
                "duration_seconds": float(len(traces[rid])) / float(cfg.fps),
                "peak_rate_hz": float(len(peaks_idx)) / max(
                    float(len(traces[rid])) / float(cfg.fps), 1e-9
                ),
                "z_threshold_used": thresh,
                "window_frames": win_frames,
            }
        )

    analysis_dir = output_dir or (manifest_path.parent / "roi_analysis_contract")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_stem = _base_stem_from_raw(raw_tiff or movie_path)
    traces_csv = analysis_dir / f"{base_stem}_roi_traces.csv"
    dff_csv = analysis_dir / f"{base_stem}_roi_dff.csv"
    corrected_traces_csv = analysis_dir / f"{base_stem}_roi_bleachcorr_traces.csv"
    slow_baseline_csv = analysis_dir / f"{base_stem}_roi_slow_baseline.csv"
    bleaching_baseline_csv = analysis_dir / f"{base_stem}_bleaching_baseline.csv"
    bleaching_fit_json = analysis_dir / f"{base_stem}_bleaching_fit.json"
    traces_plot = analysis_dir / f"{base_stem}_roi_traces.png"
    dff_plot = analysis_dir / f"{base_stem}_roi_dff.png"
    env_report_json = analysis_dir / "contract_environment.json"
    sliding_f0_csv = analysis_dir / f"{base_stem}_roi_sliding_f0.csv"
    sliding_pct_csv = analysis_dir / f"{base_stem}_roi_sliding_pct.csv"
    sliding_dff_csv = analysis_dir / f"{base_stem}_roi_sliding_dff.csv"
    roi_peaks_csv = analysis_dir / f"{base_stem}_roi_peaks.csv"
    roi_peak_counts_csv = analysis_dir / f"{base_stem}_roi_peak_counts.csv"
    roi1_f0_debug_csv = analysis_dir / f"{base_stem}_roi1_sliding_f0_debug.csv"
    roi1_f0_debug_plot = analysis_dir / f"{base_stem}_roi1_sliding_f0_debug.png"
    roi1_peaks_csv = analysis_dir / f"{base_stem}_roi1_peaks.csv"

    pd.DataFrame({rid: traces[rid] for rid in roi_ids}).to_csv(
        traces_csv, index_label="frame"
    )
    pd.DataFrame({rid: corrected_traces[rid] for rid in roi_ids}).to_csv(
        corrected_traces_csv, index_label="frame"
    )
    pd.DataFrame({rid: slow_baselines[rid] for rid in roi_ids}).to_csv(
        slow_baseline_csv, index_label="frame"
    )
    pd.DataFrame({rid: dff_traces[rid] for rid in roi_ids}).to_csv(
        dff_csv, index_label="frame"
    )
    pd.DataFrame(
        {
            "frame": np.arange(global_trace.shape[0], dtype=int),
            "global_trace": global_trace,
            "bleaching_baseline": bleaching_baseline,
            "spike_mask": spikes_union.astype(int),
        }
    ).to_csv(bleaching_baseline_csv, index=False)

    pd.DataFrame({rid: sliding_f0[rid] for rid in roi_ids}).to_csv(
        sliding_f0_csv, index_label="frame"
    )
    pd.DataFrame({rid: sliding_pct[rid] for rid in roi_ids}).to_csv(
        sliding_pct_csv, index_label="frame"
    )
    pd.DataFrame({rid: sliding_dff[rid] for rid in roi_ids}).to_csv(
        sliding_dff_csv, index_label="frame"
    )
    pd.DataFrame(peak_rows).to_csv(roi_peaks_csv, index=False)
    pd.DataFrame(peak_counts_rows).to_csv(roi_peak_counts_csv, index=False)

    bleaching_fit_json.write_text(json.dumps(fit_params, indent=2))
    env_report_json.write_text(json.dumps(env_info, indent=2))

    _plot_traces(traces, traces_plot, "ROI Fluorescence Traces (Raw)")
    _plot_dff_traces(dff_traces, dff_plot, "ROI dF/F Traces")

    qc_pdf = analysis_dir / f"{base_stem}_contract_qc.pdf"
    roi_ids_for_qc = roi_ids
    save_qc_pdf(
        out_path=qc_pdf,
        roi_ids=roi_ids_for_qc,
        raw_traces=traces,
        sliding_f0=sliding_f0,
        sliding_pct=sliding_pct,
        sliding_dff=sliding_dff,
        roi_peak_thresholds=roi_peak_thresholds,
        roi_peaks_by_roi=roi_peaks_by_roi,
    )

    spike_debug_csv = analysis_dir / f"{base_stem}_roi_spike_debug.csv"
    if roi_ids_for_qc:
        rid0 = roi_ids_for_qc[0]
        res0 = roi_spike_masks[rid0]
        pd.DataFrame(
            {
                "frame": np.arange(traces[rid0].shape[0], dtype=int),
                "raw": traces[rid0],
                "rolling_median": res0.median,
                "rolling_scale": res0.scale,
                "zscore": res0.zscore,
                "spike_mask_expanded": res0.mask.astype(int),
                "spike_mask_union": spikes_union.astype(int),
            }
        ).to_csv(spike_debug_csv, index=False)
    else:
        spike_debug_csv.write_text("No ROIs found; nothing to debug.")

    # ROI1 sliding F0 + peaks QC (reuses computed arrays, no recompute).
    if roi_ids_for_qc:
        rid0 = roi_ids_for_qc[0]
        pd.DataFrame(
            {
                "frame": np.arange(traces[rid0].shape[0], dtype=int),
                "trace": traces[rid0],
                "f0": sliding_f0[rid0],
                "percentile_used": sliding_pct[rid0],
                "dff": sliding_dff[rid0],
            }
        ).to_csv(roi1_f0_debug_csv, index=False)

        roi1_peaks = [row for row in peak_rows if row["roi"] == rid0]
        pd.DataFrame(roi1_peaks).to_csv(roi1_peaks_csv, index=False)

        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(traces[rid0], color="black", label=f"Trace (ROI {rid0})")
        ax1.plot(sliding_f0[rid0], color="darkorange", label="Adaptive F0")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        ax1.grid(alpha=0.2)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(sliding_pct[rid0], color="steelblue")
        ax2.set_ylabel("Percentile used")
        ax2.grid(alpha=0.2)

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        roi1_dff = sliding_dff[rid0]
        mean = float(np.nanmean(roi1_dff))
        std = float(np.nanstd(roi1_dff))
        thresh = roi_peak_thresholds.get(
            rid0, mean + 2.0 * std if np.isfinite(std) and std > 0 else mean
        )
        roi1_peaks_list = roi_peaks_by_roi.get(rid0, [])
        peaks_idx = [p[0] for p in roi1_peaks_list]
        peaks_vals = [p[1] for p in roi1_peaks_list]
        ax3.plot(roi1_dff, color="purple", label="dF/F (ROI1)")
        ax3.axhline(thresh, color="darkorange", linestyle="--", linewidth=1.0, label="z=2 threshold")
        if peaks_idx:
            ax3.scatter(peaks_idx, peaks_vals, color="tomato", s=15, zorder=3, label="Peaks")
        ax3.set_ylabel("dF/F")
        ax3.set_xlabel("Frame")
        ax3.legend()
        ax3.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(roi1_f0_debug_plot, dpi=200)
        plt.close()

    return ContractAnalysisOutputs(
        analysis_dir=analysis_dir,
        traces_csv=traces_csv,
        dff_csv=dff_csv,
        traces_plot=traces_plot,
        dff_plot=dff_plot,
        bleaching_baseline_csv=bleaching_baseline_csv,
        bleaching_fit_json=bleaching_fit_json,
        corrected_traces_csv=corrected_traces_csv,
        slow_baseline_csv=slow_baseline_csv,
        env_report_json=env_report_json,
        qc_pdf=qc_pdf,
        spike_debug_csv=spike_debug_csv,
        sliding_f0_csv=sliding_f0_csv,
        sliding_pct_csv=sliding_pct_csv,
        sliding_dff_csv=sliding_dff_csv,
        roi_peaks_csv=roi_peaks_csv,
        roi_peak_counts_csv=roi_peak_counts_csv,
        roi1_f0_debug_csv=roi1_f0_debug_csv,
        roi1_f0_debug_plot=roi1_f0_debug_plot,
        roi1_peaks_csv=roi1_peaks_csv,
    )


def process_project_root(
    project_root: Path,
    fps: float,
    output_dir: Path | None = None,
    config: ContractConfig | None = None,
) -> list[ContractAnalysisOutputs]:
    """Batch process every recording under a project root."""
    manifests = sorted(project_root.glob("**/processing_manifest.json"))
    eligible: list[tuple[Path, Path]] = []
    for manifest in manifests:
        rec_dir = manifest.parent
        roi_dir = rec_dir / "rois"
        roi_files = sorted(roi_dir.glob("*_roi_masks_uint16.tif"))
        roi_path = roi_files[0] if roi_files else None
        try:
            payload = json.loads(manifest.read_text())
            paths = payload.get("paths", {})
            mc_tiff = paths.get("motion_corrected_tiff")
            if not mc_tiff:
                print(f"[batch] Skipping {rec_dir.name}: missing motion_corrected_tiff in manifest.")
                continue
            if not Path(mc_tiff).exists():
                print(f"[batch] Skipping {rec_dir.name}: motion_corrected_tiff not found.")
                continue
            if roi_path is None:
                print(f"[batch] Skipping {rec_dir.name}: no ROI mask in {roi_dir}.")
                continue
            eligible.append((manifest, roi_path))
        except Exception as e:
            print(f"[batch] Skipping {rec_dir.name}: error reading manifest ({e}).")
            continue

    total = len(eligible)
    print(f"[batch] Project root: {project_root}")
    print(f"[batch] Eligible recordings: {total}")
    outputs: list[ContractAnalysisOutputs] = []
    for idx, (manifest_path, roi_path) in enumerate(eligible, start=1):
        rec_dir = manifest_path.parent
        print(f"[batch] Processing {idx}/{total}: {rec_dir.name}")
        cfg_base = config or ContractConfig()
        cfg = ContractConfig(
            z_threshold=cfg_base.z_threshold,
            spike_baseline_window_seconds=cfg_base.spike_baseline_window_seconds,
            spike_expand_seconds=cfg_base.spike_expand_seconds,
            mask_expand_min_frames=cfg_base.mask_expand_min_frames,
            lowess_window_frames=cfg_base.lowess_window_frames,
            lowess_it=cfg_base.lowess_it,
            fps=fps,
            use_motion_corrected=cfg_base.use_motion_corrected,
            f0_window_seconds=cfg_base.f0_window_seconds,
            f0_max_fraction=cfg_base.f0_max_fraction,
            f0_activity_fraction=cfg_base.f0_activity_fraction,
            f0_low_percentile=cfg_base.f0_low_percentile,
            f0_high_percentile=cfg_base.f0_high_percentile,
        )
        rec_output_dir = (output_dir / rec_dir.name) if output_dir else None
        out = process_contract_analysis(
            manifest_path=manifest_path,
            roi_path=roi_path,
            output_dir=rec_output_dir,
            config=cfg,
        )
        outputs.append(out)
    return outputs


def run_roi1_f0_debug(
    manifest_path: Path,
    roi_path: Path,
    output_dir: Path | None,
    cfg: ContractConfig,
) -> tuple[Path, Path, Path]:
    """Compute ROI1 adaptive sliding F0 (no masking/bleaching/multi-ROI)."""
    payload = json.loads(manifest_path.read_text())
    paths = payload.get("paths", {})
    raw_tiff = Path(paths["raw_tiff"]) if paths.get("raw_tiff") else None
    mc_tiff = Path(paths["motion_corrected_tiff"]) if paths.get("motion_corrected_tiff") else None
    movie_path = mc_tiff if cfg.use_motion_corrected else raw_tiff

    if movie_path is None or not movie_path.exists():
        raise FileNotFoundError("Movie TIFF not found for ROI1 F0 debug.")
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {roi_path}")

    movie = _ensure_movie_3d(tiff.imread(movie_path), "Movie").astype(float)
    roi_data = _load_roi_labels(roi_path, movie.shape)
    static_labels = roi_data.max(axis=0)
    traces, _ = extract_static_traces(movie, static_labels)
    roi_ids = sorted(traces.keys())
    if not roi_ids:
        raise ValueError("No ROI IDs found for F0 debug.")
    rid0 = roi_ids[0]
    trace = traces[rid0]

    f0, pct_used, win_frames = compute_sliding_f0_adaptive(
        trace,
        fps=cfg.fps,
        target_window_s=cfg.f0_window_seconds,
        max_fraction=cfg.f0_max_fraction,
        activity_fraction=cfg.f0_activity_fraction,
        low_percentile=cfg.f0_low_percentile,
        high_percentile=cfg.f0_high_percentile,
    )
    dff = (trace - f0) / f0
    dff_mean = float(np.nanmean(dff))
    dff_std = float(np.nanstd(dff))
    thresh = dff_mean + 2.0 * dff_std if np.isfinite(dff_std) and dff_std > 0 else dff_mean
    peak_idx, peak_props = find_peaks(dff, height=thresh)
    peak_vals = dff[peak_idx]
    peak_times = peak_idx / float(cfg.fps)

    analysis_dir = output_dir or (manifest_path.parent / "roi_analysis_contract")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    base_stem = _base_stem_from_raw(raw_tiff or movie_path)

    debug_csv = analysis_dir / f"{base_stem}_roi1_f0_debug.csv"
    debug_plot = analysis_dir / f"{base_stem}_roi1_f0_debug.png"
    peaks_csv = analysis_dir / f"{base_stem}_roi1_f0_peaks.csv"

    pd.DataFrame(
        {
            "frame": np.arange(trace.shape[0], dtype=int),
            "trace": trace,
            "f0": f0,
            "percentile_used": pct_used,
            "dff": dff,
        }
    ).to_csv(debug_csv, index=False)

    pd.DataFrame(
        {
            "frame": peak_idx.astype(int),
            "time_seconds": peak_times,
            "dff": peak_vals,
            "dff_zscore": (peak_vals - dff_mean) / (dff_std + 1e-9),
            "threshold": np.full_like(peak_vals, thresh),
        }
    ).to_csv(peaks_csv, index=False)

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(trace, color="black", label=f"Trace (ROI {rid0})")
    ax1.plot(f0, color="darkorange", label=f"Adaptive F0 (win≈{win_frames} frames)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(pct_used, color="steelblue")
    ax2.set_ylabel("Percentile used")
    ax2.grid(alpha=0.2)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(dff, color="purple", label="dF/F")
    ax3.axhline(thresh, color="darkorange", linestyle="--", linewidth=1.0, label="z=3 threshold")
    ax3.scatter(peak_idx, peak_vals, color="tomato", s=15, zorder=3, label="Peaks")
    ax3.set_ylabel("dF/F")
    ax3.set_xlabel("Frame")
    ax3.legend()
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(debug_plot, dpi=200)
    plt.close()

    return debug_csv, debug_plot, peaks_csv


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding-F0 contract pipeline (bleaching disabled). Use --project-root for batch."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Process every recording under this root (looks for processing_manifest.json and ROI mask).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=False,
        help="Single recording manifest (ignored when --project-root is supplied).",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        required=False,
        help="Single recording ROI mask (ignored when --project-root is supplied).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root; per-recording outputs go into <output-dir>/<recording>/roi_analysis_contract.",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Acquisition frame rate (Hz).")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="Local z threshold for spike masks (diagnostic only).")
    parser.add_argument(
        "--spike-baseline-sec",
        type=float,
        default=5.0,
        help="Window (s) for rolling median/MAD used in spike masking (diagnostic only).",
    )
    parser.add_argument(
        "--spike-expand-sec",
        type=float,
        default=2.5,
        help="Time (s) to expand detected spikes on each side (diagnostic only).",
    )
    parser.add_argument(
        "--mask-expand",
        type=int,
        default=0,
        help="Minimum expansion in frames added to the time-based expansion window.",
    )
    parser.add_argument("--roi1-f0-debug-only", action="store_true", help="Run only ROI1 sliding-F0 debug (no multi-ROI outputs).")
    parser.add_argument("--f0-window-sec", type=float, default=50.0, help="Sliding F0 window (s), capped at f0-max-frac of recording.")
    parser.add_argument("--f0-max-frac", type=float, default=0.4, help="Cap sliding window to this fraction of total frames.")
    parser.add_argument("--f0-activity-frac", type=float, default=0.3, help="Activity fraction used to scale percentile (not used when low=high).")
    parser.add_argument("--f0-low-pct", type=float, default=10.0, help="Lower percentile for F0 (equal to high locks F0).")
    parser.add_argument("--f0-high-pct", type=float, default=10.0, help="Upper percentile for F0 (equal to low locks F0).")
    parser.add_argument("--lowess-window", type=int, default=100, help="LOWESS window (frames) for slow baseline (diagnostic).")
    parser.add_argument("--lowess-it", type=int, default=3, help="LOWESS robust iterations (diagnostic).")
    parser.add_argument("--use-raw", action="store_true", help="Use raw movie instead of motion-corrected for trace extraction.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ContractConfig(
        z_threshold=args.z_threshold,
        spike_baseline_window_seconds=args.spike_baseline_sec,
        spike_expand_seconds=args.spike_expand_sec,
        mask_expand_min_frames=args.mask_expand,
        lowess_window_frames=args.lowess_window,
        lowess_it=args.lowess_it,
        fps=args.fps,
        use_motion_corrected=not args.use_raw,
        f0_window_seconds=args.f0_window_sec,
        f0_max_fraction=args.f0_max_frac,
        f0_activity_fraction=args.f0_activity_frac,
        f0_low_percentile=args.f0_low_pct,
        f0_high_percentile=args.f0_high_pct,
    )
    if args.project_root:
        process_project_root(
            project_root=args.project_root,
            fps=args.fps,
            output_dir=args.output_dir,
            config=config,
        )
    elif args.roi1_f0_debug_only:
        run_roi1_f0_debug(
            manifest_path=args.manifest,
            roi_path=args.roi,
            output_dir=args.output_dir,
            cfg=config,
        )
    else:
        if not args.manifest or not args.roi:
            raise SystemExit("Manifest and ROI are required unless --project-root is provided.")
        process_contract_analysis(
            manifest_path=args.manifest,
            roi_path=args.roi,
            output_dir=args.output_dir,
            config=config,
        )


if __name__ == "__main__":
    main()
