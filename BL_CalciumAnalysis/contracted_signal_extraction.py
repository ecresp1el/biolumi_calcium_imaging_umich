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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ContractConfig:
    """Configuration for the contracted preprocessing pipeline."""

    z_threshold: float = 2.0
    mask_expand: int = 1
    lowess_window_frames: int = 100
    lowess_it: int = 3
    fps: float = 1.0
    use_motion_corrected: bool = True


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
    z_threshold: float = 2.0,
    mask_expand: int = 1,
) -> np.ndarray:
    """Return boolean mask of permissive spike frames for bleaching exclusion."""
    trace = np.asarray(trace, dtype=float)
    mean = np.nanmean(trace)
    std = np.nanstd(trace)
    if not np.isfinite(std) or std == 0:
        return np.zeros(trace.shape, dtype=bool)
    zscores = (trace - mean) / std
    spikes = zscores > z_threshold
    if mask_expand > 0:
        kernel = np.ones(2 * mask_expand + 1, dtype=int)
        spikes = np.convolve(spikes.astype(int), kernel, mode="same") > 0
    return spikes


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

    spikes = detect_rough_spikes(
        global_trace,
        z_threshold=cfg.z_threshold,
        mask_expand=cfg.mask_expand,
    )
    silent_mask = ~spikes
    t = np.arange(global_trace.shape[0], dtype=float) / cfg.fps

    bleaching_baseline, fit_params = fit_bleaching_baseline(t, global_trace, silent_mask)
    corrected_traces = {
        rid: traces[rid] / bleaching_baseline for rid in traces.keys()
    }
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
            "spike_mask": spikes.astype(int),
        }
    ).to_csv(bleaching_baseline_csv, index=False)

    bleaching_fit_json.write_text(json.dumps(fit_params, indent=2))
    env_report_json.write_text(json.dumps(env_info, indent=2))

    _plot_traces(traces, traces_plot, "ROI Fluorescence Traces (Raw)")
    _plot_dff_traces(dff_traces, dff_plot, "ROI dF/F Traces (Bleaching Corrected)")

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
    )


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run contracted bleaching correction on a recording."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--mask-expand", type=int, default=1)
    parser.add_argument("--lowess-window", type=int, default=100)
    parser.add_argument("--lowess-it", type=int, default=3)
    parser.add_argument("--use-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ContractConfig(
        z_threshold=args.z_threshold,
        mask_expand=args.mask_expand,
        lowess_window_frames=args.lowess_window,
        lowess_it=args.lowess_it,
        fps=args.fps,
        use_motion_corrected=not args.use_raw,
    )
    process_contract_analysis(
        manifest_path=args.manifest,
        roi_path=args.roi,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
