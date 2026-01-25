"""Standalone DREADD stimulation validation script for the chem_dreadd_stim_projectfolder."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tifffile as tiff
from matplotlib import colormaps, colors
from scipy.signal import peak_widths
import warnings

# Ensure repo root on sys.path so BL_CalciumAnalysis imports work when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BL_CalciumAnalysis.contracted_signal_extraction import (
    ContractConfig,
    process_project_root,
)

# Fixed inputs for this validation script.
PROJECT_ROOT = Path("/Volumes/Manny4TBUM/chem_dreadd_stim_projectfolder")
FPS = 5.0
F0_WINDOW_SEC = 30.0
BLEACH_FIT_SEC = 1.0
Z_THRESHOLD = 1.0
SMOOTH_WINDOW_SEC = 1.0

# Plot styling.
COL_PRE = "#666666"
COL_POST = "#6a1b9a"
CMAP_NAME = "magma"
FAST_FPS_MULTIPLIER = 10.0
COLORBAR_FILENAME = "pre_post_motion_corrected_colorbar.png"


def label_group(rec_name: str) -> str:
    """Map recording folder name to -C21/+C21/Unknown."""
    name = rec_name.lower()
    if "before" in name or "pre" in name:
        return "- C21"
    if "after" in name or "post" in name:
        return "+ C21"
    return "Unknown"


def load_peak_counts(path: Path, group: str, smoothed: bool = False) -> list[dict]:
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "group": group,
                "roi": int(r["roi"]),
                "peak_count": float(r["peak_count"]),
                "peak_rate_hz": float(r["peak_rate_hz"]),
                "duration_seconds": float(r["duration_seconds"]),
                "smoothed": smoothed,
            }
        )
    return rows


def filter_outliers(series: pd.Series) -> pd.Series:
    """Remove values outside 1.5*IQR fences."""
    vals = series.dropna()
    if vals.empty:
        return vals
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return vals[(vals >= lower) & (vals <= upper)]


def drop_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop rows with outliers (1.5*IQR) in any of the specified columns."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for col in cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        q1, q3 = vals.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper, inclusive="both")
    return df[mask].copy()


def find_manifest_paths(root: Path) -> dict[str, Path]:
    manifests = sorted(root.glob("**/processing_manifest.json"))
    mapping: dict[str, Path] = {}
    for mf in manifests:
        mapping[mf.parent.name] = mf
    return mapping


def load_mc_movie(path: Path) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = arr.mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got {arr.shape}")
    return arr.astype(float)


def make_pre_post_movie(pre_mc: Path, post_mc: Path, out_dir: Path, fps: float) -> None:
    """Create side-by-side pre/post movies with shared normalization and colorbar."""
    pre = load_mc_movie(pre_mc)
    post = load_mc_movie(post_mc)
    if pre.shape != post.shape:
        raise ValueError(f"Pre and post movies must match shape, got {pre.shape} vs {post.shape}")

    # Normalize to the post percentile range for consistent contrast.
    lo, hi = np.nanpercentile(post, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, max(1.0, float(np.nanmax(post)))
    cmap = colormaps.get_cmap(CMAP_NAME)

    def norm_to_rgb(arr: np.ndarray) -> np.ndarray:
        norm = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
        return (cmap(norm)[..., :3] * 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    colorbar_path = out_dir / COLORBAR_FILENAME
    try:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5, top=0.9)
        norm = colors.Normalize(vmin=lo, vmax=hi)
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )
        cb.set_label("Intensity (normalized to post, 1st–99th pct)", fontsize=9)
        fig.savefig(colorbar_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[dreadd_stim] Wrote colorbar: {colorbar_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[dreadd_stim] Skipping colorbar (matplotlib backend issue?): {e}")

    def write_movie(out_path: Path, fps_value: float) -> None:
        t, _, _ = pre.shape
        try:
            with imageio.get_writer(out_path, fps=fps_value, codec="libx264", format="ffmpeg") as wri:
                for i in range(t):
                    pre_rgb = norm_to_rgb(pre[i])
                    post_rgb = norm_to_rgb(post[i])
                    frame = np.concatenate([pre_rgb, post_rgb], axis=1)
                    wri.append_data(frame)
            print(f"[dreadd_stim] Wrote movie: {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[dreadd_stim] Skipping movie (ffmpeg unavailable?): {e}")

    write_movie(out_dir / "pre_post_motion_corrected.mp4", fps_value=fps)
    write_movie(out_dir / "pre_post_motion_corrected_fast.mp4", fps_value=fps * FAST_FPS_MULTIPLIER)


def boxplot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Boxplot + jittered points for peak counts."""
    if df.empty:
        print(f"[dreadd_stim] Skipping plot (empty data): {out_path}")
        return
    df = drop_outliers(df, ["peak_count"])
    data = []
    labels = []
    for grp in ["- C21", "+ C21"]:
        vals = filter_outliers(df.loc[df["group"] == grp, "peak_count"]).tolist()
        if vals:
            data.append(vals)
            labels.append(grp)
    if not data:
        print(f"[dreadd_stim] Skipping plot (no groups with data): {out_path}")
        return

    plt.figure(figsize=(5, 4))
    bp = plt.boxplot(
        data,
        notch=False,
        patch_artist=True,
        tick_labels=labels,
        showfliers=False,
        widths=0.5,
    )
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(COL_PRE if lab == "- C21" else COL_POST)
    x_positions = np.arange(1, len(labels) + 1)
    for xpos, lab in zip(x_positions, labels):
        vals = filter_outliers(df.loc[df["group"] == lab, "peak_count"]).tolist()
        if not vals:
            continue
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
        scatter_col = COL_PRE if lab == "- C21" else COL_POST
        plt.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            color=scatter_col,
            alpha=0.75,
            s=14,
            zorder=3,
        )
    plt.xticks(x_positions, labels)
    plt.ylabel("Peak count")
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()
    print(f"[dreadd_stim] Wrote plot: {out_path}")


def compute_peak_shapes(
    trace: np.ndarray, peak_frames: np.ndarray, fps: float
) -> tuple[list[float], list[float]]:
    """Compute FWHM (s) and integrated area (dF/F * s) using peak_widths at half height."""
    if trace.size == 0 or peak_frames.size == 0:
        return [], []
    peak_frames = peak_frames[(peak_frames >= 0) & (peak_frames < trace.size)]
    # Only keep peaks with positive, finite heights to avoid zero-width warnings.
    heights_ok = np.isfinite(trace[peak_frames]) & (trace[peak_frames] > 0)
    peak_frames = peak_frames[heights_ok]
    if peak_frames.size == 0:
        return [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        widths, _, left_ips, right_ips = peak_widths(trace, peak_frames, rel_height=0.5)
    valid = (
        np.isfinite(widths)
        & (widths > 0)
        & np.isfinite(left_ips)
        & np.isfinite(right_ips)
    )
    if not np.any(valid):
        return [], []
    widths = widths[valid]
    left_ips = left_ips[valid]
    right_ips = right_ips[valid]

    fwhm_seconds = (widths / fps).tolist()
    areas: list[float] = []
    for li, ri in zip(left_ips, right_ips):
        xs = np.arange(np.floor(li), np.ceil(ri) + 1)
        if xs.size == 0:
            areas.append(np.nan)
            continue
        ys = np.interp(xs, np.arange(trace.size), trace)
        area = float(np.trapz(ys, dx=1.0) / fps)  # dF/F * seconds
        areas.append(area)
    return fwhm_seconds, areas


def summarize_metrics(
    out: "ContractAnalysisOutputs", group: str, smoothed: bool, fps: float
) -> list[dict]:
    """Collect per-ROI metrics (count, rate, amplitude, FWHM) for plotting."""
    counts_csv = out.smoothed_peak_counts_csv if smoothed else out.roi_peak_counts_csv
    peaks_csv = out.smoothed_peaks_csv if smoothed else out.roi_peaks_csv
    dff_csv = out.smoothed_dff_csv if smoothed else out.sliding_dff_csv
    if counts_csv is None or peaks_csv is None or dff_csv is None:
        return []
    counts_df = pd.read_csv(counts_csv).set_index("roi")
    peaks_df = pd.read_csv(peaks_csv)
    dff_df = pd.read_csv(dff_csv, index_col=0)
    value_col = "dff_smoothed" if smoothed else "dff"

    rows: list[dict] = []
    for roi in counts_df.index:
        roi_int = int(roi)
        # Count & rate
        count = float(counts_df.loc[roi, "peak_count"])
        rate = float(counts_df.loc[roi, "peak_rate_hz"])
        # Amplitude
        amp_vals = peaks_df.loc[peaks_df["roi"] == roi_int, value_col]
        amp_med = float(amp_vals.median()) if not amp_vals.empty else np.nan
        # FWHM and integrated area
        trace = dff_df[str(roi_int)].to_numpy()
        peak_frames = peaks_df.loc[peaks_df["roi"] == roi_int, "frame"].to_numpy(dtype=int)
        fwhm_vals, area_vals = compute_peak_shapes(trace, peak_frames, fps)
        fwhm_med = float(np.nanmedian(fwhm_vals)) if len(fwhm_vals) else np.nan
        area_med = float(np.nanmedian(area_vals)) if len(area_vals) else np.nan
        rows.append(
            {
                "group": group,
                "roi": roi_int,
                "peak_count": count,
                "peak_rate_hz": rate,
                "peak_amplitude": amp_med,
                "peak_fwhm_sec": fwhm_med,
                "peak_integrated_area": area_med,
                "smoothed": smoothed,
            }
        )
    return rows


def multi_panel_boxplots(
    df_unsmoothed: pd.DataFrame, df_smoothed: pd.DataFrame, out_path: Path
) -> None:
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude (dF/F)"),
        ("peak_fwhm_sec", "FWHM (s)"),
        ("peak_integrated_area", "Integrated amplitude (dF/F·s)"),
    ]
    dfs = [("Unsmoothed", df_unsmoothed), ("Smoothed", df_smoothed)]
    fig, axes = plt.subplots(
        nrows=len(dfs),
        ncols=len(metrics),
        figsize=(15, 6),
    )
    if len(dfs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (label, df) in enumerate(dfs):
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data = []
            labels = []
            for grp, col in (("- C21", COL_PRE), ("+ C21", COL_POST)):
                vals = filter_outliers(df.loc[df["group"] == grp, metric])
                if vals.empty:
                    continue
                data.append(vals)
                labels.append((grp, col))
            if not data:
                ax.set_visible(False)
                continue
            box = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
                widths=0.45,
            )
            for patch, (grp, col) in zip(box["boxes"], labels):
                patch.set_facecolor(col)
                patch.set_alpha(0.8)
            x_positions = np.arange(1, len(labels) + 1)
            for xpos, (grp, col) in zip(x_positions, labels):
                vals = filter_outliers(df.loc[df["group"] == grp, metric])
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
                ax.scatter(
                    np.full(len(vals), xpos) + jitter,
                    vals,
                    color=col,
                    alpha=0.75,
                    s=14,
                    zorder=3,
                )
            ax.set_xticks(x_positions)
            ax.set_xticklabels([g for g, _ in labels])
            ax.set_title(f"{label} — {title}", fontsize=9)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(False)
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"[dreadd_stim] Wrote multi-panel plot: {out_path}")


def main() -> None:
    cfg = ContractConfig(
        fps=FPS,
        f0_window_seconds=F0_WINDOW_SEC,
        bleach_fit_seconds=BLEACH_FIT_SEC,
        z_threshold=Z_THRESHOLD,
        smooth_window_seconds=SMOOTH_WINDOW_SEC,
    )

    print(f"[dreadd_stim] Processing project root: {PROJECT_ROOT}")
    outputs = process_project_root(PROJECT_ROOT, fps=FPS, config=cfg)

    rows_counts: list[dict] = []
    rows_counts_smoothed: list[dict] = []
    metrics_unsmoothed: list[dict] = []
    metrics_smoothed: list[dict] = []
    for out in outputs:
        rec_dir = out.analysis_dir.parent
        group = label_group(rec_dir.name)
        if out.roi_peak_counts_csv:
            rows_counts.extend(load_peak_counts(Path(out.roi_peak_counts_csv), group, smoothed=False))
            metrics_unsmoothed.extend(summarize_metrics(out, group, smoothed=False, fps=FPS))
        if out.smoothed_peak_counts_csv:
            rows_counts_smoothed.extend(load_peak_counts(Path(out.smoothed_peak_counts_csv), group, smoothed=True))
            metrics_smoothed.extend(summarize_metrics(out, group, smoothed=True, fps=FPS))

    # Optional: generate pre/post movie normalized to the post percentiles.
    manifests = find_manifest_paths(PROJECT_ROOT)
    pre_key = [k for k in manifests.keys() if "before" in k.lower() or "pre" in k.lower()]
    post_key = [k for k in manifests.keys() if "after" in k.lower() or "post" in k.lower()]
    if pre_key and post_key:
        pre_manifest = manifests[pre_key[0]]
        post_manifest = manifests[post_key[0]]
        pre_mc = json.loads(pre_manifest.read_text())["paths"]["motion_corrected_tiff"]
        post_mc = json.loads(post_manifest.read_text())["paths"]["motion_corrected_tiff"]
        movie_dir = PROJECT_ROOT / "roi_analysis_contract_summary"
        make_pre_post_movie(Path(pre_mc), Path(post_mc), movie_dir, fps=FPS)
    else:
        print("[dreadd_stim] Skipping movies (could not find both Pre and Post manifests)")

    df_unsmoothed = pd.DataFrame(rows_counts)
    df_smoothed = pd.DataFrame(rows_counts_smoothed)
    summary_counts = pd.concat([df_unsmoothed, df_smoothed], ignore_index=True)
    summary_counts = drop_outliers(summary_counts, ["peak_count"])

    metrics_unsmoothed_df = pd.DataFrame(metrics_unsmoothed)
    metrics_smoothed_df = pd.DataFrame(metrics_smoothed)

    metric_cols = ["peak_count", "peak_rate_hz", "peak_amplitude", "peak_fwhm_sec", "peak_integrated_area"]
    metrics_unsmoothed_df = drop_outliers(metrics_unsmoothed_df, metric_cols)
    metrics_smoothed_df = drop_outliers(metrics_smoothed_df, metric_cols)
    metrics_all = pd.concat([metrics_unsmoothed_df, metrics_smoothed_df], ignore_index=True)

    out_dir = PROJECT_ROOT / "roi_analysis_contract_summary"
    out_dir.mkdir(exist_ok=True)
    summary_csv = out_dir / "dreadd_stim_peak_counts_summary.csv"
    summary_counts.to_csv(summary_csv, index=False)
    print(f"[dreadd_stim] Wrote summary: {summary_csv}")

    metrics_csv = out_dir / "dreadd_stim_peak_metrics_summary.csv"
    metrics_all.to_csv(metrics_csv, index=False)
    print(f"[dreadd_stim] Wrote metrics: {metrics_csv}")

    boxplot_counts(
        df_unsmoothed,
        "Ca2+ event counts (unsmoothed dF/F)",
        out_dir / "dreadd_stim_peak_counts_unsmoothed.png",
    )
    boxplot_counts(
        df_smoothed,
        "Ca2+ event counts (smoothed dF/F)",
        out_dir / "dreadd_stim_peak_counts_smoothed.png",
    )

    multi_panel_boxplots(
        metrics_unsmoothed_df,
        metrics_smoothed_df,
        out_dir / "dreadd_stim_peak_metrics_panels.png",
    )


if __name__ == "__main__":
    main()
