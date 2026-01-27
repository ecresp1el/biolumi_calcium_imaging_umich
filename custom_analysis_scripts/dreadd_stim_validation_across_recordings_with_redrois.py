"""Validation across recordings using red ROIs on the green movies."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, rcParams
from scipy.signal import peak_widths
from scipy.stats import mannwhitneyu
import warnings

rcParams["svg.fonttype"] = "none"

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BL_CalciumAnalysis.contracted_signal_extraction import (
    ContractConfig,
    process_contract_analysis,
)

PROJECT_ROOT = Path("/Volumes/Manny4TBUM/c21vsnoc21_projectfolder")
FPS = 5.0
F0_WINDOW_SEC = 30.0
BLEACH_FIT_SEC = 1.0
Z_THRESHOLD = 1.0
SMOOTH_WINDOW_SEC = 1.0
COL_MEDIA = "#666666"
COL_C21 = "#6a1b9a"


def label_group(rec_name: str) -> str:
    name = rec_name.lower()
    if "media" in name or "noc21" in name or "-c21" in name or "control" in name:
        return "- C21"
    if "cn21" in name or "+c21" in name or "c21" in name:
        return "+ C21"
    return "Unknown"


def find_partner_red_roi(green_rec_dir: Path) -> Path | None:
    """Find red ROI mask for a green recording.

    Priority:
    1) red ROI saved alongside green (green_dir/rois/<green>_red_roi_masks_uint16.tif)
    2) partner red recording folder (heuristic search for matching stem containing 'Red')
    """
    green_name = green_rec_dir.name
    local_red = green_rec_dir / "rois" / f"{green_name}_red_roi_masks_uint16.tif"
    if local_red.exists():
        return local_red

    base = green_name.replace(" - Green_Confocal - Red", "").replace(" - Green", "")
    parent = green_rec_dir.parent
    candidates = []
    for d in parent.iterdir():
        if not d.is_dir():
            continue
        if base in d.name and "red" in d.name.lower():
            cand = d / "rois"
            if cand.exists():
                for tif in cand.glob("*red_roi_masks_uint16.tif"):
                    candidates.append(tif)
    if candidates:
        # pick the first sorted
        return sorted(candidates)[0]
    return None


def load_peak_counts(path: Path, group: str, smoothed: bool, modality: str, recording: str) -> list[dict]:
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
                "modality": modality,
                "recording": recording,
            }
        )
    return rows


def summarize_metrics(peaks_csv: Path, counts_csv: Path, dff_csv: Path, group: str, smoothed: bool, modality: str, recording: str) -> list[dict]:
    counts_df = pd.read_csv(counts_csv).set_index("roi")
    peaks_df = pd.read_csv(peaks_csv)
    dff_df = pd.read_csv(dff_csv, index_col=0)
    value_col = "dff_smoothed" if smoothed else "dff"

    rows: list[dict] = []
    for roi in counts_df.index:
        roi_int = int(roi)
        count = float(counts_df.loc[roi, "peak_count"])
        rate = float(counts_df.loc[roi, "peak_rate_hz"])
        amp_vals = peaks_df.loc[peaks_df["roi"] == roi_int, value_col]
        amp_med = float(amp_vals.median()) if not amp_vals.empty else np.nan
        trace = dff_df[str(roi_int)].to_numpy()
        peak_frames = peaks_df.loc[peaks_df["roi"] == roi_int, "frame"].to_numpy(dtype=int)
        fwhm_vals, area_vals = compute_peak_shapes(trace, peak_frames, FPS)
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
                "modality": modality,
                "recording": recording,
            }
        )
    return rows


def filter_outliers(series: pd.Series) -> pd.Series:
    vals = series.dropna()
    if vals.empty:
        return vals
    q1, q3 = vals.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return vals[(vals >= lower) & (vals <= upper)]

def _clean_groups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "group" not in df.columns:
        return df
    return df[df["group"].isin(["- C21", "+ C21"])].copy()


FILTER_METRICS = ["peak_count", "peak_rate_hz", "peak_amplitude"]


def drop_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    group_cols = ["group", "smoothed"] if {"group", "smoothed"}.issubset(df.columns) else []
    grouped = df.groupby(group_cols) if group_cols else [(None, df)]
    for _, sub in grouped:
        sub_idx = sub.index
        sub_mask = pd.Series(True, index=sub_idx)
        for col in cols:
            vals = sub[col].dropna()
            if vals.empty:
                continue
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            valid = sub[col].between(lower, upper, inclusive="both")
            sub_mask &= (~sub[col].notna()) | valid
        mask.loc[sub_idx] = sub_mask
    return df[mask].copy()


def boxplot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
    df = _clean_groups(df)
    if df.empty:
        return
    df = drop_outliers(df, ["peak_count"])
    data = []
    labels = []
    for grp in ["- C21", "+ C21"]:
        vals = filter_outliers(df.loc[df["group"] == grp, "peak_count"])
        if not vals.empty:
            data.append(vals)
            labels.append(grp)
    if not data:
        return
    plt.figure(figsize=(5, 4))
    bp = plt.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False, widths=0.6)
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(COL_MEDIA if lab == "- C21" else COL_C21)
    x_pos = np.arange(1, len(labels) + 1)
    for xpos, lab in zip(x_pos, labels):
        vals = filter_outliers(df.loc[df["group"] == lab, "peak_count"])
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
        color = COL_MEDIA if lab == "- C21" else COL_C21
        plt.scatter(np.full(len(vals), xpos) + jitter, vals, color=color, alpha=0.75, s=14, zorder=3)
    plt.ylabel("Peak count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.with_suffix(".svg"))
    plt.close()


def multipanel_metrics(df_unsmoothed: pd.DataFrame, df_smoothed: pd.DataFrame, out_path: Path) -> None:
    df_unsmoothed = _clean_groups(df_unsmoothed)
    df_smoothed = _clean_groups(df_smoothed)
    if df_unsmoothed.empty and df_smoothed.empty:
        print(f"[redroi] Skipping metrics panels (no data)")
        return
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude"),
        ("peak_fwhm_sec", "FWHM (s)"),
        ("peak_integrated_area", "Integrated amplitude"),
    ]
    dfs = [("Unsmoothed", df_unsmoothed), ("Smoothed", df_smoothed)]
    fig, axes = plt.subplots(len(dfs), len(metrics), figsize=(9, 5))
    if len(dfs) == 1:
        axes = np.expand_dims(axes, 0)
    for r, (label, df) in enumerate(dfs):
        if df.empty or "group" not in df.columns:
            for c in range(len(metrics)):
                axes[r, c].set_visible(False)
            continue
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            data = []
            labels = []
            for grp in ["- C21", "+ C21"]:
                series = df.loc[df["group"] == grp, metric]
                vals = filter_outliers(series) if metric in FILTER_METRICS else series.dropna()
                if not vals.empty:
                    data.append(vals)
                    labels.append(grp)
            if not data:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
            for patch, lab in zip(bp["boxes"], labels):
                patch.set_facecolor(COL_MEDIA if lab == "- C21" else COL_C21)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_title(f"{label} — {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def mann_whitney_stats(df: pd.DataFrame, metrics: list[str], smoothed_flag: bool) -> list[dict]:
    results = []
    for metric in metrics:
        grp1 = df.loc[df["group"] == "- C21", metric].dropna()
        grp2 = df.loc[df["group"] == "+ C21", metric].dropna()
        if grp1.empty or grp2.empty:
            continue
        try:
            stat, pval = mannwhitneyu(grp1, grp2, alternative="two-sided")
            results.append(
                {
                    "metric": metric,
                    "smoothed": smoothed_flag,
                    "group1": "- C21",
                    "group2": "+ C21",
                    "n_group1": int(len(grp1)),
                    "n_group2": int(len(grp2)),
                    "statistic": float(stat),
                    "p_value": float(pval),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[redroi] Stats failed for {metric} smoothed={smoothed_flag}: {exc}")
    return results


def multi_panel_boxplots_by_recording(df_unsmoothed: pd.DataFrame, df_smoothed: pd.DataFrame, out_path: Path) -> None:
    df_unsmoothed = _clean_groups(df_unsmoothed)
    df_smoothed = _clean_groups(df_smoothed)
    if df_unsmoothed.empty and df_smoothed.empty:
        return
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude"),
        ("peak_fwhm_sec", "FWHM (s)"),
        ("peak_integrated_area", "Integrated amplitude"),
    ]
    dfs = [("Unsmoothed", df_unsmoothed), ("Smoothed", df_smoothed)]
    fig, axes = plt.subplots(len(dfs), len(metrics), figsize=(15, 6))
    if len(dfs) == 1:
        axes = np.expand_dims(axes, 0)
    cmap_gray = colormaps.get_cmap("Greys")
    cmap_purp = colormaps.get_cmap("Purples")
    for r, (label, df) in enumerate(dfs):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            if df.empty or "group" not in df.columns or "recording" not in df.columns:
                ax.set_visible(False)
                continue
            medians = []
            labels = []
            for grp in ("- C21", "+ C21"):
                series = df.loc[df["group"] == grp, metric]
                vals = filter_outliers(series) if metric in FILTER_METRICS else series.dropna()
                if vals.empty:
                    continue
                medians.append(np.nanmedian(vals))
                labels.append(grp)
            if not medians:
                ax.set_visible(False)
                continue
            x_positions = np.arange(1, len(labels) + 1)
            for xpos, grp in zip(x_positions, labels):
                recs = sorted(df.loc[df["group"] == grp, "recording"].dropna().unique())
                if not recs:
                    continue
                cmap = cmap_gray if grp == "- C21" else cmap_purp
                colors = cmap(np.linspace(0.35, 0.85, len(recs)))
                color_map = dict(zip(recs, colors))
                vals = df.loc[df["group"] == grp, metric]
                vals = filter_outliers(vals) if metric in FILTER_METRICS else vals.dropna()
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
                rec_series = df.loc[df["group"] == grp, "recording"]
                for val, jit, rec in zip(vals, jitter, rec_series):
                    col = color_map.get(rec, (0, 0, 0, 0.6))
                    ax.scatter(xpos + jit, val, color=col, alpha=0.85, s=16, zorder=3)
            bp = ax.boxplot(
                [df.loc[df["group"] == grp, metric].dropna() for grp in labels],
                patch_artist=True,
                showfliers=False,
                widths=0.45,
            )
            for patch, grp in zip(bp["boxes"], labels):
                patch.set_facecolor("none")
                patch.set_edgecolor(COL_MEDIA if grp == "- C21" else COL_C21)
                patch.set_linewidth(1.2)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels)
            ax.set_title(f"{label} — {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def multi_panel_bars(df_unsmoothed: pd.DataFrame, df_smoothed: pd.DataFrame, out_path: Path) -> None:
    df_unsmoothed = _clean_groups(df_unsmoothed)
    df_smoothed = _clean_groups(df_smoothed)
    if df_unsmoothed.empty and df_smoothed.empty:
        return
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude"),
        ("peak_fwhm_sec", "FWHM (s)"),
        ("peak_integrated_area", "Integrated amplitude"),
    ]
    dfs = [("Unsmoothed", df_unsmoothed), ("Smoothed", df_smoothed)]
    fig, axes = plt.subplots(len(dfs), len(metrics), figsize=(15, 6))
    if len(dfs) == 1:
        axes = np.expand_dims(axes, 0)
    for r, (label, df) in enumerate(dfs):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            if df.empty or "group" not in df.columns:
                ax.set_visible(False)
                continue
            medians = []
            labels = []
            scatters: list[tuple[np.ndarray, str]] = []
            for grp, color in (("- C21", COL_MEDIA), ("+ C21", COL_C21)):
                vals = df.loc[df["group"] == grp, metric]
                vals = filter_outliers(vals) if metric in FILTER_METRICS else vals.dropna()
                if vals.empty:
                    continue
                medians.append(np.nanmedian(vals))
                labels.append((grp, color))
                scatters.append((vals.to_numpy(), color))
            if not medians:
                ax.set_visible(False)
                continue
            x_positions = np.arange(1, len(labels) + 1)
            ax.bar(
                x_positions,
                medians,
                color=[c for _, c in labels],
                alpha=0.35,
                edgecolor=[c for _, c in labels],
                linewidth=1.2,
            )
            for xpos, (vals, color) in zip(x_positions, scatters):
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
                ax.scatter(xpos + jitter, vals, color=color, alpha=0.75, s=14, zorder=3)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([g for g, _ in labels])
            ax.set_title(f"{label} — {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def multi_panel_bars_by_recording(df_unsmoothed: pd.DataFrame, df_smoothed: pd.DataFrame, out_path: Path) -> None:
    df_unsmoothed = _clean_groups(df_unsmoothed)
    df_smoothed = _clean_groups(df_smoothed)
    if df_unsmoothed.empty and df_smoothed.empty:
        return
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude"),
        ("peak_fwhm_sec", "FWHM (s)"),
        ("peak_integrated_area", "Integrated amplitude"),
    ]
    dfs = [("Unsmoothed", df_unsmoothed), ("Smoothed", df_smoothed)]
    fig, axes = plt.subplots(len(dfs), len(metrics), figsize=(15, 6))
    if len(dfs) == 1:
        axes = np.expand_dims(axes, 0)
    cmap_gray = colormaps.get_cmap("Greys")
    cmap_purp = colormaps.get_cmap("Purples")
    for r, (label, df) in enumerate(dfs):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r, c]
            if df.empty or "group" not in df.columns or "recording" not in df.columns:
                ax.set_visible(False)
                continue
            medians = []
            labels = []
            for grp in ("- C21", "+ C21"):
                vals = df.loc[df["group"] == grp, metric]
                vals = filter_outliers(vals) if metric in FILTER_METRICS else vals.dropna()
                if vals.empty:
                    continue
                medians.append(np.nanmedian(vals))
                labels.append(grp)
            if not medians:
                ax.set_visible(False)
                continue
            x_positions = np.arange(1, len(labels) + 1)
            ax.bar(
                x_positions,
                medians,
                color="none",
                edgecolor=[COL_MEDIA if g == "- C21" else COL_C21 for g in labels],
                linewidth=1.2,
            )
            for grp in ("- C21", "+ C21"):
                sub = df.loc[df["group"] == grp]
                if sub.empty:
                    continue
                recs = sorted(sub["recording"].dropna().unique())
                if not recs:
                    continue
                cmap = cmap_gray if grp == "- C21" else cmap_purp
                colors = cmap(np.linspace(0.35, 0.85, len(recs)))
                rec_map = dict(zip(recs, colors))
                xpos = labels.index(grp) + 1 if grp in labels else None
                if xpos is None:
                    continue
                vals = sub[metric]
                vals = filter_outliers(vals) if metric in FILTER_METRICS else vals.dropna()
                jitter = (np.random.rand(len(vals)) - 0.5) * 0.18
                for val, jit, rec in zip(vals, jitter, sub["recording"]):
                    col = rec_map.get(rec, (0, 0, 0, 0.6))
                    ax.scatter(xpos + jit, val, color=col, alpha=0.85, s=16, zorder=3)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels)
            ax.set_title(f"{label} — {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def compute_peak_shapes(trace: np.ndarray, peak_frames: np.ndarray, fps: float) -> tuple[list[float], list[float]]:
    if trace.size == 0 or peak_frames.size == 0:
        return [], []
    peak_frames = peak_frames[(peak_frames >= 0) & (peak_frames < trace.size)]
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
        area = float(np.trapz(ys, dx=1.0) / fps)
        areas.append(area)
    return fwhm_seconds, areas


def main() -> None:
    cfg = ContractConfig(
        fps=FPS,
        f0_window_seconds=F0_WINDOW_SEC,
        bleach_fit_seconds=BLEACH_FIT_SEC,
        z_threshold=Z_THRESHOLD,
        smooth_window_seconds=SMOOTH_WINDOW_SEC,
    )

    rows_counts: list[dict] = []
    rows_counts_smoothed: list[dict] = []
    metrics_unsmoothed: list[dict] = []
    metrics_smoothed: list[dict] = []

    manifests = sorted(PROJECT_ROOT.glob("**/processing_manifest.json"))
    for manifest in manifests:
        rec_dir = manifest.parent
        if "Green_Confocal - Red" in rec_dir.name:
            continue
        red_roi = find_partner_red_roi(rec_dir)
        if red_roi is None:
            print(f"[redroi] Skipping {rec_dir.name}: no red ROI found")
            continue
        print(f"[redroi] Found red ROI for {rec_dir.name}: {red_roi}")

        # Output dir to avoid clobbering green analysis
        out_dir = rec_dir / "roi_analysis_contract_redroi"
        out = process_contract_analysis(
            manifest_path=manifest,
            roi_path=red_roi,
            output_dir=out_dir,
            config=cfg,
        )

        group = label_group(rec_dir.name)
        modality = "red_roi_on_green"
        if out.roi_peak_counts_csv:
            rows_counts.extend(load_peak_counts(Path(out.roi_peak_counts_csv), group, smoothed=False, modality=modality, recording=rec_dir.name))
        if out.smoothed_peak_counts_csv:
            rows_counts_smoothed.extend(
                load_peak_counts(Path(out.smoothed_peak_counts_csv), group, smoothed=True, modality=modality, recording=rec_dir.name)
            )
        if out.roi_peaks_csv and out.roi_peak_counts_csv and out.sliding_dff_csv:
            metrics_unsmoothed.extend(
                summarize_metrics(Path(out.roi_peaks_csv), Path(out.roi_peak_counts_csv), Path(out.sliding_dff_csv), group, False, modality, rec_dir.name)
            )
        if out.smoothed_peaks_csv and out.smoothed_peak_counts_csv and out.smoothed_dff_csv:
            metrics_smoothed.extend(
                summarize_metrics(Path(out.smoothed_peaks_csv), Path(out.smoothed_peak_counts_csv), Path(out.smoothed_dff_csv), group, True, modality, rec_dir.name)
            )

    df_unsmoothed = pd.DataFrame(rows_counts)
    df_smoothed = pd.DataFrame(rows_counts_smoothed)
    metrics_unsmoothed_df = pd.DataFrame(metrics_unsmoothed)
    metrics_smoothed_df = pd.DataFrame(metrics_smoothed)
    df_unsmoothed = _clean_groups(df_unsmoothed)
    df_smoothed = _clean_groups(df_smoothed)
    metrics_unsmoothed_df = _clean_groups(metrics_unsmoothed_df)
    metrics_smoothed_df = _clean_groups(metrics_smoothed_df)

    out_dir_root = PROJECT_ROOT / "roi_analysis_contract_summary_redroi"
    out_dir_root.mkdir(exist_ok=True)
    df_unsmoothed.to_csv(out_dir_root / "redroi_peak_counts_unsmoothed.csv", index=False)
    df_smoothed.to_csv(out_dir_root / "redroi_peak_counts_smoothed.csv", index=False)
    metrics_unsmoothed_df.to_csv(out_dir_root / "redroi_metrics_unsmoothed.csv", index=False)
    metrics_smoothed_df.to_csv(out_dir_root / "redroi_metrics_smoothed.csv", index=False)
    boxplot_counts(df_unsmoothed, "Red ROI (on green movie) event counts (unsmoothed)", out_dir_root / "redroi_peak_counts_unsmoothed.png")
    boxplot_counts(df_smoothed, "Red ROI (on green movie) event counts (smoothed)", out_dir_root / "redroi_peak_counts_smoothed.png")
    multipanel_metrics(metrics_unsmoothed_df, metrics_smoothed_df, out_dir_root / "redroi_metrics_panels.png")
    multi_panel_boxplots_by_recording(metrics_unsmoothed_df, metrics_smoothed_df, out_dir_root / "redroi_metrics_panels_by_recording.png")
    multi_panel_bars(metrics_unsmoothed_df, metrics_smoothed_df, out_dir_root / "redroi_metrics_panels_bar.png")
    multi_panel_bars_by_recording(metrics_unsmoothed_df, metrics_smoothed_df, out_dir_root / "redroi_metrics_panels_bar_by_recording.png")
    stats_rows = []
    metrics_cols = ["peak_count", "peak_rate_hz", "peak_amplitude", "peak_fwhm_sec", "peak_integrated_area"]
    stats_rows += mann_whitney_stats(metrics_unsmoothed_df, metrics_cols, smoothed_flag=False)
    stats_rows += mann_whitney_stats(metrics_smoothed_df, metrics_cols, smoothed_flag=True)
    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df.to_csv(out_dir_root / "redroi_metrics_stats.csv", index=False)
    if not df_unsmoothed.empty:
        print(f"[redroi] Group counts (unsmoothed):\n{df_unsmoothed['group'].value_counts(dropna=False)}")
    if not df_smoothed.empty:
        print(f"[redroi] Group counts (smoothed):\n{df_smoothed['group'].value_counts(dropna=False)}")
    print(f"[redroi] Wrote summaries, plots, and stats to {out_dir_root}")


if __name__ == "__main__":
    main()
