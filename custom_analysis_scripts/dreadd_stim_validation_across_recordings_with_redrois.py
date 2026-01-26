"""Validation across recordings using red ROIs on the green movies."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def load_peak_counts(path: Path, group: str, smoothed: bool, modality: str) -> list[dict]:
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
            }
        )
    return rows


def summarize_metrics(peaks_csv: Path, counts_csv: Path, dff_csv: Path, group: str, smoothed: bool, modality: str) -> list[dict]:
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
        fwhm_med = np.nan
        rows.append(
            {
                "group": group,
                "roi": roi_int,
                "peak_count": count,
                "peak_rate_hz": rate,
                "peak_amplitude": amp_med,
                "peak_fwhm_sec": fwhm_med,
                "peak_integrated_area": np.nan,
                "smoothed": smoothed,
                "modality": modality,
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


def boxplot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return
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
    if df_unsmoothed.empty and df_smoothed.empty:
        print(f"[redroi] Skipping metrics panels (no data)")
        return
    metrics = [
        ("peak_count", "Peak count"),
        ("peak_rate_hz", "Peak rate (Hz)"),
        ("peak_amplitude", "Peak amplitude"),
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
                vals = filter_outliers(df.loc[df["group"] == grp, metric])
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
            rows_counts.extend(load_peak_counts(Path(out.roi_peak_counts_csv), group, smoothed=False, modality=modality))
        if out.smoothed_peak_counts_csv:
            rows_counts_smoothed.extend(
                load_peak_counts(Path(out.smoothed_peak_counts_csv), group, smoothed=True, modality=modality)
            )
        if out.roi_peaks_csv and out.roi_peak_counts_csv and out.sliding_dff_csv:
            metrics_unsmoothed.extend(
                summarize_metrics(Path(out.roi_peaks_csv), Path(out.roi_peak_counts_csv), Path(out.sliding_dff_csv), group, False, modality)
            )
        if out.smoothed_peaks_csv and out.smoothed_peak_counts_csv and out.smoothed_dff_csv:
            metrics_smoothed.extend(
                summarize_metrics(Path(out.smoothed_peaks_csv), Path(out.smoothed_peak_counts_csv), Path(out.smoothed_dff_csv), group, True, modality)
            )

    df_unsmoothed = pd.DataFrame(rows_counts)
    df_smoothed = pd.DataFrame(rows_counts_smoothed)
    metrics_unsmoothed_df = pd.DataFrame(metrics_unsmoothed)
    metrics_smoothed_df = pd.DataFrame(metrics_smoothed)

    out_dir_root = PROJECT_ROOT / "roi_analysis_contract_summary_redroi"
    out_dir_root.mkdir(exist_ok=True)
    df_unsmoothed.to_csv(out_dir_root / "redroi_peak_counts_unsmoothed.csv", index=False)
    df_smoothed.to_csv(out_dir_root / "redroi_peak_counts_smoothed.csv", index=False)
    metrics_unsmoothed_df.to_csv(out_dir_root / "redroi_metrics_unsmoothed.csv", index=False)
    metrics_smoothed_df.to_csv(out_dir_root / "redroi_metrics_smoothed.csv", index=False)
    boxplot_counts(df_unsmoothed, "Red ROI (on green movie) event counts (unsmoothed)", out_dir_root / "redroi_peak_counts_unsmoothed.png")
    boxplot_counts(df_smoothed, "Red ROI (on green movie) event counts (smoothed)", out_dir_root / "redroi_peak_counts_smoothed.png")
    multipanel_metrics(metrics_unsmoothed_df, metrics_smoothed_df, out_dir_root / "redroi_metrics_panels.png")
    if not df_unsmoothed.empty:
        print(f"[redroi] Group counts (unsmoothed):\n{df_unsmoothed['group'].value_counts(dropna=False)}")
    if not df_smoothed.empty:
        print(f"[redroi] Group counts (smoothed):\n{df_smoothed['group'].value_counts(dropna=False)}")
    print(f"[redroi] Wrote summaries and plots to {out_dir_root}")


if __name__ == "__main__":
    main()
