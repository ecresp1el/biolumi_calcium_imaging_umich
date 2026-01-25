"""Standalone DREADD stim validation script for the chem_dreadd_stim_projectfolder."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from BL_CalciumAnalysis.contracted_signal_extraction import (
    ContractConfig,
    process_project_root,
)


PROJECT_ROOT = Path("/Volumes/Manny4TBUM/chem_dreadd_stim_projectfolder")
FPS = 5.0
F0_WINDOW_SEC = 30.0
BLEACH_FIT_SEC = 1.0
Z_THRESHOLD = 1.0
SMOOTH_WINDOW_SEC = 1.0


def label_group(rec_name: str) -> str:
    name = rec_name.lower()
    if "before" in name or "pre" in name:
        return "Pre"
    if "after" in name or "post" in name:
        return "Post"
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


def main() -> None:
    cfg = ContractConfig(
        fps=FPS,
        f0_window_seconds=F0_WINDOW_SEC,
        bleach_fit_seconds=BLEACH_FIT_SEC,
        z_threshold=Z_THRESHOLD,
        smooth_window_seconds=SMOOTH_WINDOW_SEC,
        # Keep other defaults.
    )

    print(f"[dreadd_stim] Processing project root: {PROJECT_ROOT}")
    outputs = process_project_root(PROJECT_ROOT, fps=FPS, config=cfg)

    rows: list[dict] = []
    rows_smoothed: list[dict] = []
    for out in outputs:
        rec_dir = out.analysis_dir.parent
        group = label_group(rec_dir.name)
        if out.roi_peak_counts_csv:
            rows.extend(load_peak_counts(Path(out.roi_peak_counts_csv), group, smoothed=False))
        if out.smoothed_peak_counts_csv:
            rows_smoothed.extend(load_peak_counts(Path(out.smoothed_peak_counts_csv), group, smoothed=True))

    df_unsmoothed = pd.DataFrame(rows)
    df_smoothed = pd.DataFrame(rows_smoothed)
    summary = pd.concat([df_unsmoothed, df_smoothed], ignore_index=True)

    out_dir = PROJECT_ROOT / "roi_analysis_contract_summary"
    out_dir.mkdir(exist_ok=True)
    summary_csv = out_dir / "dreadd_stim_peak_counts_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[dreadd_stim] Wrote summary: {summary_csv}")

    def boxplot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
        if df.empty:
            print(f"[dreadd_stim] Skipping plot (empty data): {out_path}")
            return
        data = []
        labels = []
        for grp in ["Pre", "Post"]:
            vals = df.loc[df["group"] == grp, "peak_count"].tolist()
            if vals:
                data.append(vals)
                labels.append(grp)
        if not data:
            print(f"[dreadd_stim] Skipping plot (no groups with data): {out_path}")
            return
        plt.figure(figsize=(6, 5))
        plt.boxplot(data, labels=labels, notch=True)
        plt.ylabel("Peak count")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[dreadd_stim] Wrote plot: {out_path}")

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


if __name__ == "__main__":
    main()
