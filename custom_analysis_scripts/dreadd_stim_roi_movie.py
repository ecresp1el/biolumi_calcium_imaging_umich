"""Make 1x2 movies per recording: left = motion-corrected movie with ROI outlines, right = time-locked traces."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tifffile as tiff
from matplotlib import colormaps, rcParams
from scipy.ndimage import binary_erosion

# Keep SVG text editable if needed elsewhere.
rcParams["svg.fonttype"] = "none"

# Ensure repo root on sys.path so BL_CalciumAnalysis imports work when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BL_CalciumAnalysis.contracted_signal_extraction import (
    ContractConfig,
    process_project_root,
)

PROJECT_ROOT = Path("/Volumes/Manny4TBUM/chem_dreadd_stim_projectfolder")
FPS = 5.0
GENERATE_MOVIES = True

# Colors
ROI_OUTLINE_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # green outlines
TRACE_CMAP = colormaps.get_cmap("tab20")
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"


def load_movie(path: Path) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = arr.mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Movie must be 3D (T,Y,X), got {arr.shape}")
    return arr.astype(float)


def load_roi_mask(path: Path) -> np.ndarray:
    mask = tiff.imread(path)
    if mask.ndim == 3 and mask.shape[0] > 1:
        # Some ROI masks are saved as a time stack; collapse to a single 2D label image.
        mask = mask.max(axis=0)
    if mask.ndim != 2:
        raise ValueError(f"ROI mask must be 2D, got {mask.shape}")
    return mask


def roi_outlines(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Return per-ROI outline boolean masks."""
    outlines: dict[int, np.ndarray] = {}
    roi_ids = np.unique(mask)
    roi_ids = roi_ids[roi_ids > 0]
    for rid in roi_ids:
        region = mask == rid
        if not region.any():
            continue
        inner = binary_erosion(region, iterations=1, border_value=0)
        outline = region & ~inner
        outlines[int(rid)] = outline
    return outlines


def to_rgb_with_outlines(frame: np.ndarray, outlines: dict[int, np.ndarray], lo: float, hi: float) -> np.ndarray:
    norm = np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    for outline in outlines.values():
        rgb[outline] = ROI_OUTLINE_COLOR
    return rgb


def make_movie_for_output(out, fps: float) -> None:
    rec_dir = out.analysis_dir.parent
    manifest_path = rec_dir / "processing_manifest.json"
    if not manifest_path.exists():
        print(f"[movie] Skipping {rec_dir.name}: missing manifest.")
        return
    payload = json.loads(manifest_path.read_text())
    mc_path = payload.get("paths", {}).get("motion_corrected_tiff")
    if not mc_path or not Path(mc_path).exists():
        print(f"[movie] Skipping {rec_dir.name}: motion_corrected_tiff missing.")
        return
    roi_dir = rec_dir / "rois"
    roi_files = sorted(roi_dir.glob("*_roi_masks_uint16.tif"))
    if not roi_files:
        print(f"[movie] Skipping {rec_dir.name}: no ROI mask found.")
        return
    roi_path = roi_files[0]

    movie = load_movie(Path(mc_path))
    mask = load_roi_mask(roi_path)
    outlines = roi_outlines(mask)

    # Load dF/F traces (unsmoothed) for plotting.
    traces_df = pd.read_csv(out.sliding_dff_csv, index_col=0)
    roi_cols = sorted(traces_df.columns, key=lambda x: int(x))
    traces = traces_df[roi_cols].to_numpy()
    n_frames = min(movie.shape[0], traces.shape[0])
    movie = movie[:n_frames]
    traces = traces[:n_frames]
    times = np.arange(n_frames) / float(fps)

    lo, hi = np.nanpercentile(movie, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, float(np.nanmax(movie))

    # Precompute y-limits
    y_min = np.nanmin(traces)
    y_max = np.nanmax(traces)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        print(f"[movie] Skipping {rec_dir.name}: invalid traces.")
        return
    padding = 0.05 * (y_max - y_min + 1e-6)
    y_min -= padding
    y_max += padding

    out_dir = rec_dir / "roi_analysis_contract" / "movies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rec_dir.name}_roi_traces_movie.mp4"

    fig, (ax_img, ax_trace) = plt.subplots(
        1, 2, figsize=(12, 6), facecolor=BACKGROUND_COLOR, gridspec_kw={"width_ratios": [1, 1]}
    )
    for ax in (ax_img, ax_trace):
        ax.set_facecolor(BACKGROUND_COLOR)

    ax_img.axis("off")
    frame0_rgb = to_rgb_with_outlines(movie[0], outlines, lo, hi)
    im_artist = ax_img.imshow(frame0_rgb)

    lines = []
    for idx, col in enumerate(roi_cols):
        color = TRACE_CMAP(idx % TRACE_CMAP.N)
        line, = ax_trace.plot(times, traces[:, idx], color=color, linewidth=1.0, alpha=0.8)
        lines.append(line)
    vline = ax_trace.axvline(0, color="yellow", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.set_xlim(times[0], times[-1])
    ax_trace.set_ylim(y_min, y_max)
    ax_trace.set_xlabel("Time (s)", color=TEXT_COLOR)
    ax_trace.set_ylabel("dF/F", color=TEXT_COLOR)
    ax_trace.tick_params(colors=TEXT_COLOR)
    for spine in ax_trace.spines.values():
        spine.set_color(TEXT_COLOR)
    ax_trace.set_title(rec_dir.name, color=TEXT_COLOR)

    plt.tight_layout()

    with imageio.get_writer(out_path, fps=fps, codec="libx264", format="ffmpeg") as wri:
        for i in range(n_frames):
            rgb = to_rgb_with_outlines(movie[i], outlines, lo, hi)
            im_artist.set_data(rgb)
            vline.set_xdata([times[i], times[i]])
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            wri.append_data(frame_rgba)

    plt.close(fig)
    print(f"[movie] Wrote movie: {out_path}")


def main() -> None:
    cfg = ContractConfig(fps=FPS)
    outputs = process_project_root(PROJECT_ROOT, fps=FPS, config=cfg)
    if not GENERATE_MOVIES:
        print("[movie] GENERATE_MOVIES=False, skipping.")
        return
    for out in outputs:
        make_movie_for_output(out, fps=FPS)


if __name__ == "__main__":
    main()
