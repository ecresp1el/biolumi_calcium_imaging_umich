"""Make 1x2 movies per recording: left = motion-corrected movie with ROI outlines, right = time-locked traces."""

from __future__ import annotations

from pathlib import Path
import argparse
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
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter

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
FRAME_STRIDE = 1  # Increase to drop frames; set 1 to keep all frames.
TOP_K_HIGHLIGHT = 6  # Also render a second movie with the top-K most fluctuating ROIs; set to None to skip.
OUTLINE_THICKNESS = 1  # ROI outline thickness (pixels)
PLAYBACK_SPEED = 1.0  # Multiplier for output fps (e.g., 4.0 plays 4× faster without dropping frames)

# Colors
TRACE_CMAP = colormaps.get_cmap("tab20")
MOVIE_CMAP = colormaps.get_cmap("plasma")
BACKGROUND_COLOR = "black"
TEXT_COLOR = "white"
FRAME_SMOOTH_SIGMA = 0.6  # Gaussian sigma for light de-graining on movie frames


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


def roi_outlines(mask: np.ndarray, thickness: int = 1) -> dict[int, np.ndarray]:
    """Return per-ROI outline boolean masks with configurable thickness."""
    outlines: dict[int, np.ndarray] = {}
    roi_ids = np.unique(mask)
    roi_ids = roi_ids[roi_ids > 0]
    for rid in roi_ids:
        region = mask == rid
        if not region.any():
            continue
        inner = binary_erosion(region, iterations=1, border_value=0)
        outline = region & ~inner
        if thickness > 1:
            outline = binary_dilation(outline, iterations=thickness - 1)
        outlines[int(rid)] = outline
    return outlines


def to_rgb_with_outlines(
    frame: np.ndarray, outlines: dict[int, tuple[np.ndarray, np.ndarray]], lo: float, hi: float
) -> np.ndarray:
    if FRAME_SMOOTH_SIGMA and FRAME_SMOOTH_SIGMA > 0:
        frame = gaussian_filter(frame, sigma=FRAME_SMOOTH_SIGMA)
    norm = np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = (MOVIE_CMAP(norm)[..., :3] * 255).astype(np.uint8)
    for _, (outline_mask, color_rgb) in outlines.items():
        rgb[outline_mask] = color_rgb
    return rgb


def make_movie_for_output(
    out,
    fps: float,
    top_k: int | None = None,
    outline_thickness: int = 1,
    frame_stride: int = 1,
    playback_speed: float = 1.0,
) -> None:
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
    outlines_all = roi_outlines(mask, thickness=outline_thickness)
    roi_ids_all = sorted(outlines_all.keys())

    # Load dF/F traces (unsmoothed) for plotting.
    traces_df = pd.read_csv(out.sliding_dff_csv, index_col=0)
    roi_cols_all = sorted(traces_df.columns, key=lambda x: int(x))
    traces_all = traces_df[roi_cols_all].to_numpy()

    # Select top-K ROIs by standard deviation (fluctuations) if requested.
    if top_k is not None and top_k > 0:
        stds = []
        for idx, col in enumerate(roi_cols_all):
            stds.append((float(np.nanstd(traces_all[:, idx])), col, idx))
        stds.sort(reverse=True, key=lambda x: x[0])
        top = stds[: min(top_k, len(stds))]
        roi_cols = [col for _, col, _ in top]
        idx_map = [idx for _, _, idx in top]
    else:
        roi_cols = roi_cols_all
        idx_map = list(range(len(roi_cols_all)))

    traces = traces_all[:, idx_map]

    # Filter outlines to selected ROIs.
    outlines = {rid: outlines_all[rid] for rid in outlines_all if str(rid) in roi_cols}
    roi_ids = sorted(outlines.keys(), key=int)
    roi_colors_rgb: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, rid in enumerate(roi_ids):
        c = np.array(TRACE_CMAP(idx % TRACE_CMAP.N)[:3]) * 255.0
        roi_colors_rgb[rid] = (outlines[rid], c.astype(np.uint8))
    n_frames_full = min(movie.shape[0], traces.shape[0])
    frame_indices = np.arange(0, n_frames_full, frame_stride, dtype=int)
    movie = movie[frame_indices]
    traces = traces[frame_indices]
    times = frame_indices / float(fps)
    n_frames = len(frame_indices)

    lo, hi = np.nanpercentile(movie, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, float(np.nanmax(movie))

    # Normalize traces to 0-1 per ROI and stagger vertically.
    traces_norm = np.zeros_like(traces)
    offsets = np.arange(len(roi_cols)) * 1.2
    for idx in range(len(roi_cols)):
        t = traces[:, idx]
        t_min = np.nanmin(t)
        t_max = np.nanmax(t)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max - t_min <= 0:
            traces_norm[:, idx] = 0.0
        else:
            traces_norm[:, idx] = np.clip((t - t_min) / (t_max - t_min), 0, 1)

    out_dir = rec_dir / "roi_analysis_contract" / "movies"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_top{top_k}" if top_k is not None else "_all"
    out_path = out_dir / f"{rec_dir.name}_roi_traces_movie{suffix}.mp4"

    fig, (ax_img, ax_trace) = plt.subplots(
        1, 2, figsize=(12, 6), facecolor=BACKGROUND_COLOR, gridspec_kw={"width_ratios": [1, 1]}
    )
    for ax in (ax_img, ax_trace):
        ax.set_facecolor(BACKGROUND_COLOR)

    ax_img.axis("off")
    frame0_rgb = to_rgb_with_outlines(movie[0], roi_colors_rgb, lo, hi)
    im_artist = ax_img.imshow(frame0_rgb)

    lines = []
    for idx, col in enumerate(roi_cols):
        color = TRACE_CMAP(idx % TRACE_CMAP.N)
        line, = ax_trace.plot([], [], color=color, linewidth=1.0, alpha=0.9)
        lines.append(line)
    vline = ax_trace.axvline(0, color="yellow", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_trace.set_xlim(times[0], times[-1])
    ax_trace.set_ylim(-0.2, offsets[-1] + 1.2)
    ax_trace.set_yticks([])
    ax_trace.set_xlabel("Time (s)", color=TEXT_COLOR)
    ax_trace.set_ylabel("dF/F", color=TEXT_COLOR)
    ax_trace.tick_params(colors=TEXT_COLOR)
    for spine in ax_trace.spines.values():
        spine.set_color(TEXT_COLOR)
    # No title; filename is sufficient.

    plt.tight_layout()

    playback_fps = fps * max(playback_speed, 0.1)
    with imageio.get_writer(out_path, fps=playback_fps, codec="libx264", format="ffmpeg") as wri:
        for i in range(n_frames):
            rgb = to_rgb_with_outlines(movie[i], roi_colors_rgb, lo, hi)
            im_artist.set_data(rgb)
            vline.set_xdata([times[i], times[i]])
            for idx, line in enumerate(lines):
                line.set_data(times[: i + 1], traces_norm[: i + 1, idx] + offsets[idx])
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            wri.append_data(frame_rgba)

    plt.close(fig)
    print(f"[movie] Wrote movie: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make ROI/trace movies.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing recordings.")
    parser.add_argument("--fps", type=float, default=FPS, help="Acquisition fps.")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE, help="Render every Nth frame (speedup).")
    parser.add_argument("--playback-speed", type=float, default=PLAYBACK_SPEED, help="Playback speed multiplier (e.g., 4.0 to play 4× faster without dropping frames).")
    parser.add_argument("--top-k-highlight", type=int, default=TOP_K_HIGHLIGHT, help="Render an additional movie with top-K fluctuating ROIs (set <=0 to skip).")
    parser.add_argument("--outline-thickness", type=int, default=OUTLINE_THICKNESS, help="ROI outline thickness in pixels.")
    parser.add_argument("--generate-movies", action="store_true", default=GENERATE_MOVIES, help="Set to render movies (default from file).")
    args = parser.parse_args()

    cfg = ContractConfig(fps=args.fps)
    outputs = process_project_root(args.project_root, fps=args.fps, config=cfg)
    if not args.generate_movies:
        print("[movie] --generate-movies not set; skipping renders.")
        return
    for out in outputs:
        # Render full set
        make_movie_for_output(
            out,
            fps=args.fps,
            top_k=None,
            outline_thickness=args.outline_thickness,
            frame_stride=args.frame_stride,
            playback_speed=args.playback_speed,
        )
        # Render top-K if requested
        if args.top_k_highlight is not None and args.top_k_highlight > 0:
            make_movie_for_output(
                out,
                fps=args.fps,
                top_k=args.top_k_highlight,
                outline_thickness=args.outline_thickness,
                frame_stride=args.frame_stride,
                playback_speed=args.playback_speed,
            )


if __name__ == "__main__":
    main()
