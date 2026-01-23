"""ROI analysis utilities that mirror the notebook workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours
import tifffile as tiff

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RoiAnalysisOutputs:
    analysis_dir: Path
    raw_vs_mc_movie: Path | None
    mc_movie_uint16: Path | None
    max_projection: Path | None
    avg_projection: Path | None
    std_projection: Path | None
    static_labels_tiff: Path
    traces_csv: Path
    dff_csv: Path
    traces_plot: Path
    dff_plot: Path
    roi_grid_movie: Path | None
    movies_note: Path | None


def _to_uint16(movie: np.ndarray) -> np.ndarray:
    movie = np.asarray(movie)
    movie = np.nan_to_num(movie, nan=0.0, posinf=0.0, neginf=0.0)
    movie = np.clip(movie, 0, None)
    maxv = float(movie.max())
    if maxv <= 0:
        return movie.astype(np.uint16)
    return (movie / maxv * 65535).astype(np.uint16)


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


def _save_side_by_side_movie(
    raw_movie: np.ndarray,
    mc_movie: np.ndarray,
    out_path: Path,
    fps: int = 10,
    cmap_name: str = "magma",
    label_left: str = "ORIGINAL",
    label_right: str = "MOTION CORRECTED",
) -> Path:
    if raw_movie.shape != mc_movie.shape:
        raise ValueError("Raw and motion-corrected movies must have the same shape.")

    t_frames, height, width = raw_movie.shape
    raw_lo, raw_hi = np.percentile(raw_movie, [1, 99])
    mc_lo, mc_hi = np.percentile(mc_movie, [1, 99])

    cmap = plt.colormaps.get_cmap(cmap_name)

    def norm_and_colorize(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
        norm = np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)
        rgb = cmap(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)

    base_font_size = max(18, height // 18)
    font = None
    for font_name in ["DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"]:
        try:
            font = ImageFont.truetype(font_name, base_font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    stroke_w = 3

    def text_size(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264") as writer:
        for t in range(t_frames):
            raw_rgb = norm_and_colorize(raw_movie[t], raw_lo, raw_hi)
            mc_rgb = norm_and_colorize(mc_movie[t], mc_lo, mc_hi)
            frame_rgb = np.concatenate([raw_rgb, mc_rgb], axis=1)

            img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(img)
            height_img, width_img = frame_rgb.shape[0], frame_rgb.shape[1]
            half_width = width_img // 2
            margin = 12

            if label_left:
                w_text, h_text = text_size(draw, label_left)
                draw.text(
                    (margin, height_img - h_text - margin),
                    label_left,
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0),
                )

            if label_right:
                w_text, h_text = text_size(draw, label_right)
                draw.text(
                    (half_width + margin, height_img - h_text - margin),
                    label_right,
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0),
                )

            writer.append_data(np.array(img))

    print(f"[roi_processing] Wrote movie: {out_path}")
    return out_path


def _save_motion_corrected_movie_uint16(mc_movie_u16: np.ndarray, video_path: Path) -> Path:
    out_path = video_path.parent / f"{video_path.stem}_MCMOVIE_uint16.tif"
    tiff.imwrite(out_path, mc_movie_u16, photometric="minisblack")
    print(f"[roi_processing] Saved motion-corrected movie → {out_path}")
    return out_path


def _save_projections_uint16(mc_movie_u16: np.ndarray, video_path: Path) -> tuple[Path, Path, Path]:
    save_dir = video_path.parent
    stem = video_path.stem
    max_proj = mc_movie_u16.max(axis=0).astype(np.uint16)
    avg_proj = mc_movie_u16.mean(axis=0).astype(np.uint16)
    std_proj = mc_movie_u16.std(axis=0).astype(np.uint16)

    max_path = save_dir / f"{stem}_MAXPROJ.tif"
    avg_path = save_dir / f"{stem}_AVGPROJ.tif"
    std_path = save_dir / f"{stem}_STDPROJ.tif"

    tiff.imwrite(max_path, max_proj)
    tiff.imwrite(avg_path, avg_proj)
    tiff.imwrite(std_path, std_proj)
    print("[roi_processing] Saved projections (uint16).")
    return max_path, avg_path, std_path


def _load_roi_labels(roi_path: Path, movie_shape: tuple[int, int, int]) -> np.ndarray:
    roi_data = tiff.imread(roi_path)
    if roi_data.ndim != 3:
        raise ValueError("ROI mask must be 3D (T, Y, X). Your mask is 2D.")

    t_roi, y_roi, x_roi = roi_data.shape
    t_mc, y_mc, x_mc = movie_shape
    if (y_roi != y_mc) or (x_roi != x_mc):
        raise ValueError("ERROR: ROI (Y,X) does NOT match movie size!")
    if t_roi != t_mc:
        print("⚠️ WARNING: ROI T != Movie T — Will broadcast ROI mask across time.")
        roi_data = np.broadcast_to(roi_data[0], movie_shape)
    return roi_data


def extract_static_traces(mc_movie: np.ndarray, static_labels: np.ndarray) -> dict[int, np.ndarray]:
    """Compute raw fluorescence traces for each ROI.

    The ROIs are defined by a 2D static label image (Y, X). For each ROI ID, we
    average the motion-corrected movie intensity within that ROI on every frame,
    yielding a raw fluorescence trace (mean intensity per frame).
    """
    roi_ids = np.unique(static_labels)
    roi_ids = roi_ids[roi_ids != 0]
    traces: dict[int, np.ndarray] = {}
    for rid in roi_ids:
        mask = static_labels == rid
        trace = mc_movie[:, mask].mean(axis=1)
        traces[int(rid)] = trace
    return traces


def compute_dff(trace: np.ndarray, baseline_percentile: int = 10) -> np.ndarray:
    """Compute ΔF/F from a raw trace using a percentile baseline."""
    f0 = np.percentile(trace, baseline_percentile)
    return (trace - f0) / (f0 + 1e-9)


def _plot_traces(traces: dict[int, np.ndarray], mc_movie_u16: np.ndarray, out_path: Path) -> Path:
    """Plot raw ROI traces (mean intensity per frame), vertically offset for readability."""
    plt.figure(figsize=(6, 6))
    offset = 0
    spacing = float(np.max(mc_movie_u16)) * 0.05
    for rid in sorted(traces.keys()):
        plt.plot(traces[rid] + offset, label=f"ROI {rid}")
        offset += spacing
    plt.title("ROI Fluorescence Traces (Static ROIs Applied Across All Frames)")
    plt.xlabel("Frame")
    plt.ylabel("Mean Intensity (offset)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _plot_dff_traces(dff_traces: dict[int, np.ndarray], out_path: Path) -> Path:
    """Plot ΔF/F traces for each ROI."""
    plt.figure(figsize=(12, 6))
    for rid in sorted(dff_traces.keys()):
        plt.plot(dff_traces[rid], label=f"ROI {rid}")
    plt.title("ROI ΔF/F Traces")
    plt.xlabel("Frame")
    plt.ylabel("ΔF/F")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def make_mc_roi_trace_movie_grid_with_outlines(
    mc_movie_u16: np.ndarray,
    static_labels: np.ndarray,
    dff_traces: dict[int, np.ndarray],
    video_path: Path,
    fps: int = 10,
    cmap: str = "gray",
    n_cols: int = 5,
    inset_pad: int = 5,
) -> Path:
    print("[roi_processing] Starting ROI grid movie with outlines.")
    video_path = Path(video_path)
    save_dir = video_path.parent
    out_path = save_dir / f"{video_path.stem}_MC_ROI_TRACE_GRID_OUTLINES.mp4"
    print(f"[roi_processing] Output path: {out_path}")

    t_frames, height, width = mc_movie_u16.shape
    print(
        "[roi_processing] Movie shape (T, Y, X): "
        f"{t_frames}, {height}, {width}"
    )
    print(
        "[roi_processing] Static labels shape: "
        f"{static_labels.shape}, dtype={static_labels.dtype}"
    )
    print(
        "[roi_processing] DFF trace keys: "
        f"{sorted(dff_traces.keys())[:10]}{'...' if len(dff_traces) > 10 else ''}"
    )
    lo, hi = np.percentile(mc_movie_u16, [1, 99])
    print(f"[roi_processing] Intensity percentiles: lo={lo}, hi={hi}")

    def norm(frame: np.ndarray) -> np.ndarray:
        return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)

    roi_ids = sorted([rid for rid in np.unique(static_labels) if rid != 0])
    print(f"[roi_processing] ROI count (nonzero labels): {len(roi_ids)}")
    roi_cmap = plt.colormaps.get_cmap("tab20")

    roi_rgb = np.zeros((height, width, 3), dtype=np.float32)
    for i, rid in enumerate(roi_ids):
        roi_rgb[static_labels == rid] = roi_cmap(i)[:3]
    print("[roi_processing] ROI overlay RGB map prepared.")

    global_ymax = max(dff_traces[r].max() for r in roi_ids) * 1.1
    global_ymin = min(dff_traces[r].min() for r in roi_ids) * 1.1
    print(
        "[roi_processing] Global ΔF/F limits: "
        f"ymin={global_ymin:.4f}, ymax={global_ymax:.4f}"
    )

    n_rows = int(np.ceil(len(roi_ids) / n_cols))
    print(f"[roi_processing] ROI grid layout: {n_rows} rows x {n_cols} cols")

    roi_bounds: dict[int, tuple[int, int, int, int] | None] = {}
    for rid in roi_ids:
        ys, xs = np.where(static_labels == rid)
        if len(xs) > 0:
            y0 = max(0, ys.min() - inset_pad)
            y1 = min(height, ys.max() + inset_pad)
            x0 = max(0, xs.min() - inset_pad)
            x1 = min(width, xs.max() + inset_pad)
            roi_bounds[rid] = (y0, y1, x0, x1)
        else:
            roi_bounds[rid] = None
    preview_bounds = list(roi_bounds.items())[:5]
    if preview_bounds:
        preview_text = ", ".join(f"{rid}:{bounds}" for rid, bounds in preview_bounds)
        print(f"[roi_processing] ROI bounds preview: {preview_text}")

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264")
    for t in range(t_frames):
        if t == 0 or t == t_frames - 1 or t % 25 == 0:
            print(f"[roi_processing] Rendering frame {t + 1}/{t_frames}")
        fig = plt.figure(figsize=(16, 10))
        gs_top = fig.add_gridspec(1, 3, top=0.92, bottom=0.65, hspace=0.25)

        ax1 = fig.add_subplot(gs_top[0, 0])
        ax1.imshow(norm(mc_movie_u16[t]), cmap=cmap)
        ax1.set_title("Motion-Corrected")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs_top[0, 1])
        base = plt.colormaps.get_cmap("gray")(norm(mc_movie_u16[t]))[..., :3]
        overlay = 0.7 * base + 0.3 * roi_rgb
        ax2.imshow(overlay)
        ax2.set_title("ROI Overlay")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs_top[0, 2])
        for rid in roi_ids:
            ax3.plot(dff_traces[rid][: t + 1], linewidth=0.7)
        ax3.set_ylim(global_ymin, global_ymax)
        ax3.set_xlim(0, t_frames)
        ax3.set_title("ΔF/F (Growing)")
        ax3.set_xlabel("Frame")

        gs_grid = fig.add_gridspec(n_rows, n_cols, top=0.62, bottom=0.05, hspace=0.55)

        for i, rid in enumerate(roi_ids):
            r = i // n_cols
            c = i % n_cols
            ax = fig.add_subplot(gs_grid[r, c])

            ax.plot(dff_traces[rid][: t + 1], color=roi_cmap(i)[:3], linewidth=0.8)
            ax.set_xlim(0, t_frames)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_title(f"ROI {rid}", fontsize=9)

            bounds = roi_bounds[rid]
            if bounds is not None:
                y0, y1, x0, x1 = bounds
                inset_frame = norm(mc_movie_u16[t, y0:y1, x0:x1])
                ax_in = ax.inset_axes([0.65, 0.55, 0.3, 0.35])
                ax_in.imshow(inset_frame, cmap=cmap)
                ax_in.set_xticks([])
                ax_in.set_yticks([])

                roi_crop = static_labels[y0:y1, x0:x1] == rid
                contours = find_contours(roi_crop.astype(float), 0.5)
                for contour in contours:
                    ax_in.plot(
                        contour[:, 1],
                        contour[:, 0],
                        color=roi_cmap(i)[:3],
                        linewidth=1.5,
                    )

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        writer.append_data(frame)
        plt.close(fig)

    writer.close()
    print(f"[roi_processing] Saved GRID movie with OUTLINES: {out_path}")
    return out_path


def _base_stem_from_raw(raw_path: Path) -> str:
    stem = raw_path.stem
    if stem.endswith("_raw"):
        return stem[: -len("_raw")]
    return stem


def process_roi_analysis(
    manifest_path: Path,
    roi_path: Path,
    generate_movies: bool = False,
) -> RoiAnalysisOutputs:
    """Run ROI analysis: extract raw traces, compute ΔF/F, and write plots/CSVs.

    Analysis summary (conceptually):
      1) Load raw + motion-corrected TIFF stacks (T, Y, X) from the manifest.
      2) Normalize the motion-corrected movie to uint16 for consistent plotting.
      3) Load the ROI labels (T, Y, X), verify shape, and collapse to a 2D
         static label map by taking the max over time.
      4) For each ROI ID, compute the mean intensity inside the ROI on every
         frame (raw traces).
      5) Convert raw traces to ΔF/F using a baseline percentile (default 10th).
      6) Save CSVs and plots for raw traces and ΔF/F traces.
      7) Optionally write diagnostic movies and projection images.

    The raw trace plot is a visualization of mean intensity per frame (offset
    vertically per ROI). The ΔF/F plot is the normalized signal relative to the
    baseline percentile, highlighting activity changes over time.
    """
    print("[roi_processing] Starting ROI analysis.")
    print(f"[roi_processing] Manifest path: {manifest_path}")
    print(f"[roi_processing] ROI path: {roi_path}")
    print(f"[roi_processing] generate_movies={generate_movies}")
    payload = json.loads(manifest_path.read_text())
    paths = payload.get("paths", {})

    raw_tiff = Path(paths.get("raw_tiff", ""))
    mc_tiff = Path(paths.get("motion_corrected_tiff", ""))
    print(f"[roi_processing] Raw TIFF path: {raw_tiff}")
    print(f"[roi_processing] Motion-corrected TIFF path: {mc_tiff}")

    if not raw_tiff.exists():
        raise FileNotFoundError(f"Raw TIFF not found: {raw_tiff}")
    if not mc_tiff.exists():
        raise FileNotFoundError(f"Motion-corrected TIFF not found: {mc_tiff}")
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {roi_path}")

    analysis_dir = manifest_path.parent / "roi_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[roi_processing] Analysis directory: {analysis_dir}")

    raw_movie = tiff.imread(raw_tiff)
    mc_movie = tiff.imread(mc_tiff)
    raw_movie = _ensure_movie_3d(raw_movie, "Raw")
    mc_movie = _ensure_movie_3d(mc_movie, "Motion-corrected")
    print(
        "[roi_processing] Movie shapes (raw, mc): "
        f"{raw_movie.shape}, {mc_movie.shape}"
    )
    mc_movie_u16 = _to_uint16(mc_movie)
    print(f"[roi_processing] Converted motion-corrected movie to uint16.")

    base_stem = _base_stem_from_raw(raw_tiff)
    video_path = analysis_dir / f"{base_stem}_raw_vs_mc_withtext.mp4"
    print(f"[roi_processing] Base stem: {base_stem}")
    print(f"[roi_processing] Raw-vs-MC movie path: {video_path}")

    movie_note_path: Path | None = None
    if generate_movies:
        print("[roi_processing] Generating movies and projections.")
        _save_side_by_side_movie(raw_movie, mc_movie, video_path)
        mc_movie_uint16_path = _save_motion_corrected_movie_uint16(mc_movie_u16, video_path)
        max_path, avg_path, std_path = _save_projections_uint16(mc_movie_u16, video_path)
    else:
        mc_movie_uint16_path = None
        max_path = None
        avg_path = None
        std_path = None
        movie_note_path = analysis_dir / "movies_skipped.txt"
        movie_note_path.write_text(
            "Movie generation skipped (raw-vs-mc MP4, projections, ROI grid movie). "
            "Enable generate_movies=True to run these steps."
        )
        print(f"[roi_processing] Movie generation skipped. Note: {movie_note_path}")

    roi_data = _load_roi_labels(roi_path, mc_movie_u16.shape)
    print(f"[roi_processing] ROI mask loaded with shape: {roi_data.shape}")
    static_labels = roi_data.max(axis=0)
    static_labels_path = analysis_dir / f"{base_stem}_static_roi_labels.tif"
    tiff.imwrite(static_labels_path, static_labels.astype(np.uint16))
    print(f"[roi_processing] Saved static labels: {static_labels_path}")

    traces = extract_static_traces(mc_movie_u16, static_labels)
    dff_traces = {rid: compute_dff(trace) for rid, trace in traces.items()}
    print(f"[roi_processing] Extracted traces for {len(traces)} ROIs.")

    traces_df = pd.DataFrame({rid: traces[rid] for rid in sorted(traces.keys())})
    dff_df = pd.DataFrame({rid: dff_traces[rid] for rid in sorted(dff_traces.keys())})

    traces_csv = analysis_dir / f"{base_stem}_roi_traces.csv"
    dff_csv = analysis_dir / f"{base_stem}_roi_dff.csv"
    traces_df.to_csv(traces_csv, index_label="frame")
    dff_df.to_csv(dff_csv, index_label="frame")
    print(f"[roi_processing] Saved CSVs: {traces_csv}, {dff_csv}")

    traces_plot = analysis_dir / f"{base_stem}_roi_traces.png"
    dff_plot = analysis_dir / f"{base_stem}_roi_dff.png"
    _plot_traces(traces, mc_movie_u16, traces_plot)
    _plot_dff_traces(dff_traces, dff_plot)
    print(f"[roi_processing] Saved trace plots: {traces_plot}, {dff_plot}")

    roi_grid_movie: Path | None = None
    if generate_movies:
        print("[roi_processing] Generating ROI grid movie with outlines.")
        roi_grid_movie = make_mc_roi_trace_movie_grid_with_outlines(
            mc_movie_u16,
            static_labels,
            dff_traces,
            video_path,
            fps=10,
            n_cols=5,
        )

    payload["roi_analysis"] = {
        "analysis_dir": str(analysis_dir),
        "raw_vs_mc_movie": str(video_path) if generate_movies else None,
        "mc_movie_uint16": str(mc_movie_uint16_path) if mc_movie_uint16_path else None,
        "max_projection_uint16": str(max_path) if max_path else None,
        "avg_projection_uint16": str(avg_path) if avg_path else None,
        "std_projection_uint16": str(std_path) if std_path else None,
        "static_roi_labels": str(static_labels_path),
        "traces_csv": str(traces_csv),
        "dff_csv": str(dff_csv),
        "traces_plot": str(traces_plot),
        "dff_plot": str(dff_plot),
        "roi_grid_movie": str(roi_grid_movie) if roi_grid_movie else None,
        "movies_note": str(movie_note_path) if movie_note_path else None,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    print(f"[roi_processing] Updated manifest: {manifest_path}")

    return RoiAnalysisOutputs(
        analysis_dir=analysis_dir,
        raw_vs_mc_movie=video_path if generate_movies else None,
        mc_movie_uint16=mc_movie_uint16_path,
        max_projection=max_path,
        avg_projection=avg_path,
        std_projection=std_path,
        static_labels_tiff=static_labels_path,
        traces_csv=traces_csv,
        dff_csv=dff_csv,
        traces_plot=traces_plot,
        dff_plot=dff_plot,
        roi_grid_movie=roi_grid_movie,
        movies_note=movie_note_path,
    )
