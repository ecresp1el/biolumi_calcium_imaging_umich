"""
Napari-based ROI annotation helper.

Launches Napari with a motion-corrected movie, a max-projection image,
optionally an existing ROI labels file, and optionally saves updated labels.
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import tifffile


def _read_tiff(path: Path) -> np.ndarray:
    return tifffile.imread(str(path))


def _write_tiff(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), data)


def _summarize_roi_labels(roi_data: np.ndarray) -> str:
    unique_labels = np.unique(roi_data)
    roi_ids = unique_labels[unique_labels != 0]
    roi_count = int(roi_ids.size)
    if roi_data.ndim == 3:
        axis_hint = "(T, Y, X)"
    elif roi_data.ndim == 2:
        axis_hint = "(Y, X)"
    else:
        axis_hint = "unknown"

    return (
        "✅ ROI labels saved.\n"
        f"📐 Shape: {roi_data.shape} (axis order {axis_hint})\n"
        f"🔢 Dtype: {roi_data.dtype}\n"
        f"🎯 ROI count (nonzero labels): {roi_count}\n"
        f"🏷 Labels (nonzero): {roi_ids.tolist()}"
    )


def _validate_roi_shape(roi_data: np.ndarray, movie_data: np.ndarray, strict: bool) -> None:
    if roi_data.ndim not in {2, 3}:
        msg = f"ROI must be 2D or 3D, got shape {roi_data.shape}"
        if strict:
            raise ValueError(msg)
        print(f"⚠️ {msg}")
        return

    if roi_data.ndim == 3:
        if roi_data.shape != movie_data.shape:
            msg = f"ROI shape {roi_data.shape} != movie shape {movie_data.shape}"
            if strict:
                raise ValueError(msg)
            print(f"⚠️ {msg}")
    else:
        if roi_data.shape != movie_data.shape[1:]:
            msg = f"ROI shape {roi_data.shape} != movie (Y,X) {movie_data.shape[1:]}"
            if strict:
                raise ValueError(msg)
            print(f"⚠️ {msg}")


def launch_napari_roi_tool(
    movie_path: Path,
    max_projection_path: Path,
    roi_path: Path | None,
    save_path: Path | None,
    strict: bool,
) -> None:
    movie = _read_tiff(movie_path)
    max_projection = _read_tiff(max_projection_path)

    viewer = napari.Viewer()
    viewer.add_image(movie, name="motion_corrected", colormap="gray")
    viewer.add_image(max_projection, name="max_projection", colormap="gray")

    if roi_path is not None and roi_path.exists():
        labels_data = _read_tiff(roi_path)
        _validate_roi_shape(labels_data, movie, strict)
        viewer.add_labels(labels_data, name="roi_labels")
    else:
        viewer.add_labels(np.zeros_like(movie, dtype=np.uint16), name="roi_labels")

    napari.run()

    if save_path is not None:
        labels_layer = viewer.layers["roi_labels"]
        labels_data = np.asarray(labels_layer.data)
        _write_tiff(save_path, labels_data)
        print(_summarize_roi_labels(labels_data))
        print(f"📁 Saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Napari to create manual ROI labels from calcium imaging data."
    )
    parser.add_argument(
        "--movie",
        required=True,
        type=Path,
        help="Path to the motion-corrected TIFF movie.",
    )
    parser.add_argument(
        "--max-projection",
        required=True,
        type=Path,
        help="Path to the max projection TIFF image.",
    )
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="Optional path to an existing ROI labels TIFF.",
    )
    parser.add_argument(
        "--save-roi",
        type=Path,
        default=None,
        help="Optional output path to save ROI labels TIFF on close.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if ROI labels do not match the movie shape.",
    )

    args = parser.parse_args()

    launch_napari_roi_tool(
        movie_path=args.movie,
        max_projection_path=args.max_projection,
        roi_path=args.roi,
        save_path=args.save_roi,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
