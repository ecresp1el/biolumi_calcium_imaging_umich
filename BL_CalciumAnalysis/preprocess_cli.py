"""CLI for converting Imaris movies and running motion correction."""

from __future__ import annotations

import argparse
from pathlib import Path

from BL_CalciumAnalysis.pipeline import MotionCorrectionConfig, MotionCorrectionPipeline


def _collect_ims_paths(ims_path: Path) -> list[Path]:
    if ims_path.is_dir():
        ims_files = sorted(ims_path.glob("*.ims"))
        if not ims_files:
            raise FileNotFoundError(f"No .ims files found in directory: {ims_path}")
        return ims_files

    if not ims_path.exists():
        raise FileNotFoundError(f"IMS path not found: {ims_path}")

    if ims_path.suffix.lower() != ".ims":
        raise ValueError(f"IMS path must be a .ims file or directory, got {ims_path}")

    return [ims_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an Imaris .ims recording, run motion correction, and save projections."
    )
    parser.add_argument(
        "--ims",
        required=True,
        type=Path,
        help="Path to a .ims file or a directory containing .ims files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory for processed outputs (defaults to IMS parent).",
    )
    parser.add_argument("--res-level", type=int, default=0, help="IMS resolution level.")
    parser.add_argument("--channel", type=int, default=0, help="IMS channel index.")
    parser.add_argument(
        "--no-collapse-z",
        action="store_true",
        help="Disable Z max-projection collapse (requires 2D data).",
    )
    parser.add_argument(
        "--no-piecewise-rigid",
        action="store_true",
        help="Disable piecewise-rigid motion correction.",
    )
    parser.add_argument(
        "--max-shifts",
        nargs=2,
        type=int,
        default=(12, 12),
        metavar=("Y", "X"),
        help="Maximum shifts in pixels (y x).",
    )
    parser.add_argument(
        "--strides",
        nargs=2,
        type=int,
        default=(48, 48),
        metavar=("Y", "X"),
        help="Strides for piecewise-rigid motion correction.",
    )
    parser.add_argument(
        "--overlaps",
        nargs=2,
        type=int,
        default=(24, 24),
        metavar=("Y", "X"),
        help="Overlaps for piecewise-rigid motion correction.",
    )
    parser.add_argument(
        "--gsig-filt",
        nargs=2,
        type=int,
        default=(3, 3),
        metavar=("Y", "X"),
        help="Gaussian filter sigma for motion correction.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = MotionCorrectionConfig(
        res_level=args.res_level,
        channel=args.channel,
        collapse_z=not args.no_collapse_z,
        pw_rigid=not args.no_piecewise_rigid,
        max_shifts=tuple(args.max_shifts),
        strides=tuple(args.strides),
        overlaps=tuple(args.overlaps),
        gsig_filt=tuple(args.gsig_filt),
    )

    ims_paths = _collect_ims_paths(args.ims)
    for ims_path in ims_paths:
        pipeline = MotionCorrectionPipeline(
            ims_path=ims_path,
            output_root=args.output_root,
            config=config,
        )
        pipeline.run()


if __name__ == "__main__":
    main()
