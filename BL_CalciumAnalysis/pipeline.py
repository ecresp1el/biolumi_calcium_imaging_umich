"""Core pipeline helpers for calcium imaging preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import caiman as cm
from caiman.motion_correction import MotionCorrect
import h5py
import numpy as np
import tifffile as tiff


@dataclass(frozen=True)
class MotionCorrectionConfig:
    res_level: int = 0
    channel: int = 0
    collapse_z: bool = True
    pw_rigid: bool = True
    max_shifts: tuple[int, int] = (12, 12)
    strides: tuple[int, int] = (48, 48)
    overlaps: tuple[int, int] = (24, 24)
    gsig_filt: tuple[int, int] = (3, 3)


@dataclass(frozen=True)
class RecordingPaths:
    base_dir: Path
    raw_dir: Path
    motion_corrected_dir: Path
    projections_dir: Path
    raw_tiff: Path
    motion_corrected_tiff: Path
    max_projection: Path
    avg_projection: Path
    std_projection: Path
    manifest_path: Path

    @staticmethod
    def from_ims(ims_path: Path, output_root: Path | None = None) -> "RecordingPaths":
        output_root = output_root or ims_path.parent
        stem = ims_path.stem
        base_dir = output_root / stem
        raw_dir = base_dir / "raw"
        motion_corrected_dir = base_dir / "motion_corrected"
        projections_dir = base_dir / "projections"

        raw_tiff = raw_dir / f"{stem}_raw.tif"
        motion_corrected_tiff = motion_corrected_dir / f"{stem}_motion_corrected.tif"
        max_projection = projections_dir / f"{stem}_MAXPROJ.tif"
        avg_projection = projections_dir / f"{stem}_AVGPROJ.tif"
        std_projection = projections_dir / f"{stem}_STDPROJ.tif"
        manifest_path = base_dir / "processing_manifest.json"

        return RecordingPaths(
            base_dir=base_dir,
            raw_dir=raw_dir,
            motion_corrected_dir=motion_corrected_dir,
            projections_dir=projections_dir,
            raw_tiff=raw_tiff,
            motion_corrected_tiff=motion_corrected_tiff,
            max_projection=max_projection,
            avg_projection=avg_projection,
            std_projection=std_projection,
            manifest_path=manifest_path,
        )


class MotionCorrectionPipeline:
    """Pipeline for converting Imaris IMS movies and running motion correction."""

    def __init__(
        self,
        ims_path: str | Path,
        output_root: str | Path | None = None,
        config: MotionCorrectionConfig | None = None,
    ) -> None:
        self.ims_path = Path(ims_path)
        self.config = config or MotionCorrectionConfig()
        self.paths = RecordingPaths.from_ims(self.ims_path, Path(output_root) if output_root else None)

    def _ensure_dirs(self, dirs: Iterable[Path]) -> None:
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def convert_ims_movie_to_tiff(self) -> Path:
        if self.ims_path.suffix.lower() != ".ims":
            raise TypeError(f"Input must be a .ims file, got {self.ims_path}")

        self._ensure_dirs([self.paths.raw_dir])
        print(f"[convert_ims_movie_to_tiff] Reading {self.ims_path}")

        frames: list[np.ndarray] = []
        with h5py.File(self.ims_path, "r") as file_handle:
            dataset = file_handle["DataSet"]
            res_group = dataset[f"ResolutionLevel {self.config.res_level}"]
            tp_keys = [key for key in res_group.keys() if key.startswith("TimePoint ")]
            tp_keys = sorted(tp_keys, key=lambda key: int(key.split(" ")[1]))
            print(f"[convert_ims_movie_to_tiff] Found {len(tp_keys)} timepoints")

            for key in tp_keys:
                tp_group = res_group[key]
                channel_group = tp_group[f"Channel {self.config.channel}"]
                data = channel_group["Data"][()]

                if data.ndim == 3:
                    if self.config.collapse_z:
                        frame = data.max(axis=0) if data.shape[0] > 1 else data[0]
                    else:
                        raise ValueError(
                            f"Expected 2D frames when collapse_z=False, got {data.shape}"
                        )
                elif data.ndim == 2:
                    frame = data
                else:
                    raise ValueError(
                        f"Unexpected data shape for {key}/Channel {self.config.channel}: {data.shape}"
                    )

                frames.append(frame)

        stack = np.stack(frames, axis=0)
        print(f"[convert_ims_movie_to_tiff] Stack shape (T, Y, X): {stack.shape}")
        print(f"[convert_ims_movie_to_tiff] Writing {self.paths.raw_tiff}")
        tiff.imwrite(str(self.paths.raw_tiff), stack, bigtiff=True)
        print("[convert_ims_movie_to_tiff] Done.")
        return self.paths.raw_tiff

    def motion_correct(self, tiff_path: Path) -> tuple[Path, np.ndarray]:
        self._ensure_dirs([self.paths.motion_corrected_dir])
        fn = [str(tiff_path)]
        is_3d = False

        mc = MotionCorrect(
            fn,
            min_mov=None,
            max_shifts=self.config.max_shifts,
            strides=self.config.strides,
            overlaps=self.config.overlaps,
            pw_rigid=self.config.pw_rigid,
            is3D=is_3d,
            gSig_filt=self.config.gsig_filt,
        )
        mc.motion_correct(save_movie=True)

        corrected_file = mc.fname_tot_els[0] if self.config.pw_rigid else mc.fname_tot_rig[0]
        print(f"[motion_correct] Corrected movie: {corrected_file}")

        movie = cm.load(corrected_file)
        tiff.imwrite(str(self.paths.motion_corrected_tiff), movie.astype("float32"))
        print(f"[motion_correct] Saved corrected TIFF: {self.paths.motion_corrected_tiff}")
        return self.paths.motion_corrected_tiff, movie

    def save_projections(self, movie: np.ndarray) -> tuple[Path, Path, Path]:
        self._ensure_dirs([self.paths.projections_dir])
        max_proj = movie.max(axis=0)
        avg_proj = movie.mean(axis=0)
        std_proj = movie.std(axis=0)

        tiff.imwrite(self.paths.max_projection, max_proj.astype("float32"))
        tiff.imwrite(self.paths.avg_projection, avg_proj.astype("float32"))
        tiff.imwrite(self.paths.std_projection, std_proj.astype("float32"))

        print("[projections] Saved projections:")
        print(f"  MAX → {self.paths.max_projection}")
        print(f"  AVG → {self.paths.avg_projection}")
        print(f"  STD → {self.paths.std_projection}")
        return self.paths.max_projection, self.paths.avg_projection, self.paths.std_projection

    def write_manifest(self) -> None:
        payload = {
            "ims_path": str(self.ims_path),
            "paths": {
                "raw_tiff": str(self.paths.raw_tiff),
                "motion_corrected_tiff": str(self.paths.motion_corrected_tiff),
                "max_projection": str(self.paths.max_projection),
                "avg_projection": str(self.paths.avg_projection),
                "std_projection": str(self.paths.std_projection),
            },
            "config": asdict(self.config),
        }
        self.paths.base_dir.mkdir(parents=True, exist_ok=True)
        self.paths.manifest_path.write_text(json.dumps(payload, indent=2))
        print(f"[manifest] Saved processing manifest: {self.paths.manifest_path}")

    def run(self) -> RecordingPaths:
        raw_tiff = self.convert_ims_movie_to_tiff()
        _, movie = self.motion_correct(raw_tiff)
        self.save_projections(movie)
        self.write_manifest()
        return self.paths
