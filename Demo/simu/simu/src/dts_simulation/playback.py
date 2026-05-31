"""Offline PNG playback generation from DTS scenario snapshots.

This module repeatedly calls the non-graphical scenario runner and renders
each completed ScenarioResult as one PNG frame. It does not generate signals,
run MUSIC, solve localization, fire launchers, animate in real time, or control
hardware outside the existing scenario pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
from statistics import mean

from PIL import Image

from .models import Point3D
from .simulation import ScenarioInput, ScenarioResult, default_array_poses, run_synthetic_scenario
from .visualization import render_static_scene

DEFAULT_PLAYBACK_FRAME_COUNT = 30
DEFAULT_PLAYBACK_DURATION_SECONDS = 3.0
DEFAULT_PLAYBACK_FRAMES_PER_SECOND = 10
DEFAULT_PLAYBACK_OUTPUT_DIRECTORY = Path("outputs") / "playback"
DEFAULT_PLAYBACK_GIF_OUTPUT_PATH = Path("outputs") / "dts_playback.gif"


@dataclass(frozen=True, slots=True)
class PlaybackFrame:
    """One offline playback snapshot derived from one ScenarioResult."""

    frame_index: int
    time_seconds: float
    true_source_xyz: Point3D
    scenario_result: ScenarioResult
    output_path: Path

    @property
    def valid(self) -> bool:
        return self.scenario_result.valid

    @property
    def error_distance(self) -> float | None:
        return self.scenario_result.error_distance


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    """Summary of one deterministic PNG frame-sequence generation run."""

    frames: tuple[PlaybackFrame, ...]
    output_directory: Path

    @property
    def all_frames_valid(self) -> bool:
        return all(frame.valid for frame in self.frames)

    @property
    def error_distances(self) -> tuple[float, ...]:
        return tuple(
            frame.error_distance
            for frame in self.frames
            if frame.error_distance is not None
        )

    @property
    def max_error_distance(self) -> float | None:
        errors = self.error_distances
        if not errors:
            return None
        return max(errors)

    @property
    def mean_error_distance(self) -> float | None:
        errors = self.error_distances
        if not errors:
            return None
        return mean(errors)


@dataclass(frozen=True, slots=True)
class GifExportResult:
    """Summary of one GIF artifact export from generated playback frames."""

    output_path: Path
    frame_count: int
    succeeded: bool


def default_source_path(frame_count: int = DEFAULT_PLAYBACK_FRAME_COUNT) -> tuple[Point3D, ...]:
    """Return the deterministic path defined by the playback contract."""
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if frame_count == 1:
        return ((1.0, 0.8, 1.6),)

    positions = []
    for frame_index in range(frame_count):
        t = frame_index / (frame_count - 1)
        x = 1.0 + 2.0 * t
        y = 0.8 + 1.7 * t
        z = 2.3 + 0.7 * sin(pi * (t - 0.5))
        positions.append((round(x, 12), round(y, 12), round(z, 12)))
    return tuple(positions)


def generate_playback_frames(
    frame_count: int = DEFAULT_PLAYBACK_FRAME_COUNT,
    output_directory: str | Path = DEFAULT_PLAYBACK_OUTPUT_DIRECTORY,
    duration_seconds: float = DEFAULT_PLAYBACK_DURATION_SECONDS,
) -> PlaybackResult:
    """Generate deterministic offline playback PNG frames."""
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must not be negative")

    resolved_output_directory = Path(output_directory)
    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    source_path = default_source_path(frame_count)
    array_poses = default_array_poses()
    frames: list[PlaybackFrame] = []
    for frame_index, source_xyz in enumerate(source_path):
        if frame_count == 1:
            time_seconds = 0.0
        else:
            time_seconds = frame_index * duration_seconds / (frame_count - 1)
        scenario_result = run_synthetic_scenario(ScenarioInput(source_xyz=source_xyz))
        output_path = resolved_output_directory / f"frame_{frame_index:03d}.png"
        render_static_scene(
            scenario_result,
            output_path,
            array_poses=array_poses,
            title=(
                "DTS-Simulation offline playback frame\n"
                f"frame {frame_index + 1}/{frame_count}, t={time_seconds:.2f}s"
            ),
            status_lines=(
                "offline PNG playback frame",
                "no interactive UI / real-time window",
                "no firing / hardware control",
                "broadband and multi-bin fusion deferred",
            ),
            extra_points=source_path,
        )
        frames.append(
            PlaybackFrame(
                frame_index=frame_index,
                time_seconds=time_seconds,
                true_source_xyz=source_xyz,
                scenario_result=scenario_result,
                output_path=output_path,
            )
        )
    return PlaybackResult(frames=tuple(frames), output_directory=resolved_output_directory)


def export_playback_gif(
    playback_result: PlaybackResult,
    output_path: str | Path = DEFAULT_PLAYBACK_GIF_OUTPUT_PATH,
    frames_per_second: int = DEFAULT_PLAYBACK_FRAMES_PER_SECOND,
) -> GifExportResult:
    """Export an animated GIF from existing generated PNG playback frames."""
    if not playback_result.frames:
        raise ValueError("playback_result must contain at least one frame")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = tuple(frame.output_path for frame in playback_result.frames)
    missing_paths = [path for path in image_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"cannot export GIF; missing frame files: {missing}")

    duration_milliseconds = int(round(1000 / frames_per_second))
    images = [Image.open(path) for path in image_paths]
    try:
        first, *rest = images
        first.save(
            resolved_output_path,
            save_all=True,
            append_images=rest,
            duration=duration_milliseconds,
            loop=0,
        )
    finally:
        for image in images:
            image.close()

    return GifExportResult(
        output_path=resolved_output_path,
        frame_count=len(image_paths),
        succeeded=resolved_output_path.exists() and resolved_output_path.stat().st_size > 0,
    )


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} m"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DTS-Simulation offline playback PNG frames."
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="also export outputs/dts_playback.gif from the generated PNG frames",
    )
    args = parser.parse_args()

    result = generate_playback_frames()
    print("DTS-Simulation offline playback PNG sequence generated")
    print(f"Frames generated: {len(result.frames)}")
    print(f"Output folder: {result.output_directory}")
    print(f"All frames valid: {result.all_frames_valid}")
    print(f"Max error distance: {_format_optional_metric(result.max_error_distance)}")
    print(f"Mean error distance: {_format_optional_metric(result.mean_error_distance)}")
    if args.gif:
        gif_result = export_playback_gif(result)
        print(f"GIF output path: {gif_result.output_path}")
        print(f"GIF frames used: {gif_result.frame_count}")
        print(f"GIF creation succeeded: {gif_result.succeeded}")
    print("Interactive UI / real-time animation window: deferred")
    print("Launcher/Nerf firing and hardware control: not implemented")
    print("Broadband drone handling and multi-bin fusion: deferred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
