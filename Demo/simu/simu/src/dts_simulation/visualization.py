"""Static visual renderer for one DTS simulation scenario.

This module consumes ScenarioResult data from the non-graphical scenario
runner. It does not generate signals, run MUSIC, solve localization, animate,
fire launchers, or control hardware.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .array_geometry import default_simulation_microphone_coordinates
from .localization import direction_estimate_to_unit_vector
from .models import ArrayPose, Point3D
from .simulation import ScenarioResult, default_array_poses, run_synthetic_scenario

DEFAULT_VISUALIZATION_OUTPUT_PATH = Path("outputs") / "dts_static_scene.png"
DEFAULT_DOA_RAY_LENGTH_METERS = 4.0
DEFAULT_LAUNCHER_POSITION_XYZ: Point3D = (2.0, -1.0, 0.0)


def render_static_scene(
    scenario_result: ScenarioResult | None = None,
    output_path: str | Path = DEFAULT_VISUALIZATION_OUTPUT_PATH,
    array_poses: tuple[ArrayPose, ...] | None = None,
    title: str | None = None,
    status_lines: tuple[str, ...] = (),
    extra_points: tuple[Point3D, ...] = (),
) -> Path:
    """Render one deterministic static 3D scene and return the output path."""
    result = scenario_result or run_synthetic_scenario()
    poses = array_poses or default_array_poses()
    pose_by_id = {pose.array_id: pose for pose in poses}
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title(title or "DTS-Simulation static scenario\nsynthetic narrowband data only")
    axis.set_xlabel("+x right (m)")
    axis.set_ylabel("+y forward (m)")
    axis.set_zlabel("+z upward (m)")

    _draw_arrays(axis, poses)
    _draw_doa_rays(axis, result, pose_by_id)
    _draw_sources(axis, result)
    _draw_virtual_launcher(axis, result)
    _draw_status_text(axis, result, status_lines)
    _set_equal_axes(axis, poses, result, extra_points)
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

    figure.tight_layout()
    figure.savefig(resolved_output_path, dpi=150)
    plt.close(figure)
    return resolved_output_path


def _draw_arrays(axis, poses: tuple[ArrayPose, ...]) -> None:
    microphone_offsets = np.asarray(default_simulation_microphone_coordinates(), dtype=float)
    for pose in poses:
        position = np.asarray(pose.position_xyz, dtype=float)
        microphones = microphone_offsets + position
        axis.scatter(
            microphones[:, 0],
            microphones[:, 1],
            microphones[:, 2],
            color="tab:blue",
            s=18,
            alpha=0.8,
        )
        axis.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color="navy",
            marker="s",
            s=45,
            label="array origin" if pose == poses[0] else None,
        )
        axis.text(position[0], position[1], position[2] + 0.12, pose.array_id)


def _draw_doa_rays(axis, result: ScenarioResult, pose_by_id: dict[str, ArrayPose]) -> None:
    for estimate in result.direction_estimates:
        pose = pose_by_id.get(estimate.array_id)
        if pose is None:
            continue
        origin = np.asarray(pose.position_xyz, dtype=float)
        direction = np.asarray(direction_estimate_to_unit_vector(estimate), dtype=float)
        endpoint = origin + DEFAULT_DOA_RAY_LENGTH_METERS * direction
        line_style = "-" if estimate.valid else "--"
        alpha = 0.85 if estimate.valid else 0.35
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color="tab:orange",
            linestyle=line_style,
            alpha=alpha,
            label="MUSIC DOA ray" if estimate == result.direction_estimates[0] else None,
        )


def _draw_sources(axis, result: ScenarioResult) -> None:
    true_source = np.asarray(result.true_source_xyz, dtype=float)
    estimated = np.asarray(result.localization_result.estimated_xyz, dtype=float)
    axis.plot(
        [true_source[0], estimated[0]],
        [true_source[1], estimated[1]],
        [true_source[2], estimated[2]],
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="localization error",
    )
    axis.scatter(
        [true_source[0]],
        [true_source[1]],
        [true_source[2]],
        color="tab:green",
        marker="*",
        s=160,
        label="true source",
    )
    axis.scatter(
        [estimated[0]],
        [estimated[1]],
        [estimated[2]],
        color="tab:red",
        marker="X",
        s=90,
        label="estimated source",
    )
    axis.text(
        true_source[0] - 0.45,
        true_source[1],
        true_source[2] + 0.28,
        "true source",
    )
    axis.text(
        estimated[0] + 0.18,
        estimated[1] - 0.18,
        estimated[2] - 0.22,
        "estimated / aim target",
    )


def _draw_virtual_launcher(axis, result: ScenarioResult) -> None:
    launcher = np.asarray(DEFAULT_LAUNCHER_POSITION_XYZ, dtype=float)
    axis.scatter(
        [launcher[0]],
        [launcher[1]],
        [launcher[2]],
        color="black",
        marker="^",
        s=85,
        label="virtual launcher",
    )
    axis.text(launcher[0], launcher[1], launcher[2] + 0.12, "virtual launcher")
    if result.virtual_aim_target_xyz is None:
        return
    target = np.asarray(result.virtual_aim_target_xyz, dtype=float)
    axis.plot(
        [launcher[0], target[0]],
        [launcher[1], target[1]],
        [launcher[2], target[2]],
        color="black",
        linestyle=":",
        linewidth=1.6,
        label="virtual aim line",
    )


def _draw_status_text(
    axis,
    result: ScenarioResult,
    status_lines: tuple[str, ...] = (),
) -> None:
    error_text = "n/a" if result.error_distance is None else f"{result.error_distance:.3f} m"
    renderer_lines = status_lines or (
        "static renderer; no animation",
        "no firing / hardware control",
        "broadband and multi-bin fusion deferred",
    )
    status = (
        f"valid={result.valid}\n"
        f"error={error_text}\n"
        + "\n".join(renderer_lines)
    )
    axis.text2D(0.02, 0.02, status, transform=axis.transAxes)


def _set_equal_axes(
    axis,
    poses: tuple[ArrayPose, ...],
    result: ScenarioResult,
    extra_points: tuple[Point3D, ...] = (),
) -> None:
    points = [pose.position_xyz for pose in poses]
    points.append(result.true_source_xyz)
    points.append(result.localization_result.estimated_xyz)
    if result.virtual_aim_target_xyz is not None:
        points.append(result.virtual_aim_target_xyz)
    points.append(DEFAULT_LAUNCHER_POSITION_XYZ)
    points.extend(extra_points)
    data = np.asarray(points, dtype=float)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    padding = 0.6
    axis.set_xlim(center[0] - radius - padding, center[0] + radius + padding)
    axis.set_ylim(center[1] - radius - padding, center[1] + radius + padding)
    axis.set_zlim(center[2] - radius - padding, center[2] + radius + padding)


def main() -> int:
    output_path = render_static_scene()
    print(f"Static DTS scenario visualization written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
