# Animation Playback Contract

## Purpose

This contract defines how a future moving-drone visual simulation will generate
and consume a deterministic sequence of scenario snapshots.

The animation layer is a presentation consumer. It must consume
`ScenarioResult`-like frame data produced by the existing non-graphical
pipeline, and it must not reimplement signal generation, MUSIC, localization,
quality scoring, virtual aim target selection, launcher firing, or hardware
control.

This is a simulation contract only. It does not claim real-world accuracy,
hardware readiness, or validated physical performance.

## Scope

The first animation implementation should show a short deterministic 3D source
path over time. For each frame, the non-graphical pipeline should run the same
single-shot flow already used by the static renderer:

```text
source path point
  -> ScenarioInput(source_xyz=...)
  -> run_synthetic_scenario(...)
  -> ScenarioResult
  -> playback frame snapshot
  -> animation / export renderer
```

The future visual animation displays those snapshots over time. It does not own
the algorithms that produce them.

This contract does not implement animation, interactivity, drone path
rendering, launcher/Nerf firing visuals, real hardware control, broadband drone
handling, multi-bin MUSIC fusion, or local-to-world array orientation rotation.

## Source-Of-Truth Data Flow

Each playback frame must be derived from one completed non-graphical scenario
run:

1. Select the deterministic source position for the frame.
2. Build `ScenarioInput` with `source_xyz` set to that position.
3. Use the existing array poses unless a later contract revision supplies
   different simulation-only poses.
4. Generate synthetic samples for each array through the scenario runner.
5. Run MUSIC DOA per array through the scenario runner.
6. Run 3D localization through the scenario runner.
7. Set the virtual aim target to the estimated source position when
   localization is valid.
8. Store the resulting values in a playback frame snapshot.

The playback layer may call the scenario runner repeatedly, or it may consume a
precomputed list of frame snapshots. It must not bypass the runner by
calculating directions or source positions itself.

## Playback Data Model

The future implementation may add a thin immutable dataclass such as
`PlaybackFrame` only if code needs a flatter structure. Until then, this
contract defines the required fields:

- `frame_index`: zero-based integer frame number.
- `time_seconds`: deterministic playback time for the frame.
- `true_source_xyz`: source/drone position for this frame, in meters.
- `direction_estimates`: tuple of per-array `DirectionEstimate` outputs from
  the scenario result.
- `localization_result`: the frame's `LocalizationResult`.
- `virtual_aim_target_xyz`: estimated source position when localization is
  valid, otherwise `None`.
- `valid`: frame status derived from `ScenarioResult.valid`.
- `status_message`: short status text, such as `ok` or the localization
  failure reason.
- `error_distance`: distance between true and estimated source positions when
  available.
- `scenario_result`: optional reference to the source `ScenarioResult` when a
  renderer can consume the existing object directly.

The frame snapshot may also carry `array_poses` and microphone geometry when
that avoids implicit renderer state. Those values must match the scenario data
used to generate the frame.

## Default Drone / Source Path

The first playback path should be deterministic, short, non-degenerate, and
easy to view with the existing default triangular array layout.

Recommended default path:

```text
start_xyz = (1.0, 0.8, 1.6)
end_xyz   = (3.0, 2.5, 3.0)

for frame i in [0, frame_count - 1]:
    t = i / (frame_count - 1)
    x = 1.0 + 2.0 * t
    y = 0.8 + 1.7 * t
    z = 2.3 + 0.7 * sin(pi * (t - 0.5))
```

This path stays inside a compact visible region above the default array
triangle, changes all three coordinates, and avoids passing directly through
an array origin. It is a display path, not real-time drone physics.

The future implementation should use the same narrowband synthetic assumptions
as `run_synthetic_scenario` unless a later bounded task changes the audio
model.

## Frame Count And Timing

The first implementation should use `30` frames by default.

Acceptable first range: `20` to `40` frames. This is enough to show motion
without making Matplotlib rendering unnecessarily slow. If validation shows
runtime or file size problems, use `20` frames before changing algorithms or
adding dependencies.

Recommended playback timing:

- `frame_count = 30`
- `duration_seconds = 3.0`
- `time_seconds = frame_index * duration_seconds / (frame_count - 1)`
- `frames_per_second = 10` for GIF or preview playback

The simulation itself is step-based. It does not require real-time physics,
wall-clock timing, or a live control loop.

## First Export Format

Because the project already uses Matplotlib for the static renderer, the first
animation implementation should prefer one of these output paths:

- frame PNG sequence first, for deterministic validation and easy inspection;
- GIF second, if Matplotlib and the local environment can write it without
  adding dependencies.

MP4 and an interactive window should remain deferred until explicitly scoped,
because they may require additional local encoders, UI decisions, or runtime
dependencies.

Recommended first output layout:

```text
outputs/playback_frames/frame_000.png
outputs/playback_frames/frame_001.png
...
```

Generated playback outputs should remain under ignored `outputs/` unless a
later task explicitly changes artifact handling.

## Per-Frame Visual Updates

The future animation renderer must update these elements for each frame:

- true drone/source position;
- estimated source position when localization is valid;
- per-array DOA rays from the current `DirectionEstimate` values;
- virtual launcher aim line and target;
- localization error segment when both true and estimated positions are
  available;
- error text in meters;
- valid/status text.

The renderer may keep these elements static across frames:

- array origins;
- microphone markers;
- world axes and camera angle;
- fixed labels for synthetic narrowband processing and deferred behavior.

Optional later visual elements:

- true source trail;
- estimated source trail;
- per-frame error history.

Those optional elements must still use already-produced frame data.

## Invalid Frame Handling

If a frame has invalid localization:

- draw the true source position;
- show invalid or dimmed DOA/estimate elements as appropriate;
- set `virtual_aim_target_xyz` to `None`;
- do not draw an aim line to an invented point;
- display `status_message`;
- keep playback running through later frames.

Invalid frames must not trigger fallback localization inside the animation
layer.

## Forbidden Behavior

The animation/playback layer must not:

- generate microphone signals outside the scenario runner;
- implement or modify MUSIC;
- select frequency bins;
- implement or modify localization;
- compute estimated source position from DOA rays;
- duplicate virtual aim target selection;
- implement real firing logic;
- implement launcher/Nerf projectile behavior unless a later bounded visual
  task explicitly authorizes a visual-only projectile;
- control serial, GPIO, USB, network, motors, launchers, or other hardware;
- add telemetry, secrets, or network dependencies;
- claim broadband drone handling;
- claim multi-bin MUSIC fusion;
- claim real-world accuracy or validation;
- hide that the pipeline is synthetic, narrowband, and simulation-only.

## Acceptance Criteria For Future Implementation

A future playback implementation is acceptable when:

- it produces or consumes a list of `ScenarioResult`-derived frame snapshots;
- each frame has frame index, time, true source position, direction estimates,
  localization result, virtual aim target, validity/status text, and error
  distance;
- the default path is deterministic and non-degenerate;
- the default frame count is within `20` to `40` frames, preferably `30`;
- output is a PNG frame sequence or GIF unless a later task authorizes another
  format;
- visual updates are based only on frame snapshot values;
- no algorithm behavior is duplicated in visualization;
- no launcher firing or hardware behavior is added;
- existing tests and static rendering behavior remain valid.

## Risks And Open Decisions

- Re-running MUSIC for every frame may be slow with a fine scan grid; the
  first playback should keep the existing bounded scenario scan settings unless
  performance is measured and a separate optimization task is authorized.
- GIF support may depend on local Matplotlib writer availability; PNG sequence
  export is the safest first validation target.
- Static 3D camera framing must include the whole source path, not only the
  current frame's points.
- Local-to-world array orientation remains deferred, so playback continues to
  treat `DirectionEstimate` values as world-relative.
- The source path is a deterministic visual scenario, not a physical drone
  dynamics model.
