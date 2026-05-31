# Visual Simulation Interface Contract

## Purpose

This contract defines what the future DTS-Simulation visual layer will consume
and display. It exists so graphics can be added without duplicating or
rewriting the validated algorithmic pipeline.

The visual simulation is a presentation layer over scenario data. It is not a
signal-processing, localization, launcher-control, or hardware layer.

## Scope

The first visual implementation should render one static, single-shot scene
from one completed synthetic scenario. It should show the current algorithmic
state clearly enough for demonstration and review:

- three fixed microphone arrays;
- eight microphones per array;
- per-array MUSIC direction rays;
- true source/drone position;
- estimated source position;
- localization error distance;
- a virtual radar/launcher aiming at the estimated position;
- labels showing implemented and deferred behavior.

This contract does not authorize graphical implementation, animation, drone
path simulation, firing behavior, real hardware control, broadband drone
handling, or multi-bin MUSIC fusion.

## Source-Of-Truth Data Flow

The visual layer must consume scenario data produced by the existing
non-graphical runner:

```text
ScenarioInput
  -> run_synthetic_scenario(...)
  -> ScenarioResult
  -> visual adapter / renderer
  -> static visual scene
```

The visual layer must not reimplement:

- synthetic signal generation;
- MUSIC DOA estimation;
- frequency-bin selection;
- localization;
- quality scoring;
- virtual aim target selection.

The source-of-truth objects for the first visual implementation are:

- `ScenarioInput` from `src/dts_simulation/simulation.py`;
- `ScenarioResult` from `src/dts_simulation/simulation.py`;
- `ArrayPose` from `src/dts_simulation/models.py`;
- `DirectionEstimate` from `src/dts_simulation/models.py`;
- `LocalizationResult` from `src/dts_simulation/models.py`;
- `VirtualLauncher.target_xyz` semantics from
  `src/dts_simulation/launcher.py`.

If a future renderer needs a flatter structure, add a thin serialization or
visual-snapshot adapter around `ScenarioResult`. That adapter must preserve the
same values and must not run algorithms itself.

## Visual Input Model

For the first renderer, the visual input should be one immutable snapshot with
these fields, derived from `ScenarioResult` and the array poses used to create
it:

- `array_poses`: three `ArrayPose` values with `array_id` and `position_xyz`.
- `microphone_geometry`: eight local 3D microphone coordinates per array,
  using the current default two-layer simulation geometry unless a future task
  supplies a different geometry.
- `true_source_xyz`: the true source/drone position from `ScenarioResult`.
- `estimated_source_xyz`: `ScenarioResult.localization_result.estimated_xyz`
  when localization is valid.
- `error_distance`: `ScenarioResult.error_distance` when available.
- `direction_estimates`: one `DirectionEstimate` per array.
- `doa_rays`: derived visually from each `ArrayPose.position_xyz` plus the
  corresponding `DirectionEstimate` azimuth/elevation vector.
- `virtual_aim_target_xyz`: `ScenarioResult.virtual_aim_target_xyz`.
- `scenario_valid`: `ScenarioResult.valid`.
- `status_message`: `ScenarioResult.status_message`.
- `deferred_scope_labels`: fixed text for broadband/multi-bin fusion,
  graphical animation, launcher firing, and real hardware control.

The renderer may compute ray endpoints for drawing from source-of-truth
directions, but those endpoints are visual geometry only and must not feed back
into localization.

## Required Display Elements

The first static scene must display:

- World axes: `+x` right, `+y` forward, `+z` upward.
- Three array positions with array labels: `array-0`, `array-1`, `array-2`.
- Eight microphones per array, shown as points or small markers around each
  array position using the simulation geometry.
- One DOA ray per valid direction estimate, starting at the matching array
  position and extending toward the estimated direction.
- Invalid DOA estimates, if present, as dimmed or dashed markers with a clear
  invalid label.
- True source/drone position with a distinct marker.
- Estimated source position with a distinct marker.
- Error distance text, in meters.
- Virtual radar/launcher position and an aim line toward
  `virtual_aim_target_xyz` when available.
- Status labels explaining that current audio processing is synthetic,
  narrowband, and simulation-only.
- Deferred labels for broadband drone handling, multi-bin fusion, animation,
  firing behavior, and real hardware control.

## Coordinate And Screen Mapping Assumptions

Simulation coordinates use the existing 3D contract:

- `+x` points right.
- `+y` points forward.
- `+z` points upward.
- Units are meters.

The first visual implementation may use either:

- a static 3D Matplotlib scene with `x`, `y`, and `z` axes; or
- a 2.5D projection that maps `x` to horizontal screen position, `y` to depth
  or vertical plane position, and `z` to displayed height.

Recommended first implementation:

- Use a static Matplotlib 3D figure if the project stays in Python and the
  dependency is explicitly approved in a later implementation task.
- Keep equal or visibly comparable axis scaling so ray direction and source
  placement are not misleading.
- Use a fixed camera angle that shows the triangular array layout and source
  height.
- Draw text labels outside dense markers where possible.
- Treat microphone geometry as local coordinates added to each
  `ArrayPose.position_xyz`; array orientation rotation remains deferred until
  implemented by the algorithm layer.

If a 2.5D view is chosen instead, it must state its projection in the title or
legend so users do not mistake it for a full 3D rendering.

## Static Scene First-Phase Definition

The first visual implementation should be a static single-shot scene generated
from one `ScenarioResult`.

The first renderer should:

1. Call or receive `run_synthetic_scenario(...)`.
2. Convert the returned `ScenarioResult` plus the known array poses into a
   visual snapshot.
3. Render one figure.
4. Save or show that figure according to the future task requirements.

It should not create a real-time loop, animation timeline, interactive
controls, or drone path playback in the first phase.

## Deferred Animation And Drone-Path Behavior

Moving drone/source behavior is deferred. A future step-based playback may use
a sequence of snapshots:

```text
ScenarioFrame[0..n]
  each frame: ScenarioInput -> ScenarioResult -> visual snapshot
```

Future extension points:

- source path as a list of `Point3D` positions;
- timestamp or frame index;
- per-frame direction estimates;
- per-frame localization result;
- per-frame virtual aim target;
- optional trail for true and estimated positions;
- optional history of error distance.

Animation must still consume scenario results. It must not bypass or duplicate
the algorithm pipeline.

## Virtual Nerf / Launcher Visual Boundary

The launcher is visual-only.

Allowed visual behavior:

- display a radar/launcher marker;
- display an aim line from the launcher marker to
  `ScenarioResult.virtual_aim_target_xyz`;
- display a label such as `virtual aim target`;
- optionally display a simulated projectile only in a later explicitly scoped
  task.

Forbidden launcher behavior:

- no real firing logic;
- no serial, GPIO, USB, network, or device control;
- no physical actuation;
- no safety-critical claims;
- no implication that the virtual aim target is physically calibrated.

## Forbidden Behavior

The visual layer must not:

- implement or modify MUSIC;
- implement or modify localization;
- generate microphone signals;
- select frequency bins;
- compute source position from rays;
- add graphics dependencies without a future implementation task;
- implement drone animation in this contract task;
- implement launcher/Nerf firing behavior;
- control real hardware;
- claim real-world accuracy or validation;
- hide that the current scenario is synthetic and narrowband.

## Acceptance Criteria For Future Visual Implementation

A future first visual implementation is acceptable when:

- it consumes `ScenarioResult` or a documented adapter derived from it;
- it renders one static scene from one scenario;
- it shows three arrays, eight microphones per array, DOA rays, true source,
  estimated source, error distance, and virtual aim target;
- it labels implemented and deferred behavior clearly;
- it does not duplicate algorithm logic;
- it has a deterministic non-interactive validation path, such as verifying
  that a figure file is produced from a known scenario;
- it introduces any graphics dependency explicitly and minimally;
- it does not include real hardware or firing behavior.

## Risks And Open Decisions

- Matplotlib is a reasonable first Python graphics dependency, but it is not
  added by this contract.
- Static 3D plots can distort distance perception if axis scaling is not
  handled carefully.
- 2.5D views may be easier to read but must disclose their projection.
- Local-to-world array orientation is still deferred, so microphone markers
  should use local geometry translated to array position until orientation
  support exists.
- The current scenario is synthetic, narrowband, and far-field; visual output
  must not imply broadband or real-world validation.
- UI styling, colors, labels, export path, and file format remain future
  implementation decisions.
