# Direction of Arrival Implementation Contract

## Purpose

This contract defines the technical conventions required to implement
frequency-domain MUSIC-based three-dimensional direction of arrival (DOA)
estimation for DTS-Simulation. It is a simulation contract, not a claim about
physical hardware or measured accuracy.

## Scope

This document governs the first 3D DOA implementation for one microphone array
at a time. It fixes the coordinate, direction, geometry, sampling,
frequency-domain, scan, and output conventions needed before implementing the
MUSIC estimator.

It does not define final multi-array localization, graphical presentation,
hardware control, or realistic broadband drone-noise modeling.

## Team 1 Boundary

Team 1 receives raw samples from `3` arrays and independently estimates one
3D direction for each array:

```text
SystemSampleTensor (3 x 8 x 48000)
  -> ArraySampleMatrix (8 x 48000), processed independently per array
  -> DirectionEstimate(array_id, azimuth_deg, elevation_deg, quality_score, valid)
```

The later localization stage consumes the three direction estimates and
combines them into an estimated 3D source position. That localization stage is
not part of the first MUSIC implementation.

## Input Contracts

- `ArraySampleMatrix`: `8 x 48000`, in `microphone x sample` order, for one
  array and one one-second capture.
- `SystemSampleTensor`: `3 x 8 x 48000`, in `array x microphone x sample`
  order.
- `MicrophoneArrayConfig`: identifies an array and supplies its world
  position, orientation, microphone count, and 3D microphone coordinates.
- Raw samples remain time-domain numeric values at the module boundary.
- The DOA module must convert time-domain samples to a frequency-domain
  representation before applying MUSIC.

## Output Contracts

Each processed array returns one `DirectionEstimate`:

- `array_id`: identifier of the producing array.
- `azimuth_degrees`: DOA azimuth using the convention below, normalized to
  `[0, 360)`.
- `elevation_degrees`: DOA elevation using the convention below, in
  `[-90, 90]`.
- `quality_score`: normalized peak-dominance or equivalent deterministic
  score in `[0, 1]`, or `None` when no score can be produced.
- `valid`: indicates whether the estimator accepted the direction result.

The estimator must not present `quality_score` as a calibrated probability,
physical confidence, or real-world accuracy measure.

## Coordinate System

- The implementation target is a three-dimensional world.
- `+x` points right.
- `+y` points forward.
- `+z` points upward.
- Positions and microphone geometry are expressed in meters.

## Azimuth Convention

- `azimuth_degrees` is measured in the `x-y` plane.
- `0` degrees points along world `+x`.
- Positive azimuth rotates counterclockwise toward world `+y`.
- Results are normalized to `[0, 360)`.
- Example directions: `90` degrees points along `+y`; `180` degrees points
  along `-x`; `270` degrees points along `-y`.

## Elevation Convention

- `elevation_degrees` is measured above or below the `x-y` plane.
- Valid range is `[-90, 90]`.
- `0` degrees means horizontal in the `x-y` plane.
- `+90` degrees means straight upward along `+z`.
- `-90` degrees means straight downward along `-z`.

## Microphone Array Geometry

A flat circular or planar 8-microphone layout is not sufficient as the default
contract for stable elevation estimation, because all microphones lie in one
plane and cannot provide the same vertical aperture as a non-planar geometry.

Real hardware geometry is not known yet. For simulation, each array uses the
following explicit 3D default geometry unless a later work item replaces it
with measured hardware coordinates:

- `8` microphones arranged as two parallel square layers.
- Lower layer: `4` microphones on a square at
  `z = -layer_spacing / 2`.
- Upper layer: `4` microphones on a square at
  `z = +layer_spacing / 2`.
- Lower square half-width assumption: `0.045 m`.
- Layer spacing assumption: `0.045 m`.
- Lower layer coordinates use the square corners:
  `(half_width, half_width)`, `(-half_width, half_width)`,
  `(-half_width, -half_width)`, `(half_width, -half_width)`.
- Upper layer is rotated `45` degrees relative to the lower layer in the
  `x-y` plane, using the same radial distance from the origin as the lower
  square corners.
- Microphone indices `0` through `3` are the lower layer, counterclockwise.
- Microphone indices `4` through `7` are the upper rotated layer,
  counterclockwise.

This geometry is a simulation assumption, not a verified hardware
specification.

## Sampling Assumptions

- Sample rate: `48000 Hz`.
- Capture duration: `1 s`.
- Samples per microphone per activation: `48000`.
- Assumed speed of sound: `343.0 m/s`.
- Raw input is time-domain data.
- Initial synthetic validation may use a narrowband tone.
- Broadband drone acoustics, environmental reflections, and calibrated noise
  models are deferred.

## MUSIC / Frequency-Domain Assumptions

- MUSIC is narrowband per analyzed frequency.
- The DOA module must transform time-domain input into frequency-domain data
  using FFT/STFT or an equivalent method before applying MUSIC.
- The first implementation may validate with a synthetic narrowband tone by
  selecting the known dominant FFT bin or by accepting an internal narrowband
  complex snapshot after conversion.
- The first implementation targets 3D azimuth/elevation scanning per array.
- Initial azimuth scan resolution: `1` degree.
- Initial elevation scan resolution: `1` degree.
- Azimuth candidates cover `[0, 360)`.
- Elevation candidates cover `[-90, 90]`.
- Future broadband implementation should combine MUSIC spectra or estimates
  across selected frequency bins or bands.
- Drone-like audio is broadband and complex; handling it remains deferred
  unless explicitly implemented later.

This contract does not implement MUSIC, select covariance/windowing details,
define bin-combination logic, or assert estimator performance.

## Quality-Score Definition

For the initial MUSIC implementation, `quality_score` is a normalized,
confidence-like simulation metric in `[0, 1]` derived from relative
peak-dominance in the MUSIC spectrum or an equivalent deterministic metric.

The future estimator must document the precise formula before behavior is
accepted. The score is intended for comparing ambiguity within the simulation
and is not a calibrated probability, measured accuracy guarantee, or
hardware-quality assessment.

## Deferred Items

- Exact narrowband validation tone frequency and signal-to-noise cases.
- FFT/STFT framing, windowing, covariance, snapshot, and selected-bin logic.
- Multi-bin and broadband MUSIC spectrum or estimate combination.
- Exact peak-dominance score formula and invalid-result threshold.
- Real hardware microphone geometry.
- Multi-array 3D source localization.
- Broadband drone-noise and propagation modeling.
- Graphical simulation and virtual launcher animation.
- Any real hardware integration or control.

## Acceptance Criteria For Future MUSIC Implementation

- Accepts one valid `ArraySampleMatrix` and one compatible
  `MicrophoneArrayConfig` for a single array.
- Converts time-domain samples to a frequency-domain representation before
  applying MUSIC.
- Returns one `DirectionEstimate` with azimuth normalized to `[0, 360)` and
  elevation in `[-90, 90]`.
- Uses the documented `8`-microphone two-layer simulation geometry or an
  explicitly supplied compatible 3D configuration.
- Handles the specified narrowband synthetic simulation inputs at `48000 Hz`
  and `1 s` duration.
- Scans azimuth/elevation with the documented initial resolution unless a
  justified contract revision is recorded.
- Produces a documented peak-dominance or equivalent `quality_score` in
  `[0, 1]` without representing it as physical confidence or real-world
  validation.
- Includes deterministic tests for known synthetic azimuth/elevation
  directions and invalid or ambiguous input handling.

## Open Risks

- The `0.045 m` half-width and `0.045 m` layer spacing are unverified
  simulation assumptions and may change if hardware constraints are supplied.
- The two-layer square geometry is designed to make 3D simulation feasible,
  but it is not validated hardware.
- Narrowband tests may not predict behavior for broadband or noisy drone audio.
- Frequency-bin selection and multi-bin combination are major implementation
  choices that can affect accuracy and stability.
- The quality-score formula and validity threshold remain implementation
  decisions and can affect downstream localization behavior.
- World placement and orientation conventions must be applied consistently by
  later localization and visualization stages.

