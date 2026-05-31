# Multi-Array 3D Localization Contract

## Purpose

This contract defines the next processing layer after Team 1 direction of
arrival (DOA): combining three known microphone-array poses and three 3D
`DirectionEstimate` outputs into one estimated 3D source position.

This is a simulation contract. It does not claim real-world localization
accuracy, hardware readiness, or validated physical performance.

## Scope

The localization layer consumes already-computed direction estimates. It does
not process microphone samples, run MUSIC, control hardware, draw graphics, or
model broadband drone audio.

The future solver should estimate the 3D point that minimizes the total
weighted squared perpendicular distance to all usable DOA rays.

## Inputs

### Array Poses

Each array pose identifies the world-space origin of one DOA ray:

- `array_id`
- `position_x`
- `position_y`
- `position_z`
- optional orientation fields for future yaw/pitch/roll support

For the first implementation, `DirectionEstimate` is assumed to already be
expressed in world coordinates. Local-to-world rotation using array orientation
is deferred and must not be silently assumed.

### Direction Estimates

Each valid `DirectionEstimate` contributes one ray:

- Ray origin: the matching `ArrayPose` position.
- Ray direction: unit vector converted from `azimuth_degrees` and
  `elevation_degrees`.
- Ray weight: derived from `quality_score` if present and positive; otherwise
  fallback weight is `1.0` for valid directions.

Invalid directions must be ignored. Low-quality directions may be down-weighted
or rejected by the implementation, but the threshold must be documented before
use. `quality_score` is simulation confidence only, not a physical probability.

## Coordinate System

Localization uses the same world coordinate system as the DOA contract:

- `+x` points right.
- `+y` points forward.
- `+z` points upward.
- Positions are expressed in meters.

## Direction Conversion

Public angles are in degrees. Internal math may use radians.

```text
dx = cos(elevation) * cos(azimuth)
dy = cos(elevation) * sin(azimuth)
dz = sin(elevation)
```

`azimuth_degrees = 0` points along `+x`, positive azimuth rotates
counterclockwise toward `+y`, and `elevation_degrees = 0` lies in the `x-y`
plane. Positive elevation points upward toward `+z`.

## Ray Model

Each usable estimate forms a parametric ray:

```text
point(t) = array_position + t * direction_unit_vector
```

The future solver should treat the direction as a ray from the array toward the
source. If a least-squares line formulation is used, any handling of negative
ray parameters must be documented.

## Localization Result

The result contract contains:

- `estimated_xyz`
- `used_array_ids`
- `valid`
- `status_message`
- `residual_error`
- `quality_score`
- optional `true_xyz`
- optional `error_distance`
- contributing direction estimates

`residual_error` should represent the geometric ray-fit residual when the
solver is implemented. `quality_score`, if produced, remains simulation-only
and must not be interpreted as calibrated physical confidence.

## Minimum Valid Inputs

- The intended system design uses `3` valid directions from `3` arrays.
- At least `2` valid non-parallel rays are mathematically sufficient for a
  closest-point estimate.
- Fewer than `2` valid directions must produce an invalid localization result.
- Nearly parallel or ill-conditioned rays should produce invalid output or low
  confidence with a clear status message.
- Missing poses for valid direction estimates must produce invalid output.

## Quality Weighting

Initial weighting may use:

```text
weight = quality_score if valid and quality_score > 0 else 1.0
```

Only valid direction estimates should contribute. The fallback weight avoids
discarding a valid estimate solely because quality scoring is unavailable.
Quality values are relative simulation metrics, not physical probabilities.

## Default Simulation Array Layout

The default future test layout is simulation-only:

- `array-0`: `(0.0, 0.0, 0.0)`
- `array-1`: `(4.0, 0.0, 0.0)`
- `array-2`: `(2.0, 3.4641016151, 0.0)`

This is a non-collinear triangular layout in the `x-y` plane with all arrays at
`z = 0`. It is not verified hardware geometry and may change when real
placement data exists.

## Deferred Items

- Full least-squares ray intersection solver.
- Local-to-world orientation rotation using yaw/pitch/roll.
- Ill-conditioning thresholds and residual validity thresholds.
- Robust outlier rejection.
- Multi-array visualization.
- Real hardware placement and calibration.

## Acceptance Criteria For Future Implementation

- Accepts array poses and direction estimates.
- Uses only valid estimates with matching poses.
- Converts azimuth/elevation to unit vectors using this contract.
- Returns invalid output for fewer than two usable rays.
- Detects or reports nearly parallel/ill-conditioned ray sets.
- Produces `estimated_xyz`, `used_array_ids`, `residual_error`, `valid`, and
  `status_message`.
- Includes deterministic simulation tests with known source positions.
- Does not claim real-world accuracy.

## Open Risks

- The default triangular layout is simulation-only.
- Orientation support is deferred; this is safe only while direction estimates
  are already world-relative.
- Thresholds for parallel rays, low quality, and residual acceptance remain
  decisions for the implementation step.

