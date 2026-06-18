# Standalone Live 7-Microphone Direction Estimator

This project is a small Python prototype for estimating the approximate direction of a sound source from one UMA8-like 7-microphone array.

One microphone array can estimate direction of arrival. It does not provide distance, precise 3D position, triangulation, target lock, or impact point.

The original source artifact is kept local and is excluded from the repository.

## Architecture

```text
audio source -> latest sliding 7 x samples window -> band-limited frequency selection -> MUSIC consensus -> optional smoothing -> console output and optional visualization
```

Expected audio window shape:

```text
7 x samples
```

Rows are microphones and columns are time samples. The project uses one center microphone plus six microphones on a planar circle. The estimator requires exactly 7 microphone rows and at least 512 samples per row.

## What It Shows

The visualization is direction-only:

- 2D microphone layout
- estimated azimuth ray from the array center
- MUSIC elevation as text only, because the microphone geometry is planar
- experimental polar/elevation angle as text only when enough delay candidates are valid
- optional 3D direction ray view for azimuth plus experimental polar/elevation
- selected frequency text
- confidence text
- recent azimuth history
- simulated reference azimuth only in simulation mode

It does not show a drone position, distance, 3D coordinates, triangulated location, target lock, or firing point.

## Experimental Polar/Elevation

The main direction estimate is still MUSIC azimuth. Each audio window also computes an experimental polar/elevation angle from center-to-ring microphone delays:

```text
delta = speed_of_sound * delay
denominator = array_radius * cos(mic_angle - music_azimuth)
value = delta / denominator
candidate = arccos(value)
```

The estimator rejects a microphone when the denominator is too small, the delay is outside the physical array range, or the value is outside `[-1, 1]`. It averages the remaining candidates and reports how many microphones were valid plus a consistency-based confidence score.

This is an experimental planar-array polar-angle cue. It is not measured height, distance, 3D position, triangulation, target lock, or impact point.

## Azimuth-Only MUSIC

MUSIC can run in azimuth-only mode:

```powershell
python uma8_music.py --simulate --azimuth-only
```

In this mode, elevation is fixed at `0.0` degrees and MUSIC searches only the azimuth direction. This matches the direction ray shown in the visualization and avoids the slower elevation grid search.

For lower-latency live tracking, use azimuth-only mode with one frequency bin and no smoothing:

```powershell
python uma8_music.py --live --azimuth-only --no-smoothing --max-frequencies 1 --interval 0.02 --window-duration 0.02
```

## Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `sounddevice`
- `matplotlib`

## Safe No-Hardware Commands

Self-check:

```powershell
python uma8_music.py --self-check
```

Run simulated direction estimation:

```powershell
python uma8_music.py --simulate
```

Run simulated direction visualization:

```powershell
python uma8_music.py --2dvisualise --simulate
```

Run simulated 3D direction-ray visualization:

```powershell
python uma8_music.py --3dvisualise --simulate
```

The older `--visualize` flag is still accepted as a 2D visualization alias.

Set duration and update interval:

```powershell
python uma8_music.py --visualize --simulate --duration 3 --interval 0.5
```

Show the latest raw estimate without smoothing or history:

```powershell
python uma8_music.py --2dvisualise --simulate --duration 3 --interval 0.1 --window-duration 0.05 --no-smoothing --no-history
```

Show the raw azimuth-only estimate without smoothing or history:

```powershell
python uma8_music.py --2dvisualise --simulate --azimuth-only --duration 3 --interval 0.1 --window-duration 0.05 --no-smoothing --no-history --max-frequencies 1
```

Show the raw azimuth-only estimate with the 3D direction-ray view:

```powershell
python uma8_music.py --3dvisualise --simulate --azimuth-only --duration 3 --interval 0.1 --window-duration 0.05 --no-smoothing --no-history --max-frequencies 1
```

Tune the frequency band and number of MUSIC bins:

```powershell
python uma8_music.py --simulate --band-min 300 --band-max 3500 --max-frequencies 3
```

Selected frequency bins are combined by directional agreement. Bins that point far away from the strongest agreeing group are treated as outliers instead of being averaged into the displayed direction.

## Optional Live Commands

Live microphone direction estimate:

```powershell
python uma8_music.py --live
```

Live microphone direction visualization:

```powershell
python uma8_music.py --2dvisualise --live
python uma8_music.py --3dvisualise --live
```

Continuous live tracking uses a streaming ring buffer by default. It keeps recent audio in memory and repeatedly analyzes the latest window, instead of recording a full chunk and waiting before each estimate.

Use the older blocking live capture path only when needed:

```powershell
python uma8_music.py --visualize --live --blocking-live
```

Live capture is explicit opt-in only. Importing `uma8_music.py` never opens the microphone.

The live source expects an input device that exposes at least 7 channels at 48 kHz. If `sounddevice` is missing, no input device is available, OS permissions block capture, or the captured window has the wrong shape, the command prints a clear error.

Live microphone behavior has not been hardware validated unless a live command is explicitly run and reported on the target machine.

## Backward-Compatible Aliases

Older aliases are still accepted where reasonable:

```powershell
python uma8_music.py
python uma8_music.py --record
python uma8_music.py --track --simulate --duration 10
python uma8_music.py --track --live --duration 10 --interval 1
```

## Validation

```powershell
python --version
python -m py_compile .\uma8_music.py
python -m unittest
python uma8_music.py --self-check
python uma8_music.py --self-check --azimuth-only
python uma8_music.py --simulate
python uma8_music.py --simulate --azimuth-only
python uma8_music.py --2dvisualise --simulate --duration 3
python uma8_music.py --3dvisualise --simulate --duration 3
```

Do not run live commands as automated validation unless hardware testing is explicitly intended.

## Current Limitations

- The algorithm is a practical prototype, not a validated measurement system.
- The array model is planar, so elevation is displayed only as an estimator output with a limitation note.
- Azimuth-only MUSIC fixes elevation at `0.0` and only searches the displayed direction ray.
- Experimental polar/elevation is inferred from planar TDOA delay consistency and should be treated as a weak cue, not a reliable height or 3D direction measurement.
- The 3D visualization shows an orientation ray only. It does not show object position, range, height, triangulation, or target lock.
- Simulation uses generated narrowband tone windows, not realistic drone acoustics.
- The current filtering is frequency-bin selection inside a configurable band, not a full acoustic noise-suppression pipeline.
- Multi-frequency consensus rejects obvious directional outliers, but it is still a heuristic and depends on signal quality.
- Shorter windows and no smoothing improve responsiveness but can make estimates noisier or less reliable.
- There is no precise 3D position, distance, triangulation, target lock, or firing logic.
- There is no web app, API, deployment setup, or external service.
