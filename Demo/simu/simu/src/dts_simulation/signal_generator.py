"""Deterministic fixture data for scaffold and bounded MUSIC tests."""

from math import cos, pi, sin

import numpy as np

from .config import (
    ARRAY_COUNT,
    DEFAULT_TEST_TONE_FREQUENCY_HZ,
    MICROPHONES_PER_ARRAY,
    SAMPLE_COUNT,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_METERS_PER_SECOND,
)
from .models import ArraySampleMatrix, MicrophoneArrayConfig, SystemSampleTensor


def generate_silent_array_samples(value: float = 0.0) -> ArraySampleMatrix:
    """Generate one constant-valued input matrix for contract tests."""
    row = (float(value),) * SAMPLE_COUNT
    return ArraySampleMatrix(samples=(row,) * MICROPHONES_PER_ARRAY)


def generate_silent_system_samples(value: float = 0.0) -> SystemSampleTensor:
    """Generate a deterministic system tensor for scaffold smoke runs."""
    matrix = generate_silent_array_samples(value)
    return SystemSampleTensor(arrays=(matrix,) * ARRAY_COUNT)


def direction_unit_vector(azimuth_degrees: float, elevation_degrees: float) -> np.ndarray:
    """Convert contract azimuth/elevation degrees into a 3D unit vector."""
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    return np.array(
        [
            cos(elevation) * cos(azimuth),
            cos(elevation) * sin(azimuth),
            sin(elevation),
        ],
        dtype=float,
    )


def generate_narrowband_array_samples(
    configuration: MicrophoneArrayConfig,
    azimuth_degrees: float,
    elevation_degrees: float,
    frequency_hz: float = DEFAULT_TEST_TONE_FREQUENCY_HZ,
    amplitude: float = 1.0,
) -> ArraySampleMatrix:
    """Generate deterministic time-domain samples for one far-field tone.

    The signal is a synthetic, noise-free, narrowband plane wave used to
    validate the bounded MUSIC implementation. It is not a broadband drone
    acoustic model.
    """
    direction = direction_unit_vector(azimuth_degrees, elevation_degrees)
    microphone_positions = np.asarray(configuration.microphone_coordinates, dtype=float)
    delays = microphone_positions @ direction / SPEED_OF_SOUND_METERS_PER_SECOND
    sample_times = np.arange(SAMPLE_COUNT, dtype=float) / SAMPLE_RATE_HZ
    phase = 2.0 * pi * frequency_hz * (sample_times[None, :] - delays[:, None])
    samples = amplitude * np.cos(phase)
    return ArraySampleMatrix(
        samples=tuple(tuple(float(value) for value in row) for row in samples)
    )
