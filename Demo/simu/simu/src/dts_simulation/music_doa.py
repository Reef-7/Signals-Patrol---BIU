"""Bounded 3D narrowband frequency-domain MUSIC estimator.

This implementation is intentionally limited to one microphone array, one
dominant frequency bin, and synthetic/narrowband validation inputs. It does not
handle broadband drone audio, multi-bin fusion, localization, or visualization.
The quality score is a simulation-only peak-dominance metric:

    (best_spectrum_value - median_spectrum_value) / best_spectrum_value

clipped to [0, 1]. The result is valid only when the selected frequency-bin
snapshot has non-trivial energy and the quality score meets the configured
minimum threshold. It is not a calibrated probability or physical confidence.
"""

from dataclasses import dataclass

import numpy as np

from .config import (
    AZIMUTH_MAX_DEGREES_EXCLUSIVE,
    AZIMUTH_MIN_DEGREES,
    ELEVATION_MAX_DEGREES,
    ELEVATION_MIN_DEGREES,
    MUSIC_AZIMUTH_SCAN_STEP_DEGREES,
    MUSIC_ELEVATION_SCAN_STEP_DEGREES,
    MUSIC_MIN_SNAPSHOT_NORM,
    MUSIC_VALID_QUALITY_THRESHOLD,
    QUALITY_SCORE_MAX,
    QUALITY_SCORE_MIN,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_METERS_PER_SECOND,
)
from .models import ArraySampleMatrix, DirectionEstimate, MicrophoneArrayConfig


@dataclass(frozen=True, slots=True)
class MusicScanConfig:
    """Grid settings for bounded 3D MUSIC scanning."""

    azimuth_step_degrees: float = MUSIC_AZIMUTH_SCAN_STEP_DEGREES
    elevation_step_degrees: float = MUSIC_ELEVATION_SCAN_STEP_DEGREES

    def __post_init__(self) -> None:
        if self.azimuth_step_degrees <= 0.0:
            raise ValueError("azimuth_step_degrees must be positive")
        if self.elevation_step_degrees <= 0.0:
            raise ValueError("elevation_step_degrees must be positive")


class MusicDoaEstimator:
    """Estimate one array's 3D DOA from time-domain samples via one FFT bin."""

    def __init__(
        self,
        frequency_hz: float | None = None,
        scan_config: MusicScanConfig | None = None,
    ) -> None:
        self.frequency_hz = frequency_hz
        self.scan_config = scan_config or MusicScanConfig()

    def estimate(
        self, samples: ArraySampleMatrix, configuration: MicrophoneArrayConfig
    ) -> DirectionEstimate:
        sample_matrix = np.asarray(samples.samples, dtype=float)
        frequency_bins = np.fft.rfftfreq(sample_matrix.shape[1], d=1.0 / SAMPLE_RATE_HZ)
        spectrum = np.fft.rfft(sample_matrix, axis=1)
        bin_index = self._select_frequency_bin(spectrum, frequency_bins)
        frequency_hz = float(frequency_bins[bin_index])
        snapshot = spectrum[:, bin_index]
        if float(np.linalg.norm(snapshot)) <= MUSIC_MIN_SNAPSHOT_NORM:
            return DirectionEstimate(
                array_id=configuration.array_id,
                azimuth_degrees=0.0,
                elevation_degrees=0.0,
                quality_score=0.0,
                valid=False,
            )
        covariance = np.outer(snapshot, np.conjugate(snapshot))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        signal_subspace_size = 1
        noise_subspace = eigenvectors[:, : sample_matrix.shape[0] - signal_subspace_size]

        coordinates = np.asarray(configuration.microphone_coordinates, dtype=float)
        azimuths, elevations, directions = _candidate_grid(self.scan_config)
        steering = _steering_vectors(coordinates, directions, frequency_hz)
        projection = steering.conjugate() @ noise_subspace
        denominator = np.sum(np.abs(projection) ** 2, axis=1)
        denominator = np.maximum(denominator, np.finfo(float).eps)
        music_spectrum = 1.0 / denominator
        best_index = int(np.argmax(music_spectrum))
        best_value = float(music_spectrum[best_index])
        median_value = float(np.median(music_spectrum))
        quality_score = 0.0
        if best_value > 0.0:
            quality_score = (best_value - median_value) / best_value
        quality_score = float(np.clip(quality_score, QUALITY_SCORE_MIN, QUALITY_SCORE_MAX))
        valid = quality_score >= MUSIC_VALID_QUALITY_THRESHOLD

        return DirectionEstimate(
            array_id=configuration.array_id,
            azimuth_degrees=float(azimuths[best_index] % AZIMUTH_MAX_DEGREES_EXCLUSIVE),
            elevation_degrees=float(elevations[best_index]),
            quality_score=quality_score,
            valid=valid,
        )

    def _select_frequency_bin(self, spectrum: np.ndarray, frequency_bins: np.ndarray) -> int:
        """Select the expected bin when configured, otherwise strongest non-DC bin."""
        if self.frequency_hz is not None:
            bin_index = int(np.argmin(np.abs(frequency_bins - self.frequency_hz)))
            if bin_index == 0 and self.frequency_hz > 0.0 and frequency_bins.shape[0] > 1:
                return 1
            return bin_index
        magnitudes = np.sum(np.abs(spectrum), axis=0)
        if magnitudes.shape[0] <= 1:
            raise ValueError("frequency-domain spectrum must contain non-DC bins")
        return int(np.argmax(magnitudes[1:]) + 1)


def _candidate_grid(scan_config: MusicScanConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_values = np.arange(
        AZIMUTH_MIN_DEGREES,
        AZIMUTH_MAX_DEGREES_EXCLUSIVE,
        scan_config.azimuth_step_degrees,
        dtype=float,
    )
    elevation_values = np.arange(
        ELEVATION_MIN_DEGREES,
        ELEVATION_MAX_DEGREES + scan_config.elevation_step_degrees,
        scan_config.elevation_step_degrees,
        dtype=float,
    )
    azimuth_grid, elevation_grid = np.meshgrid(
        azimuth_values, elevation_values, indexing="xy"
    )
    azimuths = azimuth_grid.ravel()
    elevations = elevation_grid.ravel()
    azimuth_radians = np.deg2rad(azimuths)
    elevation_radians = np.deg2rad(elevations)
    directions = np.column_stack(
        (
            np.cos(elevation_radians) * np.cos(azimuth_radians),
            np.cos(elevation_radians) * np.sin(azimuth_radians),
            np.sin(elevation_radians),
        )
    )
    return azimuths, elevations, directions


def _steering_vectors(
    microphone_coordinates: np.ndarray, directions: np.ndarray, frequency_hz: float
) -> np.ndarray:
    delays = directions @ microphone_coordinates.T / SPEED_OF_SOUND_METERS_PER_SECOND
    phase = -2.0j * np.pi * frequency_hz * delays
    return np.exp(phase)
