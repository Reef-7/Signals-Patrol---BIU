"""Minimal import-safe UMA8 MUSIC direction estimator.

This module was converted from ``code.txt``. It keeps the original estimator
shape and live-recording behavior available, but live microphone capture is
only run when the CLI is called with ``--record``.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPECTED_MICROPHONES = 7
DEFAULT_BAND_MIN_HZ = 300.0
DEFAULT_BAND_MAX_HZ = 3500.0
DEFAULT_MAX_FREQUENCIES = 3
DEFAULT_CONSENSUS_DEGREES = 25.0


@dataclass(frozen=True)
class DirectionEstimate:
    """One azimuth/elevation estimate in degrees."""

    azimuth: float
    elevation: float
    frequency: Optional[float] = None
    reference_azimuth: Optional[float] = None
    reference_elevation: Optional[float] = None
    confidence: Optional[float] = None
    frequencies: Tuple[float, ...] = ()
    experimental_elevation: Optional[float] = None
    experimental_elevation_confidence: Optional[float] = None
    experimental_elevation_valid_mics: int = 0
    experimental_elevation_candidates: Tuple[float, ...] = ()


@dataclass(frozen=True)
class DirectionConsensus:
    """Consensus result from several frequency-specific MUSIC estimates."""

    azimuth: float
    elevation: float
    agreement: float
    frequencies: Tuple[float, ...]


@dataclass(frozen=True)
class ExperimentalElevationEstimate:
    """Experimental polar/elevation result inferred from planar TDOA delays."""

    elevation: Optional[float]
    confidence: float
    valid_mics: int
    candidates: Tuple[float, ...]


class UMA8MUSICEstimator:
    """MUSIC estimator for a UMA8-like 7-microphone circular layout.

    The original code uses one center microphone plus six microphones around a
    circle, even though the recorder requests eight input channels from the
    audio driver and then uses the first seven channels.
    """

    def __init__(self, fs: int = 48000, c: float = 343.0, radius: float = 0.0425):
        self.fs = fs
        self.c = c
        self.radius = radius
        self.num_mics = EXPECTED_MICROPHONES

        self.mic_coords = np.zeros((self.num_mics, 2))
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for i, angle in enumerate(angles):
            self.mic_coords[i + 1, 0] = self.radius * np.cos(angle)
            self.mic_coords[i + 1, 1] = self.radius * np.sin(angle)

    def select_target_frequency(
        self,
        audio_row: np.ndarray,
        band_min: float = DEFAULT_BAND_MIN_HZ,
        band_max: float = DEFAULT_BAND_MAX_HZ,
    ) -> float:
        """Select the strongest frequency in the configured band."""
        return self.select_target_frequencies(
            audio_row,
            band_min=band_min,
            band_max=band_max,
            max_frequencies=1,
        )[0]

    def select_target_frequencies(
        self,
        audio_row: np.ndarray,
        band_min: float = DEFAULT_BAND_MIN_HZ,
        band_max: float = DEFAULT_BAND_MAX_HZ,
        max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    ) -> Tuple[float, ...]:
        """Select the strongest FFT bins inside the configured band."""
        validate_frequency_band(band_min, band_max)
        if max_frequencies < 1:
            raise ValueError("max_frequencies must be at least 1")

        f_axis = np.fft.rfftfreq(len(audio_row), d=1 / self.fs)
        fft_mag = np.abs(np.fft.rfft(audio_row))

        valid_mask = (f_axis >= band_min) & (f_axis <= band_max)

        if not np.any(valid_mask):
            return (float(f_axis[np.argmax(fft_mag)]),)

        valid_indices = np.flatnonzero(valid_mask)
        valid_magnitudes = fft_mag[valid_indices]
        count = min(max_frequencies, len(valid_indices))
        strongest = np.argpartition(valid_magnitudes, -count)[-count:]
        ordered = strongest[np.argsort(valid_magnitudes[strongest])[::-1]]
        return tuple(float(f_axis[valid_indices[idx]]) for idx in ordered)

    def _select_target_frequency(self, audio_row: np.ndarray) -> float:
        """Backward-compatible alias from the original code."""
        return self.select_target_frequency(audio_row)

    def frequency_weights(
        self,
        audio_row: np.ndarray,
        frequencies: Sequence[float],
    ) -> np.ndarray:
        f_axis = np.fft.rfftfreq(len(audio_row), d=1 / self.fs)
        fft_mag = np.abs(np.fft.rfft(audio_row))
        weights = []
        for frequency in frequencies:
            freq_idx = int(np.argmin(np.abs(f_axis - frequency)))
            weights.append(float(fft_mag[freq_idx]))
        weights_array = np.asarray(weights, dtype=float)
        if not np.any(weights_array):
            return np.ones(len(frequencies), dtype=float)
        return weights_array

    def band_confidence(
        self,
        audio_row: np.ndarray,
        frequencies: Sequence[float],
        band_min: float = DEFAULT_BAND_MIN_HZ,
        band_max: float = DEFAULT_BAND_MAX_HZ,
    ) -> float:
        """Return a simple 0..1 peak-energy confidence score for the selected band."""
        validate_frequency_band(band_min, band_max)
        f_axis = np.fft.rfftfreq(len(audio_row), d=1 / self.fs)
        power = np.abs(np.fft.rfft(audio_row)) ** 2
        band_mask = (f_axis >= band_min) & (f_axis <= band_max)
        band_power = float(np.sum(power[band_mask]))
        if band_power <= 0.0:
            return 0.0

        selected_power = 0.0
        for frequency in frequencies:
            freq_idx = int(np.argmin(np.abs(f_axis - frequency)))
            selected_power += float(power[freq_idx])
        return round(min(1.0, selected_power / band_power), 3)

    def compute_covariance_matrix(
        self, array_data: np.ndarray, target_freq: float
    ) -> np.ndarray:
        n_samples = array_data.shape[1]
        segment_len = 512
        n_segments = n_samples // segment_len
        if n_segments == 0:
            raise ValueError("array_data must contain at least 512 samples")

        freq_axis = np.fft.rfftfreq(segment_len, d=1 / self.fs)
        freq_idx = np.argmin(np.abs(freq_axis - target_freq))

        X = np.zeros((self.num_mics, n_segments), dtype=complex)
        for s in range(n_segments):
            block = array_data[:, s * segment_len : (s + 1) * segment_len]
            X[:, s] = np.fft.rfft(block, axis=1)[:, freq_idx]

        return (X @ X.conj().T) / n_segments

    def run_music(
        self,
        array_data: np.ndarray,
        target_freq: float,
        az_step: int = 2,
        el_step: int = 2,
        azimuth_only: bool = False,
    ) -> Tuple[float, float]:
        if array_data.shape[0] != self.num_mics:
            raise ValueError(f"array_data must have {self.num_mics} microphone rows")

        Rxx = self.compute_covariance_matrix(array_data, target_freq)
        eigenvalues, eigenvectors = np.linalg.eigh(Rxx)

        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]

        n_src = 1
        Un = eigenvectors[:, : self.num_mics - n_src]
        Un_UnH = Un @ Un.conj().T

        wavenumber = 2 * np.pi * target_freq / self.c

        def steering(az_deg: float, el_deg: float) -> np.ndarray:
            az = np.radians(az_deg)
            el = np.radians(el_deg)
            u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
            delays = self.mic_coords @ u
            phase = wavenumber * delays
            return np.exp(1j * phase)

        def spectrum(az_deg: float, el_deg: float) -> float:
            a = steering(az_deg, el_deg)
            denom = np.real(a.conj() @ Un_UnH @ a)
            return float(1.0 / (denom + 1e-10))

        best_val = -1.0
        best_az, best_el = 0.0, 0.0

        if azimuth_only:
            for az in np.arange(0, 360, az_step):
                val = spectrum(az, 0.0)
                if val > best_val:
                    best_val, best_az, best_el = val, float(az), 0.0

            fine_az = np.arange(best_az - az_step, best_az + az_step + 0.1, 0.5)
            for az in fine_az:
                val = spectrum(az % 360, 0.0)
                if val > best_val:
                    best_val, best_az, best_el = val, float(az % 360), 0.0

            return round(best_az, 1), 0.0

        for az in np.arange(0, 360, az_step):
            for el in np.arange(0, 90, el_step):
                val = spectrum(az, el)
                if val > best_val:
                    best_val, best_az, best_el = val, float(az), float(el)

        fine_az = np.arange(best_az - az_step, best_az + az_step + 0.1, 0.5)
        fine_el = np.arange(max(0, best_el - el_step), best_el + el_step + 0.1, 0.5)

        for az in fine_az:
            for el in fine_el:
                val = spectrum(az % 360, el)
                if val > best_val:
                    best_val, best_az, best_el = val, float(az % 360), float(el)

        return round(best_az, 1), round(best_el, 1)

    def run_music_multifrequency(
        self,
        array_data: np.ndarray,
        target_freqs: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        az_step: int = 2,
        el_step: int = 2,
        azimuth_only: bool = False,
    ) -> Tuple[float, float]:
        """Run MUSIC at several bins and combine the resulting directions."""
        consensus = self.run_music_frequency_consensus(
            array_data,
            target_freqs,
            weights=weights,
            az_step=az_step,
            el_step=el_step,
            azimuth_only=azimuth_only,
        )
        return consensus.azimuth, consensus.elevation

    def run_music_frequency_consensus(
        self,
        array_data: np.ndarray,
        target_freqs: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        az_step: int = 2,
        el_step: int = 2,
        consensus_degrees: float = DEFAULT_CONSENSUS_DEGREES,
        azimuth_only: bool = False,
    ) -> DirectionConsensus:
        """Run MUSIC at several bins and keep the strongest agreeing group."""
        if not target_freqs:
            raise ValueError("target_freqs must contain at least one frequency")
        if consensus_degrees <= 0.0:
            raise ValueError("consensus_degrees must be positive")

        if weights is None:
            weight_array = np.ones(len(target_freqs), dtype=float)
        else:
            weight_array = np.asarray(weights, dtype=float)
            if weight_array.shape != (len(target_freqs),):
                raise ValueError("weights must match target_freqs length")
            if not np.any(weight_array):
                weight_array = np.ones(len(target_freqs), dtype=float)

        estimates = [
            self.run_music(
                array_data,
                target_freq,
                az_step=az_step,
                el_step=el_step,
                azimuth_only=azimuth_only,
            )
            for target_freq in target_freqs
        ]
        azimuth_values = np.asarray([estimate[0] for estimate in estimates], dtype=float)
        total_weight = float(np.sum(weight_array))
        best_mask = np.ones(len(target_freqs), dtype=bool)
        best_score = -1.0

        for idx, azimuth in enumerate(azimuth_values):
            distances = np.asarray(
                [angular_distance_degrees(azimuth, other) for other in azimuth_values],
                dtype=float,
            )
            mask = distances <= consensus_degrees
            score = float(np.sum(weight_array[mask]))
            if score > best_score:
                best_score = score
                best_mask = mask

        selected_weights = weight_array[best_mask]
        selected_azimuths = np.radians(azimuth_values[best_mask])
        selected_elevations = np.asarray(
            [estimate[1] for idx, estimate in enumerate(estimates) if best_mask[idx]],
            dtype=float,
        )
        x = float(np.sum(selected_weights * np.cos(selected_azimuths)))
        y = float(np.sum(selected_weights * np.sin(selected_azimuths)))
        azimuth = math.degrees(math.atan2(y, x)) % 360.0
        elevation = float(np.average(selected_elevations, weights=selected_weights))
        agreement = 1.0 if total_weight <= 0.0 else best_score / total_weight
        selected_freqs = tuple(
            round(float(freq), 1) for idx, freq in enumerate(target_freqs) if best_mask[idx]
        )
        return DirectionConsensus(
            azimuth=round(azimuth, 1),
            elevation=round(elevation, 1),
            agreement=round(min(1.0, agreement), 3),
            frequencies=selected_freqs,
        )

    def _band_limited_signal(
        self,
        signal: np.ndarray,
        band_min: float = DEFAULT_BAND_MIN_HZ,
        band_max: float = DEFAULT_BAND_MAX_HZ,
    ) -> np.ndarray:
        validate_frequency_band(band_min, band_max)
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1 / self.fs)
        mask = (freqs >= band_min) & (freqs <= band_max)
        return np.fft.irfft(spectrum * mask, n=len(signal))

    def gcc_phat_delay(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        max_tau: float,
        interpolation: int = 16,
    ) -> Tuple[float, float]:
        """Return physical delay for signal relative to reference and peak strength."""
        if interpolation < 1:
            raise ValueError("interpolation must be at least 1")

        n = signal.size + reference.size
        signal_spectrum = np.fft.rfft(signal, n=n)
        reference_spectrum = np.fft.rfft(reference, n=n)
        cross_power = signal_spectrum * np.conj(reference_spectrum)
        cross_power /= np.abs(cross_power) + 1e-12

        correlation = np.fft.irfft(cross_power, n=interpolation * n)
        max_shift = min(int(interpolation * self.fs * max_tau), correlation.size // 2)
        correlation = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
        abs_correlation = np.abs(correlation)
        shift_index = int(np.argmax(abs_correlation))

        if 0 < shift_index < len(correlation) - 1:
            left, center, right = abs_correlation[shift_index - 1 : shift_index + 2]
            denom = left - 2.0 * center + right
            offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
        else:
            offset = 0.0

        shift = shift_index - max_shift + offset
        # The simulator and steering model use t + delay, so negate the
        # correlation shift to return the same physical delay convention.
        delay = -shift / float(interpolation * self.fs)
        peak = float(abs_correlation[shift_index])
        return delay, peak

    def phase_delay_at_frequency(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        frequency: float,
    ) -> Optional[float]:
        """Estimate narrowband TDOA from phase at one selected frequency."""
        if frequency <= 0.0:
            return None
        freq_axis = np.fft.rfftfreq(len(signal), d=1 / self.fs)
        freq_idx = int(np.argmin(np.abs(freq_axis - frequency)))
        signal_bin = np.fft.rfft(signal)[freq_idx]
        reference_bin = np.fft.rfft(reference)[freq_idx]
        if abs(signal_bin) <= 1e-12 or abs(reference_bin) <= 1e-12:
            return None
        phase = float(np.angle(signal_bin * np.conj(reference_bin)))
        delay = phase / (2.0 * math.pi * frequency)
        return delay

    def estimate_experimental_elevation(
        self,
        array_data: np.ndarray,
        music_azimuth: float,
        band_min: float = DEFAULT_BAND_MIN_HZ,
        band_max: float = DEFAULT_BAND_MAX_HZ,
        interpolation: int = 16,
        min_denominator_fraction: float = 0.2,
        target_frequency: Optional[float] = None,
    ) -> ExperimentalElevationEstimate:
        """Infer experimental polar angle from center-to-ring TDOA delays."""
        array_data = validate_audio_window(array_data)
        validate_frequency_band(band_min, band_max)
        if not 0.0 < min_denominator_fraction <= 1.0:
            raise ValueError("min_denominator_fraction must be between 0 and 1")

        reference = self._band_limited_signal(array_data[0], band_min, band_max)
        max_tau = self.radius / self.c
        min_denominator = self.radius * min_denominator_fraction
        azimuth_rad = math.radians(music_azimuth)
        candidates = []

        for mic_idx in range(1, self.num_mics):
            mic_x, mic_y = self.mic_coords[mic_idx]
            mic_angle = math.atan2(float(mic_y), float(mic_x))
            denominator = self.radius * math.cos(mic_angle - azimuth_rad)
            if abs(denominator) < min_denominator:
                continue

            signal = self._band_limited_signal(array_data[mic_idx], band_min, band_max)
            delay, peak = self.gcc_phat_delay(
                signal,
                reference,
                max_tau=max_tau,
                interpolation=interpolation,
            )
            if target_frequency is not None:
                phase_delay = self.phase_delay_at_frequency(
                    signal,
                    reference,
                    frequency=target_frequency,
                )
                if phase_delay is not None and abs(phase_delay) <= max_tau * 1.2:
                    delay = phase_delay
            if peak <= 0.0 or abs(delay) > max_tau * 1.2:
                continue

            delta = self.c * delay
            value = delta / denominator
            if not np.isfinite(value) or value < -1.0 or value > 1.0:
                continue

            candidates.append(math.degrees(math.acos(max(-1.0, min(1.0, value)))))

        if not candidates:
            return ExperimentalElevationEstimate(
                elevation=None,
                confidence=0.0,
                valid_mics=0,
                candidates=(),
            )

        candidate_array = np.asarray(candidates, dtype=float)
        elevation = float(np.mean(candidate_array))
        spread = float(np.std(candidate_array)) if len(candidate_array) > 1 else 0.0
        valid_ratio = len(candidates) / 6.0
        consistency = max(0.0, 1.0 - spread / 30.0)
        confidence = round(min(1.0, valid_ratio * consistency), 3)
        return ExperimentalElevationEstimate(
            elevation=round(elevation, 1),
            confidence=confidence,
            valid_mics=len(candidates),
            candidates=tuple(round(float(candidate), 1) for candidate in candidates),
        )


def estimate_from_audio_matrix(
    audio_matrix: np.ndarray,
    fs: int = 48000,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    azimuth_only: bool = False,
) -> Tuple[float, float, float]:
    """Estimate azimuth/elevation from a 7 x samples microphone matrix."""
    estimate = estimate_direction_from_audio_matrix(
        audio_matrix,
        fs=fs,
        band_min=band_min,
        band_max=band_max,
        max_frequencies=max_frequencies,
        azimuth_only=azimuth_only,
    )
    return float(estimate.frequency or 0.0), estimate.azimuth, estimate.elevation


def estimate_direction_from_audio_matrix(
    audio_matrix: np.ndarray,
    fs: int = 48000,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    azimuth_only: bool = False,
) -> DirectionEstimate:
    """Estimate current direction with band-limited multi-frequency MUSIC."""
    audio_matrix = validate_audio_window(audio_matrix)
    validate_frequency_band(band_min, band_max)
    estimator = UMA8MUSICEstimator(fs=fs)
    target_freqs = estimator.select_target_frequencies(
        audio_matrix[0],
        band_min=band_min,
        band_max=band_max,
        max_frequencies=max_frequencies,
    )
    weights = estimator.frequency_weights(audio_matrix[0], target_freqs)
    consensus = estimator.run_music_frequency_consensus(
        audio_matrix,
        target_freqs,
        weights=weights,
        azimuth_only=azimuth_only,
    )
    frequency = float(np.average(np.asarray(target_freqs, dtype=float), weights=weights))
    band_confidence = estimator.band_confidence(
        audio_matrix[0],
        target_freqs,
        band_min=band_min,
        band_max=band_max,
    )
    confidence = round(band_confidence * consensus.agreement, 3)
    experimental_elevation = estimator.estimate_experimental_elevation(
        audio_matrix,
        music_azimuth=consensus.azimuth,
        band_min=band_min,
        band_max=band_max,
        target_frequency=frequency,
    )
    return DirectionEstimate(
        azimuth=consensus.azimuth,
        elevation=consensus.elevation,
        frequency=round(frequency, 1),
        confidence=confidence,
        frequencies=consensus.frequencies,
        experimental_elevation=experimental_elevation.elevation,
        experimental_elevation_confidence=experimental_elevation.confidence,
        experimental_elevation_valid_mics=experimental_elevation.valid_mics,
        experimental_elevation_candidates=experimental_elevation.candidates,
    )


def validate_frequency_band(band_min: float, band_max: float) -> None:
    if band_min <= 0.0:
        raise ValueError("band_min must be positive")
    if band_max <= band_min:
        raise ValueError("band_max must be greater than band_min")


def angular_distance_degrees(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def validate_window_duration(window_duration: float, fs: int = 48000) -> None:
    if window_duration <= 0.0:
        raise ValueError("window_duration must be positive")
    min_duration = 512 / fs
    if window_duration < min_duration:
        raise ValueError(
            "window_duration must provide at least 512 samples; "
            f"got {window_duration:.4f}s at {fs} Hz"
        )


def validate_audio_window(audio_matrix: np.ndarray) -> np.ndarray:
    """Validate a microphone x samples audio window.

    The estimator expects shape ``(7, samples)``: one center microphone plus six
    microphones around the UMA8-like circular layout.
    """
    matrix = np.asarray(audio_matrix)
    if matrix.ndim != 2:
        raise ValueError(
            "audio window must be a 2D matrix with shape (7, samples); "
            f"got {matrix.ndim} dimensions"
        )
    if matrix.shape[0] != EXPECTED_MICROPHONES:
        raise ValueError(
            "audio window must have 7 microphone rows; "
            f"got {matrix.shape[0]} rows"
        )
    if matrix.shape[1] < 512:
        raise ValueError(
            "audio window must contain at least 512 samples per microphone; "
            f"got {matrix.shape[1]} samples"
        )
    return matrix


class DirectionSmoother:
    """Simple exponential moving average for direction estimates."""

    def __init__(self, alpha: float = 0.35):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and less than or equal to 1")
        self.alpha = alpha
        self._current: Optional[DirectionEstimate] = None

    @property
    def current(self) -> Optional[DirectionEstimate]:
        return self._current

    def update(self, estimate: DirectionEstimate) -> DirectionEstimate:
        if self._current is None:
            self._current = estimate
            return estimate

        az_delta = ((estimate.azimuth - self._current.azimuth + 180.0) % 360.0) - 180.0
        azimuth = (self._current.azimuth + self.alpha * az_delta) % 360.0
        elevation = self._current.elevation + self.alpha * (
            estimate.elevation - self._current.elevation
        )
        self._current = DirectionEstimate(
            azimuth=round(azimuth, 3),
            elevation=round(elevation, 3),
            frequency=estimate.frequency,
            confidence=estimate.confidence,
            frequencies=estimate.frequencies,
            reference_azimuth=estimate.reference_azimuth,
            reference_elevation=estimate.reference_elevation,
            experimental_elevation=estimate.experimental_elevation,
            experimental_elevation_confidence=estimate.experimental_elevation_confidence,
            experimental_elevation_valid_mics=estimate.experimental_elevation_valid_mics,
            experimental_elevation_candidates=estimate.experimental_elevation_candidates,
        )
        return self._current


class DirectionConsoleOutput:
    """Console output for direction-only estimates."""

    def __init__(self):
        self.commands: List[DirectionEstimate] = []

    def report(self, estimate: DirectionEstimate) -> None:
        self.commands.append(estimate)
        confidence = (
            f" confidence={estimate.confidence:.2f}"
            if estimate.confidence is not None
            else ""
        )
        experimental_elevation = ""
        if estimate.experimental_elevation is not None:
            experimental_elevation = (
                " experimental_elevation="
                f"{estimate.experimental_elevation:.1f} deg"
                f" experimental_valid_mics={estimate.experimental_elevation_valid_mics}"
                f" experimental_confidence="
                f"{estimate.experimental_elevation_confidence:.2f}"
            )
        print(
            "DIRECTION "
            f"azimuth={estimate.azimuth:.1f} deg elevation={estimate.elevation:.1f} deg"
            f"{confidence}"
            f"{experimental_elevation}"
        )


class AudioWindowSource(ABC):
    """Source of one ``7 x samples`` audio window for tracking."""

    @abstractmethod
    def read_window(self) -> np.ndarray:
        """Return one audio window without changing estimator behavior."""


class MockAudioSource(AudioWindowSource):
    """Deterministic source for tests and offline validation."""

    def __init__(self, windows: Iterable[np.ndarray]):
        self._windows = iter(windows)

    def read_window(self) -> np.ndarray:
        return validate_audio_window(next(self._windows))


class SimulatedAudioSource(AudioWindowSource):
    """No-hardware moving-source tone generator for tracking validation."""

    def __init__(
        self,
        fs: int = 48000,
        window_duration: float = 0.25,
        frequency: float = 1000.0,
    ):
        validate_window_duration(window_duration, fs=fs)
        self.fs = fs
        self.window_duration = window_duration
        self.frequency = frequency
        self.estimator = UMA8MUSICEstimator(fs=fs)
        self._step = 0
        self.last_reference: Optional[DirectionEstimate] = None

    def read_window(self) -> np.ndarray:
        azimuth = (45.0 + self._step * 8.0) % 360.0
        elevation = 25.0 + 10.0 * math.sin(self._step / 4.0)
        self.last_reference = DirectionEstimate(
            azimuth=azimuth,
            elevation=round(elevation, 3),
            frequency=self.frequency,
        )
        self._step += 1
        return make_directional_tone(
            azimuth=azimuth,
            elevation=elevation,
            estimator=self.estimator,
            fs=self.fs,
            duration=self.window_duration,
            frequency=self.frequency,
        )


class AudioInputError(RuntimeError):
    """Clear live-audio error for hardware, permissions, and shape failures."""


class AudioRingBuffer:
    """Thread-safe latest-window buffer for multi-channel audio."""

    def __init__(self, channels: int, capacity_samples: int):
        if channels < EXPECTED_MICROPHONES:
            raise ValueError("ring buffer must have at least 7 channels")
        if capacity_samples < 512:
            raise ValueError("ring buffer capacity must be at least 512 samples")
        self.channels = channels
        self.capacity_samples = capacity_samples
        self._buffer = np.zeros((channels, capacity_samples), dtype=np.float32)
        self._write_index = 0
        self._samples_written = 0
        self._lock = threading.Lock()

    @property
    def samples_written(self) -> int:
        with self._lock:
            return self._samples_written

    def append(self, samples: np.ndarray) -> None:
        matrix = np.asarray(samples, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("samples must be a 2D matrix with shape (channels, samples)")
        if matrix.shape[0] < self.channels:
            raise ValueError(
                f"samples must have at least {self.channels} rows; got {matrix.shape[0]}"
            )
        matrix = matrix[: self.channels, :]
        sample_count = matrix.shape[1]
        if sample_count == 0:
            return
        if sample_count >= self.capacity_samples:
            matrix = matrix[:, -self.capacity_samples :]
            sample_count = matrix.shape[1]

        with self._lock:
            end_index = self._write_index + sample_count
            if end_index <= self.capacity_samples:
                self._buffer[:, self._write_index : end_index] = matrix
            else:
                first_count = self.capacity_samples - self._write_index
                self._buffer[:, self._write_index :] = matrix[:, :first_count]
                self._buffer[:, : end_index % self.capacity_samples] = matrix[:, first_count:]
            self._write_index = end_index % self.capacity_samples
            self._samples_written += sample_count

    def latest_window(self, window_samples: int) -> np.ndarray:
        if window_samples < 512:
            raise ValueError("window must contain at least 512 samples")
        if window_samples > self.capacity_samples:
            raise ValueError("window cannot be larger than ring buffer capacity")

        with self._lock:
            if self._samples_written < window_samples:
                raise AudioInputError(
                    "not enough live audio buffered yet; wait for the stream to fill"
                )
            start = (self._write_index - window_samples) % self.capacity_samples
            if start < self._write_index:
                window = self._buffer[:, start : self._write_index]
            else:
                window = np.hstack(
                    (self._buffer[:, start:], self._buffer[:, : self._write_index])
                )
            return validate_audio_window(window[:EXPECTED_MICROPHONES, :].copy())


class LiveSoundDeviceAudioSource(AudioWindowSource):
    """Opt-in live source; constructing it does not open microphone hardware."""

    def __init__(self, fs: int = 48000, window_duration: float = 1.0, channels: int = 7):
        validate_window_duration(window_duration, fs=fs)
        self.fs = fs
        self.window_duration = window_duration
        self.channels = channels

    def read_window(self) -> np.ndarray:
        return record_uma8_audio(
            fs=self.fs,
            duration=self.window_duration,
            channels=self.channels,
        )


class StreamingSoundDeviceAudioSource(AudioWindowSource):
    """Live source that keeps a sliding audio window in a ring buffer."""

    def __init__(
        self,
        fs: int = 48000,
        window_duration: float = 0.1,
        channels: int = 7,
        buffer_duration: float = 1.0,
        startup_timeout: float = 2.0,
    ):
        validate_window_duration(window_duration, fs=fs)
        if buffer_duration < window_duration:
            raise ValueError("buffer_duration must be at least window_duration")
        self.fs = fs
        self.window_duration = window_duration
        self.window_samples = int(round(window_duration * fs))
        self.channels = channels
        self.buffer_samples = max(int(round(buffer_duration * fs)), self.window_samples)
        self.startup_timeout = startup_timeout
        self.ring_buffer = AudioRingBuffer(channels=channels, capacity_samples=self.buffer_samples)
        self._stream = None
        self._sd = None

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioInputError(
                "sounddevice is not installed; run `python -m pip install -r requirements.txt`"
            ) from exc

        try:
            input_device = sd.query_devices(kind="input")
        except Exception as exc:
            raise AudioInputError(
                "could not query an input device; check microphone availability and OS permissions"
            ) from exc

        max_channels = int(input_device.get("max_input_channels", 0))
        if max_channels < EXPECTED_MICROPHONES:
            raise AudioInputError(
                "selected input device does not expose enough channels; "
                f"expected at least 7, got {max_channels}"
            )

        def callback(indata, frames, time_info, status):
            del frames, time_info
            if status:
                print(f"Live audio stream status: {status}")
            self.ring_buffer.append(indata.T[: self.channels, :])

        try:
            self._sd = sd
            self._stream = sd.InputStream(
                samplerate=self.fs,
                channels=self.channels,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioInputError(
                "could not start live audio stream; check device, channel count, and permissions"
            ) from exc

    def read_window(self) -> np.ndarray:
        self.start()
        deadline = time.monotonic() + self.startup_timeout
        while self.ring_buffer.samples_written < self.window_samples:
            if time.monotonic() >= deadline:
                raise AudioInputError("timed out waiting for live audio stream data")
            time.sleep(0.005)
        return self.ring_buffer.latest_window(self.window_samples)

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


def record_uma8_audio(fs: int = 48000, duration: float = 2.0, channels: int = 7) -> np.ndarray:
    """Record live audio from the default sounddevice input device."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioInputError(
            "sounddevice is not installed; run `python -m pip install -r requirements.txt`"
        ) from exc

    try:
        input_device = sd.query_devices(kind="input")
    except Exception as exc:
        raise AudioInputError(
            "could not query an input device; check microphone availability and OS permissions"
        ) from exc

    max_channels = int(input_device.get("max_input_channels", 0))
    if max_channels < EXPECTED_MICROPHONES:
        raise AudioInputError(
            "selected input device does not expose enough channels; "
            f"expected at least 7, got {max_channels}"
        )

    try:
        audio_data = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
        return validate_audio_window(audio_data.T[:EXPECTED_MICROPHONES, :])
    except ValueError:
        raise
    except Exception as exc:
        raise AudioInputError(
            "microphone capture failed; check device selection, channel count, and OS permissions"
        ) from exc


def make_directional_tone(
    azimuth: float,
    elevation: float,
    estimator: UMA8MUSICEstimator,
    fs: int = 48000,
    duration: float = 0.25,
    frequency: float = 1000.0,
) -> np.ndarray:
    """Create a simple far-field narrowband tone for no-hardware tracking."""
    t = np.arange(int(duration * fs)) / fs
    az = np.radians(azimuth)
    el = np.radians(elevation)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)])
    delays = estimator.mic_coords @ direction / estimator.c
    rows = [
        np.sin(2 * np.pi * frequency * (t + delay))
        for delay in delays
    ]
    return validate_audio_window(np.vstack(rows))


class DirectionVisualizer:
    """Matplotlib direction-only view for one planar microphone array."""

    def __init__(
        self,
        mic_coords: np.ndarray,
        history_limit: int = 60,
        show_history: bool = True,
        view_mode: str = "2d",
    ):
        import matplotlib.pyplot as plt

        if view_mode not in {"2d", "3d"}:
            raise ValueError("view_mode must be '2d' or '3d'")
        self.plt = plt
        self.mic_coords = mic_coords
        self.history_limit = history_limit
        self.show_history = show_history
        self.view_mode = view_mode
        self.azimuth_history: List[float] = []
        self.reference_history: List[float] = []
        if view_mode == "3d":
            self.fig = plt.figure(figsize=(10, 4) if show_history else (5, 4))
            self.ax_layout = self.fig.add_subplot(
                1,
                2 if show_history else 1,
                1,
                projection="3d",
            )
            self.ax_history = self.fig.add_subplot(1, 2, 2) if show_history else None
        elif show_history:
            self.fig, (self.ax_layout, self.ax_history) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            self.fig, self.ax_layout = plt.subplots(1, 1, figsize=(5, 4))
            self.ax_history = None
        self.fig.suptitle(
            "One-array direction of arrival"
            if view_mode == "2d"
            else "One-array direction of arrival: direction ray only"
        )

    def update(self, estimate: DirectionEstimate) -> None:
        self.azimuth_history.append(estimate.azimuth)
        if len(self.azimuth_history) > self.history_limit:
            self.azimuth_history = self.azimuth_history[-self.history_limit :]

        if estimate.reference_azimuth is not None:
            self.reference_history.append(estimate.reference_azimuth)
            if len(self.reference_history) > self.history_limit:
                self.reference_history = self.reference_history[-self.history_limit :]

        self.ax_layout.clear()
        if self.ax_history is not None:
            self.ax_history.clear()

        arrow_length = max(0.12, float(np.max(np.linalg.norm(self.mic_coords, axis=1))) * 2.5)
        if self.view_mode == "3d":
            self._draw_3d_layout(estimate, arrow_length)
        else:
            self._draw_2d_layout(estimate, arrow_length)

        if self.ax_history is not None:
            self.ax_history.plot(self.azimuth_history, color="tab:red", label="estimated azimuth")
            if self.reference_history:
                self.ax_history.plot(
                    self.reference_history,
                    "--",
                    color="tab:green",
                    label="simulated reference",
                )
            self.ax_history.set_ylim(0, 360)
            self.ax_history.set_xlabel("update")
            self.ax_history.set_ylabel("azimuth degrees")
            self.ax_history.set_title("Recent azimuth history")
            self.ax_history.legend(loc="upper right")

        self.fig.tight_layout()
        self.plt.pause(0.001)

    def _draw_2d_layout(self, estimate: DirectionEstimate, arrow_length: float) -> None:
        self.ax_layout.scatter(
            self.mic_coords[:, 0],
            self.mic_coords[:, 1],
            color="black",
            label="microphones",
        )
        self.ax_layout.scatter([0], [0], color="tab:blue", label="array center")

        az = np.radians(estimate.azimuth)
        self.ax_layout.arrow(
            0,
            0,
            arrow_length * np.cos(az),
            arrow_length * np.sin(az),
            color="tab:red",
            width=0.002,
            length_includes_head=True,
            label="estimated azimuth",
        )

        if estimate.reference_azimuth is not None:
            ref = np.radians(estimate.reference_azimuth)
            self.ax_layout.plot(
                [0, arrow_length * np.cos(ref)],
                [0, arrow_length * np.sin(ref)],
                "--",
                color="tab:green",
                label="simulated reference",
            )

        self.ax_layout.set_aspect("equal", adjustable="box")
        pad = arrow_length * 1.2
        self.ax_layout.set_xlim(-pad, pad)
        self.ax_layout.set_ylim(-pad, pad)
        self.ax_layout.set_xlabel("x meters")
        self.ax_layout.set_ylabel("y meters")
        self.ax_layout.set_title("Planar 7-microphone layout")
        self.ax_layout.legend(loc="upper right")
        self._draw_note(estimate)

    def _draw_3d_layout(self, estimate: DirectionEstimate, arrow_length: float) -> None:
        zeros = np.zeros(self.mic_coords.shape[0])
        self.ax_layout.scatter(
            self.mic_coords[:, 0],
            self.mic_coords[:, 1],
            zeros,
            color="black",
            label="microphones",
        )
        self.ax_layout.scatter([0], [0], [0], color="tab:blue", label="array center")

        elevation = (
            estimate.experimental_elevation
            if estimate.experimental_elevation is not None
            else estimate.elevation
        )
        az = math.radians(estimate.azimuth)
        el = math.radians(elevation)
        direction = np.array(
            [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ]
        )
        self.ax_layout.quiver(
            0,
            0,
            0,
            arrow_length * direction[0],
            arrow_length * direction[1],
            arrow_length * direction[2],
            color="tab:red",
            arrow_length_ratio=0.15,
            label="estimated direction ray",
        )
        self.ax_layout.plot(
            [0, arrow_length * math.cos(az)],
            [0, arrow_length * math.sin(az)],
            [0, 0],
            "--",
            color="tab:orange",
            label="azimuth projection",
        )
        pad = arrow_length * 1.2
        self.ax_layout.set_xlim(-pad, pad)
        self.ax_layout.set_ylim(-pad, pad)
        self.ax_layout.set_zlim(0, pad)
        self.ax_layout.set_xlabel("x meters")
        self.ax_layout.set_ylabel("y meters")
        self.ax_layout.set_zlabel("relative up axis")
        self.ax_layout.set_title("3D direction ray; no position or distance")
        self.ax_layout.legend(loc="upper right")
        self._draw_note(estimate)

    def _draw_note(self, estimate: DirectionEstimate) -> None:
        note = f"Azimuth: {estimate.azimuth:.1f} deg\n"
        note += f"Elevation: {estimate.elevation:.1f} deg (text only; planar array)"
        if estimate.frequency is not None:
            note += f"\nFrequency: {estimate.frequency:.1f} Hz"
        if estimate.confidence is not None:
            note += f"\nConfidence: {estimate.confidence:.2f}"
        if estimate.experimental_elevation is not None:
            note += (
                "\nExperimental polar/elevation: "
                f"{estimate.experimental_elevation:.1f} deg"
                f" ({estimate.experimental_elevation_valid_mics}/6 mics, "
                f"confidence {estimate.experimental_elevation_confidence:.2f})"
            )
        text = note + "\nNo distance or 3D position is estimated."
        if self.view_mode == "3d":
            self.ax_layout.text2D(
                0.02,
                0.02,
                text,
                transform=self.ax_layout.transAxes,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
            )
        else:
            self.ax_layout.text(
                0.02,
                0.02,
                text,
                transform=self.ax_layout.transAxes,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
            )

    def close(self) -> None:
        self.plt.close(self.fig)


def make_self_check_signal(fs: int = 48000, duration: float = 0.25) -> np.ndarray:
    """Create a deterministic no-hardware tone matrix for import/run validation."""
    estimator = UMA8MUSICEstimator(fs=fs)
    return make_directional_tone(
        azimuth=45.0,
        elevation=0.0,
        estimator=estimator,
        fs=fs,
        duration=duration,
        frequency=1000.0,
    )


def simulated_direction_sequence() -> Iterable[DirectionEstimate]:
    """Yield a deterministic moving direction sequence for no-hardware tracking."""
    step = 0
    while True:
        azimuth = (45.0 + step * 8.0) % 360.0
        elevation = 25.0 + 10.0 * math.sin(step / 4.0)
        yield DirectionEstimate(azimuth=azimuth, elevation=round(elevation, 3), frequency=1000.0)
        step += 1


def live_direction_source(fs: int, window_duration: float) -> Callable[[], DirectionEstimate]:
    """Create a live microphone source. The returned callable records on each call."""

    def next_estimate() -> DirectionEstimate:
        audio_matrix = record_uma8_audio(fs=fs, duration=window_duration)
        target_freq, az, el = estimate_from_audio_matrix(audio_matrix, fs=fs)
        return DirectionEstimate(azimuth=az, elevation=el, frequency=target_freq)

    return next_estimate


def audio_window_direction_source(
    audio_source: AudioWindowSource,
    fs: int = 48000,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    azimuth_only: bool = False,
) -> Callable[[], DirectionEstimate]:
    """Adapt an audio source into the direction-source API used by tracking."""

    def next_estimate() -> DirectionEstimate:
        audio_matrix = audio_source.read_window()
        return estimate_direction_from_audio_matrix(
            audio_matrix,
            fs=fs,
            band_min=band_min,
            band_max=band_max,
            max_frequencies=max_frequencies,
            azimuth_only=azimuth_only,
        )

    return next_estimate


def run_direction_loop(
    source: Callable[[], DirectionEstimate],
    output: DirectionConsoleOutput,
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
    visualizer: Optional[DirectionVisualizer] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Run continuous direction estimation and optional direction-only visualization."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")

    smoother = DirectionSmoother(alpha=smoothing_alpha) if smoothing_alpha is not None else None
    deadline = monotonic() + duration
    frames = 0

    try:
        while monotonic() < deadline:
            raw = source()
            smoothed = smoother.update(raw) if smoother is not None else raw
            print(
                f"TRACK raw=({raw.azimuth:.1f}, {raw.elevation:.1f}) "
                f"shown=({smoothed.azimuth:.1f}, {smoothed.elevation:.1f})"
            )
            output.report(smoothed)
            if visualizer is not None:
                visualizer.update(smoothed)
            frames += 1
            remaining = deadline - monotonic()
            if remaining > 0:
                sleep(min(interval, remaining))
    except KeyboardInterrupt:
        print("Tracking stopped by Ctrl+C.")

    print(f"Tracking complete. Commands issued: {frames}")
    return frames


def run_tracking_loop(
    source: Callable[[], DirectionEstimate],
    controller: DirectionConsoleOutput,
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Backward-compatible alias for the old tracking helper."""
    return run_direction_loop(
        source=source,
        output=controller,
        duration=duration,
        interval=interval,
        smoothing_alpha=smoothing_alpha,
        sleep=sleep,
        monotonic=monotonic,
    )


def run_simulated_tracking(
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
    visualize: bool = False,
    window_duration: Optional[float] = None,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    show_history: bool = True,
    azimuth_only: bool = False,
    visualization_mode: str = "2d",
) -> int:
    if window_duration is None:
        window_duration = max(512 / 48000, min(interval, 1.0))
    audio_source = SimulatedAudioSource(window_duration=window_duration)
    output = DirectionConsoleOutput()
    visualizer = (
        DirectionVisualizer(
            audio_source.estimator.mic_coords,
            show_history=show_history,
            view_mode=visualization_mode,
        )
        if visualize
        else None
    )
    try:
        return run_direction_loop(
            source=_direction_from_audio_source(
                audio_source,
                band_min=band_min,
                band_max=band_max,
                max_frequencies=max_frequencies,
                azimuth_only=azimuth_only,
            ),
            output=output,
            duration=duration,
            interval=interval,
            smoothing_alpha=smoothing_alpha,
            visualizer=visualizer,
        )
    finally:
        if visualizer is not None:
            visualizer.close()


def run_live_direction_tracking(
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
    fs: int = 48000,
    visualize: bool = False,
    window_duration: Optional[float] = None,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    show_history: bool = True,
    streaming: bool = True,
    azimuth_only: bool = False,
    visualization_mode: str = "2d",
) -> int:
    print("Starting live microphone direction tracking.")
    live_window_duration = window_duration if window_duration is not None else interval
    if streaming:
        audio_source: AudioWindowSource = StreamingSoundDeviceAudioSource(
            fs=fs,
            window_duration=live_window_duration,
            buffer_duration=max(1.0, live_window_duration * 4),
        )
        loop_interval = interval
    else:
        audio_source = LiveSoundDeviceAudioSource(
            fs=fs,
            window_duration=live_window_duration,
        )
        loop_interval = 0.001
    output = DirectionConsoleOutput()
    visualizer = (
        DirectionVisualizer(
            UMA8MUSICEstimator(fs=fs).mic_coords,
            show_history=show_history,
            view_mode=visualization_mode,
        )
        if visualize
        else None
    )
    try:
        return run_direction_loop(
            source=audio_window_direction_source(
                audio_source,
                fs=fs,
                band_min=band_min,
                band_max=band_max,
                max_frequencies=max_frequencies,
                azimuth_only=azimuth_only,
            ),
            output=output,
            duration=duration,
            interval=loop_interval,
            smoothing_alpha=smoothing_alpha,
            visualizer=visualizer,
        )
    except AudioInputError as exc:
        print(f"Live audio error: {exc}")
        return 0
    finally:
        if isinstance(audio_source, StreamingSoundDeviceAudioSource):
            audio_source.close()
        if visualizer is not None:
            visualizer.close()


def run_simulated_tracking_legacy(
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
) -> int:
    return run_simulated_tracking(
        duration=duration,
        interval=interval,
        smoothing_alpha=smoothing_alpha,
    )


def run_live_tracking(
    duration: float = 10.0,
    interval: float = 1.0,
    smoothing_alpha: Optional[float] = 0.35,
    fs: int = 48000,
) -> int:
    """Backward-compatible live tracking alias."""
    return run_live_direction_tracking(
        duration=duration,
        interval=interval,
        smoothing_alpha=smoothing_alpha,
        fs=fs,
    )


def run_one_live_estimate(fs: int = 48000, azimuth_only: bool = False) -> None:
    print("Opening UMA8/default input device for one live direction estimate.")
    try:
        audio_matrix = record_uma8_audio(fs=fs)
    except AudioInputError as exc:
        print(f"Live audio error: {exc}")
        return
    except ValueError as exc:
        print(f"Audio window shape error: {exc}")
        return

    estimate = estimate_direction_from_audio_matrix(
        audio_matrix,
        fs=fs,
        azimuth_only=azimuth_only,
    )
    print("One-shot live MUSIC direction estimate:")
    print(f"Selected dominant frequency: {estimate.frequency:.1f} Hz")
    print(f"Azimuth: {estimate.azimuth} degrees")
    print(f"Elevation: {estimate.elevation} degrees (text only; planar array)")
    if estimate.experimental_elevation is not None:
        print(
            "Experimental polar/elevation: "
            f"{estimate.experimental_elevation:.1f} degrees "
            f"({estimate.experimental_elevation_valid_mics}/6 mics, "
            f"confidence {estimate.experimental_elevation_confidence:.2f})"
        )


def run_recording() -> None:
    """Backward-compatible alias for one live estimate."""
    run_one_live_estimate()


def _direction_from_audio_source(
    audio_source: SimulatedAudioSource,
    fs: int = 48000,
    band_min: float = DEFAULT_BAND_MIN_HZ,
    band_max: float = DEFAULT_BAND_MAX_HZ,
    max_frequencies: int = DEFAULT_MAX_FREQUENCIES,
    azimuth_only: bool = False,
):
    def next_estimate() -> DirectionEstimate:
        raw_reference = audio_source.last_reference
        audio_matrix = audio_source.read_window()
        estimate = estimate_direction_from_audio_matrix(
            audio_matrix,
            fs=fs,
            band_min=band_min,
            band_max=band_max,
            max_frequencies=max_frequencies,
            azimuth_only=azimuth_only,
        )
        reference = audio_source.last_reference or raw_reference
        return DirectionEstimate(
            azimuth=estimate.azimuth,
            elevation=estimate.elevation,
            frequency=estimate.frequency,
            confidence=estimate.confidence,
            frequencies=estimate.frequencies,
            reference_azimuth=reference.azimuth if reference else None,
            reference_elevation=reference.elevation if reference else None,
            experimental_elevation=estimate.experimental_elevation,
            experimental_elevation_confidence=estimate.experimental_elevation_confidence,
            experimental_elevation_valid_mics=estimate.experimental_elevation_valid_mics,
            experimental_elevation_candidates=estimate.experimental_elevation_candidates,
        )

    return next_estimate


def run_self_check(azimuth_only: bool = False) -> None:
    audio_matrix = make_self_check_signal()
    estimate = estimate_direction_from_audio_matrix(
        audio_matrix,
        azimuth_only=azimuth_only,
    )
    print("Self-check completed without microphone input.")
    if azimuth_only:
        print("MUSIC mode: azimuth-only")
    print(f"Selected frequency: {estimate.frequency:.1f} Hz")
    print(f"Estimated azimuth: {estimate.azimuth} degrees")
    print(f"Estimated elevation: {estimate.elevation} degrees")
    if estimate.experimental_elevation is not None:
        print(
            "Experimental polar/elevation: "
            f"{estimate.experimental_elevation:.1f} degrees "
            f"({estimate.experimental_elevation_valid_mics}/6 mics, "
            f"confidence {estimate.experimental_elevation_confidence:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="UMA8 MUSIC DOA estimator")
    parser.add_argument(
        "--record",
        action="store_true",
        help="record from the default sounddevice input device before estimating DOA",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run a deterministic no-hardware validation path",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="backward-compatible alias for continuous direction tracking",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run simulated no-hardware direction tracking",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="use live microphone input; explicit opt-in only",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="backward-compatible alias for --2dvisualise",
    )
    parser.add_argument(
        "--2dvisualise",
        "--2dvisualize",
        dest="visualize_2d",
        action="store_true",
        help="show the planar 2D direction-only visualization",
    )
    parser.add_argument(
        "--3dvisualise",
        "--3dvisualize",
        dest="visualize_3d",
        action="store_true",
        help="show a 3D direction ray using azimuth and experimental elevation",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="tracking duration in seconds",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="tracking update interval or live recording window duration in seconds",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="exponential smoothing alpha for tracking estimates",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="show raw current estimates without exponential smoothing",
    )
    parser.add_argument(
        "--azimuth-only",
        action="store_true",
        help="run MUSIC with elevation fixed at 0 degrees for lower-latency direction tracking",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=None,
        help="audio analysis window duration in seconds; defaults to interval",
    )
    parser.add_argument(
        "--band-min",
        type=float,
        default=DEFAULT_BAND_MIN_HZ,
        help="minimum frequency considered for direction estimation",
    )
    parser.add_argument(
        "--band-max",
        type=float,
        default=DEFAULT_BAND_MAX_HZ,
        help="maximum frequency considered for direction estimation",
    )
    parser.add_argument(
        "--max-frequencies",
        type=int,
        default=DEFAULT_MAX_FREQUENCIES,
        help="number of strong frequency bins to combine with MUSIC",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="hide recent azimuth history and show only the current direction view",
    )
    parser.add_argument(
        "--blocking-live",
        action="store_true",
        help="use the older record-and-wait live capture instead of streaming ring-buffer capture",
    )
    args = parser.parse_args()
    smoothing_alpha = None if args.no_smoothing else args.smoothing_alpha
    visualization_mode = "3d" if args.visualize_3d else "2d"
    should_visualize = args.visualize or args.visualize_2d or args.visualize_3d

    if args.live:
        if should_visualize or args.track:
            run_live_direction_tracking(
                duration=args.duration,
                interval=args.interval,
                smoothing_alpha=smoothing_alpha,
                visualize=should_visualize,
                window_duration=args.window_duration,
                band_min=args.band_min,
                band_max=args.band_max,
                max_frequencies=args.max_frequencies,
                show_history=not args.no_history,
                streaming=not args.blocking_live,
                azimuth_only=args.azimuth_only,
                visualization_mode=visualization_mode,
            )
        else:
            run_one_live_estimate(azimuth_only=args.azimuth_only)
    elif args.record:
        run_recording()
    elif args.simulate or args.track or should_visualize:
        run_simulated_tracking(
            duration=args.duration,
            interval=args.interval,
            smoothing_alpha=smoothing_alpha,
            visualize=should_visualize,
            window_duration=args.window_duration,
            band_min=args.band_min,
            band_max=args.band_max,
            max_frequencies=args.max_frequencies,
            show_history=not args.no_history,
            azimuth_only=args.azimuth_only,
            visualization_mode=visualization_mode,
        )
    else:
        run_self_check(azimuth_only=args.azimuth_only)


if __name__ == "__main__":
    main()
