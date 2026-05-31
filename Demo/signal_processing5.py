import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from dataclasses import dataclass
import time

# ============================================================
# ADVANCED REAL-TIME DOA TRACKER SIMULATION
# ============================================================
# Features:
# - Circular angle math
# - Circular moving statistics
# - Velocity-aware Kalman filter (theta + omega)
# - Confidence gating
# - Streaming simulation
# - Real-time latency-aware processing
# - Polar visualization
# - Realistic drone motion
# - Realistic DOA noise and dropouts
# - Outlier rejection
# - Wrap-safe tracking
# ============================================================

# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 48000
BUFFER_SIZE = 1024
UPDATE_RATE = BUFFER_SIZE / SAMPLE_RATE

SIM_DURATION = 30

OUTLIER_THRESHOLD_DEG = 35
CONFIDENCE_THRESHOLD = 0.45

HISTORY_SIZE = 100

PROCESS_NOISE_ANGLE = 0.08
PROCESS_NOISE_VELOCITY = 0.03
MEASUREMENT_NOISE = 4.0

MAX_ANGULAR_VELOCITY = 120  # deg/sec

np.random.seed(42)

# ============================================================
# ANGLE UTILITIES
# ============================================================


def angle_wrap(angle):
    """Wrap angle to [-180, 180]"""
    return (angle + 180) % 360 - 180



def angle_diff(a, b):
    """Correct circular angle difference"""
    return angle_wrap(a - b)



def circular_mean(angles_deg):
    """Circular mean for angular data"""

    if len(angles_deg) == 0:
        return 0.0

    radians = np.radians(angles_deg)

    sin_sum = np.mean(np.sin(radians))
    cos_sum = np.mean(np.cos(radians))

    return np.degrees(np.arctan2(sin_sum, cos_sum))


# ============================================================
# SIMULATED DOA OUTPUT FROM STAGE 4
# ============================================================

@dataclass
class DOAMeasurement:
    timestamp: float
    angle: float
    confidence: float
    is_valid: bool


class RealisticDroneSimulator:
    """
    Simulates realistic drone DOA estimation output.

    Simulates:
    - Smooth drone movement
    - Angular acceleration
    - Acoustic noise
    - Multipath reflections
    - TDOA failures
    - Wraparound
    """

    def __init__(self):

        self.theta = -140.0
        self.angular_velocity = 22.0

        self.time = 0.0

    def update_motion(self, dt):

        # Smooth motion variation
        accel = np.sin(self.time * 0.4) * 8

        self.angular_velocity += accel * dt

        self.angular_velocity = np.clip(
            self.angular_velocity,
            -MAX_ANGULAR_VELOCITY,
            MAX_ANGULAR_VELOCITY
        )

        self.theta += self.angular_velocity * dt

        self.theta = angle_wrap(self.theta)

    def generate_measurement(self):

        self.update_motion(UPDATE_RATE)

        self.time += UPDATE_RATE

        true_angle = self.theta

        # ====================================================
        # BASE ACOUSTIC NOISE
        # ====================================================

        gaussian_noise = np.random.normal(0, 2.5)

        measured_angle = true_angle + gaussian_noise

        confidence = np.random.uniform(0.7, 1.0)

        # ====================================================
        # MULTIPATH / CORRELATION FAILURES
        # ====================================================

        failure_probability = np.random.rand()

        # Large reflection peak
        if failure_probability < 0.06:
            measured_angle += np.random.choice([-120, 120])
            confidence = np.random.uniform(0.05, 0.35)

        # Moderate correlation confusion
        elif failure_probability < 0.12:
            measured_angle += np.random.normal(0, 25)
            confidence = np.random.uniform(0.2, 0.55)

        # Temporary weak signal
        elif failure_probability < 0.17:
            confidence = np.random.uniform(0.2, 0.5)

        measured_angle = angle_wrap(measured_angle)

        return DOAMeasurement(
            timestamp=time.time(),
            angle=measured_angle,
            confidence=confidence,
            is_valid=confidence > 0.15
        )


# ============================================================
# VELOCITY-AWARE KALMAN FILTER
# ============================================================

class AngularKalmanFilter:
    """
    State:

    x = [theta,
         omega]

    theta = angle
    omega = angular velocity
    """

    def __init__(self, dt):

        self.dt = dt

        # State vector
        self.x = np.array([
            0.0,  # theta
            0.0   # omega
        ])

        # State covariance
        self.P = np.eye(2)

        # State transition matrix
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])

        # Observation matrix
        self.H = np.array([[1, 0]])

        # Process noise
        self.Q = np.array([
            [PROCESS_NOISE_ANGLE, 0],
            [0, PROCESS_NOISE_VELOCITY]
        ])

        # Measurement noise
        self.R = np.array([[MEASUREMENT_NOISE]])

    def predict(self):

        self.x = self.F @ self.x

        self.x[0] = angle_wrap(self.x[0])

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_angle, confidence=1.0):

        self.predict()

        # Adapt measurement noise dynamically
        adaptive_R = self.R / max(confidence, 0.1)

        z = np.array([measurement_angle])

        predicted_angle = self.x[0]

        innovation = angle_diff(z[0], predicted_angle)

        y = np.array([innovation])

        S = self.H @ self.P @ self.H.T + adaptive_R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)

        self.x[0] = angle_wrap(self.x[0])

        I = np.eye(2)

        self.P = (I - K @ self.H) @ self.P

        return self.x[0], self.x[1]


# ============================================================
# ADVANCED OUTLIER REJECTION
# ============================================================

class DOAStabilizer:
    """
    Handles:
    - Circular statistics
    - Velocity gating
    - Confidence gating
    - Temporal consistency
    """

    def __init__(self):

        self.history = deque(maxlen=15)

        self.last_timestamp = None

    def is_outlier(self, angle, predicted_angle, confidence):

        # ================================================
        # CONFIDENCE GATING
        # ================================================

        if confidence < CONFIDENCE_THRESHOLD:
            return True

        # ================================================
        # WARMUP
        # ================================================

        if len(self.history) < 5:
            self.history.append(angle)
            return False

        # ================================================
        # CIRCULAR MEAN
        # ================================================

        mean_angle = circular_mean(list(self.history))

        # ================================================
        # DISTANCE TO HISTORY
        # ================================================

        deviation = abs(angle_diff(angle, mean_angle))

        # ================================================
        # DISTANCE TO PREDICTION
        # ================================================

        prediction_error = abs(angle_diff(angle, predicted_angle))

        # Combined gating
        if deviation > OUTLIER_THRESHOLD_DEG and prediction_error > OUTLIER_THRESHOLD_DEG:
            return True

        self.history.append(angle)

        return False


# ============================================================
# MAIN REAL-TIME TRACKER
# ============================================================

class RealTimeDOATracker:

    def __init__(self):

        self.simulator = RealisticDroneSimulator()

        self.kalman = AngularKalmanFilter(UPDATE_RATE)

        self.stabilizer = DOAStabilizer()

        self.raw_history = deque(maxlen=HISTORY_SIZE)
        self.filtered_history = deque(maxlen=HISTORY_SIZE)
        self.true_history = deque(maxlen=HISTORY_SIZE)

        self.confidence_history = deque(maxlen=HISTORY_SIZE)

        self.time_axis = deque(maxlen=HISTORY_SIZE)

        self.start_time = time.time()

        # ====================================================
        # POLAR PLOT
        # ====================================================

        self.fig = plt.figure(figsize=(10, 10))

        self.ax = plt.subplot(111, projection='polar')

        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)

        self.ax.set_ylim(0, 1)

        self.raw_line, = self.ax.plot([], [], 'o', label='Raw DOA')
        self.filtered_line, = self.ax.plot([], [], 'o', label='Filtered Track')
        self.true_line, = self.ax.plot([], [], 'o', label='True Drone')

        self.ax.legend(loc='upper right')

    def process_measurement(self, measurement):

        predicted_angle = self.kalman.x[0]

        # ====================================================
        # OUTLIER REJECTION
        # ====================================================

        is_outlier = self.stabilizer.is_outlier(
            measurement.angle,
            predicted_angle,
            measurement.confidence
        )

        if is_outlier:

            print(
                f"OUTLIER | "
                f"RAW={measurement.angle:7.2f}° | "
                f"CONF={measurement.confidence:.2f}"
            )

            # Predict-only step
            self.kalman.predict()

            filtered_angle = self.kalman.x[0]
            angular_velocity = self.kalman.x[1]

            return filtered_angle, angular_velocity, True

        # ====================================================
        # KALMAN UPDATE
        # ====================================================

        filtered_angle, angular_velocity = self.kalman.update(
            measurement.angle,
            measurement.confidence
        )

        print(
            f"RAW={measurement.angle:7.2f}° | "
            f"FILTERED={filtered_angle:7.2f}° | "
            f"VEL={angular_velocity:7.2f}°/s | "
            f"CONF={measurement.confidence:.2f}"
        )

        return filtered_angle, angular_velocity, False

    def update_plot(self, frame):

        measurement = self.simulator.generate_measurement()

        filtered_angle, angular_velocity, rejected = self.process_measurement(measurement)

        current_time = time.time() - self.start_time

        self.raw_history.append(measurement.angle)
        self.filtered_history.append(filtered_angle)
        self.true_history.append(self.simulator.theta)

        self.confidence_history.append(measurement.confidence)

        self.time_axis.append(current_time)

        # ====================================================
        # UPDATE POLAR DISPLAY
        # ====================================================

        raw_theta = np.radians(measurement.angle)
        filtered_theta = np.radians(filtered_angle)
        true_theta = np.radians(self.simulator.theta)

        self.raw_line.set_data([raw_theta], [0.85])
        self.filtered_line.set_data([filtered_theta], [1.0])
        self.true_line.set_data([true_theta], [0.7])

        self.ax.set_title(
            f"Real-Time Acoustic Drone Tracking\n"
            f"Filtered={filtered_angle:.1f}° | "
            f"Velocity={angular_velocity:.1f}°/s | "
            f"Confidence={measurement.confidence:.2f}"
        )

        return self.raw_line, self.filtered_line, self.true_line

    def run(self):

        interval_ms = UPDATE_RATE * 1000

        animation = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False
        )

        plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    tracker = RealTimeDOATracker()

    tracker.run()
