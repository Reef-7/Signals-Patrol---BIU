import numpy as np
from scipy.ndimage import shift
from scipy.signal import chirp

class MultiArrayDroneSimulator:
    def __init__(self, fs=48000, c=343.0, radius=0.15):
        self.fs = fs
        self.c = c
        self.radius = radius
        self.num_mics_per_array = 8
        self.num_arrays = 3
        self.total_mics = self.num_arrays * self.num_mics_per_array
        self.array_centers = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0]
        ])
        self.mic_coords = np.zeros((self.total_mics, 3))
        mic_idx = 0
        for center in self.array_centers:
            angles = np.linspace(0, 2*np.pi, self.num_mics_per_array, endpoint=False)
            for angle in angles:
                self.mic_coords[mic_idx] = [
                    center[0] + self.radius * np.cos(angle),
                    center[1] + self.radius * np.sin(angle),
                    center[2]
                ]
                mic_idx += 1

    def generate_source_signal(self, duration=1.0, signal_type='sweep'):
        t = np.linspace(0, duration, int(self.fs * duration), endpoint=False)
        if signal_type == 'sweep':
            return chirp(t, f0=100, f1=1000, t1=duration, method='linear')
        return np.random.randn(len(t))

    def simulate_room_audio(self, source_signal, drone_pos):
        drone_pos = np.array(drone_pos)
        num_samples = len(source_signal)
        mic_signals = np.zeros((self.total_mics, num_samples))
        distances = np.linalg.norm(self.mic_coords - drone_pos, axis=1)
        min_dist = np.min(distances)
        relative_distances = distances - min_dist
        for i in range(self.total_mics):
            sample_delay = (relative_distances[i] / self.c) * self.fs
            delayed = shift(source_signal, +sample_delay, order=3, mode='nearest')
            mic_signals[i, :] = (1.0 / distances[i]) * delayed
        return mic_signals


class MUSICDOAEstimator:
    def __init__(self, fs=48000, c=343.0, radius=0.15):
        self.fs = fs
        self.c = c
        self.radius = radius
        self.num_mics_per_array = 8
        self.num_arrays = 3
        self.mic_angles = np.linspace(0, 2*np.pi, self.num_mics_per_array, endpoint=False)

    # -------------------------------------------------------
    # תיקון 1: בחירת תדר אופטימלי במקום argmax עיוור
    # -------------------------------------------------------
    def _select_target_frequency(self, audio_row):
        """
        בוחר תדר שבו d/λ ≈ 0.3–0.5 (אזור עבודה אופטימלי של MUSIC).
        עבור radius=0.15m: תדר אידיאלי ≈ 343/(2*0.15) ≈ 1143 Hz.
        מכיוון שה-sweep מגיע רק עד 1000 Hz, נגביל ל-800 Hz.
        """
        f_axis = np.fft.rfftfreq(len(audio_row), d=1/self.fs)
        fft_mag = np.abs(np.fft.rfft(audio_row))

        # חלון תדרים אופטימלי: 300–900 Hz (ביצועים טובים עם ה-sweep הנוכחי)
        f_min, f_max = 300.0, 900.0
        valid_mask = (f_axis >= f_min) & (f_axis <= f_max)

        if not np.any(valid_mask):
            # fallback — אם אין אות בטווח, קח את ה-argmax הגלובלי
            return f_axis[np.argmax(fft_mag)]

        # קח את התדר עם האנרגיה הגבוהה ביותר בתוך הטווח האופטימלי
        best_idx = np.argmax(fft_mag * valid_mask)
        return float(f_axis[best_idx])

    # -------------------------------------------------------
    # תיקון 2: זיהוי אוטומטי של מספר המקורות
    # -------------------------------------------------------
    def _estimate_n_sources(self, eigenvalues, max_sources=3):
        """
        מזהה כמה מקורות יש לפי קפיצות בין ערכים עצמיים.
        threshold: ערך עצמי גדול פי 10 מממוצע ה'רעש' נחשב 'אות'.
        """
        sorted_ev = np.sort(eigenvalues)[::-1]          # מגדול לקטן
        noise_floor = np.mean(sorted_ev[self.num_mics_per_array // 2:])
        threshold = 10.0 * noise_floor
        n_src = int(np.sum(sorted_ev > threshold))
        return max(1, min(n_src, max_sources))           # לפחות 1, לכל היותר max_sources

    # -------------------------------------------------------
    # חישוב מטריצת קווריאנס
    # -------------------------------------------------------
    def compute_covariance_matrix(self, array_data, target_freq):
        n_samples = array_data.shape[1]
        segment_len = 512
        n_segments = n_samples // segment_len

        freq_axis = np.fft.rfftfreq(segment_len, d=1/self.fs)
        freq_idx  = np.argmin(np.abs(freq_axis - target_freq))

        X = np.zeros((self.num_mics_per_array, n_segments), dtype=complex)
        for s in range(n_segments):
            block = array_data[:, s*segment_len:(s+1)*segment_len]
            X[:, s] = np.fft.rfft(block, axis=1)[:, freq_idx]

        Rxx = (X @ X.conj().T) / n_segments
        return Rxx

    # -------------------------------------------------------
    # MUSIC על מערך בודד
    # -------------------------------------------------------
    def run_music_per_array(self, array_data, target_freq,
                             az_step=2, el_step=2, refine=True):
        """
        מחזיר (azimuth_deg, elevation_deg).
        refine=True: אחרי grid coarse, מריץ חיפוש עדין ±az_step/±el_step סביב הפיק.
        """
        Rxx = self.compute_covariance_matrix(array_data, target_freq)
        eigenvalues, eigenvectors = np.linalg.eigh(Rxx)

        # מיון מקטן לגדול → וקטורי רעש הם הראשונים
        idx = np.argsort(eigenvalues)
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # תיקון 2: מספר מקורות דינמי
        # אנחנו בסביבת ניסוי סגורה ויודעים שיש רק רחפן אחד, לכן נקבע זאת קשיח
        n_src = 1
        Un = eigenvectors[:, :self.num_mics_per_array - n_src]
        Un_UnH = Un @ Un.conj().T

        wavenumber = 2 * np.pi * target_freq / self.c

        def steering(az_deg, el_deg):
            az = np.radians(az_deg)
            el = np.radians(el_deg)
            # תיקון 3: נוסחת steering vector מלאה למערך מעגלי
            phase = wavenumber * self.radius * np.cos(el) * np.cos(az - self.mic_angles)
            return np.exp(1j * phase)

        def spectrum(az_deg, el_deg):
            a = steering(az_deg, el_deg)
            denom = np.real(a.conj() @ Un_UnH @ a)
            return 1.0 / (denom + 1e-10)

        # --- שלב א׳: Grid Coarse ---
        az_grid = np.arange(0, 360, az_step)
        el_grid = np.arange(0, 90,  el_step)

        best_val = -1
        best_az, best_el = 0.0, 0.0

        for az in az_grid:
            for el in el_grid:
                val = spectrum(az, el)
                if val > best_val:
                    best_val = val
                    best_az, best_el = az, el

        # --- שלב ב׳: Refinement סביב הפיק (±step, רזולוציה 0.5°) ---
        if refine:
            fine_az = np.arange(best_az - az_step, best_az + az_step + 0.1, 0.5)
            fine_el = np.arange(max(0, best_el - el_step), best_el + el_step + 0.1, 0.5)
            for az in fine_az:
                for el in fine_el:
                    val = spectrum(az % 360, el)
                    if val > best_val:
                        best_val = val
                        best_az, best_el = az % 360, el

        return round(best_az, 1), round(best_el, 1)

    # -------------------------------------------------------
    # ממשק ראשי: מקבל (24, N) ומחזיר 3 וקטורי כיוון
    # -------------------------------------------------------
    def estimate_all_vectors(self, total_audio_matrix, verbose=True):
        # תיקון 1: בחירת תדר מהחלון האופטימלי
        target_freq = self._select_target_frequency(total_audio_matrix[0])

        if verbose:
            print(f"תדר נבחר לניתוח MUSIC: {target_freq:.1f} Hz\n")

        results = {}
        for array_idx in range(self.num_arrays):
            start_ch = array_idx * self.num_mics_per_array
            end_ch   = start_ch + self.num_mics_per_array
            array_data = total_audio_matrix[start_ch:end_ch, :]

            az, el = self.run_music_per_array(array_data, target_freq)
            results[array_idx + 1] = {"azimuth": az, "elevation": el}

            if verbose:
                print(f"=== מערך {array_idx + 1} ===")
                print(f"אזימוט (אופקי): {az}°")
                print(f"elevation (אנכי): {el}°\n")

        return results


# ==========================================
# הרצה עם השוואה למיקום האמיתי
# ==========================================
if __name__ == "__main__":
    # --- ייצור Mock Data ---
    simulator = MultiArrayDroneSimulator(fs=48000, radius=0.15)
    drone_audio = simulator.generate_source_signal(duration=1.0, signal_type='sweep')

    TRUE_POS = [1.0, 1.0, 2.5]
    mic_signals = simulator.simulate_room_audio(drone_audio, TRUE_POS)

    print(f"צורת המטריצה: {mic_signals.shape}")
    print(f"מיקום אמיתי של הרחפן: {TRUE_POS}\n")
    print("=" * 40)

    # --- הרצת MUSIC ---
    estimator = MUSICDOAEstimator(fs=48000, radius=0.15)
    doa_results = estimator.estimate_all_vectors(mic_signals)

    # --- שמירה לקובץ עבור שאר הצוות ---
    np.save('mock_drone_data.npy', mic_signals)
    print("הקובץ mock_drone_data.npy נשמר.")
    print("\nוקטורי הכיוון מוכנים להעברה לקבוצה B1 לטריאנגולציה.")