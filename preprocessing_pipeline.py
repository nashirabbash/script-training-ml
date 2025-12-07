"""
=============================================================================
PREPROCESSING PIPELINE - PHYSIOLOGICAL DATA
=============================================================================
Script untuk preprocessing data sensor physiological dengan activity-based windowing:
1. Deteksi periode aktif berdasarkan rolling energy ACC/BVP/EDA
2. Tentukan standard window length dari median durasi aktif
3. Ekstraksi fitur time-domain dan frequency-domain per window
4. Output: feature matrix siap untuk ML training + visualisasi

Output files:
- features_final.csv, features_midterm1.csv, features_midterm2.csv
- windowing_visualization.png (plot per subject/session)
- preprocessing_summary.txt

Author: ML Pipeline
Date: 2025-12-07
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================
DATA_ROOT = r"Data"
OUTPUT_DIR = r"preprocessed_data"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
GRADE_MAPPING_FILE = r"grade_mapping.csv"  # Mapping subject-exam ke grade/performance

# Exam duration (sesuai dataset notes)
EXAM_DURATIONS = {
    'Midterm 1': 90 * 60,  # 1.5 hours in seconds
    'Midterm 2': 90 * 60,  # 1.5 hours in seconds
    'Final': 180 * 60      # 3 hours in seconds
}
EXAM_START_HOUR = 9  # All exams start at 9:00 AM

# Signal preprocessing (sesuai paper)
GAUSSIAN_WINDOW_SEC = 180    # 3-minute Gaussian smoothing window
INTERPOLATE_LIMIT = 100      # Max consecutive NaN values to interpolate

# Windowing parameters
ROLLING_SEC = 2.0            # Rolling window untuk deteksi aktivitas (seconds)
THRESHOLD_PERCENTILE = 65    # Percentile threshold untuk aktivitas
MIN_SEGMENT_SEC = 5.0        # Minimal durasi segmen aktif (seconds)
WINDOW_SIZE_SEC = 60         # Fixed window size untuk semua data (seconds)
WINDOW_OVERLAP_SEC = 0       # Overlap antar window (0 = non-overlapping)

# Resampling
TARGET_FS = 64.0             # Target sampling frequency (Hz)

# Feature extraction
WELCH_NPERSEG = 256          # FFT window size untuk Welch
N_SPECTRAL_FEATURES = 20     # Number of PSD amplitude features (sesuai paper)

RANDOM_STATE = 42

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_dirs():
    """Buat direktori output jika belum ada"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_grade_mapping(filepath=GRADE_MAPPING_FILE):
    """Load mapping grade dan performance dari CSV"""
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: Grade mapping file not found: {filepath}")
        print(f"   Using default grades...")
        return {}
    
    df = pd.read_csv(filepath)
    mapping = {}
    for _, row in df.iterrows():
        key = f"{row['subject']}_{row['exam']}"
        mapping[key] = {
            'grade': row['grade'],
            'performance': row['performance']
        }
    return mapping

def try_load_sensor(path):
    """Load sensor CSV dengan error handling"""
    if not os.path.exists(path):
        return None
    try:
        # Skip 2 baris pertama (unix timestamp & sample rate)
        df = pd.read_csv(path, header=None, skiprows=2)
        
        if df.shape[1] == 1:
            # Single column data (e.g., EDA, BVP, HR, TEMP)
            return df.iloc[:, 0].values
        else:
            # Multiple columns (e.g., ACC: x,y,z)
            # Remove any leading/trailing whitespace and convert to float
            data = df.values.astype(float)
            return data
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
        return None

def compute_acc_magnitude(acc_data):
    """Hitung magnitude dari ACC (x,y,z)"""
    if acc_data is None or len(acc_data) == 0:
        return None
    if acc_data.ndim == 1:
        return acc_data  # Fallback
    if acc_data.shape[1] >= 3:
        x, y, z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
        mag = np.sqrt(x**2 + y**2 + z**2)
        return mag
    return acc_data[:, 0]  # Fallback ke kolom pertama

def resample_uniform(data, original_fs, target_fs):
    """Resample data ke target frequency menggunakan scipy.signal.resample"""
    if data is None or len(data) == 0:
        return None
    # If sampling rates are equal, nothing to do
    if original_fs == target_fs:
        return data

    # Try a polyphase resampling (faster and lower memory than FFT resample)
    try:
        from fractions import Fraction
        # Create a rational approximation of the ratio
        ratio = Fraction(target_fs / original_fs).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        # signal.resample_poly is efficient (C optimized)
        resampled = signal.resample_poly(data, up, down)
        return resampled
    except Exception:
        # Fallback to FFT-based resample
        n_original = len(data)
        duration = n_original / original_fs
        n_target = int(duration * target_fs)
        if n_target <= 1:
            return None
        resampled = signal.resample(data, n_target)
        return resampled

def truncate_to_exam_duration(data, fs, exam_type):
    """
    Truncate signal to exam duration only (sesuai paper step 1).
    Exam starts at 9:00 AM, duration depends on exam type.
    """
    if data is None or len(data) == 0:
        return None
    
    # Get exam duration
    duration_sec = EXAM_DURATIONS.get(exam_type, None)
    if duration_sec is None:
        print(f"    Warning: Unknown exam type '{exam_type}', skipping truncation")
        return data
    
    # Calculate expected samples
    expected_samples = int(duration_sec * fs)
    
    # Truncate or pad
    if len(data) > expected_samples:
        return data[:expected_samples]
    elif len(data) < expected_samples:
        # If shorter, pad with last value (edge padding)
        pad_length = expected_samples - len(data)
        if data.ndim == 1:
            padded = np.pad(data, (0, pad_length), mode='edge')
        else:
            padded = np.pad(data, ((0, pad_length), (0, 0)), mode='edge')
        return padded
    
    return data

def interpolate_missing(data):
    """
    Linear interpolation untuk missing data points (sesuai paper step 2).
    Handles NaN values and maintains time continuity.
    """
    if data is None or len(data) == 0:
        return None
    
    if data.ndim == 1:
        # Single dimension
        series = pd.Series(data)
        # Interpolate NaN values
        interpolated = series.interpolate(method='linear', limit=INTERPOLATE_LIMIT, limit_direction='both')
        # Fill remaining NaN with forward/backward fill
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
        return interpolated.values
    else:
        # Multi-dimensional (e.g., ACC)
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            series = pd.Series(data[:, i])
            interpolated = series.interpolate(method='linear', limit=INTERPOLATE_LIMIT, limit_direction='both')
            interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
            result[:, i] = interpolated.values
        return result

def zscore_normalize(data):
    """
    Z-score normalization (sesuai paper step 3).
    Transforms data to have mean=0 and std=1.
    """
    if data is None or len(data) == 0:
        return None
    
    if data.ndim == 1:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    else:
        # Multi-dimensional: normalize each column independently
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            if std == 0:
                result[:, i] = 0
            else:
                result[:, i] = (data[:, i] - mean) / std
        return result

def gaussian_smooth(data, fs, window_sec=GAUSSIAN_WINDOW_SEC):
    """
    Gaussian smoothing dengan 3-minute window (sesuai paper step 4).
    Removes high-frequency noise while preserving important features.
    """
    if data is None or len(data) == 0:
        return None
    
    # Use scipy.ndimage.gaussian_filter1d which is C-optimized and much faster
    try:
        from scipy.ndimage import gaussian_filter1d
        # sigma in samples; use window / 6 as earlier design
        sigma_samples = (window_sec * fs) / 6.0

        if data.ndim == 1:
            return gaussian_filter1d(data, sigma=sigma_samples, mode='reflect')
        else:
            # Multi-dimensional: smooth each column independently
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                result[:, i] = gaussian_filter1d(data[:, i], sigma=sigma_samples, mode='reflect')
            return result
    except Exception:
        # Fallback to original convolution method if gaussian_filter1d unavailable
        window_samples = int(window_sec * fs)
        if window_samples % 2 == 0:
            window_samples += 1
        sigma = window_samples / 6.0
        if data.ndim == 1:
            window = signal.windows.gaussian(window_samples, sigma)
            window = window / window.sum()
            smoothed = np.convolve(data, window, mode='same')
            return smoothed
        else:
            result = np.zeros_like(data)
            window = signal.windows.gaussian(window_samples, sigma)
            window = window / window.sum()
            for i in range(data.shape[1]):
                result[:, i] = np.convolve(data[:, i], window, mode='same')
            return result

def preprocess_signal(data, fs, exam_type):
    """
    Complete preprocessing pipeline sesuai paper:
    1. Truncate to exam duration
    2. Linear interpolation for missing data
    3. Z-score normalization
    4. Gaussian smoothing (3-minute window)
    """
    if data is None or len(data) == 0:
        return None
    
    # Step 1: Truncate
    data = truncate_to_exam_duration(data, fs, exam_type)
    if data is None:
        return None
    
    # Step 2: Interpolate missing values
    data = interpolate_missing(data)
    if data is None:
        return None
    
    # Step 3: Z-score normalization
    data = zscore_normalize(data)
    if data is None:
        return None
    
    # Step 4: Gaussian smoothing
    data = gaussian_smooth(data, fs)
    if data is None:
        return None
    
    return data

def rolling_energy(sig, fs, win_sec=ROLLING_SEC):
    """Hitung rolling energy untuk deteksi aktivitas"""
    if len(sig) == 0:
        return np.array([])
    win_samples = max(1, int(win_sec * fs))
    # Squared signal
    sq = sig ** 2
    # Moving average
    kernel = np.ones(win_samples) / win_samples
    energy = np.convolve(sq, kernel, mode='same')
    return energy

def detect_active_segments(sig, fs, threshold_pct=THRESHOLD_PERCENTILE, min_dur=MIN_SEGMENT_SEC):
    """Deteksi segmen aktif dari sinyal"""
    if len(sig) == 0:
        return []
    
    # Compute rolling energy
    en = rolling_energy(sig, fs)
    
    # Threshold
    thr = np.percentile(en, threshold_pct)
    mask = en >= thr
    
    # Find segments
    segments = []
    in_segment = False
    start_idx = 0
    
    for i in range(len(mask)):
        if mask[i] and not in_segment:
            start_idx = i
            in_segment = True
        elif not mask[i] and in_segment:
            end_idx = i - 1
            duration = (end_idx - start_idx + 1) / fs
            if duration >= min_dur:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_sec': start_idx / fs,
                    'end_sec': end_idx / fs,
                    'duration': duration
                })
            in_segment = False
    
    # Check if last segment is active
    if in_segment:
        end_idx = len(mask) - 1
        duration = (end_idx - start_idx + 1) / fs
        if duration >= min_dur:
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_sec': start_idx / fs,
                'end_sec': end_idx / fs,
                'duration': duration
            })
    
    return segments

def extract_windows_from_segments(segments, window_sec, fs):
    """Ekstraksi non-overlapping windows dari segmen aktif"""
    windows = []
    win_samples = int(window_sec * fs)
    
    for seg in segments:
        start_idx = seg['start_idx']
        end_idx = seg['end_idx']
        seg_len = end_idx - start_idx + 1
        
        # Jumlah windows yang bisa diambil dari segmen ini
        n_windows = seg_len // win_samples
        
        for i in range(n_windows):
            w_start = start_idx + i * win_samples
            w_end = w_start + win_samples - 1
            windows.append({
                'start_idx': w_start,
                'end_idx': w_end,
                'start_sec': w_start / fs,
                'end_sec': w_end / fs,
                'duration': window_sec
            })
    
    return windows

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_statistical_features(window_data, prefix=''):
    """
    Ekstraksi 12 statistical features (sesuai paper):
    1. clearance_factor, 2. crest_factor, 3. impulse_factor, 4. kurtosis,
    5. peak_value, 6. SINAD, 7. SNR, 8. shape_factor, 9. skewness,
    10. THD, 11. negative_count, 12. positive_count
    """
    features = {}
    
    if window_data is None or len(window_data) == 0:
        return features
    
    x = window_data.flatten()
    n = len(x)
    
    # Basic values needed for calculations
    abs_x = np.abs(x)
    mean_val = np.mean(x)
    mean_abs = np.mean(abs_x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(abs_x)
    
    # 1. Clearance Factor: peak / (mean(sqrt(abs(x))))^2
    mean_sqrt_abs = np.mean(np.sqrt(abs_x))
    if mean_sqrt_abs > 0:
        features[f'{prefix}clearance_factor'] = peak / (mean_sqrt_abs ** 2)
    else:
        features[f'{prefix}clearance_factor'] = 0
    
    # 2. Crest Factor: peak / RMS
    if rms > 0:
        features[f'{prefix}crest_factor'] = peak / rms
    else:
        features[f'{prefix}crest_factor'] = 0
    
    # 3. Impulse Factor: peak / mean(abs(x))
    if mean_abs > 0:
        features[f'{prefix}impulse_factor'] = peak / mean_abs
    else:
        features[f'{prefix}impulse_factor'] = 0
    
    # 4. Kurtosis (4th moment)
    features[f'{prefix}kurtosis'] = kurtosis(x)
    
    # 5. Peak Value
    features[f'{prefix}peak_value'] = peak
    
    # 6 & 7. SNR and SINAD (Signal-to-Noise metrics)
    # SNR = 10 * log10(signal_power / noise_power)
    # Estimate noise as high-frequency component
    try:
        # Simple noise estimation: signal - smoothed signal
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(x, sigma=3)
        noise = x - smoothed
        
        signal_power = np.var(smoothed)
        noise_power = np.var(noise)
        
        if noise_power > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            features[f'{prefix}snr'] = snr
            
            # SINAD: similar but includes distortion
            # SINAD = signal_power / (noise_power + distortion_power)
            # Simplified: use SNR as approximation
            features[f'{prefix}sinad'] = snr * 0.95  # Slightly lower than SNR
        else:
            features[f'{prefix}snr'] = 0
            features[f'{prefix}sinad'] = 0
    except:
        features[f'{prefix}snr'] = 0
        features[f'{prefix}sinad'] = 0
    
    # 8. Shape Factor: RMS / mean(abs(x))
    if mean_abs > 0:
        features[f'{prefix}shape_factor'] = rms / mean_abs
    else:
        features[f'{prefix}shape_factor'] = 0
    
    # 9. Skewness (3rd moment)
    features[f'{prefix}skewness'] = skew(x)
    
    # 10. Total Harmonic Distortion (THD)
    # THD = sqrt(sum(harmonics^2)) / fundamental
    try:
        # Use FFT to find harmonics
        fft_vals = np.fft.fft(x)
        fft_mag = np.abs(fft_vals[:n//2])
        
        # Fundamental frequency (largest peak)
        fundamental_idx = np.argmax(fft_mag[1:]) + 1  # Skip DC
        fundamental = fft_mag[fundamental_idx]
        
        # Harmonics (multiples of fundamental)
        harmonics_power = 0
        for h in range(2, min(6, n//(2*fundamental_idx))):  # Up to 5th harmonic
            harmonic_idx = h * fundamental_idx
            if harmonic_idx < len(fft_mag):
                harmonics_power += fft_mag[harmonic_idx] ** 2
        
        if fundamental > 0:
            thd = np.sqrt(harmonics_power) / fundamental
            features[f'{prefix}thd'] = thd
        else:
            features[f'{prefix}thd'] = 0
    except:
        features[f'{prefix}thd'] = 0
    
    # 11. Negative Count: number of samples below zero per second
    # Paper: "Mathematically calculated as the number of samples below zero per second in a signal"
    negative_count = np.sum(x < 0)
    duration_sec = n / TARGET_FS  # Convert samples to seconds
    features[f'{prefix}negative_count'] = negative_count / duration_sec if duration_sec > 0 else 0
    
    # 12. Positive Count: number of samples above zero per second
    # Paper: "Mathematically calculated as the number of samples above zero per second in a signal"
    positive_count = np.sum(x > 0)
    features[f'{prefix}positive_count'] = positive_count / duration_sec if duration_sec > 0 else 0
    
    return features

def extract_raw_signal_features(window_data, prefix=''):
    """
    Ekstraksi 4 raw signal features (sesuai paper):
    1. mean, 2. RMS, 3. std, 4. median
    """
    features = {}
    
    if window_data is None or len(window_data) == 0:
        return features
    
    x = window_data.flatten()
    
    # 1. Mean
    features[f'{prefix}mean'] = np.mean(x)
    
    # 2. RMS (Root Mean Square)
    features[f'{prefix}rms'] = np.sqrt(np.mean(x**2))
    
    # 3. Standard Deviation
    features[f'{prefix}std'] = np.std(x)
    
    # 4. Median
    features[f'{prefix}median'] = np.median(x)
    
    return features

def extract_timeseries_features(window_data, prefix=''):
    """
    Ekstraksi 7 time series features (sesuai paper):
    1. min, 2. median, 3. max, 4. Q1, 5. Q3, 6. ACF1, 7. PACF1
    """
    features = {}
    
    if window_data is None or len(window_data) == 0:
        return features
    
    x = window_data.flatten()
    
    # 1. Minimum
    features[f'{prefix}min'] = np.min(x)
    
    # 2. Median (already in raw features, but paper lists it here too)
    features[f'{prefix}median_ts'] = np.median(x)
    
    # 3. Maximum
    features[f'{prefix}max'] = np.max(x)
    
    # 4. First Quartile (Q1 - 25th percentile)
    features[f'{prefix}q1'] = np.percentile(x, 25)
    
    # 5. Third Quartile (Q3 - 75th percentile)
    features[f'{prefix}q3'] = np.percentile(x, 75)
    
    # 6. Autocorrelation Function with lag 1 (ACF1)
    try:
        if len(x) > 1:
            # ACF at lag 1
            x_shifted = x[1:]
            x_original = x[:-1]
            
            # Pearson correlation
            if np.std(x_shifted) > 0 and np.std(x_original) > 0:
                acf1 = np.corrcoef(x_original, x_shifted)[0, 1]
                features[f'{prefix}acf1'] = acf1
            else:
                features[f'{prefix}acf1'] = 0
        else:
            features[f'{prefix}acf1'] = 0
    except:
        features[f'{prefix}acf1'] = 0
    
    # 7. Partial Autocorrelation Function with lag 1 (PACF1)
    try:
        # Try using statsmodels for accurate PACF calculation
        try:
            from statsmodels.tsa.stattools import pacf
            # Calculate PACF with lag 1 (returns array: [PACF(0)=1, PACF(1), ...])
            pacf_values = pacf(x, nlags=1, method='ywunbiased')
            features[f'{prefix}pacf1'] = pacf_values[1] if len(pacf_values) > 1 else 0
        except ImportError:
            # Fallback: PACF(1) = ACF(1) for lag 1 (true untuk AR(1) model)
            features[f'{prefix}pacf1'] = features.get(f'{prefix}acf1', 0)
    except:
        features[f'{prefix}pacf1'] = 0
    
    return features

def extract_spectral_features(window_data, fs, prefix=''):
    """
    Ekstraksi 20 spectral features (sesuai paper):
    20 PSD amplitude features dari Welch method
    Paper: "Twenty spectral features: obtained using the Welch power spectral 
    density estimation method. Features were captured as the amplitude of PSD 
    without its specific frequency."
    """
    features = {}
    
    if window_data is None or len(window_data) < WELCH_NPERSEG:
        return features
    
    x = window_data.flatten()
    
    try:
        # Welch periodogram
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(WELCH_NPERSEG, len(x)))
        
        # Extract N_SPECTRAL_FEATURES amplitude values evenly distributed
        # Exclude DC component (f[0])
        n_bins = len(Pxx)
        
        if n_bins > 1:
            # Select N_SPECTRAL_FEATURES frequency bins evenly spaced
            indices = np.linspace(1, n_bins-1, N_SPECTRAL_FEATURES, dtype=int)
            
            for i, idx in enumerate(indices):
                # PSD amplitude only (sesuai paper: "without its specific frequency")
                features[f'{prefix}psd_amp_{i+1}'] = Pxx[idx]
    
    except Exception as e:
        # If extraction fails, return empty features
        pass
    
    return features

def extract_time_features(window_data, prefix=''):
    """
    Ekstraksi time-domain features sesuai paper.
    Paper categories:
    - 4 raw signal features: mean, RMS, std, median
    - 12 statistical features: clearance_factor, crest_factor, impulse_factor, 
      kurtosis, peak_value, SINAD, SNR, shape_factor, skewness, THD, 
      negative_count, positive_count
    - 7 time series features: min, median, max, Q1, Q3, ACF1, PACF1
    Total: 23 time-domain features per signal
    """
    features = {}
    
    if window_data is None or len(window_data) == 0:
        return features
    
    # Combine all time-domain features sesuai paper
    features.update(extract_raw_signal_features(window_data, prefix))      # 4 features
    features.update(extract_statistical_features(window_data, prefix))    # 12 features
    features.update(extract_timeseries_features(window_data, prefix))     # 7 features
    # Total: 23 time-domain features
    
    return features

def extract_freq_features(window_data, fs, prefix=''):
    """
    Ekstraksi frequency-domain features sesuai paper.
    Paper menggunakan 20 spectral features dari Welch PSD.
    """
    features = {}
    
    if window_data is None or len(window_data) < WELCH_NPERSEG:
        return features
    
    # Spectral features (20 PSD amplitudes) - sesuai paper
    features.update(extract_spectral_features(window_data, fs, prefix))
    
    return features

def extract_window_features(sensors_data, window, fs):
    """Ekstraksi semua fitur untuk satu window dari semua sensor"""
    features = {}
    
    start_idx = window['start_idx']
    end_idx = window['end_idx']
    
    # Metadata
    features['start_sec'] = window['start_sec']
    features['end_sec'] = window['end_sec']
    
    # Extract features per sensor
    for sensor_name, data in sensors_data.items():
        if data is None or len(data) == 0:
            continue
        
        # Get window data
        window_data = data[start_idx:end_idx+1]
        
        if len(window_data) == 0:
            continue
        
        # Handle multi-dimensional data (ACC)
        if window_data.ndim > 1:
            # Extract magnitude
            if window_data.shape[1] >= 3:
                mag = np.sqrt(window_data[:, 0]**2 + window_data[:, 1]**2 + window_data[:, 2]**2)
                prefix = f'{sensor_name}_mag_'
                features.update(extract_time_features(mag, prefix))
                features.update(extract_freq_features(mag, fs, prefix))
                
                # Individual axes
                for i, axis in enumerate(['x', 'y', 'z']):
                    prefix = f'{sensor_name}_{axis}_'
                    features.update(extract_time_features(window_data[:, i], prefix))
            else:
                prefix = f'{sensor_name}_'
                features.update(extract_time_features(window_data[:, 0], prefix))
                features.update(extract_freq_features(window_data[:, 0], fs, prefix))
        else:
            # Single dimension data
            prefix = f'{sensor_name}_'
            features.update(extract_time_features(window_data, prefix))
            features.update(extract_freq_features(window_data, fs, prefix))
    
    return features

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def get_sensor_fs(session_path, sensor_name):
    """Baca sampling frequency dari baris ke-2 file CSV"""
    # Default sampling frequencies
    default_fs = {
        'ACC': 32,
        'BVP': 64,
        'EDA': 4,
        'HR': 1,
        'TEMP': 4,
        'IBI': None  # Variable
    }
    
    # Try to read from actual file
    sensor_file = os.path.join(session_path, f'{sensor_name}.csv')
    if os.path.exists(sensor_file):
        try:
            # Read second row (sample rate)
            with open(sensor_file, 'r') as f:
                next(f)  # Skip first line (timestamp)
                fs_line = next(f).strip()
                # Handle ACC format: "32.000000, 32.000000, 32.000000"
                fs_value = float(fs_line.split(',')[0])
                return fs_value
        except Exception as e:
            pass
    
    # Fallback to default
    return default_fs.get(sensor_name, 64)

def load_session_sensors(session_path, exam_type, target_fs=TARGET_FS):
    """Load semua sensor dari satu session, resample, dan apply preprocessing pipeline"""
    sensors = {}
    
    sensor_files = {
        'ACC': 'ACC.csv',
        'BVP': 'BVP.csv',
        'EDA': 'EDA.csv',
        'HR': 'HR.csv',
        'TEMP': 'TEMP.csv'
    }
    
    for sensor_name, filename in sensor_files.items():
        path = os.path.join(session_path, filename)
        data = try_load_sensor(path)
        
        if data is not None:
            # Get original fs from file
            original_fs = get_sensor_fs(session_path, sensor_name)
            
            # Resample
            if data.ndim == 1:
                resampled = resample_uniform(data, original_fs, target_fs)
            else:
                # Multi-dimensional (ACC)
                resampled_cols = []
                for i in range(data.shape[1]):
                    r = resample_uniform(data[:, i], original_fs, target_fs)
                    if r is not None:
                        resampled_cols.append(r)
                if resampled_cols:
                    resampled = np.column_stack(resampled_cols)
                else:
                    resampled = None
            
            # Apply preprocessing pipeline (paper's method)
            if resampled is not None:
                preprocessed = preprocess_signal(resampled, target_fs, exam_type)
                sensors[sensor_name] = preprocessed
    
    return sensors

def process_subject_session(subject, session, data_root=DATA_ROOT):
    """Process satu subject-session"""
    session_path = os.path.join(data_root, subject, session)
    
    if not os.path.isdir(session_path):
        return None
    
    print(f"\n  Processing: {subject}/{session}")
    
    # Load sensors with preprocessing
    sensors = load_session_sensors(session_path, session, TARGET_FS)
    
    if not sensors:
        print(f"    No sensors loaded, skipping...")
        return None
    
    # Pilih sensor untuk activity detection (prioritas: ACC > BVP > EDA)
    activity_signal = None
    if 'ACC' in sensors and sensors['ACC'] is not None:
        activity_signal = compute_acc_magnitude(sensors['ACC'])
    elif 'BVP' in sensors and sensors['BVP'] is not None:
        activity_signal = sensors['BVP'].flatten()
    elif 'EDA' in sensors and sensors['EDA'] is not None:
        activity_signal = sensors['EDA'].flatten()
    
    if activity_signal is None:
        print(f"    No activity signal available, skipping...")
        return None
    
    # Detect active segments
    segments = detect_active_segments(activity_signal, TARGET_FS)
    
    if not segments:
        print(f"    No active segments detected, skipping...")
        return None
    
    # Hitung longest segment duration
    longest_duration = max([s['duration'] for s in segments])
    
    print(f"    Detected {len(segments)} active segments")
    print(f"    Longest segment: {longest_duration:.1f}s")
    
    return {
        'subject': subject,
        'session': session,
        'sensors': sensors,
        'activity_signal': activity_signal,
        'segments': segments,
        'longest_duration': longest_duration,
        'total_duration': len(activity_signal) / TARGET_FS
    }

def determine_standard_window(all_sessions_data):
    """
    Tentukan time range GLOBAL yang aktif di SEMUA recordings.
    Semua subject harus punya data aktif di time range ini.
    Lalu bagi time range ini menjadi fixed windows dengan ukuran sama.
    """
    print(f"\n{'='*60}")
    print(f"DETERMINING COMMON ACTIVE TIME RANGE")
    print(f"{'='*60}")
    
    # Kumpulkan semua active segments dari semua sessions
    all_segments_info = []
    
    for session_data in all_sessions_data:
        subject = session_data['subject']
        session = session_data['session']
        segments = session_data['segments']
        
        if not segments:
            continue
        
        # Untuk setiap session, cari time range keseluruhan yang aktif
        # (dari segment pertama start sampai segment terakhir end)
        start_time = segments[0]['start_sec']
        end_time = segments[-1]['end_sec']
        
        all_segments_info.append({
            'subject': subject,
            'session': session,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        })
        
        print(f"  {subject}/{session}: active {start_time:.1f}s - {end_time:.1f}s (duration: {end_time-start_time:.1f}s)")
    
    if not all_segments_info:
        print("\n❌ No active segments found!")
        return WINDOW_SIZE_SEC, 0, 0
    
    # Cari OVERLAP time range (intersection) - time yang aktif di SEMUA recordings
    global_start = max([s['start'] for s in all_segments_info])
    global_end = min([s['end'] for s in all_segments_info])
    
    if global_end <= global_start:
        print(f"\n⚠️  No common overlap found! Using longest segment approach...")
        # Fallback: gunakan rata-rata start dan durasi terpendek
        global_start = np.mean([s['start'] for s in all_segments_info])
        min_duration = min([s['duration'] for s in all_segments_info])
        global_end = global_start + min_duration
    
    common_duration = global_end - global_start
    
    print(f"\n{'─'*60}")
    print(f"COMMON ACTIVE TIME RANGE FOR ALL SUBJECTS:")
    print(f"  Start time: {global_start:.1f}s")
    print(f"  End time: {global_end:.1f}s")
    print(f"  Duration: {common_duration:.1f}s")
    print(f"  Window size: {WINDOW_SIZE_SEC}s")
    
    # Hitung jumlah windows yang bisa dibuat
    step = WINDOW_SIZE_SEC - WINDOW_OVERLAP_SEC
    n_windows = int((common_duration - WINDOW_SIZE_SEC) / step) + 1
    
    print(f"  Total windows: {n_windows}")
    print(f"{'─'*60}")
    
    return WINDOW_SIZE_SEC, global_start, global_end

def extract_all_features(all_sessions_data, window_sec, global_start, global_end, grade_mapping):
    """
    Ekstraksi fitur untuk semua sessions dengan FIXED TIME WINDOWS.
    Semua subject akan punya window di timestamp yang SAMA PERSIS.
    """
    all_features = []
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION WITH SYNCHRONIZED WINDOWS")
    print(f"{'='*60}")
    
    # Generate fixed time windows (sama untuk semua subject)
    step = window_sec - WINDOW_OVERLAP_SEC
    fixed_windows = []
    current_start = global_start
    
    while current_start + window_sec <= global_end:
        fixed_windows.append({
            'start_sec': current_start,
            'end_sec': current_start + window_sec,
            'duration': window_sec
        })
        current_start += step
    
    print(f"\nGenerated {len(fixed_windows)} synchronized windows:")
    print(f"  Window 1: {fixed_windows[0]['start_sec']:.1f}s - {fixed_windows[0]['end_sec']:.1f}s")
    if len(fixed_windows) > 1:
        print(f"  Window 2: {fixed_windows[1]['start_sec']:.1f}s - {fixed_windows[1]['end_sec']:.1f}s")
    if len(fixed_windows) > 2:
        print(f"  ...")
        print(f"  Window {len(fixed_windows)}: {fixed_windows[-1]['start_sec']:.1f}s - {fixed_windows[-1]['end_sec']:.1f}s")
    
    # Extract features dari setiap subject menggunakan SAME windows
    for session_data in all_sessions_data:
        subject = session_data['subject']
        session = session_data['session']
        sensors = session_data['sensors']
        
        print(f"\n  {subject}/{session}")
        
        # Get grade info
        key = f"{subject}_{session}"
        grade_info = grade_mapping.get(key, {'grade': 0, 'performance': 'unknown'})
        grade = grade_info['grade']
        performance = grade_info['performance']
        
        # Extract features untuk SETIAP fixed window
        extracted_count = 0
        for i, window in enumerate(fixed_windows):
            # Convert time to sample indices
            start_idx = int(window['start_sec'] * TARGET_FS)
            end_idx = int(window['end_sec'] * TARGET_FS)
            
            # Buat temporary window dict dengan indices
            window_with_idx = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_sec': window['start_sec'],
                'end_sec': window['end_sec'],
                'duration': window['duration']
            }
            
            # Extract features
            try:
                features = extract_window_features(sensors, window_with_idx, TARGET_FS)
                
                # Metadata kolom (order penting!)
                features_ordered = {
                    'window_id': f"{subject}_{i+1}",  # Format: S1_1, S1_2, ...
                    'subject': subject,
                    'exam': session,
                    'start_sec': features.pop('start_sec'),
                    'end_sec': features.pop('end_sec')
                }
                
                # Tambahkan fitur sensor
                features_ordered.update(features)
                
                # Tambahkan target kolom di akhir
                features_ordered['grade'] = grade
                features_ordered['passing_70'] = 1 if grade >= 70 else 0
                features_ordered['passing_80'] = 1 if grade >= 80 else 0
                features_ordered['passing_85'] = 1 if grade >= 85 else 0
                features_ordered['passing_90'] = 1 if grade >= 90 else 0
                features_ordered['performance'] = performance
                
                all_features.append(features_ordered)
                extracted_count += 1
                
            except Exception as e:
                print(f"    Warning: Failed to extract window {i+1} ({window['start_sec']:.1f}-{window['end_sec']:.1f}s): {e}")
                continue
        
        print(f"    ✓ Extracted {extracted_count}/{len(fixed_windows)} windows")
    
    return pd.DataFrame(all_features)

def normalize_signal(sig):
    """Normalize signal ke range [0, 1] atau [-1, 1] tergantung distribusi"""
    sig_min = np.min(sig)
    sig_max = np.max(sig)
    if sig_max - sig_min == 0:
        return np.zeros_like(sig)
    normalized = (sig - sig_min) / (sig_max - sig_min)
    # Scale to [-1, 1] if needed (for better visualization)
    # normalized = 2 * normalized - 1
    return normalized

def visualize_windowing(all_sessions_data, window_sec, output_path):
    """
    Buat visualisasi comparison raw vs normalized signals untuk setiap sensor
    Mirip dengan paper: menampilkan EDA, HR, dan TEMP per exam type
    """
    # Group by exam type
    exam_groups = {}
    for session_data in all_sessions_data:
        session = session_data['session']
        if session not in exam_groups:
            exam_groups[session] = []
        exam_groups[session].append(session_data)
    
    # Buat plot untuk setiap exam type
    for exam_type, sessions in exam_groups.items():
        # Ambil satu sample subject untuk visualisasi (subject pertama)
        if not sessions:
            continue
        
        sample_session = sessions[0]  # Ambil subject pertama
        sensors = sample_session['sensors']
        subject = sample_session['subject']
        
        # Siapkan 3 sensor untuk plot: EDA, HR (dari BVP), TEMP
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. EDA (Electrodermal Activity)
        if 'EDA' in sensors and sensors['EDA'] is not None:
            eda = sensors['EDA'].flatten()
            t = np.arange(len(eda)) / TARGET_FS
            
            ax1 = axes[0]
            ax1_twin = ax1.twinx()
            
            # Normalized (blue)
            eda_norm = normalize_signal(eda)
            ax1.plot(t, eda_norm, color='#3498db', linewidth=1.5, label='Normalized', alpha=0.8)
            ax1.set_ylabel('Normalized', color='#3498db', fontsize=11, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#3498db')
            ax1.set_ylim(-0.1, 1.1)
            
            # Raw (red)
            ax1_twin.plot(t, eda, color='#e74c3c', linewidth=1.5, label='EDA μS', alpha=0.8)
            ax1_twin.set_ylabel('EDA μS', color='#e74c3c', fontsize=11, fontweight='bold')
            ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
            
            ax1.set_title(f'Electrodermal activity - {exam_type}', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time, s', fontsize=10)
            ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        # 2. Heart Rate (from BVP or HR)
        hr_data = None
        if 'HR' in sensors and sensors['HR'] is not None:
            hr_data = sensors['HR'].flatten()
        elif 'BVP' in sensors and sensors['BVP'] is not None:
            hr_data = sensors['BVP'].flatten()
        
        if hr_data is not None:
            t = np.arange(len(hr_data)) / TARGET_FS
            
            ax2 = axes[1]
            ax2_twin = ax2.twinx()
            
            # Normalized (blue)
            hr_norm = normalize_signal(hr_data)
            ax2.plot(t, hr_norm, color='#3498db', linewidth=1.5, label='Normalized', alpha=0.8)
            ax2.set_ylabel('Normalized', color='#3498db', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='#3498db')
            ax2.set_ylim(-0.1, 1.1)
            
            # Raw (red)
            ax2_twin.plot(t, hr_data, color='#e74c3c', linewidth=1.5, label='HR BPM', alpha=0.8)
            ax2_twin.set_ylabel('HR BPM', color='#e74c3c', fontsize=11, fontweight='bold')
            ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
            
            ax2.set_title(f'Heart Rate - {exam_type}', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time, s', fontsize=10)
            ax2.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        # 3. Skin Temperature
        if 'TEMP' in sensors and sensors['TEMP'] is not None:
            temp = sensors['TEMP'].flatten()
            t = np.arange(len(temp)) / TARGET_FS
            
            ax3 = axes[2]
            ax3_twin = ax3.twinx()
            
            # Normalized (blue)
            temp_norm = normalize_signal(temp)
            ax3.plot(t, temp_norm, color='#3498db', linewidth=1.5, label='Normalized', alpha=0.8)
            ax3.set_ylabel('Normalized', color='#3498db', fontsize=11, fontweight='bold')
            ax3.tick_params(axis='y', labelcolor='#3498db')
            ax3.set_ylim(-0.1, 1.1)
            
            # Raw (red)
            ax3_twin.plot(t, temp, color='#e74c3c', linewidth=1.5, label='Skin temperature °C', alpha=0.8)
            ax3_twin.set_ylabel('Skin temperature °C', color='#e74c3c', fontsize=11, fontweight='bold')
            ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')
            
            ax3.set_title(f'Skin temperature - {exam_type}', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time, s', fontsize=10)
            ax3.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save per exam type
        filename = f"signal_comparison_{exam_type.lower().replace(' ', '_')}.png"
        filepath = os.path.join(PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 {exam_type}: {filename}")
    
    print(f"\n✅ All signal comparison plots saved to: {PLOTS_DIR}/")

def save_summary_report(all_sessions_data, window_sec, features_df, output_path):
    """Simpan summary report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PREPROCESSING SUMMARY REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Standard Window Size: {window_sec} seconds\n")
        f.write(f"Target Sampling Rate: {TARGET_FS} Hz\n")
        f.write(f"\nPreprocessing Steps (sesuai paper):\n")
        f.write(f"  1. Truncate to exam duration (Midterm: 90min, Final: 180min)\n")
        f.write(f"  2. Linear interpolation for missing data\n")
        f.write(f"  3. Z-score normalization (mean=0, std=1)\n")
        f.write(f"  4. Gaussian smoothing (window: {GAUSSIAN_WINDOW_SEC}s)\n")
        f.write(f"\nWindowing Parameters:\n")
        f.write(f"  Rolling Window (Activity Detection): {ROLLING_SEC}s\n")
        f.write(f"  Threshold Percentile: {THRESHOLD_PERCENTILE}\n")
        f.write(f"  Min Segment Duration: {MIN_SEGMENT_SEC}s\n\n")
        
        f.write("="*70 + "\n")
        f.write("PER-SESSION STATISTICS\n")
        f.write("="*70 + "\n")
        
        for session_data in all_sessions_data:
            subject = session_data['subject']
            session = session_data['session']
            segments = session_data['segments']
            windows = extract_windows_from_segments(segments, window_sec, TARGET_FS)
            
            f.write(f"\n{subject}/{session}:\n")
            f.write(f"  Total Duration: {session_data['total_duration']:.1f}s\n")
            f.write(f"  Active Segments: {len(segments)}\n")
            f.write(f"  Longest Segment: {session_data['longest_duration']:.1f}s\n")
            f.write(f"  Windows Extracted: {len(windows)}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FEATURE MATRIX SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Total Windows: {len(features_df)}\n")
        f.write(f"Total Features: {len(features_df.columns) - 3}\n")  # -3 for metadata cols
        f.write(f"\nFeature Columns ({len(features_df.columns)}):\n")
        for col in features_df.columns:
            f.write(f"  - {col}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FEATURE STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(features_df.describe().to_string())
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("Missing Values:\n")
        f.write("="*70 + "\n")
        missing = features_df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            f.write(missing.to_string())
        else:
            f.write("No missing values detected.\n")
    
    print(f"📄 Summary report saved: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  PREPROCESSING PIPELINE - PHYSIOLOGICAL DATA")
    print("="*70)
    
    create_dirs()
    
    # Collect all subject/session combinations
    subjects = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('S')])
    
    all_sessions_data = []
    
    print(f"\nFound {len(subjects)} subjects")
    
    for subject in subjects:
        subject_path = os.path.join(DATA_ROOT, subject)
        sessions = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
        
        for session in sessions:
            result = process_subject_session(subject, session)
            if result:
                all_sessions_data.append(result)
    
    if not all_sessions_data:
        print("\n❌ No valid sessions found!")
        return
    
    print(f"\n✅ Successfully processed {len(all_sessions_data)} sessions")
    
    # Load grade mapping
    grade_mapping = load_grade_mapping()
    print(f"\n📊 Loaded grades for {len(grade_mapping)} subject-exam combinations")
    
    # Determine common active time range untuk semua subjects
    window_sec, global_start, global_end = determine_standard_window(all_sessions_data)
    
    # Extract features dengan synchronized windows
    features_df = extract_all_features(all_sessions_data, window_sec, global_start, global_end, grade_mapping)
    
    if features_df.empty:
        print("\n❌ No features extracted!")
        return
    
    print(f"\n✅ Feature matrix shape: {features_df.shape}")
    
    # Visualize signal comparison (raw vs normalized)
    print(f"\n{'='*60}")
    print(f"GENERATING SIGNAL COMPARISON PLOTS")
    print(f"{'='*60}")
    visualize_windowing(all_sessions_data, window_sec, None)
    
    # Split by exam type and save
    exam_types = features_df['exam'].unique()
    
    print(f"\n{'='*60}")
    print(f"SAVING FEATURE DATASETS")
    print(f"{'='*60}")
    
    for exam in exam_types:
        exam_df = features_df[features_df['exam'] == exam].copy()
        filename = f"features_{exam.lower().replace(' ', '_')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        exam_df.to_csv(filepath, index=False)
        print(f"  ✅ {exam}: {len(exam_df)} windows → {filename}")
    
    # Save combined
    all_path = os.path.join(OUTPUT_DIR, 'features_all.csv')
    features_df.to_csv(all_path, index=False)
    print(f"  ✅ All: {len(features_df)} windows → features_all.csv")
    
    # Save summary report
    summary_path = os.path.join(OUTPUT_DIR, 'preprocessing_summary.txt')
    save_summary_report(all_sessions_data, window_sec, features_df, summary_path)
    
    print(f"\n{'='*60}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"  - Feature files: features_*.csv")
    print(f"  - Signal plots: plots/signal_comparison_*.png")
    print(f"  - Summary: preprocessing_summary.txt")
    print(f"\nNext steps:")
    print(f"  1. Check signal comparison plots in plots/ folder")
    print(f"  2. Use features_*.csv for ML training")
    print(f"  3. Run: py tuning_anova.py with new datasets")

if __name__ == "__main__":
    main()
