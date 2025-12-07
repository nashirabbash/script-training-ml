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
        df = pd.read_csv(path, header=None)
        # Asumsi: baris pertama bisa header atau langsung data
        # Coba deteksi apakah ada timestamp column
        if df.shape[1] == 1:
            # Single column data (e.g., EDA, BVP, HR, TEMP)
            return df.iloc[:, 0].values
        else:
            # Multiple columns (e.g., ACC: x,y,z atau IBI: time,value)
            return df.values
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
    n_original = len(data)
    duration = n_original / original_fs
    n_target = int(duration * target_fs)
    if n_target <= 1:
        return None
    resampled = signal.resample(data, n_target)
    return resampled

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

def extract_time_features(window_data, prefix=''):
    """Ekstraksi time-domain features"""
    features = {}
    
    if window_data is None or len(window_data) == 0:
        return features
    
    x = window_data.flatten()
    
    # Basic statistics
    features[f'{prefix}mean'] = np.mean(x)
    features[f'{prefix}median'] = np.median(x)
    features[f'{prefix}std'] = np.std(x)
    features[f'{prefix}var'] = np.var(x)
    features[f'{prefix}min'] = np.min(x)
    features[f'{prefix}max'] = np.max(x)
    features[f'{prefix}range'] = np.ptp(x)
    features[f'{prefix}rms'] = np.sqrt(np.mean(x**2))
    
    # Percentiles
    features[f'{prefix}q25'] = np.percentile(x, 25)
    features[f'{prefix}q75'] = np.percentile(x, 75)
    features[f'{prefix}iqr'] = features[f'{prefix}q75'] - features[f'{prefix}q25']
    
    # Higher order
    features[f'{prefix}skewness'] = skew(x)
    features[f'{prefix}kurtosis'] = kurtosis(x)
    
    # Energy
    features[f'{prefix}energy'] = np.sum(x**2)
    
    return features

def extract_freq_features(window_data, fs, prefix=''):
    """Ekstraksi frequency-domain features menggunakan Welch"""
    features = {}
    
    if window_data is None or len(window_data) < WELCH_NPERSEG:
        return features
    
    x = window_data.flatten()
    
    try:
        # Welch periodogram
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(WELCH_NPERSEG, len(x)))
        
        # Total power
        features[f'{prefix}total_power'] = np.trapz(Pxx, f)
        
        # Dominant frequency
        features[f'{prefix}dominant_freq'] = f[np.argmax(Pxx)]
        features[f'{prefix}peak_power'] = np.max(Pxx)
        
        # Spectral centroid
        features[f'{prefix}spectral_centroid'] = np.sum(f * Pxx) / np.sum(Pxx)
        
        # Band powers (physiological relevant bands)
        # VLF: 0.003-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
        vlf_mask = (f >= 0.003) & (f < 0.04)
        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.4)
        
        if np.any(vlf_mask):
            features[f'{prefix}vlf_power'] = np.trapz(Pxx[vlf_mask], f[vlf_mask])
        if np.any(lf_mask):
            features[f'{prefix}lf_power'] = np.trapz(Pxx[lf_mask], f[lf_mask])
        if np.any(hf_mask):
            features[f'{prefix}hf_power'] = np.trapz(Pxx[hf_mask], f[hf_mask])
        
        # LF/HF ratio (cardiac autonomic balance indicator)
        if f'{prefix}lf_power' in features and f'{prefix}hf_power' in features:
            if features[f'{prefix}hf_power'] > 0:
                features[f'{prefix}lf_hf_ratio'] = features[f'{prefix}lf_power'] / features[f'{prefix}hf_power']
        
    except Exception as e:
        print(f"  Warning: Freq feature extraction failed: {e}")
    
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

def get_sensor_fs(session_path):
    """Baca sampling frequency dari info.txt jika ada"""
    info_path = os.path.join(session_path, 'info.txt')
    fs_dict = {
        'ACC': 32,
        'BVP': 64,
        'EDA': 4,
        'HR': 1,
        'TEMP': 4,
        'IBI': None  # Variable
    }
    
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                for line in f:
                    if 'Sample rate' in line or 'Hz' in line:
                        # Parse sampling rate
                        parts = line.strip().split()
                        for p in parts:
                            try:
                                fs = float(p)
                                if fs > 0:
                                    return fs
                            except:
                                pass
        except:
            pass
    
    return fs_dict

def load_session_sensors(session_path, target_fs=TARGET_FS):
    """Load semua sensor dari satu session dan resample ke target_fs"""
    sensors = {}
    fs_dict = get_sensor_fs(session_path)
    
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
            # Get original fs
            if isinstance(fs_dict, dict):
                original_fs = fs_dict.get(sensor_name, target_fs)
            else:
                original_fs = fs_dict
            
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
            
            sensors[sensor_name] = resampled
    
    return sensors

def process_subject_session(subject, session, data_root=DATA_ROOT):
    """Process satu subject-session"""
    session_path = os.path.join(data_root, subject, session)
    
    if not os.path.isdir(session_path):
        return None
    
    print(f"\n  Processing: {subject}/{session}")
    
    # Load sensors
    sensors = load_session_sensors(session_path, TARGET_FS)
    
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
        f.write(f"Rolling Window (Activity Detection): {ROLLING_SEC}s\n")
        f.write(f"Threshold Percentile: {THRESHOLD_PERCENTILE}\n")
        f.write(f"Min Segment Duration: {MIN_SEGMENT_SEC}s\n\n")
        
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
