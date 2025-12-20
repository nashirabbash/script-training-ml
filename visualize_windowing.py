"""
=============================================================================
WINDOWING VISUALIZATION - PHYSIOLOGICAL DATA ANALYSIS
=============================================================================
Script untuk visualisasi mendalam hasil windowing preprocessing:
1. Signal comparison (raw vs normalized) per sensor per exam type
2. Window distribution analysis (jumlah window per subject/exam)
3. Feature distribution (histogram, boxplot, correlation matrix)
4. Time-series overlay (bandingkan antar subject untuk exam yang sama)
5. Statistical summary per window

Author: ML Pipeline
Date: 2025-12-08
=============================================================================
"""

import sys
import io
# Fix Windows encoding untuk Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI
# =============================================================================
DATA_ROOT = r"Data"
FEATURES_DIR = r"preprocessed_data"
OUTPUT_DIR = r"preprocessed_data/plots"
TARGET_FS = 64.0  # Sesuaikan dengan preprocessing_pipeline.py

# Sensor files yang akan divisualisasi
SENSOR_FILES = {
    'ACC': 'ACC.csv',
    'BVP': 'BVP.csv',
    'EDA': 'EDA.csv',
    'HR': 'HR.csv',
    'TEMP': 'TEMP.csv'
}

# Color palette
COLORS = {
    'Final': '#e74c3c',           # Red
    'Midterm 1': '#3498db',        # Blue
    'Midterm 2': '#2ecc71',        # Green
    'normalized': '#3498db',       # Blue
    'raw': '#e74c3c'              # Red
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_output_dir():
    """Buat direktori output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}/")


def load_features():
    """Load semua feature files"""
    features = {}
    
    feature_files = [
        'features_final.csv',
        'features_midterm_1.csv',
        'features_midterm_2.csv',
        'features_all.csv'
    ]
    
    for filename in feature_files:
        filepath = os.path.join(FEATURES_DIR, filename)
        if os.path.exists(filepath):
            exam_type = filename.replace('features_', '').replace('.csv', '')
            df = pd.read_csv(filepath)
            features[exam_type] = df
            print(f"  ✅ Loaded {exam_type}: {df.shape}")
        else:
            print(f"  ⚠️  File not found: {filename}")
    
    return features


def load_sensor_data(subject, exam):
    """
    Load raw sensor data untuk satu subject-exam
    
    FORMAT CSV EMPATICA E4:
    - Baris 1: UNIX timestamp (waktu mulai recording)
    - Baris 2: Sampling rate (Hz)
    - Baris 3+: Actual sensor data
    """
    exam_folder_map = {
        'final': 'Final',
        'midterm_1': 'Midterm 1',
        'midterm_2': 'Midterm 2'
    }
    
    exam_folder = exam_folder_map.get(exam, exam)
    session_path = os.path.join(DATA_ROOT, subject, exam_folder)
    
    if not os.path.exists(session_path):
        return None
    
    sensors = {}
    
    # Sampling rates per sensor (dari Empatica E4 specs)
    # Ini juga ada di baris ke-2 CSV, tapi kita hardcode untuk keamanan
    sensor_fs = {
        'ACC': 32,   # 32 Hz
        'BVP': 64,   # 64 Hz
        'EDA': 4,    # 4 Hz
        'HR': 1,     # 1 Hz
        'TEMP': 4    # 4 Hz
    }
    
    for sensor_name, filename in SENSOR_FILES.items():
        filepath = os.path.join(session_path, filename)
        if os.path.exists(filepath):
            try:
                # PENTING: Skip 2 baris pertama (timestamp dan sampling rate)
                data = pd.read_csv(filepath, header=None, skiprows=2)
                sensors[sensor_name] = data.values
                sensors[f'{sensor_name}_fs'] = sensor_fs.get(sensor_name, 64)
            except Exception as e:
                print(f"    Warning: Failed to load {sensor_name}: {e}")
    
    return sensors


def normalize_signal(sig):
    """Normalize signal ke range [0, 1]"""
    sig_min = np.min(sig)
    sig_max = np.max(sig)
    if sig_max - sig_min == 0:
        return np.zeros_like(sig)
    return (sig - sig_min) / (sig_max - sig_min)


def compute_acc_magnitude(acc_data):
    """Hitung magnitude dari ACC (x,y,z)"""
    if acc_data.ndim == 1:
        return acc_data
    if acc_data.shape[1] >= 3:
        x, y, z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
        return np.sqrt(x**2 + y**2 + z**2)
    return acc_data[:, 0]


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_window_distribution(features_dict):
    """
    Plot distribusi jumlah windows per subject dan per exam type
    """
    print("\n📊 Generating window distribution plots...")
    
    # Ambil data dari features_all
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # 1. Bar plot: Windows per subject
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count windows per subject
    subject_counts = df['subject'].value_counts().sort_index()
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(subject_counts)), subject_counts.values, 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Windows', fontsize=12, fontweight='bold')
    ax1.set_title('Window Distribution per Subject', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(subject_counts)))
    ax1.set_xticklabels(subject_counts.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, subject_counts.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Grouped bar: Windows per subject per exam
    ax2 = axes[1]
    
    # Pivot table: subject x exam
    pivot = df.pivot_table(index='subject', columns='exam', aggfunc='size', fill_value=0)
    
    x = np.arange(len(pivot.index))
    width = 0.25
    
    for i, exam in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2 + 0.5) * width
        color = COLORS.get(exam, '#95a5a6')
        ax2.bar(x + offset, pivot[exam].values, width, 
               label=exam, color=color, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Windows', fontsize=12, fontweight='bold')
    ax2.set_title('Window Distribution per Subject per Exam Type', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'window_distribution.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: window_distribution.png")


def plot_signal_comparison_detailed(subject='S1'):
    """
    Plot RAW signals untuk semua sensor (TANPA normalisasi misleading)
    Fokus: Lihat variasi SEBENARNYA dalam data
    """
    print(f"\n📊 Generating RAW signal plots for {subject}...")
    
    exam_types = ['final', 'midterm_1', 'midterm_2']
    
    for exam in exam_types:
        sensors = load_sensor_data(subject, exam)
        
        if not sensors:
            print(f"  ⚠️  No data for {subject}/{exam}")
            continue
        
        # Buat plot grid: 5 sensors (RAW only, no normalization!)
        fig, axes = plt.subplots(5, 1, figsize=(16, 14))
        
        sensor_configs = [
            ('EDA', 'EDA (μS)', '#27ae60', 4),      # 4 Hz
            ('BVP', 'BVP (arbitrary)', '#3498db', 64),  # 64 Hz
            ('HR', 'Heart Rate (BPM)', '#e74c3c', 1),   # 1 Hz
            ('TEMP', 'Skin Temperature (°C)', '#f39c12', 4),  # 4 Hz
            ('ACC', 'Acceleration Magnitude (g)', '#9b59b6', 32)  # 32 Hz
        ]
        
        for idx, (sensor_name, ylabel, color, default_fs) in enumerate(sensor_configs):
            ax = axes[idx]
            
            if sensor_name not in sensors:
                ax.text(0.5, 0.5, f'No {sensor_name} data', 
                       ha='center', va='center', fontsize=14, color='red')
                ax.set_title(f'{sensor_name} - Data Not Available', 
                           fontsize=12, fontweight='bold')
                continue
            
            data = sensors[sensor_name]
            
            # Get actual sampling rate
            fs = sensors.get(f'{sensor_name}_fs', default_fs)
            
            # Handle ACC magnitude
            if sensor_name == 'ACC':
                sig = compute_acc_magnitude(data)
            else:
                sig = data.flatten()
            
            # Time axis dengan sampling rate yang benar
            t = np.arange(len(sig)) / fs
            
            # Plot RAW signal ONLY (no normalization!)
            ax.plot(t, sig, color=color, linewidth=0.8, alpha=0.9)
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color=color)
            ax.set_title(f'{sensor_name} - {exam.replace("_", " ").title()} (Fs={fs} Hz, N={len(sig)})', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':', linewidth=0.7, color='gray')
            ax.tick_params(axis='y', labelcolor=color)
            
            # Add statistics text box
            stats_text = f'Mean: {np.mean(sig):.4f} | Std: {np.std(sig):.4f} | Min: {np.min(sig):.4f} | Max: {np.max(sig):.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
            
            if idx == 4:  # Last plot
                ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Raw Physiological Signals - {subject} - {exam.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        exam_clean = exam.replace('_', '')
        filepath = os.path.join(OUTPUT_DIR, f'raw_signals_{subject}_{exam_clean}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: raw_signals_{subject}_{exam_clean}.png")


def plot_feature_distributions(features_dict, top_n=20):
    """
    Plot distribusi features terpenting
    """
    print(f"\n📊 Generating feature distribution plots...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # Exclude metadata dan target columns
    exclude_cols = ['window_id', 'subject', 'exam', 'start_sec', 'end_sec',
                   'grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 1. Variance-based feature importance
    variances = df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    # 2. Histogram grid untuk top features
    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Plot histogram per exam type
        for exam_type in df['exam'].unique():
            exam_data = df[df['exam'] == exam_type][feature].dropna()
            ax.hist(exam_data, bins=30, alpha=0.5, 
                   label=exam_type, color=COLORS.get(exam_type, '#95a5a6'),
                   edgecolor='black', linewidth=0.5)
        
        ax.set_title(feature, fontsize=9, fontweight='bold')
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {top_n} Feature Distributions (by Variance)', 
                fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'feature_distributions.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: feature_distributions.png")
    
    # 3. Boxplot untuk top features
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Prepare data untuk boxplot
        data_by_exam = [df[df['exam'] == exam][feature].dropna().values 
                       for exam in df['exam'].unique()]
        
        bp = ax.boxplot(data_by_exam, labels=df['exam'].unique(),
                       patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, exam in zip(bp['boxes'], df['exam'].unique()):
            patch.set_facecolor(COLORS.get(exam, '#95a5a6'))
            patch.set_alpha(0.7)
        
        ax.set_title(feature, fontsize=9, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {top_n} Feature Boxplots by Exam Type', 
                fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'feature_boxplots.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: feature_boxplots.png")


def plot_correlation_matrix(features_dict, top_n=30):
    """
    Plot correlation matrix untuk top features
    """
    print(f"\n📊 Generating correlation matrix...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # Exclude metadata dan target columns
    exclude_cols = ['window_id', 'subject', 'exam', 'start_sec', 'end_sec',
                   'grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select top N features by variance
    variances = df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    # Compute correlation
    corr = df[top_features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 14))
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Mask upper triangle
    
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, 
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               annot=False, fmt='.2f', ax=ax,
               vmin=-1, vmax=1)
    
    ax.set_title(f'Feature Correlation Matrix (Top {top_n} by Variance)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: correlation_matrix.png")


def plot_time_series_overlay(features_dict):
    """
    Plot time-series features SETELAH windowing
    Menampilkan bagaimana nilai features berubah antar window
    Plot untuk SEMUA subjects (S1-S10) dengan style solid line tanpa markers
    """
    print(f"\n📊 Generating windowed feature time-series plots...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # Key features untuk setiap sensor (hasil ekstraksi windowing)
    key_features = [
        ('EDA_mean', 'EDA Mean (μS)', '#27ae60'),
        ('HR_mean', 'HR Mean (BPM)', '#e74c3c'),
        ('TEMP_mean', 'TEMP Mean (°C)', '#f39c12'),
        ('BVP_std', 'BVP Std', '#3498db'),
        ('ACC_mag_mean', 'ACC Magnitude Mean', '#9b59b6'),
        ('EDA_std', 'EDA Std', '#1abc9c')
    ]
    
    # Filter features yang ada
    available_features = [(f, label, color) for f, label, color in key_features if f in df.columns]
    
    if not available_features:
        print("  ⚠️  No key features found, skipping...")
        return
    
    # Plot untuk SEMUA subjects (S1-S10)
    for subject in sorted(df['subject'].unique()):  # ALL subjects, no limit
        subject_df = df[df['subject'] == subject].copy()
        
        if subject_df.empty:
            continue
        
        # Create subplots for each feature
        n_features = len(available_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(16, n_features * 2.5))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, (feature, ylabel, color) in enumerate(available_features):
            ax = axes[idx]
            
            # Plot untuk setiap exam type
            for exam_type in sorted(subject_df['exam'].unique()):
                exam_data = subject_df[subject_df['exam'] == exam_type].sort_values('start_sec')
                
                # Window number (x-axis)
                window_nums = range(1, len(exam_data) + 1)
                
                exam_color = COLORS.get(exam_type, color)
                # Style seperti raw signal: solid line, no markers
                ax.plot(window_nums, exam_data[feature].values, 
                       linewidth=0.8, label=exam_type, color=exam_color, alpha=0.9)
            
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.set_title(f'{feature} per Window - {subject}', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3, linestyle=':', linewidth=0.7)
            
            if idx == n_features - 1:
                ax.set_xlabel('Window Number', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Windowed Features Time-Series - {subject}', 
                    fontsize=14, fontweight='bold', y=1.001)
        plt.tight_layout()
        
        filepath = os.path.join(OUTPUT_DIR, f'windowed_features_{subject}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: windowed_features_{subject}.png")


def plot_grade_relationship(features_dict, top_n=12):
    """
    Plot hubungan antara features dan grade
    Scatter plots untuk top features yang paling berkorelasi dengan grade
    """
    print(f"\n📊 Generating grade relationship plots...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    if 'grade' not in df.columns:
        print("  ⚠️  Grade column not found, skipping...")
        return
    
    # Exclude metadata dan target columns
    exclude_cols = ['window_id', 'subject', 'exam', 'start_sec', 'end_sec',
                   'grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Compute correlation dengan grade
    correlations = df[feature_cols].corrwith(df['grade']).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index.tolist()
    
    # Create scatter plots
    n_cols = 3
    n_rows = int(np.ceil(top_n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Scatter plot dengan color by exam type
        for exam_type in df['exam'].unique():
            exam_data = df[df['exam'] == exam_type]
            ax.scatter(exam_data[feature], exam_data['grade'],
                      alpha=0.6, s=30, label=exam_type,
                      color=COLORS.get(exam_type, '#95a5a6'),
                      edgecolors='black', linewidth=0.5)
        
        # Regression line (overall)
        from scipy.stats import linregress
        x = df[feature].dropna()
        y = df.loc[x.index, 'grade']
        
        if len(x) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'k--', linewidth=2, alpha=0.7,
                   label=f'R={r_value:.3f}')
        
        ax.set_xlabel(feature, fontsize=9, fontweight='bold')
        ax.set_ylabel('Grade', fontsize=9, fontweight='bold')
        ax.set_title(f'{feature}\n(Corr: {correlations[feature]:.3f})', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {top_n} Features Most Correlated with Grade', 
                fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'grade_relationship.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: grade_relationship.png")


def generate_summary_statistics(features_dict):
    """
    Generate summary statistics table
    """
    print(f"\n📊 Generating summary statistics...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    summary_path = os.path.join(OUTPUT_DIR, 'windowing_statistics.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WINDOWING VISUALIZATION - SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total windows: {len(df)}\n")
        f.write(f"Total subjects: {df['subject'].nunique()}\n")
        f.write(f"Total exams: {df['exam'].nunique()}\n")
        f.write(f"Total features: {len([c for c in df.columns if c not in ['window_id', 'subject', 'exam', 'start_sec', 'end_sec', 'grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']])}\n\n")
        
        # Per exam statistics
        f.write("WINDOWS PER EXAM TYPE\n")
        f.write("-"*80 + "\n")
        for exam_type in sorted(df['exam'].unique()):
            count = len(df[df['exam'] == exam_type])
            f.write(f"  {exam_type:<20}: {count:>5} windows\n")
        f.write("\n")
        
        # Per subject statistics
        f.write("WINDOWS PER SUBJECT\n")
        f.write("-"*80 + "\n")
        subject_counts = df['subject'].value_counts().sort_index()
        for subject, count in subject_counts.items():
            f.write(f"  {subject:<10}: {count:>5} windows\n")
        f.write("\n")
        
        # Grade distribution
        if 'grade' in df.columns:
            f.write("GRADE DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(df['grade'].describe().to_string())
            f.write("\n\n")
            
            f.write("GRADE VALUE COUNTS\n")
            f.write("-"*80 + "\n")
            grade_counts = df['grade'].value_counts().sort_index()
            for grade, count in grade_counts.items():
                f.write(f"  Grade {grade:<5}: {count:>5} windows\n")
            f.write("\n")
        
        # Feature statistics
        exclude_cols = ['window_id', 'subject', 'exam', 'start_sec', 'end_sec',
                       'grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        f.write("FEATURE VARIANCE (Top 20)\n")
        f.write("-"*80 + "\n")
        variances = df[feature_cols].var().sort_values(ascending=False).head(20)
        for feat, var in variances.items():
            f.write(f"  {feat:<40}: {var:>12.4f}\n")
        f.write("\n")
        
        # Missing values
        f.write("MISSING VALUES\n")
        f.write("-"*80 + "\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            for col, count in missing.items():
                f.write(f"  {col:<40}: {count:>5} ({count/len(df)*100:.2f}%)\n")
        else:
            f.write("  No missing values detected.\n")
        f.write("\n")
    
    print(f"  ✅ Saved: windowing_statistics.txt")


def plot_windowed_heatmap(features_dict):
    """
    Plot heatmap untuk melihat pola features per window
    Visualisasi nilai features di setiap window untuk satu subject
    """
    print(f"\n📊 Generating windowed features heatmap...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # Select key features (mean dan std dari tiap sensor)
    key_features = [
        'EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max',
        'HR_mean', 'HR_std', 'HR_min', 'HR_max',
        'TEMP_mean', 'TEMP_std', 'TEMP_min', 'TEMP_max',
        'BVP_mean', 'BVP_std',
        'ACC_mag_mean', 'ACC_mag_std'
    ]
    
    # Filter features yang ada
    available_features = [f for f in key_features if f in df.columns]
    
    if not available_features:
        print("  ⚠️  No key features found, skipping...")
        return
    
    # Plot untuk SEMUA subjects (S1-S10)
    for subject in sorted(df['subject'].unique()):  # ALL subjects
        subject_df = df[df['subject'] == subject].copy()
        
        if subject_df.empty:
            continue
        
        for exam_type in sorted(subject_df['exam'].unique()):
            exam_data = subject_df[subject_df['exam'] == exam_type].sort_values('start_sec')
            
            if len(exam_data) < 2:
                continue
            
            # Create feature matrix
            feature_matrix = exam_data[available_features].values
            
            # Normalize per feature for better visualization
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_matrix_norm = scaler.fit_transform(feature_matrix)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(14, 8))
            
            im = ax.imshow(feature_matrix_norm.T, aspect='auto', cmap='RdYlBu_r',
                          vmin=-2, vmax=2)
            
            # Labels
            ax.set_yticks(range(len(available_features)))
            ax.set_yticklabels(available_features, fontsize=9)
            
            # X-axis: window numbers
            n_windows = len(exam_data)
            if n_windows > 20:
                # Show every 5th window
                tick_positions = range(0, n_windows, max(1, n_windows // 10))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([f'W{i+1}' for i in tick_positions], fontsize=8, rotation=45)
            else:
                ax.set_xticks(range(n_windows))
                ax.set_xticklabels([f'W{i+1}' for i in range(n_windows)], fontsize=8)
            
            ax.set_xlabel('Window', fontsize=11, fontweight='bold')
            ax.set_ylabel('Features', fontsize=11, fontweight='bold')
            ax.set_title(f'Windowed Features Heatmap - {subject} - {exam_type}\n(Standardized values: blue=low, red=high)', 
                        fontsize=12, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Standardized Value (z-score)', fontsize=10)
            
            plt.tight_layout()
            
            exam_clean = exam_type.replace(' ', '_').lower()
            filepath = os.path.join(OUTPUT_DIR, f'heatmap_{subject}_{exam_clean}.png')
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved: heatmap_{subject}_{exam_clean}.png")


def plot_window_samples(features_dict):
    """
    Plot sample windows - menampilkan beberapa contoh window dan statistik features-nya
    """
    print(f"\n📊 Generating window sample plots...")
    
    if 'all' not in features_dict:
        print("  ⚠️  features_all.csv not found, skipping...")
        return
    
    df = features_dict['all']
    
    # Select first subject and first exam
    subject = sorted(df['subject'].unique())[0]
    subject_df = df[df['subject'] == subject]
    exam_type = sorted(subject_df['exam'].unique())[0]
    exam_data = subject_df[subject_df['exam'] == exam_type].sort_values('start_sec')
    
    # Take first 10 windows
    sample_windows = exam_data.head(10)
    
    # Key features to show
    features_to_plot = ['EDA_mean', 'HR_mean', 'TEMP_mean', 'ACC_mag_mean']
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    
    if not features_to_plot:
        print("  ⚠️  No features to plot, skipping...")
        return
    
    # Create bar chart for each window
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        window_ids = [f"W{i+1}" for i in range(len(sample_windows))]
        values = sample_windows[feature].values
        
        bars = ax.bar(window_ids, values, color=colors[idx], edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Window', fontsize=10, fontweight='bold')
        ax.set_ylabel(feature, fontsize=10, fontweight='bold')
        ax.set_title(f'{feature} - First 10 Windows\n({subject} - {exam_type})', 
                    fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.suptitle(f'Sample Windows Feature Values - {subject} - {exam_type}', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, f'window_samples_{subject}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: window_samples_{subject}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("  WINDOWING VISUALIZATION - PHYSIOLOGICAL DATA ANALYSIS")
    print("="*80)
    
    create_output_dir()
    
    # Load features
    print(f"\n{'='*60}")
    print(f"LOADING FEATURE DATA")
    print(f"{'='*60}")
    features_dict = load_features()
    
    if not features_dict:
        print("\n❌ No feature files found!")
        return
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # 1. Window distribution
    plot_window_distribution(features_dict)
    
    # 2. Raw signal plots (SEBELUM windowing - untuk referensi)
    plot_signal_comparison_detailed(subject='S1')
    
    # 3. Feature distributions
    plot_feature_distributions(features_dict, top_n=20)
    
    # 4. Correlation matrix
    plot_correlation_matrix(features_dict, top_n=30)
    
    # 5. BARU: Windowed features time-series (SETELAH windowing)
    plot_time_series_overlay(features_dict)
    
    # 6. BARU: Windowed features heatmap
    plot_windowed_heatmap(features_dict)
    
    # 7. BARU: Window samples bar chart
    plot_window_samples(features_dict)
    
    # 8. Grade relationship
    plot_grade_relationship(features_dict, top_n=12)
    
    # 9. Summary statistics
    generate_summary_statistics(features_dict)
    
    print(f"\n{'='*60}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  RAW SIGNALS (sebelum windowing):")
    print(f"    - raw_signals_*.png")
    print(f"  ")
    print(f"  WINDOWED FEATURES (setelah windowing):")
    print(f"    - window_distribution.png      → Distribusi jumlah window")
    print(f"    - windowed_features_*.png      → Time-series features per window")
    print(f"    - heatmap_*.png                → Heatmap features per window")
    print(f"    - window_samples_*.png         → Bar chart sample windows")
    print(f"  ")
    print(f"  ANALYSIS:")
    print(f"    - feature_distributions.png    → Histogram features")
    print(f"    - feature_boxplots.png         → Boxplot per exam")
    print(f"    - correlation_matrix.png       → Korelasi antar features")
    print(f"    - grade_relationship.png       → Features vs Grade")
    print(f"    - windowing_statistics.txt     → Summary statistik")
    print(f"\nNext steps:")
    print(f"  1. Review visualizations untuk quality check")
    print(f"  2. Identify potential outliers atau data issues")
    print(f"  3. Proceed dengan ML training jika data terlihat bagus")


if __name__ == "__main__":
    main()
