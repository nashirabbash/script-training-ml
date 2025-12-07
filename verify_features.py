"""
Script untuk verifikasi jumlah fitur sesuai paper.
Expected: 43 features per signal × 7 signals = 301 total features
"""

import numpy as np
from preprocessing_pipeline import (
    extract_raw_signal_features,
    extract_statistical_features,
    extract_timeseries_features,
    extract_spectral_features,
    extract_time_features,
    extract_freq_features,
    TARGET_FS
)

# Generate sample window data (60 seconds @ 64 Hz = 3840 samples)
window_size = int(60 * TARGET_FS)
sample_data = np.random.randn(window_size)

print("="*70)
print("FEATURE EXTRACTION VERIFICATION")
print("="*70)
print(f"\nSample data: {len(sample_data)} samples ({len(sample_data)/TARGET_FS:.1f}s @ {TARGET_FS}Hz)")

# Test individual feature extraction functions
print("\n" + "="*70)
print("INDIVIDUAL FEATURE GROUPS (per signal)")
print("="*70)

# Raw signal features
raw_features = extract_raw_signal_features(sample_data, prefix='test_')
print(f"\n1. Raw Signal Features: {len(raw_features)}")
print(f"   Expected: 4 (mean, RMS, std, median)")
for name in sorted(raw_features.keys())[:10]:
    print(f"   - {name}")

# Statistical features
stat_features = extract_statistical_features(sample_data, prefix='test_')
print(f"\n2. Statistical Features: {len(stat_features)}")
print(f"   Expected: 12 (clearance, crest, impulse, kurtosis, peak, SINAD, SNR, shape, skewness, THD, neg_count, pos_count)")
for name in sorted(stat_features.keys()):
    print(f"   - {name}")

# Time series features
ts_features = extract_timeseries_features(sample_data, prefix='test_')
print(f"\n3. Time Series Features: {len(ts_features)}")
print(f"   Expected: 7 (min, median, max, Q1, Q3, ACF1, PACF1)")
for name in sorted(ts_features.keys()):
    print(f"   - {name}")

# Spectral features
spectral_features = extract_spectral_features(sample_data, TARGET_FS, prefix='test_')
print(f"\n4. Spectral Features: {len(spectral_features)}")
print(f"   Expected: 20 (PSD amplitudes)")
for name in sorted(spectral_features.keys())[:10]:
    print(f"   - {name}")
if len(spectral_features) > 10:
    print(f"   ... ({len(spectral_features) - 10} more)")

# Combined time-domain features
time_features = extract_time_features(sample_data, prefix='test_')
print(f"\n5. Combined Time-Domain Features: {len(time_features)}")
print(f"   Expected: 23 (4 raw + 12 statistical + 7 time series)")

# Combined frequency-domain features
freq_features = extract_freq_features(sample_data, TARGET_FS, prefix='test_')
print(f"\n6. Combined Frequency-Domain Features: {len(freq_features)}")
print(f"   Expected: 20 (spectral)")

# Total per signal
total_per_signal = len(time_features) + len(freq_features)
print(f"\n" + "="*70)
print(f"TOTAL FEATURES PER SIGNAL: {total_per_signal}")
print(f"Expected: 43 (23 time + 20 spectral)")
print(f"="*70)

# Paper expects 7 signals:
# 1. ACC_x, 2. ACC_y, 3. ACC_z, 4. ACC_magnitude
# 5. BVP, 6. EDA, 7. HR, 8. TEMP
# Actually paper likely uses: ACC (mag + 3 axes) + BVP + EDA + HR + TEMP = 7 signals
print(f"\nFor 7 signals (ACC_mag, ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP):")
print(f"  Total features = {total_per_signal} × 7 = {total_per_signal * 7}")
print(f"  Paper expects: 301 features")

# Breakdown
print(f"\n" + "="*70)
print("FEATURE BREAKDOWN CHECK")
print("="*70)
expected_breakdown = {
    "Raw Signal (4)": len(raw_features),
    "Statistical (12)": len(stat_features),
    "Time Series (7)": len(ts_features),
    "Spectral (20)": len(spectral_features),
}

all_match = True
for category, count in expected_breakdown.items():
    expected = int(category.split('(')[1].split(')')[0])
    status = "✓" if count == expected else "✗"
    if count != expected:
        all_match = False
    print(f"{status} {category}: {count} (expected {expected})")

print(f"\n" + "="*70)
if all_match and total_per_signal == 43:
    print("✅ SUCCESS: Feature extraction matches paper specification!")
    print(f"   {total_per_signal} features per signal × 7 signals = {total_per_signal * 7} total")
else:
    print("⚠️  WARNING: Feature count mismatch!")
    print(f"   Got {total_per_signal} per signal, expected 43")
    print(f"   Total would be {total_per_signal * 7}, expected 301")
print("="*70)
