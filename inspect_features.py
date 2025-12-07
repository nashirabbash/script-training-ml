"""
Quick inspection script untuk melihat hasil preprocessing
"""
import pandas as pd
import numpy as np

print("="*70)
print("FEATURE DATASETS INSPECTION")
print("="*70)

# Load datasets
datasets = {
    'Final': 'preprocessed_data/features_final.csv',
    'Midterm 1': 'preprocessed_data/features_midterm_1.csv',
    'Midterm 2': 'preprocessed_data/features_midterm_2.csv',
    'All': 'preprocessed_data/features_all.csv'
}

for name, path in datasets.items():
    print(f"\n{'─'*70}")
    print(f"{name}")
    print(f"{'─'*70}")
    
    df = pd.read_csv(path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nFirst 5 columns: {df.columns[:5].tolist()}")
    print(f"\nMetadata columns:")
    meta_cols = [c for c in df.columns if c in ['subject', 'session', 'window_id', 'start_sec', 'end_sec']]
    print(f"  {meta_cols}")
    
    print(f"\nFeature columns (total {len(df.columns) - len(meta_cols)}):")
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  First 10: {feat_cols[:10]}")
    
    print(f"\nSubjects: {sorted(df['subject'].unique())}")
    print(f"Windows per subject:")
    print(df['subject'].value_counts().sort_index())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  Total: {missing}")
        cols_with_missing = df.isnull().sum()[df.isnull().sum() > 0]
        print(f"  Columns with missing: {len(cols_with_missing)}")
        print(f"  Top 5 columns:")
        print(cols_with_missing.nlargest(5))
    else:
        print(f"  ✅ No missing values!")
    
    print(f"\nBasic statistics (first 3 feature columns):")
    print(df[feat_cols[:3]].describe())

print(f"\n{'='*70}")
print("✅ Inspection complete!")
print("="*70)
print("\nNext: Use these datasets with tuning scripts")
print("Example: Update tuning_anova.py to load 'preprocessed_data/features_final.csv'")
