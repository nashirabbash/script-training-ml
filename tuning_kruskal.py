"""
=============================================================================
HYPERPARAMETER TUNING WITH KRUSKAL-WALLIS FEATURE SELECTION
=============================================================================
Script ini menggunakan Kruskal-Wallis H-test untuk seleksi fitur,
kemudian membandingkan 6 classifier dengan GridSearchCV.

Kruskal-Wallis H-test:
- Non-parametric alternative untuk ANOVA
- Tidak mengasumsikan distribusi normal
- Menguji apakah distribusi fitur berbeda signifikan antar kelas
- Cocok untuk fitur KONTINYU dengan distribusi non-normal
- Univariat (mengevaluasi setiap fitur secara independen)

Author: ML Pipeline
Date: 2025
=============================================================================
"""
import sys
import io
# Fix Windows encoding untuk Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# Quick smoke-print to verify the script starts (helps debug silent runs)
print("tuning_kruskal.py: started (stdout wrapper OK)", flush=True)

import os
import warnings
import numpy as np
import pandas as pd

# PENTING: Set backend matplotlib ke 'Agg' SEBELUM import plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# PyTorch MLP (GPU-accelerated)
from pytorch_mlp_wrapper import PyTorchMLPClassifier, check_gpu_availability

# Kruskal-Wallis test
from scipy.stats import kruskal


# =============================================================================
# KONFIGURASI
# =============================================================================
RANDOM_STATE = 42
CV_FOLDS = 5
OUTPUT_DIR = "results_kruskal"
FEATURE_SELECTION_METHOD = "Kruskal-Wallis H-test"

# Berbagai rasio pembagian dataset untuk eksperimen
TEST_SIZES = [0.2, 0.25, 0.3]

# Jumlah fitur yang akan dicoba dalam feature selection
# Sesuaikan dengan jumlah fitur dataset Anda
K_FEATURES = [5, 10, 15, 20, 25, 30]

# Target column - sesuaikan dengan kebutuhan
# NOTE: passing_70 tidak direkomendasikan (semua data passing)
TARGET_COLUMN = 'passing_85'  # Opsi: passing_80, passing_85, passing_90, performance


# =============================================================================
# CUSTOM SCORING FUNCTION UNTUK KRUSKAL-WALLIS (OPTIMIZED)
# =============================================================================

# Global cache untuk menyimpan hasil scoring yang sudah dihitung
# Key: hash dari (X.tobytes(), y.tobytes()), Value: (scores, pvalues)
_kruskal_cache = {}

def kruskal_wallis_score(X, y):
    """
    Scoring function untuk Kruskal-Wallis H-test (OPTIMIZED dengan caching).
    
    Menghitung H-statistic untuk setiap fitur. H-statistic yang lebih tinggi
    menunjukkan perbedaan distribusi yang lebih signifikan antar kelas.
    
    OPTIMASI:
    - Caching hasil untuk menghindari rekomputasi pada fold yang sama
    - Vectorized pre-computation untuk class masks
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    
    Returns:
        scores: Array of H-statistics (n_features,)
    """
    global _kruskal_cache
    
    # Buat cache key dari hash data
    # Gunakan shape + sum sebagai quick hash (lebih cepat dari tobytes)
    cache_key = (X.shape, X.sum(), y.sum(), tuple(np.unique(y)))
    
    if cache_key in _kruskal_cache:
        return _kruskal_cache[cache_key]
    
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    pvalues = np.ones(n_features)
    
    # Get unique classes dan pre-compute masks (sekali saja)
    classes = np.unique(y)
    class_masks = {c: (y == c) for c in classes}
    
    # Untuk setiap fitur, hitung Kruskal-Wallis H-statistic
    for i in range(n_features):
        # Pisahkan data per kelas menggunakan pre-computed masks
        groups = [X[class_masks[c], i] for c in classes]
        
        # Kruskal-Wallis test
        try:
            # Filter out groups with all NaN or empty
            valid_groups = [g for g in groups if len(g) > 0 and not np.all(np.isnan(g))]
            
            if len(valid_groups) >= 2:
                h_stat, p_val = kruskal(*valid_groups)
                scores[i] = float(h_stat) if not np.isnan(h_stat) else 0.0
                pvalues[i] = float(p_val) if not np.isnan(p_val) else 1.0
            else:
                scores[i] = 0.0
                pvalues[i] = 1.0
        except:
            scores[i] = 0
    
    # Simpan ke cache (scores, pvalues)
    _kruskal_cache[cache_key] = (scores, pvalues)
    
    # Batasi ukuran cache (max 50 entries)
    if len(_kruskal_cache) > 50:
        # Hapus entry pertama (oldest)
        _kruskal_cache.pop(next(iter(_kruskal_cache)))
    
    return scores, pvalues


# =============================================================================
# DEFINISI MODEL DAN PARAMETER GRID (untuk Pipeline)
# =============================================================================
def get_models_and_params(n_features):
    """
    Mendefinisikan semua model beserta parameter grid untuk GridSearchCV.
    Parameter sudah di-prefix dengan 'clf__' untuk Pipeline.
    
    Args:
        n_features: Jumlah fitur dalam dataset (untuk menentukan k range)
    
    Returns:
        dict: Dictionary berisi model dan parameter grid-nya
    """
    # Sesuaikan K_FEATURES dengan jumlah fitur yang tersedia
    k_values = [k for k in K_FEATURES if k <= n_features]
    if not k_values:
        k_values = [min(10, n_features)]
    
    models_params = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'selector__k': k_values,
                # Reduced dari 5 → 3 neighbors (skip 5 & 9, keep odd numbers)
                'clf__n_neighbors': [3, 7, 11],
                'clf__weights': ['uniform', 'distance'],
                # Reduced dari 3 → 2 metrics (euclidean & manhattan cukup)
                'clf__metric': ['euclidean', 'manhattan']
            }
        },
        
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                # Reduced dari 4 → 3 C values (hapus 0.1 yang terlalu kecil)
                'clf__C': [1, 10, 100],
                # Reduced dari 4 → 2 kernels (linear & rbf paling efektif)
                # poly & sigmoid: lambat + jarang optimal untuk tabular data
                'clf__kernel': ['linear', 'rbf'],
                'clf__gamma': ['scale', 'auto']
            }
        },
        
        'MLP': {
            # OPTIMISASI MAKSIMAL: minimal grid, max_iter rendah, fokus speed
            # Early stopping akan handle convergence
            'model': PyTorchMLPClassifier(random_state=RANDOM_STATE, max_iter=200, verbose=0),
            'params': {
                # HANYA test k=[10,20,30] untuk MLP (skip 5,15,25)
                'selector__k': [k for k in [10, 20, 30] if k <= n_features],
                # 1 arsitektur saja (proven best for tabular)
                'clf__hidden_layer_sizes': [(100,50)],
                # 1 activation (ReLU industry standard)
                'clf__activation': ['tanh'],
                # 1 alpha (middle ground)
                'clf__alpha': [0.001],
                # 1 learning rate (adaptive paling robust)
                'clf__learning_rate': ['constant'],
                # Batch besar untuk GPU efficiency (skip 32)
                'clf__batch_size': [64]
            }
        },
        
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'params': {
                'selector__k': k_values,
                # Reduced dari 3 → 2 (skip 100, extreme values cukup)
                'clf__n_estimators': [50, 200],
                # Reduced dari 4 → 3 (hapus 20, fokus None/10/30)
                'clf__max_depth': [None, 10, 30],
                # Reduced dari 3 → 2 (skip 5, keep extremes)
                'clf__min_samples_split': [2, 10],
                # Reduced dari 3 → 2 (skip middle value)
                'clf__min_samples_leaf': [1, 4]
            }
        },
        
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {
                'selector__k': k_values,
                'clf__var_smoothing': np.logspace(-12, -6, 7)
            }
        },
        
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                # Reduced dari 5 → 3 (hapus 5 & 15, keep spread values)
                'clf__max_depth': [None, 10, 20],
                # Reduced dari 3 → 2 (skip 5, keep extremes)
                'clf__min_samples_split': [2, 10],
                # Reduced dari 3 → 2 (skip middle value)
                'clf__min_samples_leaf': [1, 4],
                'clf__criterion': ['gini', 'entropy']
            }
        }
    }
    
    return models_params


# =============================================================================
# FUNGSI UTILITAS
# =============================================================================
def create_output_directory(base_dir=OUTPUT_DIR):
    """Buat direktori output dengan timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def save_feature_selection_scores(X, y, feature_names, output_path):
    """
    Menghitung dan menyimpan skor feature selection ke CSV.
    Dipanggil di awal flow sebelum benchmark dimulai.
    
    Args:
        X: Feature matrix (numpy array)
        y: Target vector
        feature_names: List nama fitur
        output_path: Path direktori output
    
    Returns:
        df_scores: DataFrame berisi skor fitur
    """
    print("\n" + "=" * 70)
    print("COMPUTING FEATURE SELECTION SCORES")
    print(f"Method: {FEATURE_SELECTION_METHOD}")
    print("=" * 70)
    
    # Handle missing values dengan median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Hitung Kruskal-Wallis H-scores
    scores, pvalues = kruskal_wallis_score(X_scaled, y)
    
    # Buat DataFrame hasil
    df_scores = pd.DataFrame({
        'feature': feature_names,
        'h_statistic': scores,
        'p_value': pvalues
    })
    
    # Sort by score (descending)
    df_scores = df_scores.sort_values('h_statistic', ascending=False).reset_index(drop=True)
    df_scores['rank'] = range(1, len(df_scores) + 1)
    
    # Reorder columns
    df_scores = df_scores[['rank', 'feature', 'h_statistic', 'p_value']]
    
    # Save to CSV
    csv_path = os.path.join(output_path, 'feature_selection_scores.csv')
    df_scores.to_csv(csv_path, index=False)
    
    print(f"\nTop 10 Features by Kruskal-Wallis H-Statistic:")
    print(df_scores.head(10).to_string(index=False))
    print(f"\n>>> Feature scores saved to: {csv_path}")
    
    return df_scores


def load_and_prepare_data(filepath, target_col=TARGET_COLUMN):
    """
    Load dataset dan prepare untuk training.
    
    Args:
        filepath: Path ke file CSV
        target_col: Nama kolom target
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List nama fitur
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    print(f"Target: {target_col}")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    
    # Drop kolom yang bukan fitur
    metadata_cols = ['window_id', 'subject', 'exam', 'start_sec', 'end_sec']
    target_cols = ['grade', 'passing_70', 'passing_80', 'passing_85', 'passing_90', 'performance']
    
    # Kolom yang akan di-drop
    drop_cols = []
    for col in metadata_cols + target_cols:
        if col in df.columns and col != target_col:
            drop_cols.append(col)
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")
    
    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=drop_cols + [target_col])
    
    feature_names = X.columns.tolist()
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print(f"Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} ({count/len(y)*100:.1f}%)")
    
    # Check for missing values
    missing = X.isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠️  Warning: {missing} missing values detected")
        print(f"   Will be imputed in pipeline")
    
    return X.values, y, feature_names


def split_data(X, y, test_size=0.2):
    """
    Membagi data menjadi train dan test set.
    CATATAN: Scaling dilakukan di dalam Pipeline, bukan di sini.
    
    Args:
        X: Features
        y: Target
        test_size: Proporsi data test
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def create_pipeline(model):
    """
    Membuat Pipeline dengan: Imputer -> Scaler -> Feature Selection (Kruskal-Wallis) -> Classifier
    
    Args:
        model: Classifier model
        
    Returns:
        Pipeline object
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),                   # Standardize features
        ('selector', SelectKBest(kruskal_wallis_score)),# Kruskal-Wallis feature selection
        ('clf', model)                                  # Classifier
    ])
    
    return pipeline


# =============================================================================
# FUNGSI TRAINING DAN EVALUASI
# =============================================================================
def train_with_gridsearch(pipeline, params, X_train, y_train, model_name):
    """
    Training model dengan GridSearchCV.
    
    Args:
        pipeline: sklearn Pipeline
        params: Parameter grid
        X_train: Training features
        y_train: Training labels
        model_name: Nama model
    
    Returns:
        best_pipeline: Pipeline terbaik hasil GridSearch
        best_params: Best parameters
        cv_score: Cross-validation score
    """
    print(f"\n{'─'*70}")
    print(f"Training: {model_name}")
    print(f"{'─'*70}")
    print(f"Parameter grid size: {np.prod([len(v) for v in params.values()])} combinations")
    
    # Convert to NumPy arrays to avoid Pandas serialization issues in parallel processing
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Setup GridSearchCV dengan StratifiedKFold
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # PENTING: Gunakan n_jobs=2 untuk menghindari memory fragmentation
    # Untuk MLP: n_jobs=1 lebih baik karena PyTorch sudah parallel di GPU
    n_jobs = 1 if 'MLP' in model_name else 2
    
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring='f1',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Fit
    print(f"Running GridSearchCV with {CV_FOLDS}-fold CV...")
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Best CV Score: {grid_search.best_score_:.4f}")
    print(f"✓ Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def get_selected_features(pipeline, feature_names):
    """Ambil nama fitur yang terpilih dari pipeline"""
    try:
        selector = pipeline.named_steps['selector']
        selected_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(feature_names, selected_mask) if selected]
        return selected_features
    except:
        return []


def evaluate_model(pipeline, X_test, y_test, model_name, feature_names):
    """
    Mengevaluasi pipeline pada test set.
    
    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test target
        model_name: Nama model
        feature_names: List nama fitur
        
    Returns:
        metrics: Dictionary berisi semua metrik
        y_pred: Prediksi
    """
    y_pred = pipeline.predict(X_test)
    
    # Get selected features
    selected_features = get_selected_features(pipeline, feature_names)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'selected_features': selected_features,
        'n_features_selected': len(selected_features)
    }
    
    print(f"\nTest Results for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Features Selected: {metrics['n_features_selected']}")
    
    return metrics, y_pred


def plot_confusion_matrix(cm, model_name, output_path, test_size):
    """Membuat dan menyimpan visualisasi confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Oranges',
        xticklabels=['Not Passing (0)', 'Passing (1)'],
        yticklabels=['Not Passing (0)', 'Passing (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}\n(Kruskal-Wallis Selection, Test Size: {test_size*100:.0f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name}_test{int(test_size*100)}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def plot_feature_importance(pipeline, feature_names, model_name, output_path, test_size):
    """
    Plot skor Kruskal-Wallis untuk fitur yang dipilih.
    
    Args:
        pipeline: Trained pipeline
        feature_names: Nama fitur asli
        model_name: Nama model
        output_path: Path output
        test_size: Ukuran test set
    """
    try:
        selector = pipeline.named_steps['selector']
        scores = selector.scores_
        mask = selector.get_support()
        
        # Get selected features and their scores
        selected_indices = np.where(mask)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        selected_scores = scores[selected_indices]
        
        # Sort by score (descending)
        sorted_idx = np.argsort(selected_scores)[::-1]
        sorted_features = [selected_features[i] for i in sorted_idx]
        sorted_scores = selected_scores[sorted_idx]
        
        # Plot top 20 features
        n_top = min(20, len(sorted_features))
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(n_top), sorted_scores[:n_top], color='#e67e22', alpha=0.8)
        plt.yticks(range(n_top), sorted_features[:n_top], fontsize=9)
        plt.xlabel('Kruskal-Wallis H-statistic', fontsize=11)
        plt.title(f'Top {n_top} Features - {model_name}\n(Kruskal-Wallis Selection, Test Size: {test_size*100:.0f}%)', 
                  fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filename = f"feature_importance_{model_name}_test{int(test_size*100)}.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
        
    except Exception as e:
        print(f"  Warning: Could not plot feature importance: {e}")


def save_report(all_results, output_path, test_size, feature_names):
    """Menyimpan laporan lengkap ke file text."""
    filename = f"classification_report_test{int(test_size*100)}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT - KRUSKAL-WALLIS FEATURE SELECTION\n")
        f.write("="*70 + "\n")
        f.write(f"Feature Selection Method: {FEATURE_SELECTION_METHOD}\n")
        f.write(f"Test Size: {test_size*100:.0f}%\n")
        f.write(f"Random State: {RANDOM_STATE}\n")
        f.write(f"CV Folds: {CV_FOLDS}\n")
        f.write(f"Total Features: {len(feature_names)}\n")
        f.write("="*70 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("─"*70 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("─"*70 + "\n")
        
        for model_name, result in all_results.items():
            metrics = result['metrics']
            f.write(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}\n")
        
        f.write("─"*70 + "\n\n")
        
        # Detailed results per model
        for model_name, result in all_results.items():
            metrics = result['metrics']
            params = result['best_params']
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("="*70 + "\n")
            
            f.write(f"\nBest Parameters:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write(f"\nTest Set Performance:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            
            f.write(f"\nFeature Selection:\n")
            f.write(f"  Total features selected: {metrics['n_features_selected']}\n")
            f.write(f"  Selected features (top 20):\n")
            for feat in metrics['selected_features'][:20]:
                f.write(f"    - {feat}\n")
            if len(metrics['selected_features']) > 20:
                f.write(f"    ... and {len(metrics['selected_features']) - 20} more\n")
            
            f.write(f"\nClassification Report:\n")
            f.write(metrics['classification_report'])
            f.write("\n")
    
    print(f"\n📄 Report saved: {filename}")


def plot_comparison(all_results, output_path, test_size):
    """Membuat visualisasi perbandingan semua model."""
    models = list(all_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = {metric: [all_results[m]['metrics'][metric] for m in models] 
            for metric in metrics_names}
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#e67e22', '#3498db', '#e74c3c', '#9b59b6']  # Orange theme for Kruskal
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), 
                     color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison - Kruskal-Wallis Feature Selection\n(Test Size: {test_size*100:.0f}%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"model_comparison_test{int(test_size*100)}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved: {filename}")


# =============================================================================
# FUNGSI UTAMA PIPELINE
# =============================================================================
def run_benchmark(X, y, feature_names, test_size, output_path):
    """Menjalankan benchmark untuk satu konfigurasi test_size."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK - Test Size: {test_size*100:.0f}%")
    print(f"Feature Selection: Kruskal-Wallis H-test")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    models_params = get_models_and_params(n_features=X.shape[1])
    
    all_results = {}
    
    for model_name, config in models_params.items():
        # Create pipeline
        pipeline = create_pipeline(config['model'])
        
        # Train with GridSearchCV
        best_pipeline, best_params, cv_results = train_with_gridsearch(
            pipeline, config['params'], X_train, y_train, model_name
        )
        
        # Evaluate on test set
        metrics, y_pred = evaluate_model(best_pipeline, X_test, y_test, model_name, feature_names)
        
        # Store results
        all_results[model_name] = {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'cv_results': cv_results,
            'metrics': metrics,
            'y_pred': y_pred
        }
        
        # Generate plots
        print(f"\nGenerating plots for {model_name}...")
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, output_path, test_size)
        plot_feature_importance(best_pipeline, feature_names, model_name, output_path, test_size)
    
    # Save report and comparison
    save_report(all_results, output_path, test_size, feature_names)
    plot_comparison(all_results, output_path, test_size)
    
    return all_results


def generate_final_summary(all_benchmarks, output_path, feature_names):
    """Membuat ringkasan akhir dari semua benchmark."""
    filename = "FINAL_SUMMARY.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY - KRUSKAL-WALLIS FEATURE SELECTION BENCHMARK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Selection Method: Kruskal-Wallis H-test\n")
        f.write(f"Total Features: {len(feature_names)}\n\n")
        
        f.write("BEST MODEL PER TEST SIZE (by F1-Score)\n")
        f.write("-" * 80 + "\n")
        
        overall_best = {'f1': 0, 'model': '', 'test_size': 0}
        
        for test_size, results in all_benchmarks.items():
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1_score'])
            f1 = best_model[1]['metrics']['f1_score']
            k = best_model[1]['best_params'].get('selector__k', 'N/A')
            
            f.write(f"\nTest Size {test_size*100:.0f}%:\n")
            f.write(f"  Best Model: {best_model[0]}\n")
            f.write(f"  F1-Score:   {f1:.4f}\n")
            f.write(f"  K-Features: {k}\n")
            f.write(f"  Parameters: {best_model[1]['best_params']}\n")
            
            if f1 > overall_best['f1']:
                overall_best = {
                    'f1': f1,
                    'model': best_model[0],
                    'test_size': test_size,
                    'params': best_model[1]['best_params'],
                    'k': k
                }
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL BEST MODEL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model:     {overall_best['model']}\n")
        f.write(f"F1-Score:  {overall_best['f1']:.4f}\n")
        f.write(f"Test Size: {overall_best['test_size']*100:.0f}%\n")
        f.write(f"K-Features: {overall_best.get('k', 'N/A')}\n")
        f.write(f"Parameters: {overall_best['params']}\n")
    
    print(f"\nFinal summary saved: {filename}")
    return overall_best


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Fungsi utama yang mengorkestrasi seluruh pipeline."""
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING WITH KRUSKAL-WALLIS FEATURE SELECTION")
    print("=" * 70)
    
    # Check GPU availability for PyTorch MLP
    print("\n" + "=" * 70)
    print("GPU STATUS")
    print("=" * 70)
    check_gpu_availability()
    
    output_path = create_output_directory()
    print(f"\nOutput directory: {output_path}")
    
    data_path = "preprocessed_data/features_all.csv"
    X, y, feature_names = load_and_prepare_data(data_path, TARGET_COLUMN)
    
    # Save feature selection scores ke CSV (di awal flow)
    save_feature_selection_scores(X, y, feature_names, output_path)
    
    all_benchmarks = {}
    
    for test_size in TEST_SIZES:
        results = run_benchmark(X, y, feature_names, test_size, output_path)
        all_benchmarks[test_size] = results
    
    best = generate_final_summary(all_benchmarks, output_path, feature_names)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"""
Best Model: {best['model']}
   F1-Score:   {best['f1']:.4f}
   K-Features: {best.get('k', 'N/A')}
   Test Size:  {best['test_size']*100:.0f}%
   
All results saved in: {output_path}
    """)
    
    return all_benchmarks, best


if __name__ == "__main__":
    all_results, best_model = main()
