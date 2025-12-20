"""
=============================================================================
HYPERPARAMETER TUNING WITH CHI-SQUARE FEATURE SELECTION
=============================================================================
Script ini menggunakan Chi-Square test untuk seleksi fitur,
kemudian membandingkan 6 classifier dengan GridSearchCV.

Chi-Square Test:
- Menguji independensi antara fitur dan target
- Cocok untuk fitur NON-NEGATIF (counts, frequencies, one-hot encoded)
- Univariat (mengevaluasi setiap fitur secara independen)
- PENTING: Fitur harus >= 0, jadi kita gunakan MinMaxScaler

Author: ML Pipeline
Date: 2025
=============================================================================
"""

import sys
import io
# Fix Windows encoding untuk Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

# PENTING: Set backend matplotlib ke 'Agg' SEBELUM import plt
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
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


# =============================================================================
# KONFIGURASI
# =============================================================================
RANDOM_STATE = 42
CV_FOLDS = 5
OUTPUT_DIR = "results_chi2"
FEATURE_SELECTION_METHOD = "Chi-Square (chi2)"

# Berbagai rasio pembagian dataset untuk eksperimen
TEST_SIZES = [0.2, 0.25, 0.3]

# Jumlah fitur yang akan dicoba dalam feature selection
K_FEATURES = [5, 10, 15, 20, 25, 30]

# Target column - sesuaikan dengan kebutuhan
# NOTE: passing_70 tidak direkomendasikan (semua data passing)
TARGET_COLUMN = 'passing_85'  # Opsi: passing_80, passing_85, passing_90, performance


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
        k_values = [min(5, n_features)]
    
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
    """Membuat direktori output jika belum ada."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def save_feature_selection_scores(X, y, feature_names, output_path):
    """
    Menghitung dan menyimpan skor feature selection ke CSV.
    Dipanggil di awal flow sebelum benchmark dimulai.
    
    Args:
        X: Feature matrix
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
    
    # Chi-Square memerlukan nilai non-negatif, gunakan MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Hitung Chi-Square scores
    from sklearn.feature_selection import chi2
    scores, pvalues = chi2(X_scaled, y)
    
    # Buat DataFrame hasil
    df_scores = pd.DataFrame({
        'feature': feature_names,
        'chi2_score': scores,
        'p_value': pvalues
    })
    
    # Sort by score (descending)
    df_scores = df_scores.sort_values('chi2_score', ascending=False).reset_index(drop=True)
    df_scores['rank'] = range(1, len(df_scores) + 1)
    
    # Reorder columns
    df_scores = df_scores[['rank', 'feature', 'chi2_score', 'p_value']]
    
    # Save to CSV
    csv_path = os.path.join(output_path, 'feature_selection_scores.csv')
    df_scores.to_csv(csv_path, index=False)
    
    print(f"\nTop 10 Features by Chi-Square Score:")
    print(df_scores.head(10).to_string(index=False))
    print(f"\n>>> Feature scores saved to: {csv_path}")
    
    return df_scores


def load_and_prepare_data(filepath, target_col=TARGET_COLUMN):
    """
    Memuat dan mempersiapkan dataset versi2_gabung.csv
    
    Dataset ini memiliki kolom:
    - window_id, subject, exam: identifier (bukan fitur)
    - Fitur numerik: eda_*, hr_*, bvp_*, acc_*, temp_*
    - Target: passing_70, passing_80, passing_85, passing_90, grade, performance
    
    Args:
        filepath: Path ke file CSV
        target_col: Nama kolom target (default: passing_70)
        
    Returns:
        X: Features matrix
        y: Target vector
        feature_names: List nama fitur
    """
    print("=" * 60)
    print("LOADING DATA")
    print(f"Feature Selection: {FEATURE_SELECTION_METHOD}")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    # Kolom yang BUKAN fitur (identifier, metadata, dan target)
    non_feature_cols = [
        'window_id', 'subject', 'exam', 'start_sec', 'end_sec',  # identifier & metadata
        'grade', 'performance',  # target kategorikal
        'passing_70', 'passing_80', 'passing_85', 'passing_90'  # target biner
    ]
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan dalam dataset!")
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target column: {target_col}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target ratio: {y.value_counts(normalize=True).to_dict()}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(f"  {feature_cols[:5]}... (showing first 5)")
    
    # Check for missing values
    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        print(f"\n⚠️  Missing values detected: {n_missing} NaN values")
        print(f"   Will be handled by SimpleImputer (median strategy) in pipeline")
    else:
        print(f"\n✓ No missing values detected")
    
    # Chi-square memerlukan nilai non-negatif
    print(f"\n⚠️  Note: Chi-Square requires non-negative values.")
    print(f"   MinMaxScaler will be applied in pipeline to ensure X >= 0")
    
    return X, y, feature_cols


def split_data(X, y, test_size=0.2):
    """Membagi data menjadi train dan test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def create_pipeline(model):
    """
    Membuat Pipeline dengan: Imputer -> MinMaxScaler -> Feature Selection (Chi2) -> Classifier
    
    PENTING: Chi-Square memerlukan fitur non-negatif, jadi kita pakai MinMaxScaler
    yang akan mengubah semua nilai ke range [0, 1].
    
    Args:
        model: Classifier model
        
    Returns:
        Pipeline object
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaN dengan median
        ('scaler', MinMaxScaler()),                     # Scale ke [0,1] untuk chi-square
        ('selector', SelectKBest(chi2)),                # Chi-square feature selection
        ('clf', model)                                  # Classifier
    ])
    
    return pipeline


# =============================================================================
# FUNGSI TRAINING DAN EVALUASI
# =============================================================================
def train_with_gridsearch(pipeline, params, X_train, y_train, model_name):
    """Melatih pipeline dengan GridSearchCV."""
    print(f"\n{'─' * 50}")
    print(f"Training: {model_name}")
    print(f"Feature Selection: Chi-Square (chi2)")
    print(f"{'─' * 50}")
    
    # Convert to NumPy arrays to avoid Pandas serialization issues in parallel processing
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # PENTING: Gunakan n_jobs=2 untuk menghindari memory fragmentation
    # n_jobs=-1 menyebabkan terlalu banyak worker process yang kompetisi memory
    # Untuk MLP: n_jobs=1 lebih baik karena PyTorch sudah parallel di GPU
    n_jobs = 1 if 'MLP' in model_name else 2
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=cv,
        scoring='f1',
        n_jobs=n_jobs,  # MLP: sequential (GPU parallelism), Others: 2 workers
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    best_k = grid_search.best_params_.get('selector__k', 'N/A')
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (F1): {grid_search.best_score_:.4f}")
    print(f"Selected Features (k): {best_k}")
    
    # Deteksi overfitting
    best_idx = grid_search.best_index_
    train_score = grid_search.cv_results_['mean_train_score'][best_idx]
    test_score = grid_search.cv_results_['mean_test_score'][best_idx]
    
    gap = train_score - test_score
    if gap > 0.1:
        print(f"⚠️  WARNING: Potential OVERFITTING detected!")
        print(f"   Train Score: {train_score:.4f}, CV Score: {test_score:.4f}, Gap: {gap:.4f}")
    elif test_score < 0.5:
        print(f"⚠️  WARNING: Potential UNDERFITTING detected!")
        print(f"   CV Score: {test_score:.4f} is below 0.5")
    else:
        print(f"✓  Model looks well-fitted. Gap: {gap:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def get_selected_features(pipeline, feature_names):
    """Mendapatkan nama fitur yang terpilih dari pipeline."""
    try:
        selector = pipeline.named_steps['selector']
        mask = selector.get_support()
        selected = [f for f, m in zip(feature_names, mask) if m]
        return selected
    except Exception:
        return []


def evaluate_model(pipeline, X_test, y_test, model_name, feature_names):
    """Mengevaluasi pipeline pada test set."""
    y_pred = pipeline.predict(X_test)
    
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
        cmap='Greens',  # Warna berbeda untuk Chi2
        xticklabels=['Not Passing (0)', 'Passing (1)'],
        yticklabels=['Not Passing (0)', 'Passing (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}\n(Chi-Square Selection, Test Size: {test_size*100:.0f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name}_test{int(test_size*100)}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def plot_feature_importance(pipeline, feature_names, model_name, output_path, test_size):
    """Plot skor Chi-Square untuk fitur."""
    try:
        selector = pipeline.named_steps['selector']
        scores = selector.scores_
        
        df_scores = pd.DataFrame({
            'feature': feature_names,
            'score': scores
        }).sort_values('score', ascending=False)
        
        top_n = min(20, len(df_scores))
        df_top = df_scores.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), df_top['score'].values, color='forestgreen')
        plt.yticks(range(top_n), df_top['feature'].values)
        plt.xlabel('Chi-Square Score')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Features by Chi-Square Score - {model_name}\n(Test Size: {test_size*100:.0f}%)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = f"feature_scores_{model_name}_test{int(test_size*100)}.png"
        filepath = os.path.join(output_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
        
    except Exception as e:
        print(f"  Could not plot feature importance: {e}")


def save_report(all_results, output_path, test_size, feature_names):
    """Menyimpan laporan lengkap ke file text."""
    filename = f"classification_report_test{int(test_size*100)}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING WITH CHI-SQUARE FEATURE SELECTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Selection Method: Chi-Square (chi2)\n")
        f.write(f"Scaler: MinMaxScaler (to ensure non-negative values)\n")
        f.write(f"Test Size: {test_size*100:.0f}%\n")
        f.write(f"Cross-Validation Folds: {CV_FOLDS}\n")
        f.write(f"Total Features Available: {len(feature_names)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary Table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'K-Features':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, result in all_results.items():
            metrics = result['metrics']
            k = result['best_params'].get('selector__k', 'N/A')
            f.write(f"{model_name:<15} {str(k):<12} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Detail per model
        for model_name, result in all_results.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 80 + "\n")
            
            f.write("\nBest Parameters:\n")
            for param, value in result['best_params'].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write(f"\nBest CV Score (F1): {result['cv_score']:.4f}\n")
            
            f.write(f"\nSelected Features ({result['metrics']['n_features_selected']}):\n")
            for feat in result['metrics']['selected_features']:
                f.write(f"  - {feat}\n")
            
            f.write("\nClassification Report:\n")
            f.write(result['metrics']['classification_report'])
            
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['metrics']['confusion_matrix']))
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
    
    colors = ['#27ae60', '#2980b9', '#c0392b', '#8e44ad']  # Warna berbeda
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=metric.capitalize(), color=color)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison - Chi-Square Feature Selection\n(Test Size: {test_size*100:.0f}%)', fontsize=14)
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
    print(f"Feature Selection: Chi-Square (chi2)")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    models_params = get_models_and_params(n_features=X.shape[1])
    
    all_results = {}
    
    for model_name, config in models_params.items():
        pipeline = create_pipeline(config['model'])
        
        best_pipeline, best_params, cv_results = train_with_gridsearch(
            pipeline=pipeline,
            params=config['params'],
            X_train=X_train,
            y_train=y_train,
            model_name=model_name
        )
        
        metrics, y_pred = evaluate_model(
            best_pipeline, X_test, y_test, model_name, feature_names
        )
        
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            model_name, 
            output_path, 
            test_size
        )
        
        plot_feature_importance(
            best_pipeline,
            feature_names,
            model_name,
            output_path,
            test_size
        )
        
        all_results[model_name] = {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'cv_score': cv_results['mean_test_score'][np.argmax(cv_results['mean_test_score'])],
            'metrics': metrics
        }
    
    save_report(all_results, output_path, test_size, feature_names)
    plot_comparison(all_results, output_path, test_size)
    
    return all_results


def generate_final_summary(all_benchmarks, output_path, feature_names):
    """Membuat ringkasan akhir dari semua benchmark."""
    filename = "FINAL_SUMMARY.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY - CHI-SQUARE FEATURE SELECTION BENCHMARK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Selection Method: Chi-Square (chi2)\n")
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
        f.write("🏆 OVERALL BEST MODEL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model:     {overall_best['model']}\n")
        f.write(f"F1-Score:  {overall_best['f1']:.4f}\n")
        f.write(f"Test Size: {overall_best['test_size']*100:.0f}%\n")
        f.write(f"K-Features: {overall_best.get('k', 'N/A')}\n")
        f.write(f"Parameters: {overall_best['params']}\n")
    
    print(f"\n🏆 Final summary saved: {filename}")
    return overall_best


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Fungsi utama yang mengorkestrasi seluruh pipeline."""
    print("\n" + "█" * 70)
    print("  HYPERPARAMETER TUNING WITH CHI-SQUARE FEATURE SELECTION")
    print("█" * 70)
    
    # Check GPU availability for PyTorch MLP
    print("\n" + "=" * 70)
    print("GPU STATUS")
    print("=" * 70)
    check_gpu_availability()
    
    output_path = create_output_directory()
    print(f"\n📁 Output directory: {output_path}")
    
    data_path = "preprocessed_data/features_all.csv"
    X, y, feature_names = load_and_prepare_data(data_path, TARGET_COLUMN)
    
    # Save feature selection scores ke CSV (di awal flow)
    save_feature_selection_scores(X, y, feature_names, output_path)
    
    all_benchmarks = {}
    
    for test_size in TEST_SIZES:
        results = run_benchmark(X, y, feature_names, test_size, output_path)
        all_benchmarks[test_size] = results
    
    best = generate_final_summary(all_benchmarks, output_path, feature_names)
    
    print("\n" + "█" * 70)
    print("  BENCHMARK COMPLETE!")
    print("█" * 70)
    print(f"""
🏆 Best Model: {best['model']}
   F1-Score:   {best['f1']:.4f}
   K-Features: {best.get('k', 'N/A')}
   Test Size:  {best['test_size']*100:.0f}%
   
📁 All results saved in: {output_path}
    """)
    
    return all_benchmarks, best


if __name__ == "__main__":
    all_results, best_model = main()