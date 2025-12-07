"""
=============================================================================
HYPERPARAMETER TUNING WITH ANOVA (f_classif) FEATURE SELECTION
=============================================================================
Script ini menggunakan ANOVA F-test (f_classif) untuk seleksi fitur,
kemudian membandingkan 6 classifier dengan GridSearchCV.

ANOVA F-test:
- Menguji apakah rata-rata fitur berbeda signifikan antar kelas
- Cocok untuk fitur NUMERIK/KONTINYU
- Univariat (mengevaluasi setiap fitur secara independen)

Author: ML Pipeline
Date: 2025
=============================================================================
"""

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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# KONFIGURASI
# =============================================================================
RANDOM_STATE = 42
CV_FOLDS = 5
OUTPUT_DIR = "results_anova"
FEATURE_SELECTION_METHOD = "ANOVA (f_classif)"

# Berbagai rasio pembagian dataset untuk eksperimen
TEST_SIZES = [0.2, 0.25, 0.3]

# Jumlah fitur yang akan dicoba dalam feature selection
# Sesuaikan dengan jumlah fitur dataset Anda
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
                'clf__n_neighbors': [3, 5, 7, 9, 11],
                'clf__weights': ['uniform', 'distance'],
                'clf__metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'clf__gamma': ['scale', 'auto']
            }
        },
        
        'MLP': {
            'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'selector__k': k_values,
                'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001, 0.01],
                'clf__learning_rate': ['constant', 'adaptive']
            }
        },
        
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
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
                'clf__max_depth': [None, 5, 10, 15, 20],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
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
    
    # Pastikan target column ada
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan dalam dataset!")
    
    # Pisahkan fitur dan target
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
    
    return X, y, feature_cols


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
    Membuat Pipeline dengan: Imputer -> Scaler -> Feature Selection (ANOVA) -> Classifier
    
    Args:
        model: Classifier model
        
    Returns:
        Pipeline object
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle NaN dengan median
        ('scaler', StandardScaler()),                   # Normalisasi fitur
        ('selector', SelectKBest(f_classif)),           # ANOVA feature selection
        ('clf', model)                                  # Classifier
    ])
    
    return pipeline


# =============================================================================
# FUNGSI TRAINING DAN EVALUASI
# =============================================================================
def train_with_gridsearch(pipeline, params, X_train, y_train, model_name):
    """
    Melatih pipeline dengan GridSearchCV.
    
    Args:
        pipeline: Pipeline sklearn
        params: Parameter grid
        X_train: Training features
        y_train: Training target
        model_name: Nama model untuk logging
        
    Returns:
        best_pipeline: Pipeline dengan parameter terbaik
        best_params: Parameter terbaik
        cv_results: Hasil cross-validation
    """
    print(f"\n{'─' * 50}")
    print(f"Training: {model_name}")
    print(f"Feature Selection: ANOVA (f_classif)")
    print(f"{'─' * 50}")
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # Extract best k (jumlah fitur terpilih)
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
    """
    Mendapatkan nama fitur yang terpilih dari pipeline.
    
    Args:
        pipeline: Trained pipeline
        feature_names: List nama fitur asli
        
    Returns:
        List nama fitur terpilih
    """
    try:
        selector = pipeline.named_steps['selector']
        mask = selector.get_support()
        selected = [f for f, m in zip(feature_names, mask) if m]
        return selected
    except Exception:
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
        cmap='Blues',
        xticklabels=['Not Passing (0)', 'Passing (1)'],
        yticklabels=['Not Passing (0)', 'Passing (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}\n(ANOVA Selection, Test Size: {test_size*100:.0f}%)')
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
    Plot skor ANOVA untuk fitur yang dipilih.
    
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
        
        # Buat dataframe untuk sorting
        df_scores = pd.DataFrame({
            'feature': feature_names,
            'score': scores
        }).sort_values('score', ascending=False)
        
        # Plot top 20 fitur
        top_n = min(20, len(df_scores))
        df_top = df_scores.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), df_top['score'].values, color='steelblue')
        plt.yticks(range(top_n), df_top['feature'].values)
        plt.xlabel('ANOVA F-Score')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Features by ANOVA Score - {model_name}\n(Test Size: {test_size*100:.0f}%)')
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
        f.write("HYPERPARAMETER TUNING WITH ANOVA FEATURE SELECTION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Selection Method: ANOVA (f_classif)\n")
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
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
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
    ax.set_title(f'Model Comparison - ANOVA Feature Selection\n(Test Size: {test_size*100:.0f}%)', fontsize=14)
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
    print(f"Feature Selection: ANOVA (f_classif)")
    print("=" * 70)
    
    # Split data (scaling dilakukan di pipeline)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Get models dan params
    models_params = get_models_and_params(n_features=X.shape[1])
    
    all_results = {}
    
    for model_name, config in models_params.items():
        # Buat pipeline
        pipeline = create_pipeline(config['model'])
        
        # Training dengan GridSearchCV
        best_pipeline, best_params, cv_results = train_with_gridsearch(
            pipeline=pipeline,
            params=config['params'],
            X_train=X_train,
            y_train=y_train,
            model_name=model_name
        )
        
        # Evaluasi pada test set
        metrics, y_pred = evaluate_model(
            best_pipeline, X_test, y_test, model_name, feature_names
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            model_name, 
            output_path, 
            test_size
        )
        
        # Plot feature importance (ANOVA scores)
        plot_feature_importance(
            best_pipeline,
            feature_names,
            model_name,
            output_path,
            test_size
        )
        
        # Simpan hasil
        all_results[model_name] = {
            'best_pipeline': best_pipeline,
            'best_params': best_params,
            'cv_score': cv_results['mean_test_score'][np.argmax(cv_results['mean_test_score'])],
            'metrics': metrics
        }
    
    # Save report
    save_report(all_results, output_path, test_size, feature_names)
    
    # Plot comparison
    plot_comparison(all_results, output_path, test_size)
    
    return all_results


def generate_final_summary(all_benchmarks, output_path, feature_names):
    """Membuat ringkasan akhir dari semua benchmark."""
    filename = "FINAL_SUMMARY.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY - ANOVA FEATURE SELECTION BENCHMARK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature Selection Method: ANOVA (f_classif)\n")
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
    print("  HYPERPARAMETER TUNING WITH ANOVA FEATURE SELECTION")
    print("█" * 70)
    
    # Buat direktori output
    output_path = create_output_directory()
    print(f"\n📁 Output directory: {output_path}")
    
    # Load data
    data_path = "preprocessed_data/features_all.csv"
    X, y, feature_names = load_and_prepare_data(data_path, TARGET_COLUMN)
    
    # Jalankan benchmark untuk berbagai test size
    all_benchmarks = {}
    
    for test_size in TEST_SIZES:
        results = run_benchmark(X, y, feature_names, test_size, output_path)
        all_benchmarks[test_size] = results
    
    # Generate final summary
    best = generate_final_summary(all_benchmarks, output_path, feature_names)
    
    # Print final summary ke console
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