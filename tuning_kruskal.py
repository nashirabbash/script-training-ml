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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

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
K_FEATURES = [5, 10, 15, 20, 25, 30, 50, 75, 100]

# Target column - sesuaikan dengan kebutuhan
TARGET_COLUMN = 'passing_85'  # Opsi: passing_80, passing_85, passing_90, performance


# =============================================================================
# CUSTOM SCORING FUNCTION UNTUK KRUSKAL-WALLIS
# =============================================================================
def kruskal_wallis_score(X, y):
    """
    Scoring function untuk Kruskal-Wallis H-test.
    
    Menghitung H-statistic untuk setiap fitur. H-statistic yang lebih tinggi
    menunjukkan perbedaan distribusi yang lebih signifikan antar kelas.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    
    Returns:
        scores: Array of H-statistics (n_features,)
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    
    # Get unique classes
    classes = np.unique(y)
    
    # Untuk setiap fitur, hitung Kruskal-Wallis H-statistic
    for i in range(n_features):
        # Pisahkan data per kelas
        groups = [X[y == c, i] for c in classes]
        
        # Kruskal-Wallis test
        try:
            # Filter out groups with all NaN or empty
            valid_groups = [g for g in groups if len(g) > 0 and not np.all(np.isnan(g))]
            
            if len(valid_groups) >= 2:
                h_stat, p_val = kruskal(*valid_groups)
                scores[i] = h_stat if not np.isnan(h_stat) else 0
            else:
                scores[i] = 0
        except:
            scores[i] = 0
    
    return scores


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
                'clf__n_neighbors': [3, 5, 7, 9],
                'clf__weights': ['uniform', 'distance'],
                'clf__metric': ['euclidean', 'manhattan']
            }
        },
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['rbf', 'linear'],
                'clf__gamma': ['scale', 'auto']
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, early_stopping=True),
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
                'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'selector__k': k_values,
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4]
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
    """Split data menjadi train dan test set dengan stratifikasi"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n{'='*70}")
    print(f"DATA SPLIT (test_size={test_size})")
    print(f"{'='*70}")
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def create_pipeline(model):
    """
    Buat Pipeline dengan imputation, scaling, feature selection, dan classifier.
    
    Pipeline steps:
    1. Imputer: Handle missing values
    2. Scaler: Standardize features
    3. Selector: Kruskal-Wallis feature selection
    4. Classifier: Model
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=kruskal_wallis_score)),
        ('clf', model)
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
    
    # Setup GridSearchCV dengan StratifiedKFold
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
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
    Evaluasi model pada test set.
    
    Returns:
        dict: Dictionary berisi semua metrics
    """
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Selected features
    selected_features = get_selected_features(pipeline, feature_names)
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_test,
        'selected_features': selected_features,
        'n_selected_features': len(selected_features)
    }
    
    print(f"\n{'─'*70}")
    print(f"Test Results: {model_name}")
    print(f"{'─'*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Selected Features: {len(selected_features)}")
    
    return results


def plot_confusion_matrix(cm, model_name, output_path, test_size):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}\n(test_size={test_size})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}_test{int(test_size*100)}.png'
    plt.savefig(os.path.join(output_path, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(pipeline, feature_names, model_name, output_path, test_size):
    """Plot feature importance berdasarkan Kruskal-Wallis scores"""
    try:
        selector = pipeline.named_steps['selector']
        scores = selector.scores_
        selected_mask = selector.get_support()
        
        # Sort by scores
        indices = np.argsort(scores)[::-1][:30]  # Top 30
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), scores[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Kruskal-Wallis H-statistic')
        plt.title(f'Top 30 Features - {model_name}\n(test_size={test_size})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}_test{int(test_size*100)}.png'
        plt.savefig(os.path.join(output_path, filename), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Warning: Could not plot feature importance: {e}")


def save_report(all_results, output_path, test_size, feature_names):
    """Save text report"""
    report_file = os.path.join(output_path, f'classification_report_test{int(test_size*100)}.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"CLASSIFICATION REPORT\n")
        f.write(f"Feature Selection: {FEATURE_SELECTION_METHOD}\n")
        f.write(f"Test Size: {test_size}\n")
        f.write(f"Random State: {RANDOM_STATE}\n")
        f.write(f"CV Folds: {CV_FOLDS}\n")
        f.write("="*70 + "\n\n")
        
        for result in all_results:
            f.write(f"\n{'─'*70}\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"{'─'*70}\n")
            f.write(f"Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall:    {result['recall']:.4f}\n")
            f.write(f"F1-Score:  {result['f1_score']:.4f}\n")
            f.write(f"Selected Features: {result['n_selected_features']}\n\n")
            
            f.write("Selected Features:\n")
            for feat in result['selected_features'][:20]:  # Top 20
                f.write(f"  - {feat}\n")
            if len(result['selected_features']) > 20:
                f.write(f"  ... and {len(result['selected_features']) - 20} more\n")
            
            f.write("\n")
    
    print(f"\n📄 Report saved: {report_file}")


def plot_comparison(all_results, output_path, test_size):
    """Plot comparison of all models"""
    models = [r['model'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison - {FEATURE_SELECTION_METHOD}\n(test_size={test_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    filename = f'model_comparison_test{int(test_size*100)}.png'
    plt.savefig(os.path.join(output_path, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved: {filename}")


# =============================================================================
# FUNGSI UTAMA PIPELINE
# =============================================================================
def run_benchmark(X, y, feature_names, test_size, output_path):
    """
    Run benchmark untuk semua model dengan test_size tertentu.
    
    Returns:
        list: List of result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK: test_size = {test_size}")
    print(f"{'='*70}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Get models and params
    models_params = get_models_and_params(len(feature_names))
    
    all_results = []
    
    # Train each model
    for model_name, config in models_params.items():
        # Create pipeline
        pipeline = create_pipeline(config['model'])
        
        # GridSearch
        best_pipeline, best_params, cv_score = train_with_gridsearch(
            pipeline, config['params'], X_train, y_train, model_name
        )
        
        # Evaluate
        results = evaluate_model(best_pipeline, X_test, y_test, model_name, feature_names)
        results['cv_score'] = cv_score
        results['best_params'] = best_params
        all_results.append(results)
        
        # Plot confusion matrix
        plot_confusion_matrix(results['confusion_matrix'], model_name, output_path, test_size)
        
        # Plot feature importance
        plot_feature_importance(best_pipeline, feature_names, model_name, output_path, test_size)
    
    # Save report
    save_report(all_results, output_path, test_size, feature_names)
    
    # Plot comparison
    plot_comparison(all_results, output_path, test_size)
    
    return all_results


def generate_final_summary(all_benchmarks, output_path, feature_names):
    """Generate final summary across all test sizes"""
    summary_file = os.path.join(output_path, 'FINAL_SUMMARY.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"FINAL SUMMARY - {FEATURE_SELECTION_METHOD}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset Info:\n")
        f.write(f"  Total Features: {len(feature_names)}\n")
        f.write(f"  Test Sizes Evaluated: {TEST_SIZES}\n")
        f.write(f"  CV Folds: {CV_FOLDS}\n")
        f.write(f"  Random State: {RANDOM_STATE}\n\n")
        
        # Best model per test size
        f.write("="*70 + "\n")
        f.write("BEST MODEL PER TEST SIZE\n")
        f.write("="*70 + "\n\n")
        
        for test_size, results in all_benchmarks.items():
            best_result = max(results, key=lambda x: x['accuracy'])
            f.write(f"Test Size {test_size}:\n")
            f.write(f"  Best Model: {best_result['model']}\n")
            f.write(f"  Accuracy: {best_result['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {best_result['f1_score']:.4f}\n")
            f.write(f"  Selected Features: {best_result['n_selected_features']}\n\n")
        
        # Overall best
        f.write("="*70 + "\n")
        f.write("OVERALL BEST MODEL\n")
        f.write("="*70 + "\n\n")
        
        all_results_flat = [r for results in all_benchmarks.values() for r in results]
        overall_best = max(all_results_flat, key=lambda x: x['accuracy'])
        
        f.write(f"Model: {overall_best['model']}\n")
        f.write(f"Test Size: {[k for k, v in all_benchmarks.items() if overall_best in v][0]}\n")
        f.write(f"Accuracy: {overall_best['accuracy']:.4f}\n")
        f.write(f"Precision: {overall_best['precision']:.4f}\n")
        f.write(f"Recall: {overall_best['recall']:.4f}\n")
        f.write(f"F1-Score: {overall_best['f1_score']:.4f}\n")
        f.write(f"CV Score: {overall_best['cv_score']:.4f}\n")
        f.write(f"Selected Features: {overall_best['n_selected_features']}\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in overall_best['best_params'].items():
            f.write(f"  {param}: {value}\n")
    
    print(f"\n📋 Final summary saved: {summary_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("="*70)
    print(f"  HYPERPARAMETER TUNING - {FEATURE_SELECTION_METHOD}")
    print("="*70)
    
    # Create output directory
    output_path = create_output_directory()
    print(f"\nOutput directory: {output_path}")
    
    # Load data
    data_file = os.path.join('preprocessed_data', 'features_all.csv')
    X, y, feature_names = load_and_prepare_data(data_file)
    
    # Run benchmarks for different test sizes
    all_benchmarks = {}
    
    for test_size in TEST_SIZES:
        results = run_benchmark(X, y, feature_names, test_size, output_path)
        all_benchmarks[test_size] = results
    
    # Generate final summary
    generate_final_summary(all_benchmarks, output_path, feature_names)
    
    print(f"\n{'='*70}")
    print(f"✅ TUNING COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
