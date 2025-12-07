"""
=============================================================================
HYPERPARAMETER TUNING BENCHMARK - Machine Learning Classifier Comparison
=============================================================================
Script ini membandingkan 6 classifier dengan GridSearchCV untuk mencari
parameter optimal dan menghindari overfitting/underfitting.

Author: ML Pipeline
Date: 2025
=============================================================================
"""

import numpy as np
import pandas as pd

# PENTING: Set backend matplotlib ke 'Agg' SEBELUM import plt
# Ini mencegah konflik threading dengan tkinter saat GridSearchCV
# menggunakan multiple workers (n_jobs=-1)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, aman untuk multi-threading

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
OUTPUT_DIR = "results"

# Berbagai rasio pembagian dataset untuk eksperimen
TEST_SIZES = [0.2, 0.25, 0.3]


# =============================================================================
# DEFINISI MODEL DAN PARAMETER GRID
# =============================================================================
def get_models_and_params():
    """
    Mendefinisikan semua model beserta parameter grid untuk GridSearchCV.
    
    Analogi: Ini seperti menyiapkan "menu eksperimen" - setiap model punya
    "bumbu" (hyperparameter) yang bisa divariasikan untuk mendapat "rasa" terbaik.
    
    Returns:
        dict: Dictionary berisi model dan parameter grid-nya
    """
    models_params = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],          # Jumlah tetangga
                'weights': ['uniform', 'distance'],       # Bobot voting
                'metric': ['euclidean', 'manhattan', 'minkowski']      # Jarak yang digunakan
            }
        },
        
        'SVM': {
            'model': SVC(random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100],                   # Regularization strength
                'kernel': ['linear', 'rbf', 'poly',  'sigmoid'],      # Jenis kernel
                'gamma': ['scale', 'auto']                # Kernel coefficient
            }
        },
        
        'MLP': {
            'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [
                    (50,), (100,), (50, 50), (100, 50), (100, 100)
                ],                                         # Arsitektur layer
                'activation': ['relu', 'tanh'],           # Fungsi aktivasi
                'alpha': [0.0001, 0.001, 0.01],           # L2 regularization
                'learning_rate': ['constant', 'adaptive'] # Learning rate strategy
            }
        },
        
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],           # Jumlah pohon
                'max_depth': [None, 10, 20, 30],          # Kedalaman maksimal
                'min_samples_split': [2, 5, 10],          # Min sample untuk split
                'min_samples_leaf': [1, 2, 4]             # Min sample di leaf
            }
        },
        
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': np.logspace(-12, -6, 7)  # Smoothing parameter
            }
        },
        
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [None, 5, 10, 15, 20],       # Kedalaman maksimal
                'min_samples_split': [2, 5, 10],          # Min sample untuk split
                'min_samples_leaf': [1, 2, 4],            # Min sample di leaf
                'criterion': ['gini', 'entropy']          # Kriteria split
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


def load_and_prepare_data(filepath):
    """
    Memuat dan mempersiapkan dataset.
    
    Analogi: Seperti menyiapkan bahan masakan - kita perlu memisahkan
    "bahan utama" (features) dari "hasil akhir" (target/label).
    
    Args:
        filepath: Path ke file CSV
        
    Returns:
        X: Features matrix
        y: Target vector
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath, index_col=0)
    
    # Asumsi: kolom terakhir adalah target (Passing)
    # dan kolom 'Grades' mungkin perlu dibuang karena redundan
    if 'Grades' in df.columns:
        df = df.drop('Grades', axis=1)
    
    X = df.drop('Passing', axis=1)
    y = df['Passing']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target ratio: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y


def split_and_scale_data(X, y, test_size=0.2):
    """
    Membagi data dan melakukan scaling.
    
    Analogi: Seperti membagi adonan kue - sebagian untuk "uji coba" (test),
    sebagian untuk "latihan" (train). Scaling itu seperti memastikan semua
    bahan diukur dengan satuan yang sama.
    
    Args:
        X: Features
        y: Target
        test_size: Proporsi data test
        
    Returns:
        X_train, X_test, y_train, y_test yang sudah di-scale
    """
    # Split dengan stratify untuk menjaga proporsi kelas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y  # Penting! Menjaga balance kelas di train dan test
    )
    
    # Scaling - fit hanya di train, transform keduanya
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# FUNGSI TRAINING DAN EVALUASI
# =============================================================================
def train_with_gridsearch(model, params, X_train, y_train, model_name):
    """
    Melatih model dengan GridSearchCV.
    
    Analogi: Seperti chef yang mencoba berbagai kombinasi bumbu (parameter)
    dan memilih yang paling enak berdasarkan "voting" dari beberapa juri (CV folds).
    
    Args:
        model: Model sklearn
        params: Parameter grid
        X_train: Training features
        y_train: Training target
        model_name: Nama model untuk logging
        
    Returns:
        best_model: Model dengan parameter terbaik
        best_params: Parameter terbaik
        cv_results: Hasil cross-validation
    """
    print(f"\n{'─' * 50}")
    print(f"Training: {model_name}")
    print(f"{'─' * 50}")
    
    # StratifiedKFold untuk menjaga proporsi kelas di setiap fold
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring='f1',  # Menggunakan F1 karena lebih balanced untuk imbalanced data
        n_jobs=-1,     # Gunakan semua CPU cores
        verbose=1,
        return_train_score=True  # Untuk deteksi overfitting
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (F1): {grid_search.best_score_:.4f}")
    
    # Deteksi overfitting: bandingkan train score vs CV score
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


def evaluate_model(model, X_test, y_test, model_name):
    """
    Mengevaluasi model pada test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Nama model
        
    Returns:
        metrics: Dictionary berisi semua metrik
        y_pred: Prediksi
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\nTest Results for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    return metrics, y_pred


def plot_training_curve(model, model_name, X_train, y_train, output_path, test_size, best_params):
    """
    Membuat grafik training dengan 2 subplot: Accuracy dan Loss.
    
    Sumbu X: Iteration (0-100)
    Sumbu Y: Accuracy (0-100%) atau Loss value
    
    Args:
        model: Trained model
        model_name: Nama model
        X_train: Training features
        y_train: Training target
        output_path: Path untuk menyimpan
        test_size: Ukuran test set
        best_params: Parameter terbaik dari GridSearch
    """
    
    # Generate training history dengan iterasi bertahap
    iterations, accuracies, losses = _generate_training_history(
        model, model_name, X_train, y_train, best_params
    )
    
    # Buat figure dengan 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Plot 1: Accuracy =====
    ax1.plot(iterations, accuracies, 'b-', linewidth=2, label='Training Accuracy')
    ax1.fill_between(iterations, accuracies, alpha=0.3, color='blue')
    
    # Formatting
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Tambahkan final accuracy annotation
    final_acc = accuracies[-1]
    ax1.axhline(y=final_acc, color='green', linestyle='--', alpha=0.5)
    ax1.annotate(f'Final: {final_acc:.1f}%', 
                xy=(iterations[-1], final_acc),
                xytext=(iterations[-1]-20, final_acc-10),
                fontsize=10, color='green', fontweight='bold')
    
    # ===== Plot 2: Loss =====
    ax2.plot(iterations, losses, 'r-', linewidth=2, label='Training Loss')
    ax2.fill_between(iterations, losses, alpha=0.3, color='red')
    
    # Formatting
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Tambahkan final loss annotation
    final_loss = losses[-1]
    ax2.axhline(y=final_loss, color='darkred', linestyle='--', alpha=0.5)
    ax2.annotate(f'Final: {final_loss:.4f}', 
                xy=(iterations[-1], final_loss),
                xytext=(iterations[-1]-25, final_loss + (max(losses)-min(losses))*0.1),
                fontsize=10, color='darkred', fontweight='bold')
    
    # Title keseluruhan
    plt.suptitle(f'Training Progress - {model_name}\n(Test Size: {test_size*100:.0f}%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Simpan gambar
    filename = f"training_curve_{model_name}_test{int(test_size*100)}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def _generate_training_history(model, model_name, X_train, y_train, best_params):
    """
    Generate training history (accuracy & loss) untuk setiap model.
    
    Karena tidak semua model scikit-learn memiliki iterasi internal,
    kita simulasikan dengan melatih model secara bertahap.
    
    Returns:
        iterations: List of iteration numbers (0-100)
        accuracies: List of accuracy values (0-100%)
        losses: List of loss values
    """
    
    n_iterations = 100
    
    if model_name == 'MLP':
        # MLP memiliki loss_curve_ built-in
        return _get_mlp_history(model, n_iterations)
    
    elif model_name == 'RandomForest':
        return _get_rf_history(X_train, y_train, best_params, n_iterations)
    
    elif model_name == 'KNN':
        return _get_knn_history(X_train, y_train, best_params, n_iterations)
    
    elif model_name == 'SVM':
        return _get_svm_history(X_train, y_train, best_params, n_iterations)
    
    elif model_name == 'DecisionTree':
        return _get_dt_history(X_train, y_train, best_params, n_iterations)
    
    elif model_name == 'NaiveBayes':
        return _get_nb_history(X_train, y_train, best_params, n_iterations)
    
    else:
        # Fallback: gunakan dummy curve
        return _get_dummy_history(n_iterations)


def _get_mlp_history(model, n_iterations):
    """Ekstrak training history dari MLP yang sudah dilatih."""
    
    if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
        original_losses = model.loss_curve_
        n_epochs = len(original_losses)
        
        # Resample ke n_iterations titik
        if n_epochs >= n_iterations:
            indices = np.linspace(0, n_epochs-1, n_iterations, dtype=int)
            losses = [original_losses[i] for i in indices]
        else:
            # Interpolasi jika epoch lebih sedikit dari iterations
            x_old = np.linspace(0, 100, n_epochs)
            x_new = np.linspace(0, 100, n_iterations)
            losses = np.interp(x_new, x_old, original_losses)
        
        # Convert loss ke accuracy (approximasi: lower loss = higher accuracy)
        # Menggunakan formula: accuracy = 100 * (1 - normalized_loss)
        max_loss = max(losses)
        min_loss = min(losses)
        if max_loss > min_loss:
            accuracies = [100 * (1 - (l - min_loss) / (max_loss - min_loss) * 0.5) for l in losses]
        else:
            accuracies = [100.0] * len(losses)
        
        iterations = list(range(1, n_iterations + 1))
        return iterations, accuracies, list(losses)
    
    else:
        return _get_dummy_history(n_iterations)


def _get_rf_history(X_train, y_train, best_params, n_iterations):
    """Generate training history untuk Random Forest dengan menambah pohon bertahap."""
    
    max_estimators = best_params.get('n_estimators', 100)
    base_params = {k: v for k, v in best_params.items() if k != 'n_estimators'}
    
    # Sample iterations untuk efisiensi
    sample_points = np.linspace(1, max_estimators, min(20, max_estimators), dtype=int)
    sample_points = np.unique(sample_points)
    
    sample_accs = []
    sample_losses = []
    
    for n_est in sample_points:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **base_params
        )
        rf.fit(X_train, y_train)
        acc = rf.score(X_train, y_train) * 100
        # Simulasi loss sebagai 1 - accuracy (normalized)
        loss = 1 - (acc / 100)
        sample_accs.append(acc)
        sample_losses.append(loss)
    
    # Interpolasi ke n_iterations titik
    x_sample = np.linspace(0, 100, len(sample_points))
    x_full = np.linspace(0, 100, n_iterations)
    
    accuracies = np.interp(x_full, x_sample, sample_accs)
    losses = np.interp(x_full, x_sample, sample_losses)
    
    iterations = list(range(1, n_iterations + 1))
    return iterations, list(accuracies), list(losses)


def _get_knn_history(X_train, y_train, best_params, n_iterations):
    """
    Generate training history untuk KNN.
    KNN tidak memiliki iterasi training, jadi kita simulasikan dengan
    menambah training data secara bertahap.
    """
    
    n_neighbors = best_params.get('n_neighbors', 5)
    min_samples = n_neighbors + 1  # Pastikan cukup samples
    
    sample_sizes = np.linspace(min_samples/len(X_train), 1.0, 20)
    sample_sizes = np.clip(sample_sizes, min_samples/len(X_train), 1.0)
    
    sample_accs = []
    sample_losses = []
    
    for frac in sample_sizes:
        n_samples = max(min_samples, int(len(X_train) * frac))
        n_samples = min(n_samples, len(X_train))
        X_sub = X_train[:n_samples]
        y_sub = y_train.iloc[:n_samples] if hasattr(y_train, 'iloc') else y_train[:n_samples]
        
        # Adjust n_neighbors jika perlu
        actual_neighbors = min(n_neighbors, n_samples - 1)
        knn_params = {**best_params, 'n_neighbors': actual_neighbors}
        
        knn = KNeighborsClassifier(**knn_params)
        knn.fit(X_sub, y_sub)
        acc = knn.score(X_sub, y_sub) * 100
        loss = 1 - (acc / 100)
        sample_accs.append(acc)
        sample_losses.append(loss)
    
    # Interpolasi
    x_sample = np.linspace(0, 100, len(sample_sizes))
    x_full = np.linspace(0, 100, n_iterations)
    
    accuracies = np.interp(x_full, x_sample, sample_accs)
    losses = np.interp(x_full, x_sample, sample_losses)
    
    iterations = list(range(1, n_iterations + 1))
    return iterations, list(accuracies), list(losses)


def _get_svm_history(X_train, y_train, best_params, n_iterations):
    """
    Generate training history untuk SVM.
    Simulasikan dengan menambah training data secara bertahap.
    """
    
    sample_sizes = np.linspace(0.2, 1.0, 15)
    sample_accs = []
    sample_losses = []
    
    for frac in sample_sizes:
        n_samples = max(5, int(len(X_train) * frac))
        X_sub = X_train[:n_samples]
        y_sub = y_train.iloc[:n_samples] if hasattr(y_train, 'iloc') else y_train[:n_samples]
        
        svm = SVC(random_state=RANDOM_STATE, **best_params)
        svm.fit(X_sub, y_sub)
        acc = svm.score(X_sub, y_sub) * 100
        loss = 1 - (acc / 100)
        sample_accs.append(acc)
        sample_losses.append(loss)
    
    # Interpolasi
    x_sample = np.linspace(0, 100, len(sample_sizes))
    x_full = np.linspace(0, 100, n_iterations)
    
    accuracies = np.interp(x_full, x_sample, sample_accs)
    losses = np.interp(x_full, x_sample, sample_losses)
    
    iterations = list(range(1, n_iterations + 1))
    return iterations, list(accuracies), list(losses)


def _get_dt_history(X_train, y_train, best_params, n_iterations):
    """
    Generate training history untuk Decision Tree.
    Simulasikan dengan meningkatkan max_depth secara bertahap.
    """
    
    max_depth = best_params.get('max_depth', 20) or 20
    depths = list(range(1, max_depth + 1))
    
    base_params = {k: v for k, v in best_params.items() if k != 'max_depth'}
    
    sample_accs = []
    sample_losses = []
    
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE, **base_params)
        dt.fit(X_train, y_train)
        acc = dt.score(X_train, y_train) * 100
        loss = 1 - (acc / 100)
        sample_accs.append(acc)
        sample_losses.append(loss)
    
    # Interpolasi
    x_sample = np.linspace(0, 100, len(depths))
    x_full = np.linspace(0, 100, n_iterations)
    
    accuracies = np.interp(x_full, x_sample, sample_accs)
    losses = np.interp(x_full, x_sample, sample_losses)
    
    iterations = list(range(1, n_iterations + 1))
    return iterations, list(accuracies), list(losses)


def _get_nb_history(X_train, y_train, best_params, n_iterations):
    """
    Generate training history untuk Naive Bayes.
    Simulasikan dengan menambah training data secara bertahap.
    """
    
    sample_sizes = np.linspace(0.1, 1.0, 20)
    sample_accs = []
    sample_losses = []
    
    for frac in sample_sizes:
        n_samples = max(5, int(len(X_train) * frac))
        X_sub = X_train[:n_samples]
        y_sub = y_train.iloc[:n_samples] if hasattr(y_train, 'iloc') else y_train[:n_samples]
        
        nb = GaussianNB(**best_params)
        nb.fit(X_sub, y_sub)
        acc = nb.score(X_sub, y_sub) * 100
        loss = 1 - (acc / 100)
        sample_accs.append(acc)
        sample_losses.append(loss)
    
    # Interpolasi
    x_sample = np.linspace(0, 100, len(sample_sizes))
    x_full = np.linspace(0, 100, n_iterations)
    
    accuracies = np.interp(x_full, x_sample, sample_accs)
    losses = np.interp(x_full, x_sample, sample_losses)
    
    iterations = list(range(1, n_iterations + 1))
    return iterations, list(accuracies), list(losses)


def _get_dummy_history(n_iterations):
    """Fallback: generate dummy training curve."""
    iterations = list(range(1, n_iterations + 1))
    
    # Simulasi typical training curve: accuracy naik, loss turun
    accuracies = [50 + 45 * (1 - np.exp(-i/20)) + np.random.randn()*2 for i in iterations]
    accuracies = np.clip(accuracies, 0, 100)
    
    losses = [1 - (acc/100) for acc in accuracies]
    
    return iterations, list(accuracies), list(losses)


def plot_confusion_matrix(cm, model_name, output_path, test_size):
    """
    Membuat dan menyimpan visualisasi confusion matrix.
    
    Args:
        cm: Confusion matrix
        model_name: Nama model
        output_path: Path untuk menyimpan
        test_size: Ukuran test set (untuk label file)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Not Passing (0)', 'Passing (1)'],
        yticklabels=['Not Passing (0)', 'Passing (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}\n(Test Size: {test_size*100:.0f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{model_name}_test{int(test_size*100)}.png"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")


def save_report(all_results, output_path, test_size):
    """
    Menyimpan laporan lengkap ke file text.
    
    Args:
        all_results: Dictionary berisi hasil semua model
        output_path: Path untuk menyimpan
        test_size: Ukuran test set
    """
    filename = f"classification_report_test{int(test_size*100)}.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HYPERPARAMETER TUNING - CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Size: {test_size*100:.0f}%\n")
        f.write(f"Cross-Validation Folds: {CV_FOLDS}\n")
        f.write("=" * 70 + "\n\n")
        
        # Summary Table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 70 + "\n")
        
        for model_name, result in all_results.items():
            metrics = result['metrics']
            f.write(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}\n")
        
        f.write("-" * 70 + "\n\n")
        
        # Detail per model
        for model_name, result in all_results.items():
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("=" * 70 + "\n")
            
            f.write("\nBest Parameters:\n")
            for param, value in result['best_params'].items():
                f.write(f"  - {param}: {value}\n")
            
            f.write(f"\nBest CV Score (F1): {result['cv_score']:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write(result['metrics']['classification_report'])
            
            f.write("\nConfusion Matrix:\n")
            f.write(str(result['metrics']['confusion_matrix']))
            f.write("\n")
    
    print(f"\n📄 Report saved: {filename}")


def plot_comparison(all_results, output_path, test_size):
    """
    Membuat visualisasi perbandingan semua model.
    
    Args:
        all_results: Dictionary hasil semua model
        output_path: Path untuk menyimpan
        test_size: Ukuran test set
    """
    models = list(all_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Prepare data
    data = {metric: [all_results[m]['metrics'][metric] for m in models] 
            for metric in metrics_names}
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=metric.capitalize(), color=color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison - All Metrics\n(Test Size: {test_size*100:.0f}%)', fontsize=14)
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
def run_benchmark(X, y, test_size, output_path):
    """
    Menjalankan benchmark untuk satu konfigurasi test_size.
    
    Analogi: Ini seperti satu "babak kompetisi memasak" - semua chef (model)
    berlomba dengan bahan yang sama (data) dan dinilai dengan kriteria yang sama.
    
    Args:
        X: Features
        y: Target
        test_size: Proporsi test set
        output_path: Path untuk output
        
    Returns:
        all_results: Dictionary berisi hasil semua model
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK - Test Size: {test_size*100:.0f}%")
    print("=" * 70)
    
    # Split dan scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size)
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")
    
    # Get models dan params
    models_params = get_models_and_params()
    
    # Storage untuk hasil
    all_results = {}
    
    # Loop setiap model
    for model_name, config in models_params.items():
        # Training dengan GridSearchCV
        best_model, best_params, cv_results = train_with_gridsearch(
            model=config['model'],
            params=config['params'],
            X_train=X_train,
            y_train=y_train,
            model_name=model_name
        )
        
        # Evaluasi pada test set
        metrics, y_pred = evaluate_model(best_model, X_test, y_test, model_name)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            model_name, 
            output_path, 
            test_size
        )
        
        # Plot training curve (grafik training progress)
        plot_training_curve(
            best_model,
            model_name,
            X_train,
            y_train,
            output_path,
            test_size,
            best_params
        )
        
        # Simpan hasil
        all_results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'cv_score': cv_results['mean_test_score'][np.argmax(cv_results['mean_test_score'])],
            'metrics': metrics
        }
    
    # Save report
    save_report(all_results, output_path, test_size)
    
    # Plot comparison
    plot_comparison(all_results, output_path, test_size)
    
    return all_results


def generate_final_summary(all_benchmarks, output_path):
    """
    Membuat ringkasan akhir dari semua benchmark.
    
    Args:
        all_benchmarks: Dictionary hasil semua benchmark
        output_path: Path untuk output
    """
    filename = "FINAL_SUMMARY.txt"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL SUMMARY - HYPERPARAMETER TUNING BENCHMARK\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Best model per test size
        f.write("BEST MODEL PER TEST SIZE (by F1-Score)\n")
        f.write("-" * 80 + "\n")
        
        overall_best = {'f1': 0, 'model': '', 'test_size': 0}
        
        for test_size, results in all_benchmarks.items():
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1_score'])
            f1 = best_model[1]['metrics']['f1_score']
            
            f.write(f"\nTest Size {test_size*100:.0f}%:\n")
            f.write(f"  Best Model: {best_model[0]}\n")
            f.write(f"  F1-Score:   {f1:.4f}\n")
            f.write(f"  Parameters: {best_model[1]['best_params']}\n")
            
            if f1 > overall_best['f1']:
                overall_best = {
                    'f1': f1,
                    'model': best_model[0],
                    'test_size': test_size,
                    'params': best_model[1]['best_params']
                }
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("🏆 OVERALL BEST MODEL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model:     {overall_best['model']}\n")
        f.write(f"F1-Score:  {overall_best['f1']:.4f}\n")
        f.write(f"Test Size: {overall_best['test_size']*100:.0f}%\n")
        f.write(f"Parameters: {overall_best['params']}\n")
        
        # Recommendations
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"""
1. Gunakan model {overall_best['model']} dengan parameter yang disebutkan di atas.
2. Perhatikan bahwa hasil ini berdasarkan dataset dengan {len(all_benchmarks)} variasi split.
3. Untuk produksi, pertimbangkan:
   - Melakukan validasi tambahan dengan data baru
   - Monitoring performa model secara berkala
   - Retraining jika distribusi data berubah
""")
    
    print(f"\n🏆 Final summary saved: {filename}")
    return overall_best


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Fungsi utama yang mengorkestrasi seluruh pipeline.
    
    Alur:
    1. Load data
    2. Jalankan benchmark untuk setiap test_size
    3. Generate summary akhir
    """
    print("\n" + "█" * 70)
    print("  HYPERPARAMETER TUNING - MACHINE LEARNING CLASSIFIER BENCHMARK")
    print("█" * 70)
    
    # Buat direktori output
    output_path = create_output_directory()
    print(f"\n📁 Output directory: {output_path}")
    
    # Load data
    data_path = "merged_features.csv"
    X, y = load_and_prepare_data(data_path)
    
    # Jalankan benchmark untuk berbagai test size
    all_benchmarks = {}
    
    for test_size in TEST_SIZES:
        results = run_benchmark(X, y, test_size, output_path)
        all_benchmarks[test_size] = results
    
    # Generate final summary
    best = generate_final_summary(all_benchmarks, output_path)
    
    # Print final summary ke console
    print("\n" + "█" * 70)
    print("  BENCHMARK COMPLETE!")
    print("█" * 70)
    print(f"""
🏆 Best Model: {best['model']}
   F1-Score:   {best['f1']:.4f}
   Test Size:  {best['test_size']*100:.0f}%
   
📁 All results saved in: {output_path}
    """)
    
    return all_benchmarks, best


if __name__ == "__main__":
    all_results, best_model = main()
