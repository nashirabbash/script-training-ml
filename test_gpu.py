"""
GPU Detection & Benchmark Script
Test apakah RTX 3050 terdeteksi dan bandingkan performa CPU vs GPU
"""

import time
import numpy as np
import pandas as pd

print("=" * 70)
print("GPU DETECTION & BENCHMARK TEST")
print("=" * 70)

# ============================================================================
# 1. CHECK GPU AVAILABILITY
# ============================================================================
print("\n[1] Checking GPU Availability...")

try:
    import cuml
    import cudf
    import numba.cuda
    
    print("✓ cuML installed successfully!")
    print(f"  cuML version: {cuml.__version__}")
    
    if numba.cuda.is_available():
        print("✓ GPU is available!")
        gpu_device = numba.cuda.get_current_device()
        print(f"  GPU Name: {gpu_device.name.decode()}")
        print(f"  Compute Capability: {gpu_device.compute_capability}")
        
        # Get GPU memory info
        ctx = numba.cuda.current_context()
        total_memory = ctx.get_memory_info()[1] / (1024**3)  # Convert to GB
        free_memory = ctx.get_memory_info()[0] / (1024**3)
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Free Memory: {free_memory:.2f} GB")
        
        GPU_AVAILABLE = True
    else:
        print("⚠️  cuML installed but GPU not detected!")
        print("   Check NVIDIA drivers and CUDA installation")
        GPU_AVAILABLE = False
        
except ImportError as e:
    print("⚠️  cuML not installed")
    print("   Install via: conda install -c rapidsai -c conda-forge -c nvidia cuml")
    GPU_AVAILABLE = False

# ============================================================================
# 2. BENCHMARK: CPU vs GPU
# ============================================================================
print("\n[2] Running Performance Benchmark...")
print("    Testing KNN classifier with synthetic data")

# Generate synthetic dataset
print("\n  Generating synthetic data (10,000 samples, 100 features)...")
np.random.seed(42)
n_samples = 10000
n_features = 100
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, 2, size=n_samples).astype(np.int32)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- CPU Benchmark (sklearn) ----
print("\n  [CPU] Training KNN with sklearn...")
from sklearn.neighbors import KNeighborsClassifier as CPU_KNN

cpu_start = time.time()
cpu_model = CPU_KNN(n_neighbors=5)
cpu_model.fit(X_train, y_train)
cpu_pred = cpu_model.predict(X_test)
cpu_time = time.time() - cpu_start

from sklearn.metrics import accuracy_score
cpu_accuracy = accuracy_score(y_test, cpu_pred)

print(f"    Time: {cpu_time:.4f}s | Accuracy: {cpu_accuracy:.4f}")

# ---- GPU Benchmark (cuML) ----
if GPU_AVAILABLE:
    print("\n  [GPU] Training KNN with cuML...")
    from cuml.neighbors import KNeighborsClassifier as GPU_KNN
    
    # Convert to cuDF
    import cudf
    X_train_gpu = cudf.DataFrame(X_train)
    y_train_gpu = cudf.Series(y_train)
    X_test_gpu = cudf.DataFrame(X_test)
    
    gpu_start = time.time()
    gpu_model = GPU_KNN(n_neighbors=5)
    gpu_model.fit(X_train_gpu, y_train_gpu)
    gpu_pred = gpu_model.predict(X_test_gpu).to_numpy()
    gpu_time = time.time() - gpu_start
    
    gpu_accuracy = accuracy_score(y_test, gpu_pred)
    
    print(f"    Time: {gpu_time:.4f}s | Accuracy: {gpu_accuracy:.4f}")
    
    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\n  🚀 GPU Speedup: {speedup:.2f}x faster!")
    
    if speedup < 1:
        print("  ⚠️  Note: GPU slower for small datasets due to transfer overhead")
        print("     GPU advantage increases with larger datasets (>50k samples)")
else:
    print("\n  ⚠️  GPU benchmark skipped (cuML not available)")

# ============================================================================
# 3. MEMORY TEST
# ============================================================================
if GPU_AVAILABLE:
    print("\n[3] GPU Memory Test...")
    
    # Try progressively larger datasets
    test_sizes = [1000, 5000, 10000, 20000]
    
    for size in test_sizes:
        try:
            X_test_mem = np.random.randn(size, 100).astype(np.float32)
            X_gpu_test = cudf.DataFrame(X_test_mem)
            print(f"  ✓ Successfully allocated {size:,} samples on GPU")
            del X_gpu_test  # Free memory
        except Exception as e:
            print(f"  ✗ Failed at {size:,} samples: {e}")
            break

# ============================================================================
# 4. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if GPU_AVAILABLE:
    print("✓ GPU Status: READY (RTX 3050)")
    print("✓ Scripts will automatically use GPU acceleration")
    print("\nNext steps:")
    print("  1. Run: python tuning_anova.py")
    print("  2. Check output for 'Compute Device: GPU (RTX 3050)'")
    print("  3. Enjoy faster training! 🚀")
else:
    print("⚠️  GPU Status: NOT AVAILABLE")
    print("⚠️  Scripts will fall back to CPU (sklearn)")
    print("\nTo enable GPU:")
    print("  1. Install cuML: conda install -c rapidsai -c conda-forge -c nvidia cuml")
    print("  2. Verify CUDA: nvidia-smi")
    print("  3. Re-run this test")

print("=" * 70)
