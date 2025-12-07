# 🚀 GPU Acceleration Setup Guide (RTX 3050)

## Prerequisites

- **NVIDIA GPU**: RTX 3050 (sudah tersedia) ✅
- **CUDA Toolkit**: Version 11.8 atau 12.x
- **Conda/Miniconda**: Untuk instalasi cuML

---

## 📦 Instalasi cuML (RAPIDS)

### Opsi 1: Install via Conda (Recommended)

```bash
# Create new conda environment with Python 3.10
conda create -n ml_gpu python=3.10 -y
conda activate ml_gpu

# Install cuML for CUDA 12.x (sesuaikan dengan CUDA version Anda)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.12 python=3.10 cudatoolkit=12.0 -y

# Install dependencies lainnya
pip install scikit-learn pandas matplotlib seaborn skrebate
```

### Opsi 2: Install via pip (Experimental)

```bash
pip install cuml-cu12  # untuk CUDA 12.x
# atau
pip install cuml-cu11  # untuk CUDA 11.x
```

---

## 🔍 Verifikasi Instalasi

Jalankan script berikut untuk mengecek GPU:

```python
import cuml
import cudf
import torch

# Check cuML version
print(f"cuML version: {cuml.__version__}")

# Check GPU availability
import numba.cuda
print(f"GPU Available: {numba.cuda.is_available()}")
print(f"GPU Name: {numba.cuda.get_current_device().name.decode()}")

# Check CUDA version
print(f"CUDA Version: {torch.version.cuda}")
```

---

## ⚡ Penggunaan

### Automatic Detection

Script sudah dikonfigurasi untuk **auto-detect** GPU:

- Jika cuML terinstall → otomatis pakai **GPU (RTX 3050)**
- Jika cuML tidak ada → fallback ke **CPU (sklearn)**

### Running Training

```bash
# Activate environment
conda activate ml_gpu

# Run training dengan GPU acceleration
python tuning_anova.py
python tuning_chi2.py
python tuning_relieff.py
```

Output akan menunjukkan:

```
✓ cuML detected - GPU acceleration ENABLED (RTX 3050)
Compute Device: GPU (RTX 3050)
```

---

## 🎯 Model GPU Support

| Model            | GPU Support | Library |
| ---------------- | ----------- | ------- |
| **KNN**          | ✅ YES      | cuML    |
| **SVM**          | ✅ YES      | cuML    |
| **RandomForest** | ✅ YES      | cuML    |
| **MLP**          | ❌ CPU only | sklearn |
| **NaiveBayes**   | ❌ CPU only | sklearn |
| **DecisionTree** | ❌ CPU only | sklearn |

**Note**: MLP, NaiveBayes, dan DecisionTree akan tetap berjalan di CPU karena cuML belum support model tersebut.

---

## 🔧 Troubleshooting

### cuML tidak terdeteksi setelah install

```bash
# Check installation
conda list | grep cuml

# Reinstall jika perlu
conda install -c rapidsai -c conda-forge -c nvidia cuml --force-reinstall
```

### CUDA version mismatch

```bash
# Check CUDA version di system
nvidia-smi

# Install cuML sesuai CUDA version:
# CUDA 11.x:
conda install -c rapidsai -c conda-forge -c nvidia cuml cudatoolkit=11.8

# CUDA 12.x:
conda install -c rapidsai -c conda-forge -c nvidia cuml cudatoolkit=12.0
```

### Out of Memory (OOM)

Jika RTX 3050 (4GB VRAM) kehabisan memory:

1. **Reduce batch size** dalam GridSearchCV:

   ```python
   # Tambahkan di script
   CV_FOLDS = 3  # reduce dari 5 ke 3
   ```

2. **Reduce parameter grid**:

   ```python
   K_FEATURES = [5, 10, 15]  # reduce options
   ```

3. **Train model one-by-one**:
   ```python
   # Comment out model yang tidak dibutuhkan
   # models_params = {'KNN': {...}}  # hanya train 1 model
   ```

---

## 📊 Expected Performance Gain

Dengan RTX 3050, ekspektasi speedup:

- **KNN**: 5-10x faster
- **SVM (RBF kernel)**: 10-20x faster
- **RandomForest**: 3-7x faster

**Dataset size impact**:

- Small (<10k samples): Speedup minimal (overhead transfer)
- Medium (10k-100k): Speedup 5-15x
- Large (>100k): Speedup 10-50x

Dataset Anda (~5k samples) akan mendapat speedup **5-10x** untuk KNN/SVM/RF.

---

## 🆚 Fallback Behavior

Jika script tidak menemukan cuML:

```
⚠️  cuML not found - falling back to CPU (sklearn)
   Install cuML: conda install -c rapidsai -c conda-forge -c nvidia cuml
Compute Device: CPU
```

Script tetap berjalan normal dengan sklearn (CPU), tidak ada error.

---

## 📝 Notes

1. **First run lebih lambat**: Kompilasi CUDA kernels (warmup)
2. **Memory transfer overhead**: Untuk dataset kecil, GPU bisa lebih lambat
3. **Mixed CPU-GPU**: Model yang tidak support GPU (MLP, NB, DT) otomatis pakai CPU
4. **GridSearchCV parallelism**: cuML GridSearchCV pakai GPU parallelism, lebih efisien dari `n_jobs=-1` di CPU

---

## 🔗 Resources

- [RAPIDS cuML Docs](https://docs.rapids.ai/api/cuml/stable/)
- [cuML GitHub](https://github.com/rapidsai/cuml)
- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
