"""
Test PyTorch GPU Setup
"""
import sys

print("="*70)
print("  TESTING PyTorch GPU SETUP")
print("="*70)

# Test 1: Import PyTorch
print("\n[1/5] Testing PyTorch import...")
try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"   Version: {torch.__version__}")
except ImportError as e:
    print(f"❌ Failed to import PyTorch: {e}")
    print("\n   Please install PyTorch:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Checking CUDA availability...")
if torch.cuda.is_available():
    print("✅ CUDA is available")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  CUDA not available - will use CPU")
    print("   This is okay but slower for training")

# Test 3: Import wrapper
print("\n[3/5] Testing PyTorch MLP wrapper import...")
try:
    from pytorch_mlp_wrapper import PyTorchMLPClassifier, check_gpu_availability
    print("✅ PyTorch MLP wrapper imported successfully")
except ImportError as e:
    print(f"❌ Failed to import wrapper: {e}")
    print("   Make sure pytorch_mlp_wrapper.py is in the same directory")
    sys.exit(1)

# Test 4: Create model
print("\n[4/5] Testing model creation...")
try:
    model = PyTorchMLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=10,
        random_state=42,
        verbose=0
    )
    print("✅ Model created successfully")
    print(f"   Device: {model.device}")
except Exception as e:
    print(f"❌ Failed to create model: {e}")
    sys.exit(1)

# Test 5: Test training
print("\n[5/5] Testing model training on dummy data...")
try:
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_classes=2, random_state=42)
    
    # Train
    print("   Training...")
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X[:10])
    
    print("✅ Model training successful")
    print(f"   Predictions: {y_pred[:5]}")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("  TEST SUMMARY")
print("="*70)
print("✅ All tests passed!")
print("\nYour setup is ready for GPU-accelerated MLP training!")
print("\nNext steps:")
print("  1. Run: python tuning_anova.py")
print("  2. Watch for GPU utilization with: nvidia-smi -l 1")
print("  3. Compare training speed with CPU version")
print("="*70)
