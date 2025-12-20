"""
=============================================================================
PYTORCH MLP WRAPPER - GPU ACCELERATED
=============================================================================
Wrapper untuk PyTorch MLP agar kompatibel dengan scikit-learn API.
Support GPU CUDA untuk training yang lebih cepat.

Author: ML Pipeline
Date: 2025-12-08
=============================================================================
"""
import sys
import io
# Fix Windows encoding untuk Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    PyTorch MLP Classifier dengan GPU support yang kompatibel dengan scikit-learn.
    
    Parameters:
    -----------
    hidden_layer_sizes : tuple, default=(100,)
        Ukuran hidden layers. Contoh: (100,) atau (100, 50)
    activation : str, default='relu'
        Fungsi aktivasi: 'relu', 'tanh', 'sigmoid'
    alpha : float, default=0.0001
        L2 regularization parameter
    learning_rate : str, default='constant'
        Learning rate schedule: 'constant' atau 'adaptive'
    max_iter : int, default=1000
        Maximum number of epochs
    random_state : int, default=None
        Random seed
    batch_size : int, default=32
        Batch size untuk training
    learning_rate_init : float, default=0.001
        Initial learning rate
    early_stopping : bool, default=False
        Whether to use early stopping
    validation_fraction : float, default=0.1
        Fraction of training data untuk validation (jika early_stopping=True)
    """
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 alpha=0.0001, learning_rate='constant', max_iter=1000,
                 random_state=None, batch_size=32, learning_rate_init=0.001,
                 early_stopping=False, validation_fraction=0.1, verbose=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        
        # Device detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model akan di-initialize saat fit
        self.model_ = None
        self.label_encoder_ = None
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
    
    def _build_model(self, n_features, n_classes):
        """Build PyTorch model architecture."""
        layers = []
        
        # Input layer
        if isinstance(self.hidden_layer_sizes, int):
            hidden_sizes = [self.hidden_layer_sizes]
        else:
            hidden_sizes = list(self.hidden_layer_sizes)
        
        # Build hidden layers
        prev_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))
        
        # Create sequential model
        model = nn.Sequential(*layers)
        
        return model.to(self.device)
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Store input features
        self.n_features_in_ = X.shape[1]
        
        # Build model
        self.model_ = self._build_model(self.n_features_in_, self.n_classes_)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Split for validation if early stopping
        if self.early_stopping:
            n_samples = X.shape[0]
            n_val = int(n_samples * self.validation_fraction)
            indices = np.random.permutation(n_samples)
            
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            
            X_train = X_tensor[train_idx]
            y_train = y_tensor[train_idx]
            X_val = X_tensor[val_idx]
            y_val = y_tensor[val_idx]
        else:
            X_train = X_tensor
            y_train = y_tensor
            X_val = None
            y_val = None
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), 
                              lr=self.learning_rate_init,
                              weight_decay=self.alpha)
        
        # Learning rate scheduler
        if self.learning_rate == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                             factor=0.5, patience=10)
        else:
            scheduler = None
        
        # Training loop
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // self.batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(self.max_iter):
            self.model_.train()
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= n_batches
            
            # Validation
            if self.early_stopping and X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_outputs = self.model_(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    if self.verbose > 0:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if scheduler is not None:
                    scheduler.step(val_loss)
            
            # Print progress
            if self.verbose > 0 and (epoch + 1) % 100 == 0:
                if self.early_stopping and X_val is not None:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {epoch_loss:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        self.model_.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        return self.label_encoder_.inverse_transform(predicted)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        self.model_.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model_(X_tensor)
            proba = torch.softmax(outputs, dim=1)
            proba = proba.cpu().numpy()
        
        return proba
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'batch_size': self.batch_size,
            'learning_rate_init': self.learning_rate_init,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def __del__(self):
        """Cleanup GPU memory when object is deleted."""
        # PENTING: Defensive check untuk avoid error saat Python shutdown
        # torch module bisa jadi None saat garbage collection
        if torch is None:
            return
        
        if hasattr(self, 'model_') and self.model_ is not None:
            del self.model_
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Utility function untuk cek GPU
def check_gpu_availability():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠️  GPU not available, using CPU")
        return False
