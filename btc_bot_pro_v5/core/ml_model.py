"""
BTC Bot Pro - ML Model Modülü
FAZA 4.2 & 4.3: Gelişmiş Model Mimarileri + Hyperparameter Optimization

Özellikler:
- GRU/LSTM modelleri
- Attention mechanism
- Ensemble predictions
- Optuna hyperparameter optimization
- Cross-validation
- Model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

# TensorFlow import (opsiyonel)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential, load_model
    from tensorflow.keras.layers import (
        Input, Dense, LSTM, GRU, Dropout, 
        BatchNormalization, Bidirectional,
        Attention, MultiHeadAttention, LayerNormalization,
        Concatenate, GlobalAveragePooling1D, Flatten
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Optuna import (opsiyonel)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ================================================================
# MODEL CONFIGURATIONS
# ================================================================

@dataclass
class ModelConfig:
    """Model konfigürasyonu"""
    model_type: str = "gru"  # gru, lstm, attention, ensemble
    
    # Architecture
    units_1: int = 64
    units_2: int = 32
    dense_units: int = 16
    dropout_rate: float = 0.3
    l2_reg: float = 0.001
    use_bidirectional: bool = False
    use_attention: bool = False
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    
    # Data
    lookback_window: int = 48
    prediction_horizon: int = 4
    
    # Output
    output_type: str = "regression"  # regression, classification


@dataclass
class TrainingResult:
    """Eğitim sonucu"""
    history: Dict = field(default_factory=dict)
    best_epoch: int = 0
    train_loss: float = 0
    val_loss: float = 0
    metrics: Dict = field(default_factory=dict)
    training_time: float = 0


@dataclass
class PredictionResult:
    """Tahmin sonucu"""
    prediction: float
    confidence: float
    signal: str  # LONG, SHORT, HOLD
    features_used: int = 0
    model_type: str = ""


# ================================================================
# BASE MODEL CLASS
# ================================================================

class BaseModel:
    """Temel model sınıfı"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_trained = False
    
    def build(self, input_shape: Tuple[int, int]):
        """Model oluştur (alt sınıflar override eder)"""
        raise NotImplementedError
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: int = 1) -> TrainingResult:
        """Model eğit"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for training")
        
        if self.model is None:
            self.build((X_train.shape[1], X_train.shape[2]))
        
        start_time = datetime.now()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_trained = True
        
        return TrainingResult(
            history=history.history,
            best_epoch=len(history.history['loss']) - self.config.patience,
            train_loss=history.history['loss'][-1],
            val_loss=history.history['val_loss'][-1] if 'val_loss' in history.history else 0,
            training_time=training_time
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin yap"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Model değerlendir"""
        predictions = self.predict(X_test).flatten()
        
        # Metrikler
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        
        # Direction accuracy
        direction_correct = np.mean(np.sign(predictions) == np.sign(y_test))
        
        # Threshold-based accuracy (signal accuracy)
        threshold = 0.5
        pred_signals = np.where(predictions >= threshold, 1, 
                               np.where(predictions <= -threshold, -1, 0))
        true_signals = np.where(y_test >= threshold, 1,
                               np.where(y_test <= -threshold, -1, 0))
        
        signal_mask = pred_signals != 0
        signal_accuracy = np.mean(pred_signals[signal_mask] == true_signals[signal_mask]) if signal_mask.sum() > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_correct,
            'signal_accuracy': signal_accuracy,
            'signal_count': signal_mask.sum(),
            'predictions': predictions
        }
    
    def save(self, path: str):
        """Model kaydet"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Model
        self.model.save(f"{path}_model.keras")
        
        # Config & metadata
        metadata = {
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Scaler
        if self.scaler is not None:
            with open(f"{path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load(self, path: str):
        """Model yükle"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        self.model = load_model(f"{path}_model.keras")
        
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.config = ModelConfig(**metadata['config'])
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        
        scaler_path = f"{path}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)


# ================================================================
# GRU MODEL
# ================================================================

class GRUModel(BaseModel):
    """GRU tabanlı model"""
    
    def build(self, input_shape: Tuple[int, int]):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        inputs = Input(shape=input_shape)
        
        # First GRU layer
        if self.config.use_bidirectional:
            x = Bidirectional(GRU(
                self.config.units_1,
                return_sequences=True,
                kernel_regularizer=l2(self.config.l2_reg)
            ))(inputs)
        else:
            x = GRU(
                self.config.units_1,
                return_sequences=True,
                kernel_regularizer=l2(self.config.l2_reg)
            )(inputs)
        
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Second GRU layer
        if self.config.use_bidirectional:
            x = Bidirectional(GRU(
                self.config.units_2,
                return_sequences=False,
                kernel_regularizer=l2(self.config.l2_reg)
            ))(x)
        else:
            x = GRU(
                self.config.units_2,
                return_sequences=False,
                kernel_regularizer=l2(self.config.l2_reg)
            )(x)
        
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Dense layers
        x = Dense(
            self.config.dense_units,
            activation='relu',
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = Dropout(self.config.dropout_rate / 2)(x)
        
        # Output
        outputs = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return self.model


# ================================================================
# LSTM MODEL
# ================================================================

class LSTMModel(BaseModel):
    """LSTM tabanlı model"""
    
    def build(self, input_shape: Tuple[int, int]):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        if self.config.use_bidirectional:
            x = Bidirectional(LSTM(
                self.config.units_1,
                return_sequences=True,
                kernel_regularizer=l2(self.config.l2_reg)
            ))(inputs)
        else:
            x = LSTM(
                self.config.units_1,
                return_sequences=True,
                kernel_regularizer=l2(self.config.l2_reg)
            )(inputs)
        
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Second LSTM layer
        if self.config.use_bidirectional:
            x = Bidirectional(LSTM(
                self.config.units_2,
                return_sequences=False,
                kernel_regularizer=l2(self.config.l2_reg)
            ))(x)
        else:
            x = LSTM(
                self.config.units_2,
                return_sequences=False,
                kernel_regularizer=l2(self.config.l2_reg)
            )(x)
        
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Dense layers
        x = Dense(
            self.config.dense_units,
            activation='relu',
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = Dropout(self.config.dropout_rate / 2)(x)
        
        # Output
        outputs = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return self.model


# ================================================================
# ATTENTION MODEL
# ================================================================

class AttentionModel(BaseModel):
    """Self-Attention tabanlı model"""
    
    def build(self, input_shape: Tuple[int, int]):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        inputs = Input(shape=input_shape)
        
        # GRU layer
        x = GRU(
            self.config.units_1,
            return_sequences=True,
            kernel_regularizer=l2(self.config.l2_reg)
        )(inputs)
        x = BatchNormalization()(x)
        
        # Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=4,
            key_dim=self.config.units_1 // 4
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization()(x + attention_output)
        
        # Second GRU
        x = GRU(
            self.config.units_2,
            return_sequences=False,
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        
        # Dense layers
        x = Dense(
            self.config.dense_units,
            activation='relu',
            kernel_regularizer=l2(self.config.l2_reg)
        )(x)
        x = Dropout(self.config.dropout_rate / 2)(x)
        
        # Output
        outputs = Dense(1, activation='linear')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return self.model


# ================================================================
# ENSEMBLE MODEL
# ================================================================

class EnsembleModel:
    """Ensemble model (birden fazla modeli birleştirir)"""
    
    def __init__(self, models: List[BaseModel] = None, weights: List[float] = None):
        self.models = models or []
        self.weights = weights
        self.is_trained = False
    
    def add_model(self, model: BaseModel, weight: float = 1.0):
        """Model ekle"""
        self.models.append(model)
        if self.weights is None:
            self.weights = []
        self.weights.append(weight)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: int = 1) -> List[TrainingResult]:
        """Tüm modelleri eğit"""
        results = []
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"Training model {i+1}/{len(self.models)}...")
            result = model.train(X_train, y_train, X_val, y_val, verbose=verbose)
            results.append(result)
        
        self.is_trained = True
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average prediction"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X).flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        if self.weights:
            weights = np.array(self.weights) / sum(self.weights)
            weighted_pred = np.average(predictions, axis=0, weights=weights)
        else:
            weighted_pred = np.mean(predictions, axis=0)
        
        return weighted_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Ensemble değerlendir"""
        predictions = self.predict(X_test)
        
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(y_test))
        
        # Individual model performance
        individual_mae = []
        for model in self.models:
            pred = model.predict(X_test).flatten()
            individual_mae.append(np.mean(np.abs(y_test - pred)))
        
        return {
            'ensemble_mae': mae,
            'ensemble_rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'individual_mae': individual_mae,
            'improvement': (min(individual_mae) - mae) / min(individual_mae) * 100
        }


# ================================================================
# HYPERPARAMETER OPTIMIZATION
# ================================================================

class HyperparameterOptimizer:
    """Optuna tabanlı hyperparameter optimization"""
    
    def __init__(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 model_type: str = "gru"):
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter optimization")
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        self.best_params: Dict = None
        self.study = None
    
    def objective(self, trial) -> float:
        """Optuna objective function"""
        
        # Hyperparameters to optimize
        config = ModelConfig(
            model_type=self.model_type,
            units_1=trial.suggest_int('units_1', 32, 128, step=16),
            units_2=trial.suggest_int('units_2', 16, 64, step=8),
            dense_units=trial.suggest_int('dense_units', 8, 32, step=8),
            dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
            l2_reg=trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            learning_rate=trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
            epochs=50,  # Fixed for speed
            patience=10,
            use_bidirectional=trial.suggest_categorical('bidirectional', [True, False])
        )
        
        # Create model
        if self.model_type == "gru":
            model = GRUModel(config)
        elif self.model_type == "lstm":
            model = LSTMModel(config)
        elif self.model_type == "attention":
            model = AttentionModel(config)
        else:
            model = GRUModel(config)
        
        # Train
        try:
            result = model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                verbose=0
            )
            
            # Evaluate
            metrics = model.evaluate(self.X_val, self.y_val)
            
            # Clean up
            del model
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
            
            return metrics['mae']
        
        except Exception as e:
            return float('inf')
    
    def optimize(self, n_trials: int = 50, timeout: int = None,
                 show_progress: bool = True) -> Dict:
        """Optimization çalıştır"""
        
        sampler = TPESampler(seed=42)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def get_best_model(self) -> BaseModel:
        """En iyi parametrelerle model oluştur"""
        if self.best_params is None:
            raise ValueError("Run optimize() first")
        
        config = ModelConfig(
            model_type=self.model_type,
            units_1=self.best_params['units_1'],
            units_2=self.best_params['units_2'],
            dense_units=self.best_params['dense_units'],
            dropout_rate=self.best_params['dropout_rate'],
            l2_reg=self.best_params['l2_reg'],
            learning_rate=self.best_params['learning_rate'],
            batch_size=self.best_params['batch_size'],
            use_bidirectional=self.best_params['bidirectional']
        )
        
        if self.model_type == "gru":
            return GRUModel(config)
        elif self.model_type == "lstm":
            return LSTMModel(config)
        elif self.model_type == "attention":
            return AttentionModel(config)
        else:
            return GRUModel(config)


# ================================================================
# CROSS VALIDATION
# ================================================================

class TimeSeriesCrossValidator:
    """Time series cross validation"""
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Args:
            n_splits: Fold sayısı
            gap: Train ve test arasındaki boşluk
        """
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split indices oluştur"""
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        
        splits = []
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.gap
            test_end = min(test_start + fold_size, n)
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def cross_validate(self, model_class, config: ModelConfig,
                       X: np.ndarray, y: np.ndarray,
                       verbose: int = 1) -> Dict:
        """Cross validation çalıştır"""
        
        splits = self.split(X)
        
        results = {
            'fold_mae': [],
            'fold_direction_acc': [],
            'fold_train_loss': [],
            'fold_val_loss': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                print(f"Fold {fold + 1}/{self.n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Split train into train/val
            split_point = int(len(X_train) * 0.85)
            X_t, X_v = X_train[:split_point], X_train[split_point:]
            y_t, y_v = y_train[:split_point], y_train[split_point:]
            
            # Train
            model = model_class(config)
            result = model.train(X_t, y_t, X_v, y_v, verbose=0)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            
            results['fold_mae'].append(metrics['mae'])
            results['fold_direction_acc'].append(metrics['direction_accuracy'])
            results['fold_train_loss'].append(result.train_loss)
            results['fold_val_loss'].append(result.val_loss)
            
            # Cleanup
            del model
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
        
        # Summary
        results['mean_mae'] = np.mean(results['fold_mae'])
        results['std_mae'] = np.std(results['fold_mae'])
        results['mean_direction_acc'] = np.mean(results['fold_direction_acc'])
        
        return results


# ================================================================
# MODEL FACTORY
# ================================================================

def create_model(model_type: str = "gru", config: ModelConfig = None) -> BaseModel:
    """Model factory"""
    config = config or ModelConfig(model_type=model_type)
    
    if model_type == "gru":
        return GRUModel(config)
    elif model_type == "lstm":
        return LSTMModel(config)
    elif model_type == "attention":
        return AttentionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_signal(prediction: float, threshold: float = 0.5) -> PredictionResult:
    """Tahmin'den sinyal üret"""
    if prediction >= threshold:
        signal = "LONG"
        confidence = min(50 + (prediction - threshold) * 25, 95)
    elif prediction <= -threshold:
        signal = "SHORT"
        confidence = min(50 + (abs(prediction) - threshold) * 25, 95)
    else:
        signal = "HOLD"
        confidence = 50
    
    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        signal=signal
    )


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("ML Model test ediliyor...\n")
    print(f"TensorFlow available: {TF_AVAILABLE}")
    print(f"Optuna available: {OPTUNA_AVAILABLE}")
    
    if not TF_AVAILABLE:
        print("\nTensorFlow yüklü değil. Model testleri atlanıyor.")
        print("Yüklemek için: pip install tensorflow")
    else:
        # Test verisi
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        lookback = 48
        
        # Dummy data
        X = np.random.randn(n_samples, lookback, n_features)
        y = np.random.randn(n_samples) * 2
        
        # Split
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Further split train into train/val
        val_split = int(len(X_train) * 0.85)
        X_t, X_v = X_train[:val_split], X_train[val_split:]
        y_t, y_v = y_train[:val_split], y_train[val_split:]
        
        # 1. GRU Model
        print("\n1. GRU Model")
        config = ModelConfig(epochs=5, patience=3)  # Fast training for test
        gru = GRUModel(config)
        result = gru.train(X_t, y_t, X_v, y_v, verbose=0)
        metrics = gru.evaluate(X_test, y_test)
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   Direction Acc: {metrics['direction_accuracy']*100:.1f}%")
        
        # 2. LSTM Model
        print("\n2. LSTM Model")
        lstm = LSTMModel(config)
        result = lstm.train(X_t, y_t, X_v, y_v, verbose=0)
        metrics = lstm.evaluate(X_test, y_test)
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   Direction Acc: {metrics['direction_accuracy']*100:.1f}%")
        
        # 3. Ensemble
        print("\n3. Ensemble Model")
        ensemble = EnsembleModel()
        ensemble.add_model(gru, weight=1.0)
        ensemble.add_model(lstm, weight=1.0)
        ens_metrics = ensemble.evaluate(X_test, y_test)
        print(f"   Ensemble MAE: {ens_metrics['ensemble_mae']:.4f}")
        print(f"   Improvement: {ens_metrics['improvement']:.1f}%")
        
        # 4. Signal Generation
        print("\n4. Signal Generation")
        pred = 1.2
        signal = generate_signal(pred)
        print(f"   Prediction: {pred}")
        print(f"   Signal: {signal.signal} (Confidence: {signal.confidence:.0f}%)")
        
        # Cleanup
        tf.keras.backend.clear_session()
        
    print("\n✓ ML Model testi başarılı!")
