import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using fallback prediction methods.")

import joblib
import os
from dataclasses import dataclass
import logging

@dataclass
class PredictionConfig:
    sequence_length: int = 60
    prediction_days: int = 5
    features: List[str] = None
    model_type: str = 'lstm'
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['Close', 'Volume', 'High', 'Low', 'Open']

class LSTMPredictor:
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scalers = {}
        self.is_trained = False
        self.model_path = "models/saved_models"
        self.scaler_path = "models/scalers"
        
        # Create directories if they don't exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Technical indicators for feature engineering
        self.technical_indicators = [
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Volume_SMA', 'Price_Volume_Trend'
        ]
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for feature engineering"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Price_Volume_Trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Calculate technical indicators
        df = self.calculate_technical_indicators(data)
        
        # Select features
        feature_columns = self.config.features + [col for col in self.technical_indicators if col in df.columns]
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < self.config.sequence_length + self.config.prediction_days:
            raise ValueError(f"Insufficient data. Need at least {self.config.sequence_length + self.config.prediction_days} rows.")
        
        # Prepare features and target
        features = df[feature_columns].values
        target = df[target_column].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.config.sequence_length, len(scaled_features) - self.config.prediction_days + 1):
            # Input sequence
            X.append(scaled_features[i-self.config.sequence_length:i])
            
            # Target (next prediction_days prices)
            y.append(target[i:i+self.config.prediction_days])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.logger.info(f"Data prepared: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Please install tensorflow.")
        
        model = Sequential([
            # First LSTM layer
            LSTM(units=128, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(units=64, return_sequences=True),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(units=32, return_sequences=False),
            Dropout(self.config.dropout_rate),
            
            # Dense layers
            Dense(units=50, activation='relu'),
            Dropout(self.config.dropout_rate),
            
            Dense(units=25, activation='relu'),
            
            # Output layer
            Dense(units=self.config.prediction_days, activation='linear')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, symbol: str, target_column: str = 'Close') -> Dict:
        """Train the LSTM model"""
        try:
            self.logger.info(f"Starting training for {symbol}")
            
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_data(data, target_column)
            
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            val_loss = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Save model and scaler
            self.save_model(symbol)
            
            self.is_trained = True
            
            training_results = {
                'symbol': symbol,
                'train_loss': train_loss[0],
                'train_mae': train_loss[1],
                'val_loss': val_loss[0],
                'val_mae': val_loss[1],
                'epochs_trained': len(history.history['loss']),
                'training_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"Training completed for {symbol}. Val Loss: {val_loss[0]:.6f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Training failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def predict(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """Make predictions using trained model"""
        try:
            # Load model if not in memory
            if self.model is None and symbol:
                self.load_model(symbol)
            
            if self.model is None:
                return self._fallback_prediction(data)
            
            # Prepare data for prediction
            df = self.calculate_technical_indicators(data)
            
            # Select features
            feature_columns = self.config.features + [col for col in self.technical_indicators if col in df.columns]
            feature_columns = [col for col in feature_columns if col in df.columns]
            
            # Get last sequence
            df = df.dropna()
            if len(df) < self.config.sequence_length:
                return self._fallback_prediction(data)
            
            features = df[feature_columns].tail(self.config.sequence_length).values
            scaled_features = self.scaler.transform(features)
            
            # Reshape for prediction
            X = scaled_features.reshape(1, self.config.sequence_length, len(feature_columns))
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            predicted_prices = prediction[0]
            
            # Calculate confidence based on recent volatility
            recent_volatility = df['Close'].pct_change().tail(20).std()
            confidence = max(0.6, 1 - (recent_volatility * 10))  # Simple confidence metric
            
            # Generate prediction dates
            last_date = df.index[-1]
            prediction_dates = []
            for i in range(1, self.config.prediction_days + 1):
                next_date = last_date + timedelta(days=i)
                # Skip weekends
                while next_date.weekday() >= 5:
                    next_date += timedelta(days=1)
                prediction_dates.append(next_date)
            
            # Format results
            predictions = []
            current_price = df['Close'].iloc[-1]
            
            for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
                direction = 'up' if price > current_price else 'down'
                change_pct = ((price - current_price) / current_price) * 100
                
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': round(float(price), 2),
                    'direction': direction,
                    'change_percent': round(change_pct, 2),
                    'confidence': round(confidence * 100, 1)
                })
                
                current_price = price  # Update for next iteration
            
            return {
                'symbol': symbol,
                'current_price': round(df['Close'].iloc[-1], 2),
                'predictions': predictions,
                'model_type': 'LSTM',
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction(data, symbol)
    
    def _fallback_prediction(self, data: pd.DataFrame, symbol: str = None) -> Dict:
        """Fallback prediction method when LSTM is not available"""
        try:
            df = data.copy()
            current_price = df['Close'].iloc[-1]
            
            # Simple trend-based prediction
            recent_trend = df['Close'].tail(10).pct_change().mean()
            volatility = df['Close'].pct_change().tail(20).std()
            
            predictions = []
            
            for i in range(1, self.config.prediction_days + 1):
                # Simple random walk with trend
                trend_component = recent_trend * i
                random_component = np.random.normal(0, volatility) * 0.5
                
                predicted_change = trend_component + random_component
                predicted_price = current_price * (1 + predicted_change)
                
                direction = 'up' if predicted_price > current_price else 'down'
                change_pct = ((predicted_price - current_price) / current_price) * 100
                
                # Generate date
                last_date = df.index[-1]
                pred_date = last_date + timedelta(days=i)
                while pred_date.weekday() >= 5:
                    pred_date += timedelta(days=1)
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_price': round(predicted_price, 2),
                    'direction': direction,
                    'change_percent': round(change_pct, 2),
                    'confidence': 65.0  # Lower confidence for fallback
                })
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'predictions': predictions,
                'model_type': 'Fallback',
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}")
            return {'error': str(e)}
    
    def save_model(self, symbol: str):
        """Save trained model and scaler"""
        try:
            if self.model and TENSORFLOW_AVAILABLE:
                model_file = os.path.join(self.model_path, f"{symbol}_lstm_model.h5")
                self.model.save(model_file)
                
            scaler_file = os.path.join(self.scaler_path, f"{symbol}_scaler.pkl")
            joblib.dump(self.scaler, scaler_file)
            
            self.logger.info(f"Model and scaler saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model for {symbol}: {e}")
    
    def load_model(self, symbol: str) -> bool:
        """Load trained model and scaler"""
        try:
            model_file = os.path.join(self.model_path, f"{symbol}_lstm_model.h5")
            scaler_file = os.path.join(self.scaler_path, f"{symbol}_scaler.pkl")
            
            if os.path.exists(model_file) and TENSORFLOW_AVAILABLE:
                self.model = load_model(model_file)
                self.is_trained = True
                
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                
            self.logger.info(f"Model and scaler loaded for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model for {symbol}: {e}")
            return False
    
    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance (simplified version)"""
        try:
            df = self.calculate_technical_indicators(data)
            
            # Calculate correlation with target (Close price)
            correlations = {}
            for feature in self.config.features + self.technical_indicators:
                if feature in df.columns and feature != 'Close':
                    corr = df[feature].corr(df['Close'])
                    if not np.isnan(corr):
                        correlations[feature] = abs(corr)
            
            # Normalize to sum to 1
            total = sum(correlations.values())
            if total > 0:
                correlations = {k: v/total for k, v in correlations.items()}
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def backtest(self, data: pd.DataFrame, symbol: str, test_days: int = 30) -> Dict:
        """Backtest the model on historical data"""
        try:
            # Split data for backtesting
            train_data = data.iloc[:-test_days]
            test_data = data.iloc[-test_days:]
            
            # Train on historical data
            training_results = self.train(train_data, symbol)
            
            if 'error' in training_results:
                return training_results
            
            # Make predictions on test data
            predictions = []
            actual_prices = []
            
            for i in range(len(test_data) - self.config.prediction_days):
                # Use data up to current point
                current_data = data.iloc[:-(test_days-i)]
                
                # Make prediction
                pred_result = self.predict(current_data, symbol)
                
                if 'predictions' in pred_result and pred_result['predictions']:
                    predicted_price = pred_result['predictions'][0]['predicted_price']
                    actual_price = test_data.iloc[i + 1]['Close']
                    
                    predictions.append(predicted_price)
                    actual_prices.append(actual_price)
            
            if predictions and actual_prices:
                # Calculate metrics
                mse = mean_squared_error(actual_prices, predictions)
                mae = mean_absolute_error(actual_prices, predictions)
                
                # Calculate directional accuracy
                correct_direction = 0
                for i in range(1, len(predictions)):
                    pred_direction = predictions[i] > predictions[i-1]
                    actual_direction = actual_prices[i] > actual_prices[i-1]
                    if pred_direction == actual_direction:
                        correct_direction += 1
                
                directional_accuracy = correct_direction / (len(predictions) - 1) if len(predictions) > 1 else 0
                
                return {
                    'symbol': symbol,
                    'test_period_days': test_days,
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'directional_accuracy': directional_accuracy,
                    'predictions': predictions,
                    'actual_prices': actual_prices,
                    'backtest_time': datetime.now().isoformat()
                }
            else:
                return {'error': 'No valid predictions generated during backtest'}
                
        except Exception as e:
            self.logger.error(f"Backtesting failed for {symbol}: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Create predictor
    config = PredictionConfig(
        sequence_length=60,
        prediction_days=5,
        epochs=50,
        batch_size=32
    )
    
    predictor = LSTMPredictor(config)
    
    # Generate sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate stock price data
    price = 1000
    prices = [price]
    
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(100000, 1000000) for _ in prices]
    }, index=dates)
    
    print("Sample data created:")
    print(sample_data.tail())
    
    # Train model
    print("\nTraining model...")
    training_results = predictor.train(sample_data, 'TEST_STOCK')
    print("Training results:", training_results)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(sample_data, 'TEST_STOCK')
    print("Predictions:", predictions)