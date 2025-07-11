import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, GJR_GARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("ARCH package not available. Using fallback volatility methods.")

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from dataclasses import dataclass
import logging

@dataclass
class VolatilityConfig:
    lookback_period: int = 252  # Trading days in a year
    prediction_horizon: int = 22  # Trading days in a month
    garch_p: int = 1  # GARCH lag order
    garch_q: int = 1  # ARCH lag order
    confidence_levels: List[float] = None
    vol_models: List[str] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]
        if self.vol_models is None:
            self.vol_models = ['GARCH', 'EWMA', 'Historical', 'ML']

class VolatilityPredictor:
    def __init__(self, config: VolatilityConfig = None):
        self.config = config or VolatilityConfig()
        self.models = {}
        self.scalers = {}
        self.is_fitted = {}
        self.model_path = "models/volatility_models"
        
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize volatility models"""
        self.models = {
            'GARCH': None,
            'EWMA': None,
            'Historical': None,
            'ML': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        self.is_fitted = {model: False for model in self.models.keys()}
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 22) -> pd.Series:
        """Calculate realized volatility using rolling standard deviation"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def calculate_parkinson_volatility(self, data: pd.DataFrame, window: int = 22) -> pd.Series:
        """Calculate Parkinson volatility estimator using high-low prices"""
        if 'High' not in data.columns or 'Low' not in data.columns:
            return None
        
        hl_ratio = np.log(data['High'] / data['Low'])
        parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * hl_ratio**2)
        return parkinson_vol.rolling(window=window).mean() * np.sqrt(252)
    
    def calculate_garman_klass_volatility(self, data: pd.DataFrame, window: int = 22) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            return None
        
        hl = np.log(data['High'] / data['Low'])
        co = np.log(data['Close'] / data['Open'])
        
        gk_vol = np.sqrt(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)
        return gk_vol.rolling(window=window).mean() * np.sqrt(252)
    
    def fit_garch_model(self, returns: pd.Series, symbol: str) -> Dict:
        """Fit GARCH model to returns"""
        try:
            if not ARCH_AVAILABLE:
                return {'error': 'ARCH package not available'}
            
            # Remove any infinite or NaN values
            clean_returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_returns) < 100:
                return {'error': 'Insufficient data for GARCH model'}
            
            # Convert to percentage returns for better numerical stability
            clean_returns = clean_returns * 100
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(
                clean_returns, 
                vol='Garch', 
                p=self.config.garch_p, 
                q=self.config.garch_q,
                dist='normal'
            )
            
            garch_fit = garch_model.fit(disp='off')
            
            # Store the fitted model
            self.models['GARCH'] = garch_fit
            self.is_fitted['GARCH'] = True
            
            # Calculate model diagnostics
            aic = garch_fit.aic
            bic = garch_fit.bic
            log_likelihood = garch_fit.loglikelihood
            
            return {
                'model_type': 'GARCH',
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'parameters': garch_fit.params.to_dict(),
                'fitted_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GARCH model fitting failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def fit_ewma_model(self, returns: pd.Series, lambda_param: float = 0.94) -> Dict:
        """Fit Exponentially Weighted Moving Average model"""
        try:
            # Calculate EWMA variance
            ewma_var = returns.ewm(alpha=1-lambda_param).var()
            
            self.models['EWMA'] = {
                'lambda': lambda_param,
                'last_variance': ewma_var.iloc[-1],
                'last_return': returns.iloc[-1]
            }
            self.is_fitted['EWMA'] = True
            
            return {
                'model_type': 'EWMA',
                'lambda_parameter': lambda_param,
                'last_variance': ewma_var.iloc[-1],
                'fitted_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"EWMA model fitting failed: {e}")
            return {'error': str(e)}
    
    def fit_historical_model(self, returns: pd.Series) -> Dict:
        """Fit historical volatility model"""
        try:
            # Calculate rolling volatilities for different windows
            vol_windows = [22, 66, 252]  # 1 month, 3 months, 1 year
            
            historical_vols = {}
            for window in vol_windows:
                if len(returns) >= window:
                    vol = returns.rolling(window=window).std() * np.sqrt(252)
                    historical_vols[f'vol_{window}d'] = vol.iloc[-1]
            
            self.models['Historical'] = historical_vols
            self.is_fitted['Historical'] = True
            
            return {
                'model_type': 'Historical',
                'volatilities': historical_vols,
                'fitted_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Historical model fitting failed: {e}")
            return {'error': str(e)}
    
    def prepare_ml_features(self, data: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Prepare features for ML volatility model"""
        features = pd.DataFrame(index=returns.index)
        
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Squared returns (proxy for volatility)
        for lag in [1, 2, 3, 5, 10]:
            features[f'squared_return_lag_{lag}'] = (returns**2).shift(lag)
        
        # Rolling volatilities
        for window in [5, 10, 22, 66]:
            features[f'rolling_vol_{window}'] = returns.rolling(window=window).std()
        
        # Price-based features if available
        if 'High' in data.columns and 'Low' in data.columns:
            features['high_low_ratio'] = np.log(data['High'] / data['Low'])
            features['high_low_vol'] = features['high_low_ratio'].rolling(window=22).std()
        
        if 'Volume' in data.columns:
            features['volume'] = data['Volume']
            features['volume_ma'] = data['Volume'].rolling(window=22).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_ma']
        
        # Market microstructure features
        features['return_skewness'] = returns.rolling(window=22).skew()
        features['return_kurtosis'] = returns.rolling(window=22).kurt()
        
        # VIX-like features (fear index)
        features['abs_return'] = returns.abs()
        features['abs_return_ma'] = features['abs_return'].rolling(window=22).mean()
        
        return features.dropna()
    
    def fit_ml_model(self, data: pd.DataFrame, returns: pd.Series, symbol: str) -> Dict:
        """Fit machine learning model for volatility prediction"""
        try:
            # Prepare features
            features_df = self.prepare_ml_features(data, returns)
            
            if len(features_df) < 100:
                return {'error': 'Insufficient data for ML model'}
            
            # Target: future realized volatility
            target_window = 22  # 1 month ahead
            target = returns.rolling(window=target_window).std().shift(-target_window) * np.sqrt(252)
            
            # Align features and target
            common_index = features_df.index.intersection(target.dropna().index)
            X = features_df.loc[common_index]
            y = target.loc[common_index]
            
            if len(X) < 50:
                return {'error': 'Insufficient aligned data for ML model'}
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            self.models['ML'].fit(X_train_scaled, y_train)
            self.scalers['ML'] = scaler
            self.is_fitted['ML'] = True
            
            # Evaluate
            y_pred = self.models['ML'].predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, self.models['ML'].feature_importances_))
            
            return {
                'model_type': 'ML',
                'mse': mse,
                'mae': mae,
                'feature_importance': feature_importance,
                'fitted_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ML model fitting failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def fit_all_models(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Fit all volatility models"""
        results = {}
        
        # Calculate returns
        returns = self.calculate_returns(data['Close'])
        
        # Fit each model
        for model_name in self.config.vol_models:
            self.logger.info(f"Fitting {model_name} model for {symbol}")
            
            if model_name == 'GARCH':
                results[model_name] = self.fit_garch_model(returns, symbol)
            elif model_name == 'EWMA':
                results[model_name] = self.fit_ewma_model(returns)
            elif model_name == 'Historical':
                results[model_name] = self.fit_historical_model(returns)
            elif model_name == 'ML':
                results[model_name] = self.fit_ml_model(data, returns, symbol)
        
        return results
    
    def predict_volatility(self, data: pd.DataFrame, symbol: str = None, horizon: int = None) -> Dict:
        """Predict volatility using all fitted models"""
        try:
            horizon = horizon or self.config.prediction_horizon
            returns = self.calculate_returns(data['Close'])
            
            predictions = {}
            
            # GARCH predictions
            if self.is_fitted.get('GARCH', False) and self.models['GARCH']:
                try:
                    garch_forecast = self.models['GARCH'].forecast(horizon=horizon)
                    garch_vol = np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(252)
                    predictions['GARCH'] = {
                        'volatility_forecast': garch_vol.tolist(),
                        'confidence_intervals': self._calculate_confidence_intervals(garch_vol, 'GARCH')
                    }
                except Exception as e:
                    self.logger.warning(f"GARCH prediction failed: {e}")
            
            # EWMA predictions
            if self.is_fitted.get('EWMA', False) and self.models['EWMA']:
                try:
                    ewma_model = self.models['EWMA']
                    lambda_param = ewma_model['lambda']
                    last_var = ewma_model['last_variance']
                    
                    # Simple EWMA forecast (assumes mean reversion)
                    ewma_vol = [np.sqrt(last_var * np.sqrt(252))] * horizon
                    predictions['EWMA'] = {
                        'volatility_forecast': ewma_vol,
                        'confidence_intervals': self._calculate_confidence_intervals(np.array(ewma_vol), 'EWMA')
                    }
                except Exception as e:
                    self.logger.warning(f"EWMA prediction failed: {e}")
            
            # Historical predictions
            if self.is_fitted.get('Historical', False) and self.models['Historical']:
                try:
                    hist_model = self.models['Historical']
                    # Use average of different window volatilities
                    avg_vol = np.mean(list(hist_model.values()))
                    hist_vol = [avg_vol] * horizon
                    predictions['Historical'] = {
                        'volatility_forecast': hist_vol,
                        'confidence_intervals': self._calculate_confidence_intervals(np.array(hist_vol), 'Historical')
                    }
                except Exception as e:
                    self.logger.warning(f"Historical prediction failed: {e}")
            
            # ML predictions
            if self.is_fitted.get('ML', False) and self.models['ML'] and 'ML' in self.scalers:
                try:
                    # Prepare features for prediction
                    features_df = self.prepare_ml_features(data, returns)
                    if len(features_df) > 0:
                        last_features = features_df.iloc[-1:]
                        scaled_features = self.scalers['ML'].transform(last_features)
                        
                        ml_vol_pred = self.models['ML'].predict(scaled_features)[0]
                        ml_vol = [ml_vol_pred] * horizon
                        predictions['ML'] = {
                            'volatility_forecast': ml_vol,
                            'confidence_intervals': self._calculate_confidence_intervals(np.array(ml_vol), 'ML')
                        }
                except Exception as e:
                    self.logger.warning(f"ML prediction failed: {e}")
            
            # Ensemble prediction (average of all models)
            if predictions:
                ensemble_vol = self._calculate_ensemble_prediction(predictions, horizon)
                predictions['Ensemble'] = {
                    'volatility_forecast': ensemble_vol,
                    'confidence_intervals': self._calculate_confidence_intervals(np.array(ensemble_vol), 'Ensemble')
                }
            
            # Current volatility metrics
            current_vol = returns.tail(22).std() * np.sqrt(252)
            
            # Generate prediction dates
            last_date = data.index[-1]
            prediction_dates = []
            for i in range(1, horizon + 1):
                next_date = last_date + timedelta(days=i)
                while next_date.weekday() >= 5:  # Skip weekends
                    next_date += timedelta(days=1)
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))
            
            return {
                'symbol': symbol,
                'current_volatility': round(current_vol, 4),
                'prediction_horizon_days': horizon,
                'prediction_dates': prediction_dates,
                'model_predictions': predictions,
                'volatility_regime': self._classify_volatility_regime(current_vol),
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Volatility prediction failed: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_intervals(self, volatility: np.ndarray, model_type: str) -> Dict:
        """Calculate confidence intervals for volatility predictions"""
        try:
            confidence_intervals = {}
            
            for conf_level in self.config.confidence_levels:
                # Simple approach: assume normal distribution
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                
                # Estimate uncertainty based on model type
                if model_type == 'GARCH':
                    uncertainty = 0.1  # 10% uncertainty
                elif model_type == 'ML':
                    uncertainty = 0.15  # 15% uncertainty
                else:
                    uncertainty = 0.2   # 20% uncertainty
                
                lower_bound = volatility * (1 - z_score * uncertainty)
                upper_bound = volatility * (1 + z_score * uncertainty)
                
                confidence_intervals[f'{int(conf_level*100)}%'] = {
                    'lower': lower_bound.tolist() if hasattr(lower_bound, 'tolist') else [lower_bound],
                    'upper': upper_bound.tolist() if hasattr(upper_bound, 'tolist') else [upper_bound]
                }
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return {}
    
    def _calculate_ensemble_prediction(self, predictions: Dict, horizon: int) -> List[float]:
        """Calculate ensemble prediction from multiple models"""
        try:
            # Weight models based on their typical performance
            model_weights = {
                'GARCH': 0.3,
                'EWMA': 0.2,
                'Historical': 0.2,
                'ML': 0.3
            }
            
            ensemble_vol = np.zeros(horizon)
            total_weight = 0
            
            for model_name, pred_data in predictions.items():
                if model_name in model_weights and 'volatility_forecast' in pred_data:
                    weight = model_weights[model_name]
                    vol_forecast = np.array(pred_data['volatility_forecast'])
                    
                    if len(vol_forecast) == horizon:
                        ensemble_vol += weight * vol_forecast
                        total_weight += weight
            
            if total_weight > 0:
                ensemble_vol /= total_weight
            
            return ensemble_vol.tolist()
            
        except Exception as e:
            self.logger.warning(f"Ensemble calculation failed: {e}")
            return [0.2] * horizon  # Fallback to 20% volatility
    
    def _classify_volatility_regime(self, current_vol: float) -> str:
        """Classify current volatility regime"""
        if current_vol < 0.15:
            return 'Low Volatility'
        elif current_vol < 0.25:
            return 'Normal Volatility'
        elif current_vol < 0.40:
            return 'High Volatility'
        else:
            return 'Extreme Volatility'
    
    def calculate_var_es(self, returns: pd.Series, confidence_level: float = 0.05) -> Dict:
        """Calculate Value at Risk and Expected Shortfall"""
        try:
            # Sort returns
            sorted_returns = returns.sort_values()
            
            # Calculate VaR
            var_index = int(confidence_level * len(sorted_returns))
            var = sorted_returns.iloc[var_index]
            
            # Calculate Expected Shortfall (Conditional VaR)
            es = sorted_returns.iloc[:var_index].mean()
            
            return {
                'VaR': var,
                'Expected_Shortfall': es,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"VaR/ES calculation failed: {e}")
            return {'error': str(e)}
    
    def save_models(self, symbol: str):
        """Save fitted models"""
        try:
            model_file = os.path.join(self.model_path, f"{symbol}_volatility_models.pkl")
            
            save_data = {
                'models': self.models,
                'scalers': self.scalers,
                'is_fitted': self.is_fitted,
                'config': self.config
            }
            
            joblib.dump(save_data, model_file)
            self.logger.info(f"Volatility models saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models for {symbol}: {e}")
    
    def load_models(self, symbol: str) -> bool:
        """Load fitted models"""
        try:
            model_file = os.path.join(self.model_path, f"{symbol}_volatility_models.pkl")
            
            if os.path.exists(model_file):
                save_data = joblib.load(model_file)
                
                self.models = save_data['models']
                self.scalers = save_data['scalers']
                self.is_fitted = save_data['is_fitted']
                
                self.logger.info(f"Volatility models loaded for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load models for {symbol}: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create volatility predictor
    config = VolatilityConfig(
        lookback_period=252,
        prediction_horizon=22,
        vol_models=['GARCH', 'EWMA', 'Historical', 'ML']
    )
    
    predictor = VolatilityPredictor(config)
    
    # Generate sample data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate stock price with time-varying volatility
    price = 1000
    prices = [price]
    volatility = 0.2
    
    for i in range(len(dates) - 1):
        # Time-varying volatility
        volatility += np.random.normal(0, 0.01)
        volatility = max(0.1, min(0.5, volatility))  # Keep volatility reasonable
        
        # Price change
        change = np.random.normal(0, volatility / np.sqrt(252))
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
    
    # Fit models
    print("\nFitting volatility models...")
    fitting_results = predictor.fit_all_models(sample_data, 'TEST_STOCK')
    print("Fitting results:")
    for model, result in fitting_results.items():
        print(f"{model}: {result}")
    
    # Predict volatility
    print("\nPredicting volatility...")
    vol_predictions = predictor.predict_volatility(sample_data, 'TEST_STOCK')
    print("Volatility predictions:")
    print(vol_predictions)