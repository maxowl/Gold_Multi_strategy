"""
Hidden Markov Model Regime Detector.
Detects market regimes (BULL, BEAR, SIDEWAY) using GaussianHMM.
[FINAL FIX] Uses state mapping instead of model reordering for hmmlearn 0.3.0+ compatibility
"""
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Tuple, Dict
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    REGIME_NAMES = {0: 'BULL_TREND', 1: 'BEAR_TREND', 2: 'SIDEWAY'}
    
    def __init__(self, model_path: str = "hmm_regime_model.pkl", n_states: int = 3):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.n_states = n_states
        self.model = None
        self.feature_scaler = None
        self.state_mapping = None  # Maps internal state ID to regime name
        self.is_trained = False
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load pre-trained HMM model with backward compatibility."""
        try:
            data = joblib.load(self.model_path)
            
            if isinstance(data, dict):
                self.model = data.get('model')
                self.feature_scaler = data.get('scaler')
                self.state_mapping = data.get('state_mapping', {0: 0, 1: 1, 2: 2})
            else:
                # Legacy format
                self.model = data
                self.feature_scaler = None
                self.state_mapping = {0: 0, 1: 1, 2: 2}
            
            if self.model is not None:
                self.is_trained = True
                self.logger.info(f"[OK] Loaded HMM model from {self.model_path}")
            else:
                self.logger.warning(f"[WARN] Model file exists but is invalid")
                
        except FileNotFoundError:
            self.logger.info(f"[INFO] No pre-trained HMM model found at {self.model_path}")
        except Exception as e:
            self.logger.warning(f"[WARN] Failed to load HMM model: {e}")
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM: returns, volatility, hurst, volume z-score, ATR percentile."""
        if df is None or len(df) < 100:
            return np.array([])
        
        try:
            close = df['close'].to_numpy().astype(float)
            
            # Feature 1: Log returns
            returns = np.diff(np.log(close + 1e-10))
            returns = np.insert(returns, 0, 0.0)
            
            # Feature 2: Rolling volatility (20-bar)
            vol = pd.Series(returns).rolling(20).std().to_numpy()
            vol = np.nan_to_num(vol, nan=0.0)
            
            # Feature 3: Hurst exponent (optimized)
            hurst = np.full(len(close), 0.5)
            window_size = 50
            lags = np.array([2, 4, 6, 8])
            
            for i in range(window_size, len(close)):
                window = close[i-window_size:i]
                try:
                    tau = []
                    for lag in lags:
                        if lag < len(window):
                            diffs = window[lag:] - window[:-lag]
                            std = np.std(diffs)
                            if std > 0:
                                tau.append((lag, std))
                    
                    if len(tau) >= 2:
                        log_lags = np.log([t[0] for t in tau])
                        log_tau = np.log([t[1] for t in tau])
                        slope, _ = np.polyfit(log_lags, log_tau, 1)
                        hurst[i] = max(0.0, min(1.0, slope))
                except Exception:
                    hurst[i] = 0.5
            
            # Feature 4: Volume z-score
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            else:
                volume = np.ones(len(close))
            vol_mean = pd.Series(volume).rolling(20).mean().to_numpy()
            vol_std = pd.Series(volume).rolling(20).std().to_numpy()
            volume_z = (volume - vol_mean) / (vol_std + 1e-10)
            volume_z = np.nan_to_num(volume_z, nan=0.0)
            
            # Feature 5: ATR percentile
            from core.atr_cache import ATRCache
            atr = ATRCache.get_atr(df, 14).to_numpy()
            atr_percentile = pd.Series(atr).rolling(100).rank(pct=True).to_numpy()
            atr_percentile = np.nan_to_num(atr_percentile, nan=0.5)
            
            # Stack features
            features = np.column_stack([returns, vol, hurst, volume_z, atr_percentile])
            
            # Remove rows with NaN
            features = features[~np.isnan(features).any(axis=1)]
            
            return features
            
        except Exception as e:
            self.logger.error(f"[FAIL] Feature extraction error: {e}")
            return np.array([])
    
    def train(self, df: pd.DataFrame, n_iter: int = 100):
        """
        Train HMM model on historical data.
        [FINAL FIX] Uses state mapping instead of reordering model internals.
        This works with all hmmlearn versions without validation errors.
        """
        features = self.extract_features(df)
        if len(features) < 100:
            self.logger.error("[FAIL] Insufficient data for training")
            return
        
        try:
            # Scale features
            self.feature_scaler = StandardScaler()
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Train GaussianHMM (no state reordering in model itself)
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type='diag',
                n_iter=n_iter,
                random_state=42
            )
            self.model.fit(features_scaled)
            
            # Predict states to determine their characteristics
            states = self.model.predict(features_scaled)
            
            # Calculate mean returns for each state
            mean_returns = []
            for i in range(self.n_states):
                mask = states == i
                if mask.sum() > 0:
                    mean_returns.append(features_scaled[mask, 0].mean())
                else:
                    mean_returns.append(0.0)
            
            # Create state mapping: internal_state_id -> regime_id
            # Sort by mean return: lowest = BEAR (1), middle = SIDEWAY (2), highest = BULL (0)
            sorted_indices = np.argsort(mean_returns, kind='stable')
            
            # Mapping: internal_state -> regime_id
            # sorted_indices[0] has lowest mean return -> BEAR_TREND (1)
            # sorted_indices[1] has middle mean return -> SIDEWAY (2)
            # sorted_indices[2] has highest mean return -> BULL_TREND (0)
            self.state_mapping = {
                int(sorted_indices[0]): 1,  # BEAR
                int(sorted_indices[1]): 2,  # SIDEWAY
                int(sorted_indices[2]): 0   # BULL
            }
            
            # Save model with state mapping
            joblib.dump({
                'model': self.model,
                'scaler': self.feature_scaler,
                'state_mapping': self.state_mapping
            }, self.model_path)
            
            self.is_trained = True
            self.logger.info(f"[OK] HMM model trained and saved to {self.model_path}")
            
            # Log state mapping for verification
            self.logger.info(f"[INFO] State mapping (internal_state -> regime):")
            for internal_state, regime_id in self.state_mapping.items():
                regime_name = self.REGIME_NAMES.get(regime_id, 'UNKNOWN')
                mean_ret = mean_returns[internal_state]
                self.logger.info(f"  Internal State {internal_state} -> {regime_name} (mean_return={mean_ret:.4f})")
            
        except Exception as e:
            self.logger.error(f"[FAIL] HMM training error: {e}", exc_info=True)
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, Dict]:
        """
        Predict current market regime.
        Uses state mapping to translate internal state IDs to regime IDs.
        """
        if not self.is_trained:
            return 2, 0.0, {'reason': 'not_trained'}
        
        features = self.extract_features(df)
        if len(features) == 0:
            return 2, 0.0, {'reason': 'feature_failed'}
        
        # Handle legacy models without scaler
        if self.feature_scaler is not None:
            features_scaled = self.feature_scaler.transform(features)
        else:
            features_scaled = features
        
        try:
            recent = features_scaled[-50:]
            internal_states = self.model.predict(recent)
            
            # Get the last internal state
            last_internal_state = int(internal_states[-1])
            
            # Map internal state to regime ID using our mapping
            if self.state_mapping is None:
                self.state_mapping = {0: 0, 1: 1, 2: 2}  # Default identity mapping
            
            current_regime_id = self.state_mapping.get(last_internal_state, 2)
            
            # Calculate confidence based on recent state consistency
            mapped_states = [self.state_mapping.get(int(s), 2) for s in internal_states[-20:]]
            counts = np.bincount(mapped_states, minlength=self.n_states)
            confidence = float(counts[current_regime_id] / 20.0)
            
            regime_name = self.REGIME_NAMES.get(current_regime_id, 'UNKNOWN')
            
            return current_regime_id, confidence, {'regime_name': regime_name}
            
        except Exception as e:
            self.logger.error(f"[FAIL] HMM prediction error: {e}")
            return 2, 0.0, {'reason': 'predict_error'}