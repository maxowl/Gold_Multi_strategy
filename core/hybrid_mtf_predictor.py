"""
Hybrid Multi-Timeframe Predictor - LightGBM Based.

Uses LightGBM models trained on MTF features to predict market regime.

Features:
  - Multi-timeframe feature extraction
  - LightGBM ensemble prediction
  - Regime confidence scoring
  - Fallback to rule-based classification

Used by:
  - RegimeRouter (ensemble with HMM and Rule-based)
"""
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, List, Optional, Tuple

from config import config
from core.mtf_feature_engineer import MTFFeatureEngineer


class HybridMTFPredictor:
    """
    LightGBM-based multi-timeframe regime predictor.
    
    Features:
      - Multi-timeframe feature extraction
      - LightGBM ensemble (one model per regime)
      - Confidence scoring
      - Graceful degradation to rule-based
    """

    def __init__(self, models_path: str = None):
        """
        Initialize HybridMTFPredictor.
        
        Args:
            models_path: Path to LightGBM models (defaults to config.lightgbm_models_path)
        """
        if models_path is None:
            models_path = config.lightgbm_models_path
            
        self.models_path = models_path
        self.logger = logging.getLogger(self.__class__.__name__)

        # Feature engineer
        self.feature_engineer = MTFFeatureEngineer()

        # Models storage
        self.models = {}
        self.is_loaded = False

        # Regime mapping
        self.regime_names = {
            0: 'HEALTHY_UPTREND',
            1: 'HEALTHY_DOWNTREND',
            2: 'QUIET_RALLY',
            3: 'SLOW_BLEED',
            4: 'CLASSIC_RANGE',
            5: 'TIGHT_RANGE',
            6: 'PRE_BREAKOUT',
            7: 'CONSOLIDATING_BULL',
            8: 'CONSOLIDATING_BEAR',
            9: 'PARABOLIC_RALLY',
            10: 'PANIC_CAPITULATION',
            11: 'VOLATILE_CHOP',
            12: 'WHIPSAW_MARKET',
            13: 'OVERSOLD_BOUNCE',
            14: 'EXHAUSTED_BULL',
            15: 'EXHAUSTED_BEAR',
            16: 'ANOMALY_BULL',
            17: 'ANOMALY_BEAR'
        }

        self._try_load_models()

    def _try_load_models(self):
        """Attempt to load pre-trained LightGBM models."""
        try:
            import lightgbm as lgb

            data = joblib.load(self.models_path)

            if isinstance(data, dict):
                self.models = data.get('models', {})
                self.is_loaded = len(self.models) > 0
                if self.is_loaded:
                    self.logger.info(
                        f"[LGBM] Loaded {len(self.models)} LightGBM models from {self.models_path}"
                    )
            else:
                self.logger.warning(f"[LGBM] Model file exists but is invalid")

        except FileNotFoundError:
            self.logger.info(f"[LGBM] No pre-trained LightGBM models found at {self.models_path}")
        except ImportError:
            self.logger.warning("[LGBM] lightgbm not installed, predictor disabled")
        except Exception as e:
            self.logger.warning(f"[LGBM] Failed to load models: {e}")

    def predict(
        self,
        df_m15: pd.DataFrame,
        df_m5: pd.DataFrame = None,
        df_h1: pd.DataFrame = None
    ) -> Dict:
        """
        Predict current regime using LightGBM ensemble.
        
        Args:
            df_m15: M15 DataFrame
            df_m5: M5 DataFrame (optional)
            df_h1: H1 DataFrame (optional)
            
        Returns:
            Dict with regime, confidence, and method
        """
        if not self.is_loaded:
            return self._fallback_prediction('models_not_loaded')

        # Extract features
        features_df = self.feature_engineer.extract_all_features(df_m15, df_m5, df_h1)

        if features_df.empty:
            return self._fallback_prediction('feature_extraction_failed')

        # Get latest features
        try:
            latest_features = features_df.iloc[[-1]].drop(columns=['time'], errors='ignore')
        except Exception as e:
            self.logger.error(f"[LGBM] Feature selection error: {e}")
            return self._fallback_prediction('feature_selection_failed')

        # Predict with each model
        predictions = {}
        confidences = {}

        try:
            import lightgbm as lgb

            for regime_id, model in self.models.items():
                try:
                    # Predict probability
                    proba = model.predict_proba(latest_features)[0]

                    # For binary classifier, probability is [not_regime, is_regime]
                    if len(proba) == 2:
                        predictions[regime_id] = proba[1]
                    else:
                        predictions[regime_id] = proba[0]

                except Exception as e:
                    self.logger.debug(f"[LGBM] Prediction error for regime {regime_id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"[LGBM] Prediction error: {e}")
            return self._fallback_prediction('prediction_failed')

        if not predictions:
            return self._fallback_prediction('no_valid_predictions')

        # Get regime with highest probability
        best_regime_id = max(predictions.keys(), key=lambda k: predictions[k])
        best_confidence = predictions[best_regime_id]

        # Normalize confidence to 0-1
        confidence = min(1.0, max(0.0, best_confidence))

        regime_name = self.regime_names.get(best_regime_id, 'UNKNOWN')

        return {
            'regime': regime_name,
            'regime_id': best_regime_id,
            'confidence': confidence,
            'method': 'LIGHTGBM',
            'all_predictions': {
                self.regime_names.get(k, 'UNKNOWN'): round(v, 3)
                for k, v in predictions.items()
            }
        }

    def _fallback_prediction(self, reason: str) -> Dict:
        """
        Fallback to rule-based prediction when LightGBM fails.
        
        Args:
            reason: Reason for fallback
            
        Returns:
            Dict with regime info
        """
        try:
            from core.regime_classifier import RegimeClassifier
            classifier = RegimeClassifier()
            result = classifier.classify(df_m5=None)

            return {
                'regime': result.get('regime_name', 'UNKNOWN'),
                'confidence': 0.5,
                'method': 'RULE_BASED_FALLBACK',
                'reason': reason
            }
        except Exception as e:
            self.logger.error(f"[LGBM] Fallback failed: {e}")
            return {
                'regime': 'UNKNOWN',
                'confidence': 0.0,
                'method': 'NONE',
                'reason': f'All methods failed: {reason}'
            }

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dict with model information
        """
        if not self.is_loaded:
            return {
                'loaded': False,
                'model_count': 0,
                'regimes': []
            }

        return {
            'loaded': True,
            'model_count': len(self.models),
            'regimes': [self.regime_names.get(k, 'UNKNOWN') for k in self.models.keys()],
            'models_path': self.models_path
        }