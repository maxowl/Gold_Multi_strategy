"""
Hybrid Multi-Timeframe Predictor.
Combines HMM regime detection with LightGBM classifiers for directional prediction.
"""
import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Optional


class HybridMTFPredictor:
    LABELS = ['DOWN', 'SIDEWAY', 'UP']
    
    def __init__(self, hmm_detector, model_path: str = "lightgbm_regime_models.pkl"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hmm_detector = hmm_detector
        self.model_path = model_path
        self.lightgbm_models = {}
        self.feature_names = None
        self._try_load_models()
    
    def _try_load_models(self):
        """Attempt to load pre-trained LightGBM models."""
        try:
            data = joblib.load(self.model_path)
            if isinstance(data, dict):
                self.lightgbm_models = data.get('models', data)
                self.feature_names = data.get('feature_names')
                self.logger.info(f"[OK] Loaded {len(self.lightgbm_models)} LightGBM models")
            else:
                self.logger.warning("[WARN] Invalid LightGBM model format")
        except FileNotFoundError:
            self.logger.info(f"[INFO] No LightGBM models found at {self.model_path}")
        except Exception as e:
            self.logger.warning(f"[WARN] Failed to load LightGBM models: {e}")
    
    def predict(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None, df_h1: pd.DataFrame = None) -> Dict:
        """
        Predict direction using hybrid HMM + LightGBM ensemble.
        [FIX] Uses .predict() instead of .predict_proba() for native LightGBM Booster.
        """
        # Get HMM regime
        regime_id, regime_conf, regime_details = (2, 0.5, {})
        if self.hmm_detector is not None and self.hmm_detector.is_trained:
            regime_id, regime_conf, regime_details = self.hmm_detector.predict(df_m15 if df_m15 is not None else df_m5)
        
        # Get LightGBM prediction
        lgbm_pred, lgbm_probs = None, None
        if regime_id in self.lightgbm_models:
            try:
                from core.mtf_feature_engineer import MTFFeatureEngineer
                engineer = MTFFeatureEngineer()
                features = engineer.extract_all_features(df_m5, df_m15, df_h1)
                
                if not features.empty:
                    latest = features.iloc[[-1]].copy()
                    
                    # Align columns with training features
                    if self.feature_names:
                        for c in set(self.feature_names) - set(latest.columns):
                            latest[c] = 0
                        latest = latest[[c for c in self.feature_names if c in latest.columns]]
                    
                    # [FIX] Native LightGBM Booster uses .predict() which returns probabilities directly for multiclass
                    # The output shape is (n_samples, n_classes), so we take [0] to get the first (and only) row
                    probs = self.lightgbm_models[regime_id].predict(latest)[0]
                    
                    lgbm_pred = self.LABELS[np.argmax(probs)]
                    lgbm_probs = {
                        'DOWN': float(probs[0]),
                        'SIDEWAY': float(probs[1]),
                        'UP': float(probs[2])
                    }
            except Exception as e:
                self.logger.error(f"[FAIL] LightGBM prediction error: {e}", exc_info=True)
        
        # Meta-ensemble
        result = self._meta_ensemble(regime_id, regime_conf, lgbm_pred, lgbm_probs)
        result['regime_id'] = regime_id
        result['regime_name'] = regime_details.get('regime_name', 'UNKNOWN')
        
        return result
    
    def _meta_ensemble(self, hmm_id: int, hmm_conf: float,
                       lgbm_pred: Optional[str], lgbm_probs: Optional[dict]) -> Dict:
        """
        Combine HMM and LightGBM predictions with confidence thresholds.
        [FIX] Applies minimum confidence threshold and uses predicted class probability.
        """
        MIN_CONFIDENCE = 0.4  # Minimum confidence to trust a prediction
        
        if lgbm_pred is None or lgbm_probs is None:
            # Fallback to HMM only
            if hmm_conf < MIN_CONFIDENCE:
                return {'prediction': 'SIDEWAY', 'confidence': 0.5, 'reason': 'low_hmm_confidence'}
            
            if hmm_id == 0:
                return {'prediction': 'UP', 'confidence': hmm_conf}
            elif hmm_id == 1:
                return {'prediction': 'DOWN', 'confidence': hmm_conf}
            else:
                return {'prediction': 'SIDEWAY', 'confidence': hmm_conf}
        
        # Map HMM to direction
        hmm_direction = {0: 'UP', 1: 'DOWN', 2: 'SIDEWAY'}.get(hmm_id, 'SIDEWAY')
        
        # [FIX] Get probability of the predicted class, not max probability
        lgbm_conf = lgbm_probs.get(lgbm_pred, 0.0)
        
        # Agreement check
        if lgbm_pred == hmm_direction:
            # Both agree - boost confidence
            combined_conf = min(0.95, (hmm_conf + lgbm_conf) / 2 + 0.1)
            return {'prediction': lgbm_pred, 'confidence': combined_conf}
        else:
            # Disagreement - trust higher confidence
            if lgbm_conf > hmm_conf and lgbm_conf >= MIN_CONFIDENCE:
                return {'prediction': lgbm_pred, 'confidence': lgbm_conf * 0.8}
            elif hmm_conf >= MIN_CONFIDENCE:
                return {'prediction': hmm_direction, 'confidence': hmm_conf * 0.8}
            else:
                # Both have low confidence - default to SIDEWAY
                return {'prediction': 'SIDEWAY', 'confidence': 0.5, 'reason': 'both_low_confidence'}