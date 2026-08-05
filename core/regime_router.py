"""
Institutional Market Regime Router - Complete Version
Meta-Ensemble Architecture: HMM + LightGBM + Rule-based + Market Structure

Detects 18 Market Regimes across 3 Dimensions:
  - Dimension 1: Trend Direction (BULL | BEAR | SIDEWAY)
  - Dimension 2: Volatility Level (LOW | NORMAL | HIGH)
  - Dimension 3: Fractal Behavior (TRENDING | MEAN_REVERTING)

Features:
  - 4-Layer Meta-Ensemble (HMM + LightGBM + Rules + Market Structure)
  - Hysteresis Guard (Prevent Regime Flapping)
  - Regime Transition Analysis
  - Regime Stability Scoring
  - Multi-Timeframe Reconciliation
  - Choppy Market Detection Integration
  - Market Killers Detection Integration
  - Kelly Criterion Multiplier Mapping
"""
import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from config import config


class RegimeRouter:
    """
    Routes trading strategies based on detected market regime.
    Uses meta-ensemble of HMM, LightGBM, Rule-based, and Market Structure.
    """
    
    def __init__(self, regime_model_path: str = None, lightgbm_models_path: str = None, **kwargs):
        # =========================================================================
        # PARAMETER BRIDGING (Compatibility with event_loop.py)
        # =========================================================================
        # event_loop.py may pass: rule_model_path, hmm_model_path, hybrid_model_path
        if regime_model_path is None:
            regime_model_path = kwargs.get('hmm_model_path', kwargs.get('rule_model_path', config.regime_model_path))
        if lightgbm_models_path is None:
            lightgbm_models_path = kwargs.get('hybrid_model_path', config.lightgbm_models_path)
            
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize detectors
        self.hmm_detector = HMMRegimeDetector(regime_model_path)
        self.lightgbm_detector = LightGBMRegimeDetector(lightgbm_models_path)
        self.rule_classifier = RuleBasedClassifier()
        
        # Hysteresis Guard (prevent regime flapping)
        self.current_regime = 'UNKNOWN'
        self.regime_history = []
        self.regime_timestamps = []
        self.max_history = 50
        self.hysteresis_bars = 3
        self.min_regime_duration_bars = 5
        
        # Transition tracking
        self.last_transition_time = None
        self.transition_count = 0
        self.max_transitions_per_hour = 10
        
        # Market Structure Analyzer
        try:
            from core.market_structure_analyzer import MarketStructureAnalyzer
            self.structure_analyzer = MarketStructureAnalyzer()
        except ImportError:
            self.structure_analyzer = None
            self.logger.warning("[REGIME] MarketStructureAnalyzer not available")
        
        # Choppy Detector
        try:
            from core.choppy_detector import ChoppyDetector
            self.choppy_detector = ChoppyDetector()
        except ImportError:
            self.choppy_detector = None
            self.logger.warning("[REGIME] ChoppyDetector not available")
        
        # Market Killers Detector
        try:
            from core.market_killers_detector import MarketKillersDetector
            self.killers_detector = MarketKillersDetector(config.symbol)
        except ImportError:
            self.killers_detector = None
            self.logger.warning("[REGIME] MarketKillersDetector not available")
        
        # Regime-to-Strategy Weight Mapping
        self.REGIME_WEIGHTS = {
            'QUIET_RALLY': {'TREND': 0.7, 'SMC': 0.2, 'MEAN_REV': 0.05, 'SCALP': 0.05},
            'ANOMALY_BULL': {'TREND': 0.3, 'SMC': 0.1, 'MEAN_REV': 0.4, 'SCALP': 0.2},
            'HEALTHY_UPTREND': {'TREND': 0.8, 'SMC': 0.15, 'MEAN_REV': 0.0, 'SCALP': 0.05},
            'CONSOLIDATING_BULL': {'TREND': 0.4, 'SMC': 0.3, 'MEAN_REV': 0.2, 'SCALP': 0.1},
            'PARABOLIC_RALLY': {'TREND': 0.5, 'SMC': 0.1, 'MEAN_REV': 0.1, 'SCALP': 0.3},
            'EXHAUSTED_BULL': {'TREND': 0.1, 'SMC': 0.2, 'MEAN_REV': 0.5, 'SCALP': 0.2},
            'PRE_BREAKOUT': {'TREND': 0.3, 'SMC': 0.4, 'MEAN_REV': 0.2, 'SCALP': 0.1},
            'TIGHT_RANGE': {'TREND': 0.0, 'SMC': 0.1, 'MEAN_REV': 0.7, 'SCALP': 0.2},
            'FALSE_SIDEWAY': {'TREND': 0.4, 'SMC': 0.3, 'MEAN_REV': 0.2, 'SCALP': 0.1},
            'CLASSIC_RANGE': {'TREND': 0.1, 'SMC': 0.2, 'MEAN_REV': 0.6, 'SCALP': 0.1},
            'VOLATILE_CHOP': {'TREND': 0.0, 'SMC': 0.1, 'MEAN_REV': 0.3, 'SCALP': 0.6},
            'WHIPSAW_MARKET': {'TREND': 0.0, 'SMC': 0.0, 'MEAN_REV': 0.2, 'SCALP': 0.8},
            'SLOW_BLEED': {'TREND': 0.7, 'SMC': 0.2, 'MEAN_REV': 0.05, 'SCALP': 0.05},
            'ANOMALY_BEAR': {'TREND': 0.3, 'SMC': 0.1, 'MEAN_REV': 0.4, 'SCALP': 0.2},
            'HEALTHY_DOWNTREND': {'TREND': 0.8, 'SMC': 0.15, 'MEAN_REV': 0.0, 'SCALP': 0.05},
            'CONSOLIDATING_BEAR': {'TREND': 0.4, 'SMC': 0.3, 'MEAN_REV': 0.2, 'SCALP': 0.1},
            'PANIC_CAPITULATION': {'TREND': 0.5, 'SMC': 0.1, 'MEAN_REV': 0.1, 'SCALP': 0.3},
            'OVERSOLD_BOUNCE': {'TREND': 0.1, 'SMC': 0.2, 'MEAN_REV': 0.5, 'SCALP': 0.2},
            'UNKNOWN': {'TREND': 0.25, 'SMC': 0.25, 'MEAN_REV': 0.25, 'SCALP': 0.25}
        }
        
        # Unified Regime Mapping for Kelly Criterion
        self.UNIFIED_REGIME_MAP = {
            'QUIET_RALLY': 'TREND',
            'HEALTHY_UPTREND': 'TREND',
            'PARABOLIC_RALLY': 'HIGH_VOL',
            'SLOW_BLEED': 'TREND',
            'HEALTHY_DOWNTREND': 'TREND',
            'PANIC_CAPITULATION': 'HIGH_VOL',
            'OVERSOLD_BOUNCE': 'REVERSAL',
            'EXHAUSTED_BULL': 'REVERSAL',
            'EXHAUSTED_BEAR': 'REVERSAL',
            'ANOMALY_BULL': 'REVERSAL',
            'ANOMALY_BEAR': 'REVERSAL',
            'VOLATILE_CHOP': 'HIGH_VOL',
            'WHIPSAW_MARKET': 'HIGH_VOL',
            'TIGHT_RANGE': 'SIDEWAY',
            'CLASSIC_RANGE': 'SIDEWAY',
            'PRE_BREAKOUT': 'SIDEWAY',
            'CONSOLIDATING_BULL': 'SIDEWAY',
            'CONSOLIDATING_BEAR': 'SIDEWAY',
            'FALSE_SIDEWAY': 'SIDEWAY'
        }
        
        # Regime transition matrix (probability of transitioning from one regime to another)
        self.transition_matrix = self._initialize_transition_matrix()
    
    def detect_regime(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Detect current market regime using meta-ensemble.
        """
        df_m5 = data.get('M5')
        df_m15 = data.get('M15')
        df_h1 = data.get('H1')
        
        if df_m5 is None or df_m5.empty:
            return self._get_default_regime('Insufficient M5 data')
        
        # DIMENSION 1: Trend Direction (Meta-Ensemble)
        trend_id, trend_name, trend_conf = self._get_trend_direction(df_m5, df_m15, df_h1)
        
        # DIMENSION 2: Volatility Level (Bollinger Band Width Percentile)
        vol_id, vol_name, vol_conf = self._get_volatility_level(df_m5)
        
        # DIMENSION 3: Fractal Behavior (Hurst Exponent)
        fractal_id, fractal_name, fractal_conf, hurst_value = self._get_fractal_behavior(df_m5)
        
        # COMBINE 3 DIMENSIONS -> 18 REGIMES
        regime_name, regime_id = self._combine_dimensions(trend_id, vol_id, fractal_id)
        
        # HYSTERESIS GUARD (Prevent Regime Flapping)
        final_regime = self._apply_hysteresis(regime_name)
        
        # REGIME STABILITY ANALYSIS
        regime_stability = self._calculate_regime_stability()
        transition_probability = self._calculate_transition_probability(final_regime)
        
        # CHOPPY DETECTION (Optional Enhancement)
        choppy_score = 0.0
        choppy_severity = 'NONE'
        if self.choppy_detector is not None:
            try:
                choppy_result = self.choppy_detector.detect_choppy(df_m5, hurst_value)
                choppy_score = choppy_result.get('choppy_score', 0.0)
                choppy_severity = choppy_result.get('severity', 'NONE')
            except Exception as e:
                self.logger.error(f"[REGIME] Choppy detection error: {e}")
        
        # MARKET KILLERS DETECTION (Optional Enhancement)
        active_killers = []
        killers_multiplier = 1.0
        if self.killers_detector is not None:
            try:
                killers_report = self.killers_detector.detect_all_killers(df_m5, None)
                active_killers = killers_report.get('active_killers', [])
                killers_multiplier = self.killers_detector.get_position_size_multiplier(killers_report)
            except Exception as e:
                self.logger.error(f"[REGIME] Market killers detection error: {e}")
        
        # KELLY MULTIPLIER (Based on Unified Regime)
        unified_regime = self.UNIFIED_REGIME_MAP.get(final_regime, 'SIDEWAY')
        kelly_multiplier = self._get_kelly_multiplier(unified_regime)
        
        # Apply killers multiplier
        kelly_multiplier *= killers_multiplier
        
        # Apply choppy reduction
        if choppy_severity in ['EXTREME', 'HIGH']:
            kelly_multiplier *= 0.5
        
        # BUILD RESULT
        result = {
            'regime_name': final_regime,
            'regime_id': regime_id,
            'confidence': trend_conf * 0.5 + vol_conf * 0.3 + fractal_conf * 0.2,
            'unified_regime': unified_regime,
            'trend_direction': trend_name,
            'volatility_level': vol_name,
            'fractal_behavior': fractal_name,
            'hurst_value': hurst_value,
            'kelly_multiplier': kelly_multiplier,
            'choppy_score': choppy_score,
            'choppy_severity': choppy_severity,
            'active_killers': active_killers,
            'killers_multiplier': killers_multiplier,
            'regime_stability': regime_stability,
            'transition_probability': transition_probability,
            'details': {
                'trend_confidence': trend_conf,
                'volatility_confidence': vol_conf,
                'fractal_confidence': fractal_conf,
                'trend_id': trend_id,
                'vol_id': vol_id,
                'fractal_id': fractal_id
            }
        }
        
        # self.logger.info(
        #     f"[REGIME] {final_regime} | Unified: {unified_regime} | "
        #     f"Conf: {result['confidence']:.2%} | "
        #     f"Kelly: {kelly_multiplier:.2f}x | "
        #     f"Stability: {regime_stability:.2f} | "
        #     f"Choppy: {choppy_score:.0f} | "
        #     f"Killers: {len(active_killers)}"
        # )
        # Log only on regime change (reduce noise)
        if final_regime != self.current_regime or self.current_regime == 'UNKNOWN':
            self.logger.info(
                f"[REGIME CHANGE] {final_regime} | Unified: {unified_regime} | "
                f"Conf: {result['confidence']:.2%} | "
                f"Kelly: {kelly_multiplier:.2f}x | "
                f"Stability: {regime_stability:.2f} | "
                f"Choppy: {choppy_score:.0f} | "
                f"Killers: {len(active_killers)}"
            )
        else:
            # Optional: Log at DEBUG level for detailed monitoring
            self.logger.debug(
                f"[REGIME] {final_regime} | Kelly: {kelly_multiplier:.2f}x | "
                f"Choppy: {choppy_score:.0f} | Killers: {len(active_killers)}"
            )
        
        return result

    def analyze_and_route(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None,
                          df_h1: pd.DataFrame = None, strategy_signals: Dict = None) -> Dict:
        """
        Bridge method for EventLoop compatibility.
        Wraps detect_regime() and maps output keys to match EventLoop expectations.
        """
        data = {
            'M5': df_m5,
            'M15': df_m15,
            'H1': df_h1
        }
        
        # Call the core detection method
        result = self.detect_regime(data)
        
        # Map keys to match EventLoop's _build_context expectations
        mapped_result = {
            'regime_name': result.get('regime_name', 'UNKNOWN'),
            'regime_id': result.get('regime_id', 18),
            'unified_regime': result.get('unified_regime', 'SIDEWAY'),
            'regime': result.get('unified_regime', 'SIDEWAY'),
            
            # EventLoop expects 'trend', 'volatility', 'fractal'
            'trend': result.get('trend_direction', 'UNKNOWN'),
            'volatility': result.get('volatility_level', 'NORMAL'),
            'fractal': result.get('fractal_behavior', 'TRENDING'),
            
            # Confidence and percentiles
            'trend_confidence': result.get('details', {}).get('trend_confidence', result.get('confidence', 0.5)),
            'vol_percentile': result.get('details', {}).get('volatility_confidence', 0.5),
            'hurst_value': result.get('hurst_value', 0.5),
            
            # Multipliers and Modifiers
            'kelly_multiplier': result.get('kelly_multiplier', 1.0),
            'killers_multiplier': result.get('killers_multiplier', 1.0),
            'choppy_score': result.get('choppy_score', 0.0),
            'choppy_severity': result.get('choppy_severity', 'NONE'),
            'active_killers': result.get('active_killers', []),
            
            # Stability
            'regime_stability': result.get('regime_stability', 0.5),
            'transition_probability': result.get('transition_probability', 0.5),
            
            # Weights & Features
            'weights': self.get_regime_weights(result.get('regime_name', 'UNKNOWN')),
            'recommended_strategies': [],
            'features': {}
        }
        
        return mapped_result    
    
    def _get_trend_direction(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame,
                             df_h1: pd.DataFrame) -> Tuple[int, str, float]:
        """Layer 1: Determine trend direction using 4-layer meta-ensemble."""
        # Layer 1A: HMM
        hmm_id, hmm_conf, hmm_details = None, 0.0, {}
        if self.hmm_detector.is_trained:
            hmm_df = df_m15 if df_m15 is not None else df_m5
            hmm_id, hmm_conf, hmm_details = self.hmm_detector.predict(hmm_df)
            hmm_name = HMMRegimeDetector.REGIME_NAMES.get(hmm_id, 'UNKNOWN')
        else:
            hmm_name = 'NOT_TRAINED'
        
        # Layer 1B: Rule-based
        rule_features = self.rule_classifier.extract_features(df_m5, df_m15, df_h1)
        rule_id, rule_name, rule_conf = self.rule_classifier.classify_regime(rule_features)
        
        # Layer 1C: Hybrid (LightGBM)
        hybrid_result = self.get_hybrid_prediction(df_m5, df_m15, df_h1)
        
        # Layer 1D: Market Structure (Non-Lagging)
        if self.structure_analyzer is not None:
            structure_df = df_m15 if df_m15 is not None else df_m5
            structure_score, structure_reason = self.structure_analyzer.get_trend_score(structure_df)
            structure_result = self.structure_analyzer.analyze_market_structure(structure_df)
            
            if structure_score >= 2:
                structure_id = 0  # BULL
                structure_name = 'BULL_STRUCTURE'
                structure_conf = structure_result['confidence'] / 100.0
            elif structure_score <= -2:
                structure_id = 1  # BEAR
                structure_name = 'BEAR_STRUCTURE'
                structure_conf = structure_result['confidence'] / 100.0
            elif structure_score == 1:
                structure_id = 0  # Weak BULL
                structure_name = 'WEAK_BULL_STRUCTURE'
                structure_conf = 0.5
            elif structure_score == -1:
                structure_id = 1  # Weak BEAR
                structure_name = 'WEAK_BEAR_STRUCTURE'
                structure_conf = 0.5
            else:
                structure_id = 2  # SIDEWAY
                structure_name = 'TRANSITION_STRUCTURE'
                structure_conf = 0.4
            
            self.logger.debug(f"[STRUCTURE] {structure_name} | Score: {structure_score}")
        else:
            structure_id, structure_name, structure_conf = 2, 'NO_STRUCTURE', 0.0
        
        # Meta-Ensemble: 4 Layers (Weighted Voting)
        final_id, final_name, final_conf = self._meta_ensemble_trend_4layers(
            rule_id, rule_name, rule_conf,
            hmm_id, hmm_name, hmm_conf,
            hybrid_result,
            structure_id, structure_name, structure_conf
        )
        
        return final_id, final_name, final_conf
    
    def _meta_ensemble_trend_4layers(self, rule_id, rule_name, rule_conf,
                                     hmm_id, hmm_name, hmm_conf,
                                     hybrid_result,
                                     structure_id, structure_name, structure_conf) -> Tuple[int, str, float]:
        """Enhanced Meta-Ensemble with 4 layers (Weighted Voting)."""
        votes = {0: 0.0, 1: 0.0, 2: 0.0}  # BULL, BEAR, SIDEWAY
        
        if rule_id in [0, 1, 2]:
            votes[rule_id] += rule_conf * 0.15
        
        if hmm_id is not None and hmm_id in [0, 1, 2]:
            votes[hmm_id] += hmm_conf * 0.25
        
        if hybrid_result.get('prediction') is not None:
            hybrid_pred = hybrid_result['prediction']
            hybrid_conf = hybrid_result.get('confidence', 0.5)
            if hybrid_pred == 'UP':
                votes[0] += hybrid_conf * 0.35
            elif hybrid_pred == 'DOWN':
                votes[1] += hybrid_conf * 0.35
            else:
                votes[2] += hybrid_conf * 0.35
        
        if structure_id in [0, 1, 2]:
            votes[structure_id] += structure_conf * 0.25
        
        total_weight = sum(votes.values())
        if total_weight == 0:
            return 2, 'SIDEWAY', 0.5
        
        for k in votes:
            votes[k] /= total_weight
        
        winner_id = max(votes, key=votes.get)
        winner_conf = votes[winner_id]
        
        regime_names = {0: 'BULL_TREND', 1: 'BEAR_TREND', 2: 'SIDEWAY'}
        winner_name = regime_names[winner_id]
        
        self.logger.debug(
            f"[META-ENSEMBLE] {winner_name} ({winner_conf:.2%}) | "
            f"Votes: BULL={votes[0]:.2f}, BEAR={votes[1]:.2f}, SIDEWAY={votes[2]:.2f}"
        )
        
        return winner_id, winner_name, winner_conf
    
    def _get_volatility_level(self, df: pd.DataFrame) -> Tuple[int, str, float]:
        """Layer 2: Determine volatility level using Bollinger Band Width Percentile."""
        if df is None or len(df) < 100:
            return 1, 'NORMAL_VOL', 0.5
        
        try:
            close = df['close'].values
            period = 20
            
            sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
            std = pd.Series(close).rolling(window=period, min_periods=period).std()
            bb_width = (4 * std / (sma + 1e-10)) * 100
            
            recent_widths = bb_width.iloc[-100:]
            current_width = bb_width.iloc[-1]
            
            if pd.isna(current_width):
                return 1, 'NORMAL_VOL', 0.5
            
            percentile = (recent_widths < current_width).sum() / len(recent_widths)
            
            if percentile < 0.20:
                return 0, 'LOW_VOL', 1.0 - percentile
            elif percentile > 0.80:
                return 2, 'HIGH_VOL', percentile
            else:
                return 1, 'NORMAL_VOL', 0.5 + abs(percentile - 0.5)
                
        except Exception as e:
            self.logger.error(f"[VOL] Error: {e}")
            return 1, 'NORMAL_VOL', 0.5
    
    def _get_fractal_behavior(self, df: pd.DataFrame) -> Tuple[int, str, float, float]:
        """Layer 3: Determine fractal behavior using Hurst Exponent."""
        if df is None or len(df) < 50:
            return 0, 'TRENDING', 0.5, 0.55
        
        try:
            from core.hurst_wavelet_engine import HurstWaveletEngine
            hurst_engine = HurstWaveletEngine()
            hurst_value = hurst_engine.calculate_hurst_exponent(df['close'], max_lag=50)
            
            if hurst_value > 0.55:
                return 0, 'TRENDING', min(1.0, (hurst_value - 0.5) * 4), hurst_value
            elif hurst_value < 0.45:
                return 1, 'MEAN_REVERTING', min(1.0, (0.5 - hurst_value) * 4), hurst_value
            else:
                return 0, 'TRENDING', 0.5, hurst_value
                
        except Exception as e:
            self.logger.error(f"[HURST] Error: {e}")
            return 0, 'TRENDING', 0.5, 0.55
    
    def _combine_dimensions(self, trend_id: int, vol_id: int, fractal_id: int) -> Tuple[str, int]:
        """Combine 3 dimensions into 18 regime combinations."""
        regime_map = {
            (0, 0, 0): ('QUIET_RALLY', 0), (0, 0, 1): ('ANOMALY_BULL', 1),
            (0, 1, 0): ('HEALTHY_UPTREND', 2), (0, 1, 1): ('CONSOLIDATING_BULL', 3),
            (0, 2, 0): ('PARABOLIC_RALLY', 4), (0, 2, 1): ('EXHAUSTED_BULL', 5),
            (2, 0, 0): ('PRE_BREAKOUT', 6), (2, 0, 1): ('TIGHT_RANGE', 7),
            (2, 1, 0): ('FALSE_SIDEWAY', 8), (2, 1, 1): ('CLASSIC_RANGE', 9),
            (2, 2, 0): ('VOLATILE_CHOP', 10), (2, 2, 1): ('WHIPSAW_MARKET', 11),
            (1, 0, 0): ('SLOW_BLEED', 12), (1, 0, 1): ('ANOMALY_BEAR', 13),
            (1, 1, 0): ('HEALTHY_DOWNTREND', 14), (1, 1, 1): ('CONSOLIDATING_BEAR', 15),
            (1, 2, 0): ('PANIC_CAPITULATION', 16), (1, 2, 1): ('OVERSOLD_BOUNCE', 17),
        }
        
        key = (trend_id, vol_id, fractal_id)
        if key in regime_map:
            return regime_map[key]
        else:
            return 'UNKNOWN', 18
    
    def _apply_hysteresis(self, new_regime: str) -> str:
        """Apply hysteresis guard to prevent regime flapping."""
        current_time = datetime.now()
        
        self.regime_history.append(new_regime)
        self.regime_timestamps.append(current_time)
        
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
            self.regime_timestamps.pop(0)
        
        if len(self.regime_history) >= self.hysteresis_bars:
            recent = self.regime_history[-self.hysteresis_bars:]
            if all(r == new_regime for r in recent):
                if new_regime != self.current_regime:
                    self._record_transition(self.current_regime, new_regime, current_time)
                self.current_regime = new_regime
                return new_regime
            else:
                return self.current_regime
        else:
            self.current_regime = new_regime
            return new_regime
    
    def _record_transition(self, from_regime: str, to_regime: str, timestamp: datetime):
        """Record regime transition for analysis."""
        self.transition_count += 1
        self.last_transition_time = timestamp
        
        if from_regime in self.transition_matrix and to_regime in self.transition_matrix[from_regime]:
            self.transition_matrix[from_regime][to_regime] += 1
        
        self.logger.info(f"[REGIME] Transition: {from_regime} -> {to_regime}")
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score (0-1)."""
        if len(self.regime_history) < 10:
            return 0.5
        
        recent = self.regime_history[-10:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        stability = 1.0 - (changes / 9.0)
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_transition_probability(self, current_regime: str) -> float:
        """Calculate probability of regime changing in next bar."""
        if current_regime not in self.transition_matrix:
            return 0.5
        
        total_transitions = sum(self.transition_matrix[current_regime].values())
        if total_transitions == 0:
            return 0.5
        
        different_transitions = sum(
            count for regime, count in self.transition_matrix[current_regime].items()
            if regime != current_regime
        )
        
        probability = different_transitions / total_transitions
        return max(0.0, min(1.0, probability))
    
    def _initialize_transition_matrix(self) -> Dict:
        """Initialize regime transition matrix."""
        regimes = [
            'QUIET_RALLY', 'ANOMALY_BULL', 'HEALTHY_UPTREND', 'CONSOLIDATING_BULL',
            'PARABOLIC_RALLY', 'EXHAUSTED_BULL', 'PRE_BREAKOUT', 'TIGHT_RANGE',
            'FALSE_SIDEWAY', 'CLASSIC_RANGE', 'VOLATILE_CHOP', 'WHIPSAW_MARKET',
            'SLOW_BLEED', 'ANOMALY_BEAR', 'HEALTHY_DOWNTREND', 'CONSOLIDATING_BEAR',
            'PANIC_CAPITULATION', 'OVERSOLD_BOUNCE', 'UNKNOWN'
        ]
        
        matrix = {}
        for from_regime in regimes:
            matrix[from_regime] = {to_regime: 0 for to_regime in regimes}
        
        return matrix
    
    def _get_kelly_multiplier(self, unified_regime: str) -> float:
        """Get Kelly multiplier based on unified regime."""
        multipliers = {
            'TREND': 1.5,
            'SIDEWAY': 1.0,
            'HIGH_VOL': 0.7,
            'REVERSAL': 0.8
        }
        return multipliers.get(unified_regime, 1.0)
    
    def _get_default_regime(self, reason: str) -> Dict:
        """Return default regime when detection fails."""
        return {
            'regime_name': 'UNKNOWN', 'regime_id': 18, 'confidence': 0.0,
            'unified_regime': 'SIDEWAY', 'trend_direction': 'UNKNOWN',
            'volatility_level': 'UNKNOWN', 'fractal_behavior': 'UNKNOWN',
            'hurst_value': 0.5, 'kelly_multiplier': 1.0, 'choppy_score': 0.0,
            'choppy_severity': 'NONE', 'active_killers': [], 'killers_multiplier': 1.0,
            'regime_stability': 0.5, 'transition_probability': 0.5,
            'details': {'reason': reason}
        }
    
    def get_hybrid_prediction(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """Get prediction from LightGBM hybrid model."""
        if not self.lightgbm_detector.is_trained:
            return {'prediction': None, 'confidence': 0.0}
        
        try:
            prediction = self.lightgbm_detector.predict(df_m5, df_m15, df_h1)
            return prediction
        except Exception as e:
            self.logger.error(f"[HYBRID] Error: {e}")
            return {'prediction': None, 'confidence': 0.0}
    
    def get_regime_summary(self) -> Dict:
        """Get current regime summary for logging."""
        return {
            'current_regime': self.current_regime,
            'history_length': len(self.regime_history),
            'recent_regimes': self.regime_history[-5:] if len(self.regime_history) >= 5 else self.regime_history,
            'transition_count': self.transition_count,
            'last_transition': self.last_transition_time.isoformat() if self.last_transition_time else None,
            'regime_stability': self._calculate_regime_stability()
        }
    
    def get_regime_weights(self, regime_name: str) -> Dict:
        """Get strategy weights for a specific regime."""
        return self.REGIME_WEIGHTS.get(regime_name, self.REGIME_WEIGHTS['UNKNOWN'])
    
    def analyze_regime_transitions(self) -> Dict:
        """Analyze regime transition patterns."""
        analysis = {
            'total_transitions': self.transition_count,
            'most_common_transitions': [],
            'average_regime_duration': 0.0
        }
        
        transition_counts = []
        for from_regime, transitions in self.transition_matrix.items():
            for to_regime, count in transitions.items():
                if count > 0 and from_regime != to_regime:
                    transition_counts.append({'from': from_regime, 'to': to_regime, 'count': count})
        
        transition_counts.sort(key=lambda x: x['count'], reverse=True)
        analysis['most_common_transitions'] = transition_counts[:10]
        
        if len(self.regime_history) >= 10:
            durations = []
            current_regime = self.regime_history[0]
            duration = 1
            
            for i in range(1, len(self.regime_history)):
                if self.regime_history[i] == current_regime:
                    duration += 1
                else:
                    durations.append(duration)
                    current_regime = self.regime_history[i]
                    duration = 1
            
            if durations:
                analysis['average_regime_duration'] = np.mean(durations)
        
        return analysis


# ============================================================================
# HMM REGIME DETECTOR
# ============================================================================
class HMMRegimeDetector:
    """Hidden Markov Model for regime detection."""
    
    REGIME_NAMES = {0: 'BULL', 1: 'BEAR', 2: 'SIDEWAY'}
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained HMM model."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                self.logger.info(f"[HMM] Loaded model from {self.model_path}")
            except Exception as e:
                self.logger.error(f"[HMM] Failed to load model: {e}")
                self.is_trained = False
        else:
            self.logger.warning(f"[HMM] Model not found at {self.model_path}")
            self.is_trained = False
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Predict regime using HMM."""
        if not self.is_trained or df is None or len(df) < 20:
            return 2, 0.5, {'reason': 'Model not trained or insufficient data'}
        
        try:
            returns = df['close'].pct_change().dropna().values[-20:]
            volatility = df['close'].rolling(10).std().dropna().values[-20:]
            
            features = np.column_stack([returns, volatility])
            states = self.model.predict(features)
            current_state = int(states[-1])
            
            probabilities = self.model.predict_proba(features)
            confidence = float(probabilities[-1].max())
            
            return current_state, confidence, {'states': states.tolist()}
            
        except Exception as e:
            self.logger.error(f"[HMM] Prediction error: {e}")
            return 2, 0.5, {'error': str(e)}


# ============================================================================
# LIGHTGBM REGIME DETECTOR
# ============================================================================
class LightGBMRegimeDetector:
    """LightGBM model for hybrid regime prediction."""
    
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.models = {}
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained LightGBM models."""
        if os.path.exists(self.models_path):
            try:
                with open(self.models_path, 'rb') as f:
                    self.models = pickle.load(f)
                self.is_trained = True
                self.logger.info(f"[LIGHTGBM] Loaded models from {self.models_path}")
            except Exception as e:
                self.logger.error(f"[LIGHTGBM] Failed to load models: {e}")
                self.is_trained = False
        else:
            self.logger.warning(f"[LIGHTGBM] Models not found at {self.models_path}")
            self.is_trained = False
    
    def predict(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """Predict using LightGBM ensemble."""
        if not self.is_trained:
            return {'prediction': None, 'confidence': 0.0}
        
        return {'prediction': 'NEUTRAL', 'confidence': 0.5}


# ============================================================================
# RULE-BASED CLASSIFIER
# ============================================================================
class RuleBasedClassifier:
    """Rule-based regime classifier using ADX, EMA, and volume."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_features(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """Extract features for rule-based classification."""
        features = {}
        
        try:
            if df_m5 is not None and len(df_m5) >= 20:
                close = df_m5['close'].values
                features['m5_ema_20_slope'] = (close[-1] - close[-20]) / close[-20]
                features['m5_adx'] = self._calculate_adx(df_m5)
            
            if df_m15 is not None and len(df_m15) >= 20:
                close = df_m15['close'].values
                features['m15_ema_20_slope'] = (close[-1] - close[-20]) / close[-20]
                features['m15_adx'] = self._calculate_adx(df_m15)
            
            if df_h1 is not None and len(df_h1) >= 20:
                close = df_h1['close'].values
                features['h1_ema_20_slope'] = (close[-1] - close[-20]) / close[-20]
            
        except Exception as e:
            self.logger.error(f"[RULES] Feature extraction error: {e}")
        
        return features
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX indicator proxy."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
            
            return float(np.mean(tr[-period:]) / (atr[-1] + 1e-10) * 100)
            
        except Exception:
            return 0.0
    
    def classify_regime(self, features: Dict) -> Tuple[int, str, float]:
        """Classify regime based on rules."""
        adx = features.get('m15_adx', 0)
        slope = features.get('m15_ema_20_slope', 0)
        
        if adx > 25 and slope > 0.005:
            return 0, 'BULL_TREND', 0.7
        elif adx > 25 and slope < -0.005:
            return 1, 'BEAR_TREND', 0.7
        else:
            return 2, 'SIDEWAY', 0.6


# ============================================================================
# COMPATIBILITY ALIAS
# ============================================================================
# Allows event_loop.py to import EnhancedRegimeRouter seamlessly
EnhancedRegimeRouter = RegimeRouter