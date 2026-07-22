"""
Enhanced Regime Router.
The central brain that combines AI predictions, volatility, and fractal analysis 
into an 18-regime classification system, then allocates strategy weights via Kelly Criterion.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from core.hmm_regime_detector import HMMRegimeDetector
from core.hybrid_mtf_predictor import HybridMTFPredictor
from core.regime_classifier import RegimeClassifier
from core.hurst_wavelet_engine import HurstWaveletEngine
from core.strategy_allocator import EnhancedStrategyAllocator
from core.kelly_criterion import KellyCriterionEngine
from core.strategy_performance_tracker import StrategyPerformanceTracker


class EnhancedRegimeRouter:
    def __init__(self, rule_model_path: str, hmm_model_path: str, hybrid_model_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize AI and Math engines
        self.hmm = HMMRegimeDetector(hmm_model_path)
        self.hybrid = HybridMTFPredictor(self.hmm, hybrid_model_path)
        self.rule_classifier = RegimeClassifier()
        self.hurst_engine = HurstWaveletEngine()
        self.allocator = EnhancedStrategyAllocator()
        self.kelly_engine = KellyCriterionEngine()
        self.perf_tracker = StrategyPerformanceTracker()

        # State tracking
        self.current_regime = None
        self.bar_counter = 0
        self.last_regime_change_bar = 0

    def analyze_and_route(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame, 
                          df_h1: pd.DataFrame, strategy_signals: Dict) -> Dict:
        """
        Main entry point for regime analysis. Called on every new M15 bar.
        Returns a comprehensive dictionary of regime context and strategy weights.
        """
        self.bar_counter += 1
        
        # 1. Trend & Direction (Hybrid AI or Rule Fallback)
        if self.hmm.is_trained:
            pred = self.hybrid.predict(df_m5, df_m15, df_h1)
            trend_name = pred.get('prediction', 'SIDEWAY')
            trend_conf = pred.get('confidence', 0.5)
        else:
            rule_res = self.rule_classifier.classify(df_m5, df_m15, df_h1)
            trend_name = rule_res.get('trend', 'SIDEWAY')
            trend_conf = 0.5

        # 2. Volatility Regime
        vol_id, vol_name, vol_percentile = self._get_volatility_regime(df_m5)

        # 3. Fractal / Hurst Exponent
        hurst_val = 0.5
        if df_m5 is not None and len(df_m5) > 100:
            hurst_val = self.hurst_engine.calculate_hurst_exponent(df_m5['close'])
            
        if hurst_val > 0.55:
            fractal_name = 'TRENDING'
        elif hurst_val < 0.45:
            fractal_name = 'MEAN_REVERTING'
        else:
            fractal_name = 'RANDOM_WALK'

        # 4. Combine to 18-Regime Name
        regime_name = self._combine_regimes(trend_name, vol_name, fractal_name)

        # Track regime persistence
        if self.current_regime is None or self.current_regime.get('name') != regime_name:
            self.last_regime_change_bar = self.bar_counter

        # 5. Calculate Kelly Multiplier based on historical regime performance
        base_stats = self.perf_tracker.get_strategy_stats('ALL', regime_name)
        kelly_risk, kelly_frac, kelly_reason = self.kelly_engine.calculate_kelly_risk(
            base_stats['winrate'], base_stats['avg_win'], base_stats['avg_loss'], 1.0, base_stats['trades']
        )
        # Map Kelly fraction to a safe position multiplier (0.5x to 2.0x)
        kelly_mult = max(0.5, min(2.0, kelly_frac * 4.0))

        # Store current regime state
        self.current_regime = {
            'name': regime_name,
            'trend': trend_name,
            'volatility': vol_name,
            'fractal': fractal_name,
            'trend_confidence': trend_conf,
            'vol_percentile': vol_percentile,
            'hurst_value': hurst_val,
            'kelly_multiplier': kelly_mult,
            'kelly_reason': kelly_reason
        }

        # 6. Allocate Strategy Weights
        weights = self.allocator.allocate_weights_18_regime(regime_name, kelly_mult, strategy_signals)

        return {**self.current_regime, 'weights': weights}

    def _get_volatility_regime(self, df_m5: pd.DataFrame) -> Tuple[int, str, float]:
        """
        Layer 2: Classify volatility regime using Bollinger Band Width Percentile.
        [FIX] Added dropna() and NaN checks to prevent crashes on invalid width data.
        """
        if df_m5 is None or len(df_m5) < 100:
            return 1, 'NORMAL_VOL', 0.5
        
        try:
            close = df_m5['close'].to_numpy().astype(float)
            
            # Calculate Bollinger Band Width
            sma_20 = pd.Series(close).rolling(20, min_periods=20).mean()
            std_20 = pd.Series(close).rolling(20, min_periods=20).std()
            
            # BB Width as percentage of price
            bb_width = (4 * std_20 / (sma_20 + 1e-10)) * 100
            
            # Drop NaN values before calculating percentile
            valid_widths = bb_width.dropna()
            
            if len(valid_widths) < 50:
                return 1, 'NORMAL_VOL', 0.5
                
            current_width = valid_widths.iloc[-1]
            
            # Safety check for NaN current width
            if np.isnan(current_width):
                return 1, 'NORMAL_VOL', 0.5
            
            recent_widths = valid_widths.iloc[-100:]
            
            # Calculate percentile rank (0.0 to 1.0)
            percentile = float((recent_widths < current_width).sum() / len(recent_widths))
            
            # Classification
            if percentile < 0.20:
                return 0, 'LOW_VOL', percentile
            elif percentile > 0.80:
                return 2, 'HIGH_VOL', percentile
            else:
                return 1, 'NORMAL_VOL', percentile
                
        except Exception as e:
            self.logger.error(f"[FAIL] Volatility regime error: {e}")
            return 1, 'NORMAL_VOL', 0.5

    def _combine_regimes(self, trend: str, vol: str, fractal: str) -> str:
        """
        Map the 3 core dimensions (Trend, Volatility, Fractal) into one of the 18 named regimes.
        """
        if trend == 'UP':
            if vol == 'HIGH_VOL': 
                return 'PARABOLIC_UP' if fractal == 'TRENDING' else 'FOMO_BUY'
            if vol == 'LOW_VOL': 
                return 'QUIET_RALLY'
            return 'HEALTHY_UPTREND'
            
        elif trend == 'DOWN':
            if vol == 'HIGH_VOL': 
                return 'PANIC_SELL' if fractal == 'TRENDING' else 'WHIPSAW'
            if vol == 'LOW_VOL': 
                return 'SLOW_BLEED'
            return 'HEALTHY_DOWNTREND'
            
        else:  # SIDEWAY
            if vol == 'HIGH_VOL': 
                return 'WIDE_RANGE'
            if vol == 'LOW_VOL': 
                return 'TIGHT_RANGE'
            return 'CLASSIC_RANGE'

    def get_regime_summary(self) -> Dict:
        """
        Return a standardized summary of the current regime for logging and context building.
        [FIX] Uses .get() with defaults to prevent KeyError if dict is partially populated.
        """
        if self.current_regime is None:
            return {
                'regime': 'UNKNOWN', 'regime_name': 'UNKNOWN', 'confidence': '50%',
                'source': 'default', 'kelly_multiplier': 1.0
            }
        
        return {
            'regime': self.current_regime.get('name', 'UNKNOWN'),
            'regime_name': self.current_regime.get('name', 'UNKNOWN'),
            'trend': self.current_regime.get('trend', 'UNKNOWN'),
            'volatility': self.current_regime.get('volatility', 'NORMAL'),
            'fractal': self.current_regime.get('fractal', 'TRENDING'),
            'confidence': f"{self.current_regime.get('trend_confidence', 0.5):.1%}",
            'kelly_multiplier': self.current_regime.get('kelly_multiplier', 1.0),
            'bars_in_regime': self.bar_counter - self.last_regime_change_bar,
            'hurst': self.current_regime.get('hurst_value', 0.5),
            'vol_percentile': self.current_regime.get('vol_percentile', 0.5)
        }