"""
Regime Router - 18-Regime Detection & Routing System.

Detects current market regime using multiple models and routes signals
to appropriate strategies based on regime compatibility.

18 Market Regimes:
  TREND (5):
    1. HEALTHY_UPTREND: Strong bullish trend with healthy volume
    2. HEALTHY_DOWNTREND: Strong bearish trend with healthy volume
    3. QUIET_RALLY: Slow, steady uptrend with low volatility
    4. SLOW_BLEED: Slow, steady downtrend with low volatility
    5. FALSE_SIDEWAY: Range that's actually trending slowly

  HIGH_VOL (4):
    6. PARABOLIC_RALLY: Extreme bullish momentum, overextended
    7. PANIC_CAPITULATION: Extreme bearish momentum, panic selling
    8. VOLATILE_CHOP: High volatility with no clear direction
    9. WHIPSAW_MARKET: Rapid direction changes, trap-heavy

  SIDEWAY (5):
    10. TIGHT_RANGE: Very narrow range, consolidation
    11. CLASSIC_RANGE: Normal range-bound market
    12. PRE_BREAKOUT: Range tightening before breakout
    13. CONSOLIDATING_BULL: Bullish consolidation (flag/pennant)
    14. CONSOLIDATING_BEAR: Bearish consolidation (flag/pennant)

  REVERSAL (4):
    15. OVERSOLD_BOUNCE: Oversold condition, bounce expected
    16. EXHAUSTED_BULL: Bull trend exhausted, reversal likely
    17. EXHAUSTED_BEAR: Bear trend exhausted, reversal likely
    18. ANOMALY: Abnormal price action, unpredictable

Unified Regimes (for Kelly sizing):
  - TREND: For trend-following strategies
  - SIDEWAY: For mean-reversion strategies
  - HIGH_VOL: For scalping strategies
  - REVERSAL: For counter-trend strategies
"""
import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

from config import config


class RegimeRouter:
    """
    Detects current market regime and provides routing context.
    
    Features:
      - Multi-model ensemble (HMM, Rule-based, LightGBM)
      - 18-regime classification
      - Choppy score detection (0-100)
      - Market killers detection
      - Regime stability tracking
      - Kelly multiplier calculation
      - Comprehensive regime context
    """

    # 18 Regime definitions
    REGIMES = {
        # TREND regimes
        'HEALTHY_UPTREND': {'category': 'TREND', 'strength': 0.8},
        'HEALTHY_DOWNTREND': {'category': 'TREND', 'strength': 0.8},
        'QUIET_RALLY': {'category': 'TREND', 'strength': 0.6},
        'SLOW_BLEED': {'category': 'TREND', 'strength': 0.6},
        'FALSE_SIDEWAY': {'category': 'TREND', 'strength': 0.4},

        # HIGH_VOL regimes
        'PARABOLIC_RALLY': {'category': 'HIGH_VOL', 'strength': 0.9},
        'PANIC_CAPITULATION': {'category': 'HIGH_VOL', 'strength': 0.9},
        'VOLATILE_CHOP': {'category': 'HIGH_VOL', 'strength': 0.3},
        'WHIPSAW_MARKET': {'category': 'HIGH_VOL', 'strength': 0.2},

        # SIDEWAY regimes
        'TIGHT_RANGE': {'category': 'SIDEWAY', 'strength': 0.7},
        'CLASSIC_RANGE': {'category': 'SIDEWAY', 'strength': 0.6},
        'PRE_BREAKOUT': {'category': 'SIDEWAY', 'strength': 0.5},
        'CONSOLIDATING_BULL': {'category': 'SIDEWAY', 'strength': 0.6},
        'CONSOLIDATING_BEAR': {'category': 'SIDEWAY', 'strength': 0.6},

        # REVERSAL regimes
        'OVERSOLD_BOUNCE': {'category': 'REVERSAL', 'strength': 0.7},
        'EXHAUSTED_BULL': {'category': 'REVERSAL', 'strength': 0.6},
        'EXHAUSTED_BEAR': {'category': 'REVERSAL', 'strength': 0.6},
        'ANOMALY': {'category': 'REVERSAL', 'strength': 0.3}
    }

    def __init__(self):
        """Initialize RegimeRouter with all sub-engines."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load regime detection models
        self.hmm_detector = None
        self.rule_classifier = None
        self.lgbm_predictor = None

        self._load_models()

        # Regime history for stability tracking (last 20 detections)
        self._regime_history = deque(maxlen=20)
        self._last_regime = 'UNKNOWN'
        self._last_regime_time = None

        # Choppy detector
        try:
            from core.choppy_detector import ChoppyDetector
            self.choppy_detector = ChoppyDetector()
        except ImportError:
            self.choppy_detector = None

        # Market killers detector
        try:
            from core.market_killers_detector import MarketKillersDetector
            self.killers_detector = MarketKillersDetector()
        except ImportError:
            self.killers_detector = None

        # Session volatility manager
        try:
            from core.session_volatility import SessionVolatilityManager
            self.session_mgr = SessionVolatilityManager()
        except ImportError:
            self.session_mgr = None

        self.logger.info("[REGIME_ROUTER] Initialized with 18-regime detection")

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_models(self):
        """Load regime detection models."""
        # HMM Detector
        try:
            from core.hmm_regime_detector import HMMRegimeDetector
            self.hmm_detector = HMMRegimeDetector()
            self.logger.info("[REGIME_ROUTER] HMM detector loaded")
        except Exception as e:
            self.logger.warning(f"[REGIME_ROUTER] HMM detector not available: {e}")

        # Rule-based Classifier
        try:
            from core.regime_classifier import RegimeClassifier
            self.rule_classifier = RegimeClassifier()
            self.logger.info("[REGIME_ROUTER] Rule classifier loaded")
        except Exception as e:
            self.logger.warning(f"[REGIME_ROUTER] Rule classifier not available: {e}")

        # LightGBM Predictor
        try:
            from core.hybrid_mtf_predictor import HybridMTFPredictor
            self.lgbm_predictor = HybridMTFPredictor()
            self.logger.info("[REGIME_ROUTER] LightGBM predictor loaded")
        except Exception as e:
            self.logger.warning(f"[REGIME_ROUTER] LightGBM predictor not available: {e}")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def detect_regime(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Detect current market regime from multi-timeframe data.
        
        Args:
            data: Dict of timeframe -> DataFrame (M1, M5, M15, H1, etc.)
            
        Returns:
            Comprehensive regime context dict with:
              - regime_name: One of 18 regimes
              - regime_category: TREND, SIDEWAY, HIGH_VOL, REVERSAL
              - confidence: Detection confidence (0-1)
              - choppy_score: Choppy market score (0-100)
              - active_killers: List of market killer conditions
              - kelly_multiplier: Position sizing multiplier
              - session: Current trading session
              - volatility_percentile: Current volatility vs historical
        """
        # Get primary timeframe data
        df_m15 = data.get('M15')
        df_m5 = data.get('M5')
        df_h1 = data.get('H1')

        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            return self._build_default_context('UNKNOWN', 'Unknown regime - insufficient data')

        # =========================================================================
        # STEP 1: Ensemble Regime Detection
        # =========================================================================
        regime_name, confidence = self._ensemble_regime_detection(
            df_m15, df_m5, df_h1
        )

        # =========================================================================
        # STEP 2: Get Regime Category
        # =========================================================================
        regime_info = self.REGIMES.get(regime_name, {'category': 'UNKNOWN', 'strength': 0.5})
        regime_category = regime_info['category']
        regime_strength = regime_info['strength']

        # =========================================================================
        # STEP 3: Detect Choppy Score
        # =========================================================================
        choppy_score = self._detect_choppy(df_m15, df_m5)

        # =========================================================================
        # STEP 4: Detect Market Killers
        # =========================================================================
        active_killers = self._detect_market_killers(df_m15, df_m5)

        # =========================================================================
        # STEP 5: Track Regime Stability
        # =========================================================================
        stability_info = self._track_regime_stability(regime_name)

        # =========================================================================
        # STEP 6: Calculate Kelly Multiplier
        # =========================================================================
        kelly_multiplier = self._calculate_kelly_multiplier(
            regime_category, regime_strength, choppy_score, len(active_killers)
        )

        # =========================================================================
        # STEP 7: Get Session Info
        # =========================================================================
        session_info = self._get_session_info(df_m15)

        # =========================================================================
        # STEP 8: Build Comprehensive Context
        # =========================================================================
        context = {
            'regime_name': regime_name,
            'regime_category': regime_category,
            'regime_strength': regime_strength,
            'confidence': confidence,
            'choppy_score': choppy_score,
            'active_killers': active_killers,
            'kelly_multiplier': kelly_multiplier,
            'session': session_info['session'],
            'session_quality': session_info['quality_score'],
            'volatility_percentile': session_info['volatility_percentile'],
            'regime_stability': stability_info['stability_score'],
            'regime_changes_last_20': stability_info['change_count'],
            'is_stable_regime': stability_info['is_stable'],
            'timestamp': datetime.now().isoformat()
        }

        # Log regime detection
        self.logger.info(
            f"[REGIME_ROUTER] {regime_name} | "
            f"Category: {regime_category} | "
            f"Confidence: {confidence:.2f} | "
            f"Choppy: {choppy_score:.0f} | "
            f"Killers: {len(active_killers)} | "
            f"Kelly: {kelly_multiplier:.2f}x"
        )

        return context

    # =========================================================================
    # ENSEMBLE REGIME DETECTION
    # =========================================================================

    def _ensemble_regime_detection(self, df_m15: pd.DataFrame,
                                     df_m5: pd.DataFrame = None,
                                     df_h1: pd.DataFrame = None) -> Tuple[str, float]:
        """
        Ensemble multiple regime detection models.
        
        Voting weights:
          - HMM: 0.4 (statistical)
          - Rule-based: 0.35 (interpretable)
          - LightGBM: 0.25 (ML)
        
        Returns:
            Tuple of (regime_name, confidence)
        """
        votes = {}
        confidences = {}

        # HMM Detection
        if self.hmm_detector is not None:
            try:
                hmm_result = self.hmm_detector.detect_regime(df_m15)
                if hmm_result and 'regime' in hmm_result:
                    regime = hmm_result['regime']
                    conf = hmm_result.get('confidence', 0.7)
                    votes['HMM'] = regime
                    confidences['HMM'] = conf
            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] HMM detection error: {e}")

        # Rule-based Detection
        if self.rule_classifier is not None:
            try:
                rule_result = self.rule_classifier.classify(df_m15)
                if rule_result and 'regime' in rule_result:
                    regime = rule_result['regime']
                    conf = rule_result.get('confidence', 0.6)
                    votes['RULE'] = regime
                    confidences['RULE'] = conf
            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] Rule detection error: {e}")

        # LightGBM Detection
        if self.lgbm_predictor is not None:
            try:
                lgbm_result = self.lgbm_predictor.predict(df_m15, df_m5, df_h1)
                if lgbm_result and 'regime' in lgbm_result:
                    regime = lgbm_result['regime']
                    conf = lgbm_result.get('confidence', 0.65)
                    votes['LGBM'] = regime
                    confidences['LGBM'] = conf
            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] LightGBM detection error: {e}")

        # No votes - return UNKNOWN
        if not votes:
            return 'UNKNOWN', 0.0

        # Weighted voting
        weights = {'HMM': 0.4, 'RULE': 0.35, 'LGBM': 0.25}

        # Count weighted votes per regime
        regime_scores = {}
        for model, regime in votes.items():
            weight = weights.get(model, 0.3)
            conf = confidences.get(model, 0.5)
            score = weight * conf

            if regime not in regime_scores:
                regime_scores[regime] = 0
            regime_scores[regime] += score

        # Get winning regime
        if not regime_scores:
            return 'UNKNOWN', 0.0

        winning_regime = max(regime_scores.keys(), key=lambda r: regime_scores[r])
        winning_score = regime_scores[winning_regime]

        # Calculate confidence (normalized to 0-1)
        max_possible_score = sum(weights.values())
        confidence = min(1.0, winning_score / max_possible_score)

        # Validate regime is in our 18 regimes
        if winning_regime not in self.REGIMES:
            # Try to map to closest regime
            winning_regime = self._map_to_18_regimes(winning_regime, df_m15)

        return winning_regime, confidence

    def _map_to_18_regimes(self, raw_regime: str, df: pd.DataFrame) -> str:
        """Map raw regime name to one of 18 defined regimes."""
        # Common mappings
        mappings = {
            'TRENDING_UP': 'HEALTHY_UPTREND',
            'TRENDING_DOWN': 'HEALTHY_DOWNTREND',
            'RANGING': 'CLASSIC_RANGE',
            'HIGH_VOLATILITY': 'VOLATILE_CHOP',
            'LOW_VOLATILITY': 'TIGHT_RANGE',
            'BREAKOUT': 'PRE_BREAKOUT',
            'REVERSAL_UP': 'OVERSOLD_BOUNCE',
            'REVERSAL_DOWN': 'EXHAUSTED_BULL'
        }

        if raw_regime in mappings:
            return mappings[raw_regime]

        # Fallback: analyze df to determine regime
        if df is not None and len(df) >= 20:
            return self._classify_from_price_action(df)

        return 'CLASSIC_RANGE'

    def _classify_from_price_action(self, df: pd.DataFrame) -> str:
        """Classify regime from price action when models fail."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 20:
                return 'CLASSIC_RANGE'

            # Calculate basic metrics
            recent_close = close[-1]
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20

            # Price range
            range_20 = np.max(high[-20:]) - np.min(low[-20:])
            range_pct = range_20 / recent_close * 100

            # Trend detection
            if recent_close > sma_20 > sma_50:
                if range_pct > 5:
                    return 'HEALTHY_UPTREND'
                else:
                    return 'QUIET_RALLY'
            elif recent_close < sma_20 < sma_50:
                if range_pct > 5:
                    return 'HEALTHY_DOWNTREND'
                else:
                    return 'SLOW_BLEED'
            elif range_pct < 2:
                return 'TIGHT_RANGE'
            elif range_pct < 4:
                return 'CLASSIC_RANGE'
            else:
                return 'VOLATILE_CHOP'

        except Exception:
            return 'CLASSIC_RANGE'

    # =========================================================================
    # CHOPPY DETECTION
    # =========================================================================

    def _detect_choppy(self, df_m15: pd.DataFrame,
                        df_m5: pd.DataFrame = None) -> float:
        """
        Calculate choppy score (0-100).
        
        High choppy score (>65) indicates market is too choppy for most strategies.
        """
        if self.choppy_detector is not None:
            try:
                result = self.choppy_detector.calculate_choppy_score(df_m15, df_m5)
                if result and 'score' in result:
                    return float(result['score'])
            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] Choppy detection error: {e}")

        # Fallback: Simple choppy calculation
        return self._simple_choppy_score(df_m15)

    def _simple_choppy_score(self, df: pd.DataFrame) -> float:
        """Simple choppy score based on direction changes."""
        try:
            if df is None or len(df) < 20:
                return 50.0

            close = df['close'].values.astype(float)
            changes = np.diff(close)

            # Count direction changes
            direction_changes = np.sum(np.diff(np.sign(changes)) != 0)
            max_changes = len(changes) - 1

            if max_changes == 0:
                return 50.0

            choppy_ratio = direction_changes / max_changes
            score = choppy_ratio * 100

            return max(0, min(100, score))

        except Exception:
            return 50.0

    # =========================================================================
    # MARKET KILLERS DETECTION
    # =========================================================================

    def _detect_market_killers(self, df_m15: pd.DataFrame,
                                 df_m5: pd.DataFrame = None) -> List[str]:
        """
        Detect market killer conditions that make trading dangerous.
        
        Returns:
            List of active killer condition names
        """
        if self.killers_detector is not None:
            try:
                result = self.killers_detector.detect_killers(df_m15, df_m5)
                if result and 'active_killers' in result:
                    return result['active_killers']
            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] Killers detection error: {e}")

        # Fallback: Basic killer detection
        return self._simple_killers_detection(df_m15)

    def _simple_killers_detection(self, df: pd.DataFrame) -> List[str]:
        """Simple market killers detection."""
        killers = []

        try:
            if df is None or len(df) < 20:
                return killers

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Volume spike (if available)
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]

                if current_volume > avg_volume * 3:
                    killers.append('VOLUME_SPIKE')

            # Price gap
            if len(close) >= 2:
                gap = abs(close[-1] - close[-2])
                avg_range = np.mean(high[-20:] - low[-20:])

                if gap > avg_range * 2:
                    killers.append('PRICE_GAP')

            # Extreme volatility
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            current_tr = tr[-1]
            avg_tr = np.mean(tr[-20:])

            if current_tr > avg_tr * 3:
                killers.append('EXTREME_VOLATILITY')

        except Exception:
            pass

        return killers

    # =========================================================================
    # REGIME STABILITY TRACKING
    # =========================================================================

    def _track_regime_stability(self, current_regime: str) -> Dict:
        """
        Track regime stability over time.
        
        Returns:
            Dict with stability metrics
        """
        # Add to history
        self._regime_history.append(current_regime)

        # Count regime changes in last 20
        change_count = 0
        for i in range(1, len(self._regime_history)):
            if self._regime_history[i] != self._regime_history[i-1]:
                change_count += 1

        # Calculate stability score (0-100)
        # Fewer changes = higher stability
        stability_score = max(0, 100 - (change_count * 10))

        # Determine if regime is stable
        is_stable = stability_score >= 60

        # Track regime transition
        if self._last_regime != current_regime and self._last_regime != 'UNKNOWN':
            self.logger.info(
                f"[REGIME_ROUTER] Regime transition: "
                f"{self._last_regime} -> {current_regime}"
            )

        self._last_regime = current_regime
        self._last_regime_time = datetime.now()

        return {
            'stability_score': stability_score,
            'change_count': change_count,
            'is_stable': is_stable,
            'history_length': len(self._regime_history)
        }

    # =========================================================================
    # KELLY MULTIPLIER CALCULATION
    # =========================================================================

    def _calculate_kelly_multiplier(self, regime_category: str,
                                     regime_strength: float,
                                     choppy_score: float,
                                     num_killers: int) -> float:
        """
        Calculate Kelly position sizing multiplier based on regime.
        
        Returns:
            Multiplier (0.0 to 2.0)
        """
        # Base multiplier by regime category
        base_multipliers = {
            'TREND': 1.5,
            'SIDEWAY': 1.0,
            'HIGH_VOL': 0.7,
            'REVERSAL': 1.2,
            'UNKNOWN': 0.5
        }

        base_mult = base_multipliers.get(regime_category, 1.0)

        # Adjust by regime strength
        strength_mult = 0.5 + (regime_strength * 0.5)  # 0.5 to 1.0

        # Adjust by choppy score
        if choppy_score > 65:
            choppy_mult = 0.5  # High choppy = reduce size
        elif choppy_score > 50:
            choppy_mult = 0.75
        else:
            choppy_mult = 1.0

        # Adjust by killers
        killer_mult = max(0.3, 1.0 - (num_killers * 0.2))

        # Final multiplier
        final_mult = base_mult * strength_mult * choppy_mult * killer_mult

        # Clamp to reasonable range
        return max(0.0, min(2.0, final_mult))

    # =========================================================================
    # SESSION INFO
    # =========================================================================

    def _get_session_info(self, df: pd.DataFrame) -> Dict:
        """Get current session information."""
        result = {
            'session': 'OTHER',
            'quality_score': 50.0,
            'volatility_percentile': 50.0
        }

        if self.session_mgr is not None:
            try:
                session = self.session_mgr.get_session_from_dataframe(df)
                quality = self.session_mgr.get_session_quality_score(session, df)
                volatility = self.session_mgr.calculate_session_volatility(df, session)

                result['session'] = session
                result['quality_score'] = quality.get('quality_score', 50.0)
                result['volatility_percentile'] = volatility.get('volatility_percentile', 50.0)

            except Exception as e:
                self.logger.debug(f"[REGIME_ROUTER] Session info error: {e}")

        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _build_default_context(self, regime_name: str, reason: str) -> Dict:
        """Build default context when detection fails."""
        return {
            'regime_name': regime_name,
            'regime_category': 'UNKNOWN',
            'regime_strength': 0.5,
            'confidence': 0.0,
            'choppy_score': 50.0,
            'active_killers': [],
            'kelly_multiplier': 1.0,
            'session': 'OTHER',
            'session_quality': 50.0,
            'volatility_percentile': 50.0,
            'regime_stability': 50.0,
            'regime_changes_last_20': 0,
            'is_stable_regime': True,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    def get_regime_stats(self) -> Dict:
        """
        Get statistics on regime detection performance.
        
        Returns:
            Dict with regime statistics
        """
        if not self._regime_history:
            return {
                'total_detections': 0,
                'current_regime': 'UNKNOWN',
                'regime_distribution': {}
            }

        # Count regime occurrences
        regime_counts = {}
        for regime in self._regime_history:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate percentages
        total = len(self._regime_history)
        regime_pcts = {
            r: round(c / total * 100, 1) for r, c in regime_counts.items()
        }

        return {
            'total_detections': total,
            'current_regime': self._last_regime,
            'regime_distribution': regime_counts,
            'regime_percentages': regime_pcts,
            'unique_regimes': len(regime_counts)
        }

    def format_regime_log(self, context: Dict) -> str:
        """
        Format a concise log string for regime context.
        
        Args:
            context: Regime context dict
            
        Returns:
            Formatted log string
        """
        regime = context.get('regime_name', 'UNKNOWN')
        category = context.get('regime_category', 'UNKNOWN')
        conf = context.get('confidence', 0)
        choppy = context.get('choppy_score', 0)
        killers = context.get('active_killers', [])
        kelly = context.get('kelly_multiplier', 1.0)
        session = context.get('session', 'OTHER')

        killer_str = f", Killers: {len(killers)}" if killers else ""

        return (
            f"[REGIME] {regime} ({category}) | "
            f"Conf: {conf:.0%} | "
            f"Choppy: {choppy:.0f} | "
            f"Kelly: {kelly:.2f}x | "
            f"Session: {session}{killer_str}"
        )