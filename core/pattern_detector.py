"""
Pattern Detector - Trap & Fakeout Detection.

Detects trap patterns that commonly fool retail traders.
These patterns are used by market makers and institutions to
harvest liquidity from obvious stop loss levels.

Pattern Types:
  1. BULL_TRAP: False breakout above resistance, then reversal down
  2. BEAR_TRAP: False breakdown below support, then reversal up
  3. FALSE_BREAKOUT: Breakout that quickly fails
  4. STOP_HUNT: Sweep of obvious SL levels before reversal
  5. LIQUIDITY_GRAB: Grab liquidity pool then reverse
  6. FAKEOUT: Quick spike and immediate reversal
  7. WYCKOFF_SPRING: Spring below support with low volume
  8. WYCKOFF_UPTHrust: Upthrust above resistance with low volume

Pattern Confidence (0-100):
  Based on:
    - Price action confirmation
    - Volume pattern
    - Structure context
    - Multi-timeframe alignment
    - Historical success rate

Actions:
  - AVOID_ENTRY: Don't enter in trap direction
  - COUNTER_TRADE: Consider trading opposite direction
  - WAIT: Wait for pattern to fully develop
  - MONITOR: Watch for confirmation
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import config


class PatternDetector:
    """
    Detects trap patterns that fool retail traders.
    
    Provides:
      - 8 trap pattern detections
      - Confidence scoring (0-100)
      - Multi-timeframe confirmation
      - Volume confirmation
      - Structure context analysis
      - Action recommendations
    """

    # Pattern types
    PATTERN_BULL_TRAP = 'BULL_TRAP'
    PATTERN_BEAR_TRAP = 'BEAR_TRAP'
    PATTERN_FALSE_BREAKOUT = 'FALSE_BREAKOUT'
    PATTERN_STOP_HUNT = 'STOP_HUNT'
    PATTERN_LIQUIDITY_GRAB = 'LIQUIDITY_GRAB'
    PATTERN_FAKEOUT = 'FAKEOUT'
    PATTERN_WYCKOFF_SPRING = 'WYCKOFF_SPRING'
    PATTERN_WYCKOFF_UPTHrust = 'WYCKOFF_UPTHrust'

    # Action types
    ACTION_AVOID = 'AVOID_ENTRY'
    ACTION_COUNTER = 'COUNTER_TRADE'
    ACTION_WAIT = 'WAIT'
    ACTION_MONITOR = 'MONITOR'

    def __init__(self):
        """Initialize PatternDetector with configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Detection parameters
        self.breakout_threshold_pct = 0.3  # 0.3% for breakout
        self.reversal_threshold_pct = 0.2  # 0.2% reversal to confirm trap
        self.volume_threshold = 1.5  # Volume must be 1.5x average

        # Swing detection parameters
        self.swing_lookback = 20
        self.swing_order = 3  # Number of bars on each side

        # Pattern history for analysis
        self._pattern_history: List[Dict] = []
        self.max_history = 200

        self.logger.info("[PATTERN] Initialized with 8 trap pattern detections")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def detect_patterns(self, df_m15: pd.DataFrame,
                          df_m5: pd.DataFrame = None,
                          df_h1: pd.DataFrame = None) -> Dict:
        """
        Main entry point: Detect all trap patterns.
        
        Args:
            df_m15: Primary timeframe data (M15)
            df_m5: Secondary timeframe (M5, optional)
            df_h1: Higher timeframe (H1, optional)
            
        Returns:
            Dict with:
              - active_patterns: List of detected pattern names
              - pattern_details: Detailed info per pattern
              - trap_detected: Boolean flag
              - recommendations: Action recommendations
              - structure_context: Current market structure
        """
        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            return self._build_default_result()

        try:
            # Extract price data
            close = df_m15['close'].values.astype(float)
            high = df_m15['high'].values.astype(float)
            low = df_m15['low'].values.astype(float)
            open_ = df_m15['open'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=close[0])
            high = np.nan_to_num(high, nan=high[0])
            low = np.nan_to_num(low, nan=low[0])
            open_ = np.nan_to_num(open_, nan=open_[0])

            # Get volume data
            volume = None
            if 'tick_volume' in df_m15.columns:
                volume = df_m15['tick_volume'].values.astype(float)
                volume = np.nan_to_num(volume, nan=1.0)

            # Detect structure context
            structure = self._analyze_structure(close, high, low)

            # =========================================================================
            # PATTERN DETECTIONS
            # =========================================================================
            active_patterns = []
            pattern_details = {}

            # Pattern 1: Bull Trap
            bull_trap = self._detect_bull_trap(close, high, low, open_, volume, structure, df_m5)
            if bull_trap['detected']:
                active_patterns.append(self.PATTERN_BULL_TRAP)
                pattern_details[self.PATTERN_BULL_TRAP] = bull_trap

            # Pattern 2: Bear Trap
            bear_trap = self._detect_bear_trap(close, high, low, open_, volume, structure, df_m5)
            if bear_trap['detected']:
                active_patterns.append(self.PATTERN_BEAR_TRAP)
                pattern_details[self.PATTERN_BEAR_TRAP] = bear_trap

            # Pattern 3: False Breakout
            false_breakout = self._detect_false_breakout(close, high, low, open_, volume, structure)
            if false_breakout['detected']:
                active_patterns.append(self.PATTERN_FALSE_BREAKOUT)
                pattern_details[self.PATTERN_FALSE_BREAKOUT] = false_breakout

            # Pattern 4: Stop Hunt
            stop_hunt = self._detect_stop_hunt(close, high, low, open_, volume, structure)
            if stop_hunt['detected']:
                active_patterns.append(self.PATTERN_STOP_HUNT)
                pattern_details[self.PATTERN_STOP_HUNT] = stop_hunt

            # Pattern 5: Liquidity Grab
            liquidity_grab = self._detect_liquidity_grab(close, high, low, open_, volume, structure)
            if liquidity_grab['detected']:
                active_patterns.append(self.PATTERN_LIQUIDITY_GRAB)
                pattern_details[self.PATTERN_LIQUIDITY_GRAB] = liquidity_grab

            # Pattern 6: Fakeout
            fakeout = self._detect_fakeout(close, high, low, open_, volume)
            if fakeout['detected']:
                active_patterns.append(self.PATTERN_FAKEOUT)
                pattern_details[self.PATTERN_FAKEOUT] = fakeout

            # Pattern 7: Wyckoff Spring
            wyckoff_spring = self._detect_wyckoff_spring(close, high, low, open_, volume, structure)
            if wyckoff_spring['detected']:
                active_patterns.append(self.PATTERN_WYCKOFF_SPRING)
                pattern_details[self.PATTERN_WYCKOFF_SPRING] = wyckoff_spring

            # Pattern 8: Wyckoff Upthrust
            wyckoff_upthrust = self._detect_wyckoff_upthrust(close, high, low, open_, volume, structure)
            if wyckoff_upthrust['detected']:
                active_patterns.append(self.PATTERN_WYCKOFF_UPTHrust)
                pattern_details[self.PATTERN_WYCKOFF_UPTHrust] = wyckoff_upthrust

            # =========================================================================
            # AGGREGATE RESULTS
            # =========================================================================
            trap_detected = len(active_patterns) > 0

            # Generate recommendations
            recommendations = self._generate_recommendations(active_patterns, pattern_details)

            result = {
                'active_patterns': active_patterns,
                'pattern_count': len(active_patterns),
                'pattern_details': pattern_details,
                'trap_detected': trap_detected,
                'structure_context': structure,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }

            # Record to history
            self._record_to_history(result)

            # Log if patterns detected
            if trap_detected:
                self.logger.info(
                    f"[PATTERN] Detected {len(active_patterns)} pattern(s): "
                    f"{', '.join(active_patterns)}"
                )

            return result

        except Exception as e:
            self.logger.error(f"[PATTERN] Detection error: {e}")
            return self._build_default_result()

    # =========================================================================
    # STRUCTURE ANALYSIS
    # =========================================================================

    def _analyze_structure(self, close: np.ndarray, high: np.ndarray,
                            low: np.ndarray) -> Dict:
        """
        Analyze current market structure.
        
        Returns:
            Dict with structure info (swing highs, swing lows, support, resistance)
        """
        structure = {
            'swing_highs': [],
            'swing_lows': [],
            'resistance': None,
            'support': None,
            'trend': 'UNKNOWN'
        }

        try:
            # Find swing highs
            swing_highs = []
            for i in range(self.swing_order, len(high) - self.swing_order):
                if high[i] == np.max(high[i-self.swing_order:i+self.swing_order+1]):
                    swing_highs.append({'index': i, 'price': float(high[i])})
            structure['swing_highs'] = swing_highs[-5:]  # Last 5

            # Find swing lows
            swing_lows = []
            for i in range(self.swing_order, len(low) - self.swing_order):
                if low[i] == np.min(low[i-self.swing_order:i+self.swing_order+1]):
                    swing_lows.append({'index': i, 'price': float(low[i])})
            structure['swing_lows'] = swing_lows[-5:]  # Last 5

            # Determine resistance and support
            if swing_highs:
                structure['resistance'] = swing_highs[-1]['price']
            if swing_lows:
                structure['support'] = swing_lows[-1]['price']

            # Determine trend
            if len(close) >= 50:
                sma_20 = np.mean(close[-20:])
                sma_50 = np.mean(close[-50:])
                current = close[-1]

                if current > sma_20 > sma_50:
                    structure['trend'] = 'UPTREND'
                elif current < sma_20 < sma_50:
                    structure['trend'] = 'DOWNTREND'
                else:
                    structure['trend'] = 'SIDEWAY'

        except Exception as e:
            self.logger.debug(f"[PATTERN] Structure analysis error: {e}")

        return structure

    # =========================================================================
    # PATTERN 1: BULL TRAP
    # =========================================================================

    def _detect_bull_trap(self, close: np.ndarray, high: np.ndarray,
                            low: np.ndarray, open_: np.ndarray,
                            volume: np.ndarray, structure: Dict,
                            df_m5: pd.DataFrame = None) -> Dict:
        """
        Detect Bull Trap: False breakout above resistance.
        
        Criteria:
          1. Price breaks above recent swing high
          2. Price quickly reverses (within 3-5 bars)
          3. Closes below breakout level
          4. Volume spike on breakout, declining on reversal
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'BEARISH',
            'details': {}
        }

        try:
            resistance = structure.get('resistance')
            if resistance is None:
                return result

            # Check if recent high broke resistance
            recent_high = np.max(high[-5:])
            current_close = close[-1]

            if recent_high <= resistance:
                return result

            # Check if price reversed
            breakout_amount = (recent_high - resistance) / resistance * 100
            reversal_amount = (recent_high - current_close) / recent_high * 100

            if reversal_amount < self.reversal_threshold_pct:
                return result

            # Check volume pattern
            volume_confirmed = True
            if volume is not None and len(volume) >= 10:
                avg_volume = np.mean(volume[-10:-3])
                breakout_volume = np.max(volume[-5:-1])
                current_volume = volume[-1]

                # Volume should spike on breakout and decline on reversal
                volume_confirmed = (breakout_volume > avg_volume * self.volume_threshold and
                                    current_volume < breakout_volume * 0.7)

            # M5 confirmation
            m5_confirmed = False
            if df_m5 is not None and len(df_m5) >= 20:
                m5_close = df_m5['close'].values.astype(float)
                m5_reversal = (np.max(m5_close[-10:]) - m5_close[-1]) / np.max(m5_close[-10:]) * 100
                m5_confirmed = m5_reversal > self.reversal_threshold_pct * 1.5

            # Calculate confidence
            confidence = self._score_pattern(
                base_score=60,
                breakout_bonus=10 if breakout_amount > 0.5 else 0,
                reversal_bonus=10 if reversal_amount > 0.5 else 0,
                volume_bonus=10 if volume_confirmed else 0,
                mtf_bonus=10 if m5_confirmed else 0
            )

            if confidence >= 50:
                result['detected'] = True
                result['confidence'] = confidence
                result['details'] = {
                    'resistance': round(resistance, 2),
                    'breakout_high': round(recent_high, 2),
                    'current_close': round(current_close, 2),
                    'breakout_pct': round(breakout_amount, 2),
                    'reversal_pct': round(reversal_amount, 2),
                    'volume_confirmed': volume_confirmed,
                    'm5_confirmed': m5_confirmed
                }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Bull trap detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 2: BEAR TRAP
    # =========================================================================

    def _detect_bear_trap(self, close: np.ndarray, high: np.ndarray,
                            low: np.ndarray, open_: np.ndarray,
                            volume: np.ndarray, structure: Dict,
                            df_m5: pd.DataFrame = None) -> Dict:
        """
        Detect Bear Trap: False breakdown below support.
        
        Criteria:
          1. Price breaks below recent swing low
          2. Price quickly reverses (within 3-5 bars)
          3. Closes above breakdown level
          4. Volume spike on breakdown, declining on reversal
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'BULLISH',
            'details': {}
        }

        try:
            support = structure.get('support')
            if support is None:
                return result

            # Check if recent low broke support
            recent_low = np.min(low[-5:])
            current_close = close[-1]

            if recent_low >= support:
                return result

            # Check if price reversed
            breakdown_amount = (support - recent_low) / support * 100
            reversal_amount = (current_close - recent_low) / recent_low * 100

            if reversal_amount < self.reversal_threshold_pct:
                return result

            # Check volume pattern
            volume_confirmed = True
            if volume is not None and len(volume) >= 10:
                avg_volume = np.mean(volume[-10:-3])
                breakdown_volume = np.max(volume[-5:-1])
                current_volume = volume[-1]

                volume_confirmed = (breakdown_volume > avg_volume * self.volume_threshold and
                                    current_volume < breakdown_volume * 0.7)

            # M5 confirmation
            m5_confirmed = False
            if df_m5 is not None and len(df_m5) >= 20:
                m5_close = df_m5['close'].values.astype(float)
                m5_reversal = (m5_close[-1] - np.min(m5_close[-10:])) / np.min(m5_close[-10:]) * 100
                m5_confirmed = m5_reversal > self.reversal_threshold_pct * 1.5

            # Calculate confidence
            confidence = self._score_pattern(
                base_score=60,
                breakout_bonus=10 if breakdown_amount > 0.5 else 0,
                reversal_bonus=10 if reversal_amount > 0.5 else 0,
                volume_bonus=10 if volume_confirmed else 0,
                mtf_bonus=10 if m5_confirmed else 0
            )

            if confidence >= 50:
                result['detected'] = True
                result['confidence'] = confidence
                result['details'] = {
                    'support': round(support, 2),
                    'breakdown_low': round(recent_low, 2),
                    'current_close': round(current_close, 2),
                    'breakdown_pct': round(breakdown_amount, 2),
                    'reversal_pct': round(reversal_amount, 2),
                    'volume_confirmed': volume_confirmed,
                    'm5_confirmed': m5_confirmed
                }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Bear trap detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 3: FALSE BREAKOUT
    # =========================================================================

    def _detect_false_breakout(self, close: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, open_: np.ndarray,
                                 volume: np.ndarray, structure: Dict) -> Dict:
        """
        Detect False Breakout: Breakout that quickly fails.
        
        More general than Bull/Bear Trap - can occur at any level.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'UNKNOWN',
            'details': {}
        }

        try:
            if len(close) < 10:
                return result

            # Look at last 5 bars
            recent_high = np.max(high[-5:])
            recent_low = np.min(low[-5:])
            prev_high = np.max(high[-10:-5])
            prev_low = np.min(low[-10:-5])

            current_close = close[-1]

            # Check for upside false breakout
            if recent_high > prev_high and current_close < prev_high:
                breakout_pct = (recent_high - prev_high) / prev_high * 100
                fail_pct = (recent_high - current_close) / recent_high * 100

                if breakout_pct > 0.2 and fail_pct > 0.3:
                    confidence = self._score_pattern(
                        base_score=55,
                        breakout_bonus=10 if breakout_pct > 0.4 else 0,
                        reversal_bonus=10 if fail_pct > 0.5 else 0
                    )

                    if confidence >= 50:
                        result['detected'] = True
                        result['confidence'] = confidence
                        result['direction'] = 'BEARISH'
                        result['details'] = {
                            'type': 'UPSIDE_FALSE_BREAKOUT',
                            'breakout_pct': round(breakout_pct, 2),
                            'fail_pct': round(fail_pct, 2)
                        }

            # Check for downside false breakout
            elif recent_low < prev_low and current_close > prev_low:
                breakout_pct = (prev_low - recent_low) / prev_low * 100
                fail_pct = (current_close - recent_low) / recent_low * 100

                if breakout_pct > 0.2 and fail_pct > 0.3:
                    confidence = self._score_pattern(
                        base_score=55,
                        breakout_bonus=10 if breakout_pct > 0.4 else 0,
                        reversal_bonus=10 if fail_pct > 0.5 else 0
                    )

                    if confidence >= 50:
                        result['detected'] = True
                        result['confidence'] = confidence
                        result['direction'] = 'BULLISH'
                        result['details'] = {
                            'type': 'DOWNSIDE_FALSE_BREAKOUT',
                            'breakout_pct': round(breakout_pct, 2),
                            'fail_pct': round(fail_pct, 2)
                        }

        except Exception as e:
            self.logger.debug(f"[PATTERN] False breakout detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 4: STOP HUNT
    # =========================================================================

    def _detect_stop_hunt(self, close: np.ndarray, high: np.ndarray,
                            low: np.ndarray, open_: np.ndarray,
                            volume: np.ndarray, structure: Dict) -> Dict:
        """
        Detect Stop Hunt: Sweep of obvious SL levels.
        
        Criteria:
          1. Price sweeps round number or obvious level
          2. Quick reversal with wick
          3. Low volume on sweep
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'UNKNOWN',
            'details': {}
        }

        try:
            current_price = close[-1]
            current_high = high[-1]
            current_low = low[-1]

            # Check for round number levels
            round_levels = self._find_round_levels(current_price)

            for level in round_levels:
                # Upside stop hunt (above round number)
                if current_high > level and current_close < level:
                    wick_size = current_high - max(open_[-1], close[-1])
                    body_size = abs(close[-1] - open_[-1])

                    if wick_size > body_size * 2:
                        confidence = self._score_pattern(
                            base_score=50,
                            breakout_bonus=15,
                            reversal_bonus=10
                        )

                        if confidence >= 50:
                            result['detected'] = True
                            result['confidence'] = confidence
                            result['direction'] = 'BEARISH'
                            result['details'] = {
                                'hunt_level': round(level, 2),
                                'wick_size': round(wick_size, 2),
                                'body_size': round(body_size, 2)
                            }
                        break

                # Downside stop hunt (below round number)
                elif current_low < level and current_close > level:
                    wick_size = min(open_[-1], close[-1]) - current_low
                    body_size = abs(close[-1] - open_[-1])

                    if wick_size > body_size * 2:
                        confidence = self._score_pattern(
                            base_score=50,
                            breakout_bonus=15,
                            reversal_bonus=10
                        )

                        if confidence >= 50:
                            result['detected'] = True
                            result['confidence'] = confidence
                            result['direction'] = 'BULLISH'
                            result['details'] = {
                                'hunt_level': round(level, 2),
                                'wick_size': round(wick_size, 2),
                                'body_size': round(body_size, 2)
                            }
                        break

        except Exception as e:
            self.logger.debug(f"[PATTERN] Stop hunt detection error: {e}")

        return result

    def _find_round_levels(self, price: float) -> List[float]:
        """Find round number levels near current price."""
        levels = []
        base = int(price / 10) * 10  # Nearest 10

        for offset in [-20, -10, 0, 10, 20]:
            level = base + offset
            levels.append(float(level))

        return levels

    # =========================================================================
    # PATTERN 5: LIQUIDITY GRAB
    # =========================================================================

    def _detect_liquidity_grab(self, close: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, open_: np.ndarray,
                                 volume: np.ndarray, structure: Dict) -> Dict:
        """
        Detect Liquidity Grab: Grab liquidity pool then reverse.
        
        Similar to stop hunt but focuses on liquidity pools.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'UNKNOWN',
            'details': {}
        }

        try:
            # Check if price swept swing high/low and reversed
            swing_highs = structure.get('swing_highs', [])
            swing_lows = structure.get('swing_lows', [])

            if not swing_highs and not swing_lows:
                return result

            current_close = close[-1]

            # Check upside liquidity grab
            if swing_highs:
                recent_swing_high = swing_highs[-1]['price']
                recent_high = np.max(high[-3:])

                if recent_high > recent_swing_high and current_close < recent_swing_high:
                    grab_pct = (recent_high - recent_swing_high) / recent_swing_high * 100

                    if grab_pct > 0.1:
                        confidence = self._score_pattern(
                            base_score=55,
                            breakout_bonus=10,
                            reversal_bonus=15
                        )

                        if confidence >= 50:
                            result['detected'] = True
                            result['confidence'] = confidence
                            result['direction'] = 'BEARISH'
                            result['details'] = {
                                'swing_high': round(recent_swing_high, 2),
                                'grab_high': round(recent_high, 2),
                                'grab_pct': round(grab_pct, 2)
                            }

            # Check downside liquidity grab
            if swing_lows and not result['detected']:
                recent_swing_low = swing_lows[-1]['price']
                recent_low = np.min(low[-3:])

                if recent_low < recent_swing_low and current_close > recent_swing_low:
                    grab_pct = (recent_swing_low - recent_low) / recent_swing_low * 100

                    if grab_pct > 0.1:
                        confidence = self._score_pattern(
                            base_score=55,
                            breakout_bonus=10,
                            reversal_bonus=15
                        )

                        if confidence >= 50:
                            result['detected'] = True
                            result['confidence'] = confidence
                            result['direction'] = 'BULLISH'
                            result['details'] = {
                                'swing_low': round(recent_swing_low, 2),
                                'grab_low': round(recent_low, 2),
                                'grab_pct': round(grab_pct, 2)
                            }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Liquidity grab detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 6: FAKEOUT
    # =========================================================================

    def _detect_fakeout(self, close: np.ndarray, high: np.ndarray,
                          low: np.ndarray, open_: np.ndarray,
                          volume: np.ndarray) -> Dict:
        """
        Detect Fakeout: Quick spike and immediate reversal.
        
        Single-bar or two-bar pattern.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'UNKNOWN',
            'details': {}
        }

        try:
            if len(close) < 3:
                return result

            # Check last bar
            current_open = open_[-1]
            current_high = high[-1]
            current_low = low[-1]
            current_close = close[-1]

            prev_close = close[-2]

            # Upside fakeout: spike up then close down
            if current_high > prev_close and current_close < current_open:
                spike_pct = (current_high - prev_close) / prev_close * 100
                reversal_pct = (current_high - current_close) / current_high * 100

                if spike_pct > 0.3 and reversal_pct > 0.4:
                    confidence = self._score_pattern(
                        base_score=50,
                        breakout_bonus=10 if spike_pct > 0.5 else 0,
                        reversal_bonus=15 if reversal_pct > 0.6 else 0
                    )

                    if confidence >= 50:
                        result['detected'] = True
                        result['confidence'] = confidence
                        result['direction'] = 'BEARISH'
                        result['details'] = {
                            'type': 'UPSIDE_FAKEOUT',
                            'spike_pct': round(spike_pct, 2),
                            'reversal_pct': round(reversal_pct, 2)
                        }

            # Downside fakeout: spike down then close up
            elif current_low < prev_close and current_close > current_open:
                spike_pct = (prev_close - current_low) / prev_close * 100
                reversal_pct = (current_close - current_low) / current_low * 100

                if spike_pct > 0.3 and reversal_pct > 0.4:
                    confidence = self._score_pattern(
                        base_score=50,
                        breakout_bonus=10 if spike_pct > 0.5 else 0,
                        reversal_bonus=15 if reversal_pct > 0.6 else 0
                    )

                    if confidence >= 50:
                        result['detected'] = True
                        result['confidence'] = confidence
                        result['direction'] = 'BULLISH'
                        result['details'] = {
                            'type': 'DOWNSIDE_FAKEOUT',
                            'spike_pct': round(spike_pct, 2),
                            'reversal_pct': round(reversal_pct, 2)
                        }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Fakeout detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 7: WYCKOFF SPRING
    # =========================================================================

    def _detect_wyckoff_spring(self, close: np.ndarray, high: np.ndarray,
                                 low: np.ndarray, open_: np.ndarray,
                                 volume: np.ndarray, structure: Dict) -> Dict:
        """
        Detect Wyckoff Spring: Spring below support with low volume.
        
        Classic accumulation pattern.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'BULLISH',
            'details': {}
        }

        try:
            support = structure.get('support')
            if support is None:
                return result

            current_low = low[-1]
            current_close = close[-1]

            # Check if price sprang below support
            if current_low >= support:
                return result

            # Check if price recovered
            if current_close < support:
                return result

            # Check volume (should be low on spring)
            low_volume = False
            if volume is not None and len(volume) >= 20:
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                low_volume = current_volume < avg_volume * 0.8

            if not low_volume:
                return result

            # Calculate spring depth
            spring_depth = (support - current_low) / support * 100

            confidence = self._score_pattern(
                base_score=60,
                breakout_bonus=10 if spring_depth > 0.3 else 0,
                reversal_bonus=15,
                volume_bonus=10  # Low volume is good for spring
            )

            if confidence >= 50:
                result['detected'] = True
                result['confidence'] = confidence
                result['details'] = {
                    'support': round(support, 2),
                    'spring_low': round(current_low, 2),
                    'spring_depth_pct': round(spring_depth, 2),
                    'low_volume': True
                }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Wyckoff spring detection error: {e}")

        return result

    # =========================================================================
    # PATTERN 8: WYCKOFF UPTHrust
    # =========================================================================

    def _detect_wyckoff_upthrust(self, close: np.ndarray, high: np.ndarray,
                                   low: np.ndarray, open_: np.ndarray,
                                   volume: np.ndarray, structure: Dict) -> Dict:
        """
        Detect Wyckoff Upthrust: Upthrust above resistance with low volume.
        
        Classic distribution pattern.
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'direction': 'BEARISH',
            'details': {}
        }

        try:
            resistance = structure.get('resistance')
            if resistance is None:
                return result

            current_high = high[-1]
            current_close = close[-1]

            # Check if price thrust above resistance
            if current_high <= resistance:
                return result

            # Check if price recovered
            if current_close > resistance:
                return result

            # Check volume (should be low on upthrust)
            low_volume = False
            if volume is not None and len(volume) >= 20:
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                low_volume = current_volume < avg_volume * 0.8

            if not low_volume:
                return result

            # Calculate upthrust height
            upthrust_height = (current_high - resistance) / resistance * 100

            confidence = self._score_pattern(
                base_score=60,
                breakout_bonus=10 if upthrust_height > 0.3 else 0,
                reversal_bonus=15,
                volume_bonus=10  # Low volume is good for upthrust
            )

            if confidence >= 50:
                result['detected'] = True
                result['confidence'] = confidence
                result['details'] = {
                    'resistance': round(resistance, 2),
                    'upthrust_high': round(current_high, 2),
                    'upthrust_height_pct': round(upthrust_height, 2),
                    'low_volume': True
                }

        except Exception as e:
            self.logger.debug(f"[PATTERN] Wyckoff upthrust detection error: {e}")

        return result

    # =========================================================================
    # PATTERN SCORING
    # =========================================================================

    def _score_pattern(self, base_score: float = 50,
                        breakout_bonus: float = 0,
                        reversal_bonus: float = 0,
                        volume_bonus: float = 0,
                        mtf_bonus: float = 0) -> float:
        """
        Calculate pattern confidence score.
        
        Returns:
            Score 0-100
        """
        score = base_score + breakout_bonus + reversal_bonus + volume_bonus + mtf_bonus
        return max(0, min(100, score))

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def _generate_recommendations(self, active_patterns: List[str],
                                    pattern_details: Dict) -> Dict:
        """
        Generate action recommendations based on detected patterns.
        
        Returns:
            Dict with recommendations
        """
        recommendations = {
            'should_enter': True,
            'avoid_direction': None,
            'preferred_direction': None,
            'actions': [],
            'warnings': []
        }

        if not active_patterns:
            recommendations['actions'].append('No trap patterns detected - normal trading')
            return recommendations

        # Analyze pattern directions
        bullish_patterns = [p for p in active_patterns
                            if pattern_details.get(p, {}).get('direction') == 'BULLISH']
        bearish_patterns = [p for p in active_patterns
                            if pattern_details.get(p, {}).get('direction') == 'BEARISH']

        # If bearish patterns detected, avoid buying
        if bearish_patterns:
            recommendations['avoid_direction'] = 'BUY'
            recommendations['preferred_direction'] = 'SELL'
            recommendations['actions'].append(
                f"Bearish trap detected ({', '.join(bearish_patterns)}) - avoid BUY entries"
            )
            recommendations['warnings'].append('Potential bull trap - sellers in control')

        # If bullish patterns detected, avoid selling
        if bullish_patterns:
            recommendations['avoid_direction'] = 'SELL'
            recommendations['preferred_direction'] = 'BUY'
            recommendations['actions'].append(
                f"Bullish trap detected ({', '.join(bullish_patterns)}) - avoid SELL entries"
            )
            recommendations['warnings'].append('Potential bear trap - buyers in control')

        # If conflicting patterns, wait
        if bullish_patterns and bearish_patterns:
            recommendations['should_enter'] = False
            recommendations['avoid_direction'] = 'BOTH'
            recommendations['preferred_direction'] = None
            recommendations['actions'].append('Conflicting patterns - wait for clarity')

        # High confidence patterns suggest counter-trades
        for pattern_name, details in pattern_details.items():
            if details.get('confidence', 0) >= 75:
                direction = details.get('direction')
                if direction == 'BULLISH':
                    recommendations['actions'].append(
                        f"High confidence {pattern_name} - consider BUY counter-trade"
                    )
                elif direction == 'BEARISH':
                    recommendations['actions'].append(
                        f"High confidence {pattern_name} - consider SELL counter-trade"
                    )

        return recommendations

    # =========================================================================
    # HISTORY MANAGEMENT
    # =========================================================================

    def _record_to_history(self, result: Dict):
        """Record detection result to history."""
        self._pattern_history.append({
            'timestamp': datetime.now().isoformat(),
            'active_patterns': result['active_patterns'],
            'pattern_count': result['pattern_count'],
            'trap_detected': result['trap_detected']
        })

        if len(self._pattern_history) > self.max_history:
            self._pattern_history = self._pattern_history[-self.max_history:]

    def get_pattern_history(self, limit: int = 20) -> List[Dict]:
        """Get recent pattern history."""
        return self._pattern_history[-limit:]

    def get_pattern_stats(self) -> Dict:
        """Get statistics on pattern detections."""
        if not self._pattern_history:
            return {
                'total_detections': 0,
                'pattern_frequency': {}
            }

        pattern_counts = {}
        for record in self._pattern_history:
            for pattern in record['active_patterns']:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return {
            'total_detections': len(self._pattern_history),
            'pattern_frequency': pattern_counts,
            'most_common': max(pattern_counts.keys(), key=lambda k: pattern_counts[k]) if pattern_counts else None
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _build_default_result(self) -> Dict:
        """Build default result when detection fails."""
        return {
            'active_patterns': [],
            'pattern_count': 0,
            'pattern_details': {},
            'trap_detected': False,
            'structure_context': {},
            'recommendations': {
                'should_enter': True,
                'actions': ['Normal conditions - no traps detected']
            },
            'timestamp': datetime.now().isoformat()
        }

    def format_pattern_log(self, result: Dict) -> str:
        """
        Format a concise log string for pattern detection.
        
        Args:
            result: Result from detect_patterns
            
        Returns:
            Formatted log string
        """
        patterns = result.get('active_patterns', [])
        count = result.get('pattern_count', 0)

        if not patterns:
            return "[PATTERN] No traps detected"

        patterns_str = ', '.join(patterns[:3])
        if len(patterns) > 3:
            patterns_str += f", +{len(patterns) - 3} more"

        return f"[PATTERN] {count} trap(s): {patterns_str}"