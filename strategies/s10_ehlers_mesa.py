"""
S10_EhlersMESA - Ehlers MESA Adaptive Moving Average Strategy.

Trend-following strategy that uses Ehlers MESA Adaptive Moving Average
and Instantaneous Trendline for trend identification and entry timing.

Strategy Logic:
  1. Calculate MESA Adaptive Moving Average (MAMA)
  2. Calculate Instantaneous Trendline
  3. Detect dominant cycle using MESA
  4. Determine trend direction
  5. Generate entry signal on trend confirmation

MESA (Maximum Entropy Spectral Analysis):
  An adaptive moving average that adjusts its period based on
  the dominant cycle of the market. Uses Hilbert Transform for
  instantaneous phase calculation.
  
  - MAMA: MESA Adaptive Moving Average (fast line)
  - FAMA: Following Adaptive Moving Average (slow line)
  - When MAMA crosses FAMA, trend changes

Used Engines:
  - EhlersDSPEngine: MESA and Instantaneous Trendline
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.dsp_ehlers_engine import EhlersDSPEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S10_EhlersMESA(BaseStrategy):
    """
    Ehlers MESA Adaptive Moving Average Strategy.
    
    This strategy uses Ehlers MESA Adaptive Moving Average and
    Instantaneous Trendline for trend identification.
    
    MESA Definition:
      MESA (Maximum Entropy Spectral Analysis) is an adaptive
      moving average that adjusts its period based on the
      dominant cycle of the market.
      
    MAMA/FAMA Crossover:
      - MAMA (fast line) crosses above FAMA (slow line) → BUY signal
      - MAMA crosses below FAMA → SELL signal
      
    Instantaneous Trendline:
      A trendline that adjusts based on the current market cycle,
      providing dynamic support/resistance levels.
      
    Entry Criteria:
      - MAMA/FAMA crossover
      - Trend direction confirmed
      - Dominant cycle detected
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S10_EhlersMESA strategy."""
        super().__init__(
            strategy_name='S10_EhlersMESA',
            strategy_category='TREND',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.ehlers_engine = EhlersDSPEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.mesa_lookback = 100  # Lookback for MESA calculation
        self.fast_limit = 0.5  # MESA fast limit
        self.slow_limit = 0.05  # MESA slow limit
        self.min_cycle_period = 6  # Minimum cycle period
        self.max_cycle_period = 50  # Maximum cycle period

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m15: pd.DataFrame,
        df_m5: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S10_EhlersMESA.
        
        Args:
            df_m15: M15 DataFrame
            df_m5: M5 DataFrame (optional)
            regime_context: Current regime information
            
        Returns:
            Signal dict with entry/exit information
        """
        # Default neutral signal
        default_signal = self._create_neutral_signal()

        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < self.mesa_lookback:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate MESA
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            mesa_result = self.ehlers_engine.ehlers_mesa(
                close, fast_limit=self.fast_limit, slow_limit=self.slow_limit
            )

            if mesa_result is None or mesa_result[0] is None:
                return default_signal

            mama, fama = mesa_result

            # =========================================================================
            # STEP 2: Calculate Instantaneous Trendline
            # =========================================================================
            trendline = self.ehlers_engine.instantaneous_trendline(close)

            # =========================================================================
            # STEP 3: Detect Dominant Cycle
            # =========================================================================
            cycle_info = self._detect_cycle(mama, fama, close)

            # =========================================================================
            # STEP 4: Determine Trend Direction
            # =========================================================================
            trend_info = self._determine_direction(mama, fama, trendline, close)

            if trend_info is None:
                return default_signal

            # =========================================================================
            # STEP 5: Check Crossover
            # =========================================================================
            crossover = self._check_crossover(mama, fama)

            if crossover is None:
                return default_signal

            # =========================================================================
            # STEP 6: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, trend_info):
                    return default_signal

            # =========================================================================
            # STEP 7: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, trend_info, crossover,
                                            cycle_info, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S10_MESA] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def _detect_cycle(
        self, mama: np.ndarray, fama: np.ndarray, close: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect dominant cycle using MESA.
        
        Args:
            mama: MAMA array
            fama: FAMA array
            close: Close price array
            
        Returns:
            Cycle info dict or None
        """
        try:
            # Calculate MAMA-FAMA difference (indicates cycle phase)
            diff = mama - fama

            # Find zero crossings (cycle changes)
            zero_crossings = []
            for i in range(1, len(diff)):
                if diff[i-1] * diff[i] < 0:
                    zero_crossings.append(i)

            if len(zero_crossings) < 2:
                return {'cycle_detected': False, 'cycle_period': None}

            # Calculate cycle period from zero crossings
            periods = []
            for i in range(1, len(zero_crossings)):
                period = zero_crossings[i] - zero_crossings[i-1]
                periods.append(period)

            avg_period = np.mean(periods)

            # Validate cycle period
            if self.min_cycle_period <= avg_period <= self.max_cycle_period:
                return {
                    'cycle_detected': True,
                    'cycle_period': float(avg_period),
                    'zero_crossings': len(zero_crossings),
                    'current_phase': float(diff[-1])
                }

            return {'cycle_detected': False, 'cycle_period': None}

        except Exception as e:
            self.logger.debug(f"[S10_MESA] Cycle detection error: {e}")
            return None

    # =========================================================================
    # TREND DIRECTION
    # =========================================================================

    def _determine_direction(
        self, mama: np.ndarray, fama: np.ndarray,
        trendline: Optional[np.ndarray], close: np.ndarray
    ) -> Optional[Dict]:
        """
        Determine trend direction.
        
        Args:
            mama: MAMA array
            fama: FAMA array
            trendline: Instantaneous Trendline array
            close: Close price array
            
        Returns:
            Trend info dict or None
        """
        try:
            current_price = close[-1]
            current_mama = mama[-1]
            current_fama = fama[-1]

            # Check MAMA-FAMA relationship
            if current_mama > current_fama:
                # MAMA above FAMA = uptrend
                direction = 'BUY'
                trend_strength = (current_mama - current_fama) / current_price * 100
            elif current_mama < current_fama:
                # MAMA below FAMA = downtrend
                direction = 'SELL'
                trend_strength = (current_fama - current_mama) / current_price * 100
            else:
                return None  # No clear trend

            # Check trendline confirmation
            if trendline is not None:
                current_trendline = trendline[-1]

                if direction == 'BUY' and current_price < current_trendline:
                    # Price below trendline in uptrend = weak
                    trend_strength *= 0.5
                elif direction == 'SELL' and current_price > current_trendline:
                    # Price above trendline in downtrend = weak
                    trend_strength *= 0.5

            # Calculate trend strength
            trend_strength = min(1.0, trend_strength / 5.0)  # Normalize

            if trend_strength < 0.2:
                return None  # Weak trend

            return {
                'direction': direction,
                'trend_strength': float(trend_strength),
                'mama': float(current_mama),
                'fama': float(current_fama),
                'mama_above_fama': current_mama > current_fama
            }

        except Exception as e:
            self.logger.debug(f"[S10_MESA] Direction error: {e}")
            return None

    # =========================================================================
    # CROSSOVER DETECTION
    # =========================================================================

    def _check_crossover(self, mama: np.ndarray, fama: np.ndarray) -> Optional[Dict]:
        """
        Check for MAMA/FAMA crossover.
        
        Args:
            mama: MAMA array
            fama: FAMA array
            
        Returns:
            Crossover dict or None
        """
        try:
            if len(mama) < 2 or len(fama) < 2:
                return None

            # Check current crossover status
            prev_mama = mama[-2]
            prev_fama = fama[-2]
            curr_mama = mama[-1]
            curr_fama = fama[-1]

            # Bullish crossover: MAMA crosses above FAMA
            if prev_mama <= prev_fama and curr_mama > curr_fama:
                return {
                    'crossover_detected': True,
                    'crossover_type': 'BULLISH',
                    'direction': 'BUY',
                    'crossover_price': float(curr_mama)
                }

            # Bearish crossover: MAMA crosses below FAMA
            elif prev_mama >= prev_fama and curr_mama < curr_fama:
                return {
                    'crossover_detected': True,
                    'crossover_type': 'BEARISH',
                    'direction': 'SELL',
                    'crossover_price': float(curr_mama)
                }

            # No crossover, but check if trend is established
            elif curr_mama > curr_fama:
                return {
                    'crossover_detected': False,
                    'crossover_type': 'BULLISH_TREND',
                    'direction': 'BUY',
                    'crossover_price': float(curr_mama)
                }
            elif curr_mama < curr_fama:
                return {
                    'crossover_detected': False,
                    'crossover_type': 'BEARISH_TREND',
                    'direction': 'SELL',
                    'crossover_price': float(curr_mama)
                }

            return None

        except Exception:
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, trend_info: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            trend_info: Trend information
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = trend_info.get('direction', 'BUY')

            # Check M5 momentum aligns with trend direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                return momentum > 0  # Bullish momentum on M5
            else:
                return momentum < 0  # Bearish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, trend_info: Dict, crossover: Dict,
        cycle_info: Optional[Dict], regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            trend_info: Trend information
            crossover: Crossover information
            cycle_info: Cycle information
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = trend_info.get('direction', 'BUY')
            trend_strength = trend_info.get('trend_strength', 0.5)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            if direction == 'BUY':
                sl_price = entry_price - atr * 2.0
            else:  # SELL
                sl_price = entry_price + atr * 2.0

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, direction == 'BUY',
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Fixed R:R
                risk = abs(entry_price - sl_price)
                if direction == 'BUY':
                    tp_price = entry_price + risk * 2.0
                else:
                    tp_price = entry_price - risk * 2.0

            # Calculate confidence
            cycle_bonus = 0.1 if cycle_info and cycle_info.get('cycle_detected', False) else 0.0
            crossover_bonus = 0.2 if crossover.get('crossover_detected', False) else 0.1

            confidence = min(1.0, 0.3 + trend_strength * 0.3 + crossover_bonus + cycle_bonus)

            # Build signal
            signal = {
                'signal': f'{direction}_MARKET',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'crossover_type': crossover.get('crossover_type', 'UNKNOWN'),
                    'trend_strength': trend_strength,
                    'mama': trend_info.get('mama', 0),
                    'fama': trend_info.get('fama', 0),
                    'cycle_period': cycle_info.get('cycle_period', 0) if cycle_info else 0,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S10_MESA] Signal generated: {direction} | "
                f"Crossover: {crossover.get('crossover_type')} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Cycle: {cycle_info.get('cycle_period', 0) if cycle_info else 0} bars | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S10_MESA] Signal generation error: {e}")
            return self._create_neutral_signal()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_regime_compatible(self, regime_context: Dict) -> bool:
        """
        Check if current regime is compatible with this strategy.
        
        Args:
            regime_context: Current regime information
            
        Returns:
            True if compatible
        """
        regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # TREND strategies work best in trending regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'FALSE_SIDEWAY', 'PRE_BREAKOUT'
        ]

        return regime_name in compatible_regimes

    def _create_neutral_signal(self) -> Dict:
        """Create neutral signal."""
        return {
            'signal': 'NEUTRAL',
            'meta': {
                'strategy': self.strategy_name,
                'strategy_category': self.strategy_category,
                'confidence': 0.0
            }
        }