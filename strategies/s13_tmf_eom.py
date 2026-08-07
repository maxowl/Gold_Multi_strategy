"""
S13_TMF_EOM - True Money Flow + Ease of Movement Strategy.

Trend-following strategy that combines True Money Flow (TMF) with
Ease of Movement (EOM) for trend identification and entry timing.

Strategy Logic:
  1. Calculate True Money Flow (TMF) for money flow direction
  2. Calculate Ease of Movement (EOM) for price movement efficiency
  3. Combine both indicators for trend confirmation
  4. Generate entry signal when both align

True Money Flow (TMF):
  Measures the flow of money into and out of a security.
  Formula: TMF = ((Close - Low) - (High - Close)) / (High - Low) * Volume
  
  Positive TMF = Money flowing in (accumulation)
  Negative TMF = Money flowing out (distribution)

Ease of Movement (EOM):
  Measures the relationship between volume and price movement.
  High EOM = Price moves easily (low resistance)
  Low EOM = Price moves with difficulty (high resistance)

Used Engines:
  - VolumeIndicatorsEngine: TMF and EOM calculation
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.volume_indicators import VolumeIndicatorsEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S13_TMF_EOM(BaseStrategy):
    """
    True Money Flow + Ease of Movement Strategy.
    
    This strategy combines True Money Flow (TMF) with Ease of
    Movement (EOM) for trend identification.
    
    TMF Definition:
      True Money Flow measures the flow of money into and out
      of a security, considering the true range of price movement.
      
    EOM Definition:
      Ease of Movement measures how easily price can move.
      High EOM indicates low resistance (easy movement),
      Low EOM indicates high resistance (difficult movement).
      
    Entry Criteria:
      - TMF confirms money flow direction
      - EOM confirms movement efficiency
      - Both indicators align
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S13_TMF_EOM strategy."""
        super().__init__(
            strategy_name='S13_TMF_EOM',
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
        self.volume_engine = VolumeIndicatorsEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.tmf_period = 21  # TMF lookback period
        self.eom_period = 14  # EOM lookback period
        self.tmf_threshold = 0.05  # TMF threshold for signal
        self.eom_threshold = 0  # EOM threshold for signal

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
        Main analysis method for S13_TMF_EOM.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate TMF
            # =========================================================================
            tmf_result = self.volume_engine.calculate_tmf(df_m15, period=self.tmf_period)

            if tmf_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Calculate EOM
            # =========================================================================
            eom_result = self.volume_engine.calculate_eom(df_m15, period=self.eom_period)

            if eom_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Trend
            # =========================================================================
            trend_info = self._detect_trend(tmf_result, eom_result, df_m15)

            if trend_info is None:
                return default_signal

            # =========================================================================
            # STEP 4: Confirm Signal
            # =========================================================================
            if not self._confirm_signal(tmf_result, eom_result, trend_info):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, trend_info):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, trend_info, tmf_result, eom_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S13_TMF] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # TREND DETECTION
    # =========================================================================

    def _detect_trend(
        self, tmf_result: Dict, eom_result: Dict, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Detect trend from TMF and EOM.
        
        Args:
            tmf_result: TMF calculation result
            eom_result: EOM calculation result
            df: DataFrame with OHLCV data
            
        Returns:
            Trend info dict or None
        """
        try:
            tmf = tmf_result.get('tmf', [])
            tmf_trend = tmf_result.get('tmf_trend', 'NEUTRAL')
            current_tmf = tmf_result.get('current_tmf', 0)

            eom = eom_result.get('eom', [])
            eom_trend = eom_result.get('eom_trend', 'NEUTRAL')
            current_eom = eom_result.get('current_eom', 0)

            if len(tmf) == 0 or len(eom) == 0:
                return None

            # Determine direction based on TMF
            if tmf_trend == 'STRONG_BULLISH' or tmf_trend == 'BULLISH':
                direction = 'BUY'
            elif tmf_trend == 'STRONG_BEARISH' or tmf_trend == 'BEARISH':
                direction = 'SELL'
            else:
                return None  # No clear TMF direction

            # Check EOM alignment
            # For BUY: EOM should be positive (easy upward movement)
            # For SELL: EOM should be negative (easy downward movement)
            if direction == 'BUY' and current_eom <= self.eom_threshold:
                return None  # EOM doesn't support BUY
            elif direction == 'SELL' and current_eom >= self.eom_threshold:
                return None  # EOM doesn't support SELL

            # Calculate trend strength
            tmf_strength = abs(current_tmf)
            eom_strength = abs(current_eom) / 100  # Normalize EOM

            trend_strength = min(1.0, tmf_strength * 0.6 + eom_strength * 0.4)

            if trend_strength < 0.3:
                return None  # Weak trend

            return {
                'direction': direction,
                'trend_strength': float(trend_strength),
                'tmf': float(current_tmf),
                'tmf_trend': tmf_trend,
                'eom': float(current_eom),
                'eom_trend': eom_trend
            }

        except Exception as e:
            self.logger.debug(f"[S13_TMF] Trend detection error: {e}")
            return None

    # =========================================================================
    # SIGNAL CONFIRMATION
    # =========================================================================

    def _confirm_signal(
        self, tmf_result: Dict, eom_result: Dict, trend_info: Dict
    ) -> bool:
        """
        Confirm signal with additional checks.
        
        Args:
            tmf_result: TMF calculation result
            eom_result: EOM calculation result
            trend_info: Trend information
            
        Returns:
            True if signal is confirmed
        """
        try:
            direction = trend_info.get('direction', 'BUY')
            current_tmf = tmf_result.get('current_tmf', 0)
            current_eom = eom_result.get('current_eom', 0)

            # Check TMF threshold
            if direction == 'BUY' and current_tmf < self.tmf_threshold:
                return False
            elif direction == 'SELL' and current_tmf > -self.tmf_threshold:
                return False

            # Check EOM alignment
            if direction == 'BUY' and current_eom < 0:
                return False
            elif direction == 'SELL' and current_eom > 0:
                return False

            return True

        except Exception:
            return False

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
        self, df: pd.DataFrame, trend_info: Dict,
        tmf_result: Dict, eom_result: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            trend_info: Trend information
            tmf_result: TMF calculation result
            eom_result: EOM calculation result
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
                sl_price = entry_price - atr * 1.5
            else:  # SELL
                sl_price = entry_price + atr * 1.5

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
            tmf_strength = abs(trend_info.get('tmf', 0))
            confidence = min(1.0, 0.4 + trend_strength * 0.4 + tmf_strength * 0.2)

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
                    'tmf': trend_info.get('tmf', 0),
                    'tmf_trend': trend_info.get('tmf_trend', 'UNKNOWN'),
                    'eom': trend_info.get('eom', 0),
                    'eom_trend': trend_info.get('eom_trend', 'UNKNOWN'),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S13_TMF] Signal generated: {direction} | "
                f"TMF: {trend_info.get('tmf', 0):.3f} | "
                f"EOM: {trend_info.get('eom', 0):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S13_TMF] Signal generation error: {e}")
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