"""
S20_VFIAccumulation - Volume Flow Indicator Accumulation Strategy.

Trend-following strategy that uses Volume Flow Indicator (VFI) to
detect accumulation patterns and trade with smart money.

Strategy Logic:
  1. Calculate Volume Flow Indicator (VFI)
  2. Detect accumulation patterns (smart money buying)
  3. Confirm trend direction with VFI
  4. Generate entry signal on accumulation confirmation

Volume Flow Indicator (VFI):
  Measures the flow of volume into and out of a security.
  Positive VFI = Volume flowing in (accumulation)
  Negative VFI = Volume flowing out (distribution)

Accumulation Pattern:
  Accumulation occurs when smart money is buying:
    - Price is stable or declining slightly
    - Volume is increasing
    - VFI is positive (money flowing in)
    
  This indicates institutional buying before the real move up.

Distribution Pattern:
  Distribution occurs when smart money is selling:
    - Price is stable or rising slightly
    - Volume is increasing
    - VFI is negative (money flowing out)
    
  This indicates institutional selling before the real move down.

Used Engines:
  - VolumeFlowEngine: VFI calculation and accumulation detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.volume_flow_engine import VolumeFlowEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S20_VFIAccumulation(BaseStrategy):
    """
    Volume Flow Indicator Accumulation Strategy.
    
    This strategy uses VFI to detect accumulation patterns
    and trade with smart money.
    
    VFI Definition:
      Volume Flow Indicator measures the flow of volume into
      and out of a security. Positive VFI indicates accumulation
      (smart money buying), negative indicates distribution.
      
    Accumulation Definition:
      Accumulation occurs when smart money is quietly buying:
        - Price stable or slightly down
        - Volume increasing
        - VFI positive
        
    Entry Criteria:
      - VFI positive and rising
      - Accumulation pattern detected
      - Trend direction confirmed
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S20_VFIAccumulation strategy."""
        super().__init__(
            strategy_name='S20_VFIAccumulation',
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
        self.volume_flow_engine = VolumeFlowEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.vfi_period = 130  # VFI lookback period
        self.vfi_ma_period = 5  # VFI moving average period
        self.ad_lookback = 50  # Accumulation/distribution lookback
        self.min_vfi_threshold = 0.1  # Minimum VFI for signal
        self.min_accumulation_strength = 0.4  # Minimum accumulation strength

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
        Main analysis method for S20_VFIAccumulation.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < 150:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Calculate VFI
            # =========================================================================
            vfi_result = self.volume_flow_engine.calculate_vfi(
                df_m15, period=self.vfi_period, ma_period=self.vfi_ma_period
            )

            if vfi_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Accumulation/Distribution
            # =========================================================================
            accumulation_result = self.volume_flow_engine.detect_accumulation(
                df_m15, lookback=self.ad_lookback
            )

            distribution_result = self.volume_flow_engine.detect_distribution(
                df_m15, lookback=self.ad_lookback
            )

            # =========================================================================
            # STEP 3: Analyze Smart Money
            # =========================================================================
            smart_money = self._analyze_smart_money(
                vfi_result, accumulation_result, distribution_result
            )

            if smart_money is None:
                return default_signal

            # =========================================================================
            # STEP 4: Confirm Trend
            # =========================================================================
            if not self._confirm_trend(df_m15, smart_money):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, smart_money):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, smart_money, vfi_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S20_VFI] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SMART MONEY ANALYSIS
    # =========================================================================

    def _analyze_smart_money(
        self, vfi_result: Dict, accumulation_result: Dict, distribution_result: Dict
    ) -> Optional[Dict]:
        """
        Analyze smart money activity.
        
        Args:
            vfi_result: VFI calculation result
            accumulation_result: Accumulation detection result
            distribution_result: Distribution detection result
            
        Returns:
            Smart money dict or None
        """
        try:
            current_vfi = vfi_result.get('current_vfi', 0)
            vfi_trend = vfi_result.get('vfi_trend', 'NEUTRAL')
            vfi_above_ma = vfi_result.get('vfi_above_ma', False)

            is_accumulating = accumulation_result.get('is_accumulating', False)
            accumulation_strength = accumulation_result.get('accumulation_strength', 0)

            is_distributing = distribution_result.get('is_distributing', False)
            distribution_strength = distribution_result.get('distribution_strength', 0)

            # Determine smart money direction
            if is_accumulating and current_vfi > self.min_vfi_threshold and vfi_above_ma:
                # Accumulation detected with positive VFI
                direction = 'BUY'
                activity_type = 'ACCUMULATION'
                strength = accumulation_strength
            elif is_distributing and current_vfi < -self.min_vfi_threshold and not vfi_above_ma:
                # Distribution detected with negative VFI
                direction = 'SELL'
                activity_type = 'DISTRIBUTION'
                strength = distribution_strength
            else:
                return None  # No clear smart money activity

            # Check minimum strength
            if strength < self.min_accumulation_strength:
                return None

            return {
                'direction': direction,
                'activity_type': activity_type,
                'strength': float(strength),
                'vfi': float(current_vfi),
                'vfi_trend': vfi_trend,
                'vfi_above_ma': vfi_above_ma,
                'is_accumulating': is_accumulating,
                'is_distributing': is_distributing
            }

        except Exception as e:
            self.logger.debug(f"[S20_VFI] Smart money analysis error: {e}")
            return None

    # =========================================================================
    # TREND CONFIRMATION
    # =========================================================================

    def _confirm_trend(self, df: pd.DataFrame, smart_money: Dict) -> bool:
        """
        Confirm trend direction aligns with smart money.
        
        Args:
            df: DataFrame with OHLCV data
            smart_money: Smart money dict
            
        Returns:
            True if trend confirmed
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = smart_money.get('direction', 'BUY')

            # Calculate recent trend
            recent_close = close[-20:]
            trend = recent_close[-1] - recent_close[0]
            trend_pct = trend / recent_close[0] * 100

            # For accumulation (BUY): Trend should be stable or slightly up
            if direction == 'BUY':
                # Price not falling too much (accumulation before move)
                return trend_pct > -2.0
            else:  # SELL (distribution)
                # Price not rising too much (distribution before move)
                return trend_pct < 2.0

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, smart_money: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            smart_money: Smart money dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = smart_money.get('direction', 'BUY')

            # Check M5 momentum aligns with smart money direction
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
        self, df: pd.DataFrame, smart_money: Dict, vfi_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            smart_money: Smart money dict
            vfi_result: VFI calculation result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = smart_money.get('direction', 'BUY')
            strength = smart_money.get('strength', 0.5)

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
            vfi_bonus = min(0.2, abs(smart_money.get('vfi', 0)) * 0.5)
            confidence = min(1.0, 0.4 + strength * 0.3 + vfi_bonus)

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
                    'activity_type': smart_money.get('activity_type', 'UNKNOWN'),
                    'vfi': smart_money.get('vfi', 0),
                    'vfi_trend': smart_money.get('vfi_trend', 'UNKNOWN'),
                    'accumulation_strength': strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S20_VFI] Signal generated: {direction} | "
                f"Activity: {smart_money.get('activity_type')} | "
                f"VFI: {smart_money.get('vfi', 0):.2f} | "
                f"Strength: {strength:.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S20_VFI] Signal generation error: {e}")
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
            'FALSE_SIDEWAY', 'PRE_BREAKOUT',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR'
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