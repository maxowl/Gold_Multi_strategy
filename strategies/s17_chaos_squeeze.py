"""
S17_ChaosSqueeze - Chaos Theory Squeeze Strategy.

Trend-following strategy that uses Chaos Theory indicators and
volatility squeeze detection for breakout trading.

Strategy Logic:
  1. Detect volatility squeeze (BB inside KC)
  2. Calculate momentum for breakout direction
  3. Detect squeeze breakout
  4. Generate entry signal on breakout confirmation

Chaos Theory Concepts:
  Gaussian Squeeze:
    Detects when Bollinger Bands are inside Keltner Channels,
    indicating volatility compression and potential breakout.
    
  Squeeze ON: BB inside KC (low volatility, compression)
  Squeeze OFF: BB outside KC (high volatility, expansion)
  
  Momentum:
    Linear regression slope of price over lookback period.
    Positive momentum = upward breakout expected
    Negative momentum = downward breakout expected

Used Engines:
  - KalmanSqueezeEngine: Squeeze detection and momentum
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.kalman_squeeze_engine import KalmanSqueezeEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S17_ChaosSqueeze(BaseStrategy):
    """
    Chaos Theory Squeeze Strategy.
    
    This strategy uses Chaos Theory indicators and volatility
    squeeze detection for breakout trading.
    
    Squeeze Definition:
      A squeeze occurs when Bollinger Bands are inside Keltner
      Channels, indicating volatility compression. This is
      often followed by a breakout with significant momentum.
      
    Squeeze Phases:
      1. Squeeze ON: BB inside KC (compression)
      2. Squeeze Building: Squeeze count increasing
      3. Squeeze Release: BB crosses outside KC
      4. Breakout: Price moves with momentum
      
    Entry Criteria:
      - Squeeze detected and building
      - Squeeze released (BB outside KC)
      - Momentum confirms direction
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S17_ChaosSqueeze strategy."""
        super().__init__(
            strategy_name='S17_ChaosSqueeze',
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
        self.kalman_engine = KalmanSqueezeEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.bb_period = 20  # Bollinger Bands period
        self.bb_std = 2.0  # Bollinger Bands std deviation
        self.kc_period = 20  # Keltner Channel period
        self.kc_atr_mult = 1.5  # Keltner Channel ATR multiplier
        self.momentum_period = 12  # Momentum lookback
        self.min_squeeze_count = 5  # Minimum squeeze bars
        self.breakout_threshold = 0.5  # Minimum breakout momentum

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
        Main analysis method for S17_ChaosSqueeze.
        
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
            # STEP 1: Detect Squeeze
            # =========================================================================
            squeeze_result = self.kalman_engine.detect_squeeze(df_m15)

            if squeeze_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Check Squeeze Status
            # =========================================================================
            squeeze_on = squeeze_result.get('squeeze_on', False)
            squeeze_count = squeeze_result.get('squeeze_count', 0)
            momentum = squeeze_result.get('momentum', 0)
            breakout_direction = squeeze_result.get('breakout_direction', 'UNKNOWN')

            # Check if squeeze is building or just released
            if squeeze_on and squeeze_count < self.min_squeeze_count:
                return default_signal  # Squeeze not mature yet

            # =========================================================================
            # STEP 3: Calculate Momentum
            # =========================================================================
            momentum_result = self.kalman_engine.calculate_momentum(
                df_m15, period=self.momentum_period
            )

            if momentum_result is None:
                return default_signal

            current_momentum = momentum_result[-1] if len(momentum_result) > 0 else 0

            # =========================================================================
            # STEP 4: Detect Breakout
            # =========================================================================
            breakout = self._detect_breakout(squeeze_result, momentum_result, df_m15)

            if breakout is None or not breakout.get('breakout_detected', False):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, breakout):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, breakout, squeeze_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S17_CHAOS] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # BREAKOUT DETECTION
    # =========================================================================

    def _detect_breakout(
        self, squeeze_result: Dict, momentum_result: np.ndarray, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Detect squeeze breakout.
        
        Args:
            squeeze_result: Squeeze detection result
            momentum_result: Momentum array
            df: DataFrame with OHLCV data
            
        Returns:
            Breakout dict or None
        """
        try:
            squeeze_on = squeeze_result.get('squeeze_on', False)
            squeeze_count = squeeze_result.get('squeeze_count', 0)
            breakout_direction = squeeze_result.get('breakout_direction', 'UNKNOWN')
            current_momentum = momentum_result[-1] if len(momentum_result) > 0 else 0

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Breakout detected when:
            # 1. Squeeze just ended (squeeze_count == 0 and was on before)
            # 2. OR breakout_direction is UP or DOWN
            # 3. Momentum confirms direction

            if breakout_direction == 'UP' and current_momentum > self.breakout_threshold:
                return {
                    'breakout_detected': True,
                    'direction': 'BUY',
                    'momentum': float(current_momentum),
                    'squeeze_count': squeeze_count,
                    'breakout_type': 'UP_BREAKOUT'
                }
            elif breakout_direction == 'DOWN' and current_momentum < -self.breakout_threshold:
                return {
                    'breakout_detected': True,
                    'direction': 'SELL',
                    'momentum': float(current_momentum),
                    'squeeze_count': squeeze_count,
                    'breakout_type': 'DOWN_BREAKOUT'
                }
            elif not squeeze_on and squeeze_count > 0:
                # Squeeze just ended, check momentum for direction
                if current_momentum > self.breakout_threshold:
                    return {
                        'breakout_detected': True,
                        'direction': 'BUY',
                        'momentum': float(current_momentum),
                        'squeeze_count': squeeze_count,
                        'breakout_type': 'SQUEEZE_RELEASE_UP'
                    }
                elif current_momentum < -self.breakout_threshold:
                    return {
                        'breakout_detected': True,
                        'direction': 'SELL',
                        'momentum': float(current_momentum),
                        'squeeze_count': squeeze_count,
                        'breakout_type': 'SQUEEZE_RELEASE_DOWN'
                    }

            return {'breakout_detected': False}

        except Exception as e:
            self.logger.debug(f"[S17_CHAOS] Breakout detection error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, breakout: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            breakout: Breakout dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = breakout.get('direction', 'BUY')

            # Check M5 momentum aligns with breakout direction
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
        self, df: pd.DataFrame, breakout: Dict, squeeze_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            breakout: Breakout dict
            squeeze_result: Squeeze detection result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = breakout.get('direction', 'BUY')
            momentum = breakout.get('momentum', 0)

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
            squeeze_bonus = 0.1 if squeeze_result.get('squeeze_count', 0) >= 10 else 0.0
            momentum_bonus = min(0.2, abs(momentum) / 10.0)

            confidence = min(1.0, 0.4 + squeeze_bonus + momentum_bonus + 0.2)

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
                    'breakout_type': breakout.get('breakout_type', 'UNKNOWN'),
                    'momentum': momentum,
                    'squeeze_count': squeeze_result.get('squeeze_count', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S17_CHAOS] Signal generated: {direction} | "
                f"Breakout: {breakout.get('breakout_type')} | "
                f"Momentum: {momentum:.2f} | "
                f"Squeeze: {squeeze_result.get('squeeze_count', 0)} bars | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S17_CHAOS] Signal generation error: {e}")
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

        # TREND strategies work best in trending and breakout regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'PRE_BREAKOUT', 'FALSE_SIDEWAY',
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