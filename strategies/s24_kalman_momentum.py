"""
S24_KalmanMomentum - Kalman Filter Momentum Strategy.

Trend-following strategy that uses Kalman filter for optimal momentum
tracking and trend identification.

Strategy Logic:
  1. Apply Kalman filter to price momentum
  2. Track filtered momentum state
  3. Detect momentum changes and trend shifts
  4. Generate entry signal on momentum confirmation

Kalman Filter Definition:
  The Kalman filter is an optimal recursive filter that estimates
  the true state of a system from noisy measurements. It provides:
    - Optimal state estimation
    - Noise reduction
    - Real-time updates
    
  For momentum tracking:
    - State = True momentum
    - Measurement = Observed price change
    - Filter output = Smoothed momentum

Kalman Filter Equations:
  Predict: x_pred = x_prev, P_pred = P_prev + Q
  Update: K = P_pred / (P_pred + R), x = x_pred + K*(z - x_pred)
  
  Where:
    x = State estimate (momentum)
    P = Error covariance
    Q = Process noise
    R = Measurement noise
    K = Kalman gain
    z = Measurement (price change)

Used Engines:
  - KalmanSqueezeEngine: Kalman filter application
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


class S24_KalmanMomentum(BaseStrategy):
    """
    Kalman Filter Momentum Strategy.
    
    This strategy uses Kalman filter for optimal momentum tracking
    and trend identification.
    
    Kalman Filter Definition:
      An optimal recursive filter that estimates the true momentum
      from noisy price measurements, providing smooth and accurate
      momentum estimates.
      
    Kalman Momentum Benefits:
      - Noise reduction: Filters out price noise
      - Optimal estimation: Minimum variance estimate
      - Adaptive: Adjusts to changing market conditions
      - Real-time: Updates with each new price
      
    Entry Criteria:
      - Kalman-filtered momentum positive (BUY) or negative (SELL)
      - Momentum above threshold
      - Trend direction confirmed
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S24_KalmanMomentum strategy."""
        super().__init__(
            strategy_name='S24_KalmanMomentum',
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
        self.kalman_lookback = 50  # Lookback for Kalman filter
        self.process_noise = 0.01  # Q - process noise covariance
        self.measurement_noise = 0.1  # R - measurement noise covariance
        self.momentum_threshold = 0.5  # Minimum momentum for signal
        self.trend_period = 20  # Trend confirmation period

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
        Main analysis method for S24_KalmanMomentum.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < 100:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Apply Kalman Filter
            # =========================================================================
            close = df_m15['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            kalman_result = self._apply_kalman(close)

            if kalman_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Calculate Momentum
            # =========================================================================
            momentum_result = self._calculate_momentum(kalman_result)

            if momentum_result is None:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Trend
            # =========================================================================
            trend_result = self._detect_trend(momentum_result, df_m15)

            if trend_result is None:
                return default_signal

            # =========================================================================
            # STEP 4: Detect Momentum Change
            # =========================================================================
            momentum_change = self._detect_momentum_change(momentum_result)

            if momentum_change is None:
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, trend_result):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, trend_result, momentum_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S24_KALMAN] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # KALMAN FILTER APPLICATION
    # =========================================================================

    def _apply_kalman(self, close: np.ndarray) -> Optional[Dict]:
        """
        Apply Kalman filter to price data.
        
        Args:
            close: Close price array
            
        Returns:
            Kalman result dict or None
        """
        try:
            n = len(close)
            if n < self.kalman_lookback:
                return None

            # Calculate price changes (measurements)
            changes = np.diff(close)
            changes = np.insert(changes, 0, 0)

            # Apply Kalman filter
            kalman_engine = self.kalman_engine
            filtered = kalman_engine.apply_kalman_filter(
                changes, process_noise=self.process_noise,
                measurement_noise=self.measurement_noise
            )

            if filtered is None:
                return None

            return {
                'filtered': filtered,
                'raw': changes,
                'current_filtered': float(filtered[-1]),
                'current_raw': float(changes[-1])
            }

        except Exception as e:
            self.logger.debug(f"[S24_KALMAN] Kalman application error: {e}")
            return None

    # =========================================================================
    # MOMENTUM CALCULATION
    # =========================================================================

    def _calculate_momentum(self, kalman_result: Dict) -> Optional[Dict]:
        """
        Calculate momentum from Kalman-filtered data.
        
        Args:
            kalman_result: Kalman filter result
            
        Returns:
            Momentum dict or None
        """
        try:
            filtered = kalman_result.get('filtered', [])
            if len(filtered) == 0:
                return None

            current_momentum = filtered[-1]
            prev_momentum = filtered[-5] if len(filtered) > 5 else filtered[0]

            # Calculate momentum change
            momentum_change = current_momentum - prev_momentum

            # Calculate momentum strength
            momentum_std = np.std(filtered[-20:]) if len(filtered) >= 20 else np.std(filtered)
            momentum_strength = abs(current_momentum) / (momentum_std + 1e-10)

            return {
                'current_momentum': float(current_momentum),
                'prev_momentum': float(prev_momentum),
                'momentum_change': float(momentum_change),
                'momentum_strength': float(momentum_strength),
                'momentum_series': filtered
            }

        except Exception as e:
            self.logger.debug(f"[S24_KALMAN] Momentum calculation error: {e}")
            return None

    # =========================================================================
    # TREND DETECTION
    # =========================================================================

    def _detect_trend(self, momentum_result: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect trend from Kalman-filtered momentum.
        
        Args:
            momentum_result: Momentum calculation result
            df: DataFrame with OHLCV data
            
        Returns:
            Trend dict or None
        """
        try:
            current_momentum = momentum_result.get('current_momentum', 0)
            momentum_strength = momentum_result.get('momentum_strength', 0)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Determine direction based on momentum
            if current_momentum > self.momentum_threshold:
                direction = 'BUY'
            elif current_momentum < -self.momentum_threshold:
                direction = 'SELL'
            else:
                return None  # No clear trend

            # Confirm with price trend
            recent_close = close[-self.trend_period:]
            price_trend = recent_close[-1] - recent_close[0]

            if direction == 'BUY' and price_trend <= 0:
                return None  # Momentum says BUY but price is falling
            elif direction == 'SELL' and price_trend >= 0:
                return None  # Momentum says SELL but price is rising

            return {
                'direction': direction,
                'momentum': float(current_momentum),
                'momentum_strength': float(momentum_strength),
                'price_trend': float(price_trend)
            }

        except Exception as e:
            self.logger.debug(f"[S24_KALMAN] Trend detection error: {e}")
            return None

    # =========================================================================
    # MOMENTUM CHANGE DETECTION
    # =========================================================================

    def _detect_momentum_change(self, momentum_result: Dict) -> Optional[Dict]:
        """
        Detect momentum changes.
        
        Args:
            momentum_result: Momentum calculation result
            
        Returns:
            Momentum change dict or None
        """
        try:
            momentum_change = momentum_result.get('momentum_change', 0)
            current_momentum = momentum_result.get('current_momentum', 0)

            # Detect momentum shift
            if current_momentum > 0 and momentum_change > 0:
                change_type = 'ACCELERATING_UP'
            elif current_momentum > 0 and momentum_change < 0:
                change_type = 'DECELERATING_UP'
            elif current_momentum < 0 and momentum_change < 0:
                change_type = 'ACCELERATING_DOWN'
            elif current_momentum < 0 and momentum_change > 0:
                change_type = 'DECELERATING_DOWN'
            else:
                change_type = 'NEUTRAL'

            return {
                'change_type': change_type,
                'momentum_change': float(momentum_change),
                'current_momentum': float(current_momentum)
            }

        except Exception as e:
            self.logger.debug(f"[S24_KALMAN] Momentum change error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, trend_result: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            trend_result: Trend detection result
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = trend_result.get('direction', 'BUY')

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
        self, df: pd.DataFrame, trend_result: Dict, momentum_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            trend_result: Trend detection result
            momentum_result: Momentum calculation result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = trend_result.get('direction', 'BUY')
            momentum_strength = trend_result.get('momentum_strength', 0.5)

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
            momentum_bonus = min(0.2, momentum_strength * 0.2)
            confidence = min(1.0, 0.4 + momentum_bonus + 0.2)

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
                    'momentum': trend_result.get('momentum', 0),
                    'momentum_strength': momentum_strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S24_KALMAN] Signal generated: {direction} | "
                f"Momentum: {trend_result.get('momentum', 0):.2f} | "
                f"Strength: {momentum_strength:.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S24_KALMAN] Signal generation error: {e}")
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