"""
Entry Optimizer - Micro-Account-Only Edition.

Optimizes entry execution by converting Market Orders to Limit Orders
when market conditions favor it. Critical for Micro-Accounts where
every tick of slippage impacts profitability.

Optimization Triggers:
  1. Overextended RSI (pullback expected)
  2. Price far from structural support/resistance
  3. Low momentum (range-bound market)
  4. Wide spread (wait for spread to normalize)

Execution Methods:
  - MARKET: Execute immediately at current price
  - LIMIT: Place limit order at optimized price with expiration

Expiration Logic:
  - SCALP strategies: 15-30 minutes
  - TREND strategies: 45-90 minutes
  - SMC strategies: 30-60 minutes
  - MEAN_REVERSION: 20-40 minutes
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Optional, Tuple

from config import config


class EntryOptimizer:
    """
    Optimizes entry execution for Micro-Account trading.
    
    Converts Market Orders to Limit Orders when conditions favor
    better entry prices, reducing slippage and improving fill quality.
    """

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize EntryOptimizer.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # RSI thresholds for pullback detection
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.rsi_period = 14

        # Minimum distance from current price to limit price (USD)
        self.min_limit_distance = 1.0

        # Maximum distance from current price to limit price (USD)
        self.max_limit_distance = 8.0

        # Expiration times per strategy category (minutes)
        self.expiration_times = {
            'SCALP': (15, 30),
            'TREND': (45, 90),
            'SMC': (30, 60),
            'MEAN_REVERSION': (20, 40),
            'GENERAL': (25, 50)
        }

        # Momentum threshold (ADX)
        self.momentum_threshold = 30  # Above this = strong trend, don't use limit

        # Cache
        self._symbol_info_cache = None
        self._symbol_info_time = 0

        self.logger.info(f"[ENTRY_OPT] Initialized | Min Distance: {self.min_limit_distance} USD")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def optimize_entry(self, signal: Dict, df_m5: pd.DataFrame = None) -> Dict:
        """
        Main entry point: Optimize signal entry method.
        
        Args:
            signal: Signal dict with meta
            df_m5: M5 DataFrame for analysis
            
        Returns:
            Modified signal dict with optimized entry method
        """
        meta = signal.get('meta', {})
        signal_type = signal.get('signal', '')

        # Skip optimization for pending orders (already optimized)
        if 'LIMIT' in signal_type or 'STOP' in signal_type:
            meta['execution_method'] = 'PENDING'
            signal['meta'] = meta
            return signal

        # Skip if no df available
        if df_m5 is None or df_m5.empty or len(df_m5) < 30:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = 'Insufficient data'
            signal['meta'] = meta
            return signal

        # Get current price
        tick = self._get_current_tick()
        if tick is None:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = 'Cannot get tick'
            signal['meta'] = meta
            return signal

        is_buy = 'BUY' in signal_type
        current_price = tick.ask if is_buy else tick.bid
        strategy_category = meta.get('strategy_category', 'GENERAL')

        # =========================================================================
        # CHECK 1: Momentum Filter (Strong trend = Market order)
        # =========================================================================
        momentum = self._calculate_momentum(df_m5)
        if momentum['is_strong_trend']:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = f"Strong trend (ADX {momentum['adx']:.1f})"
            signal['meta'] = meta
            self.logger.info(
                f"[ENTRY_OPT] Market order: Strong trend ADX={momentum['adx']:.1f}"
            )
            return signal

        # =========================================================================
        # CHECK 2: Should convert to limit order?
        # =========================================================================
        should_limit, limit_reason = self._should_convert_to_limit(
            df_m5, is_buy, current_price, strategy_category
        )

        if not should_limit:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = limit_reason
            signal['meta'] = meta
            return signal

        # =========================================================================
        # CHECK 3: Calculate optimal limit price
        # =========================================================================
        limit_price = self._calculate_limit_price(
            df_m5, is_buy, current_price, strategy_category
        )

        if limit_price is None or limit_price <= 0:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = 'Failed to calculate limit price'
            signal['meta'] = meta
            return signal

        # Validate distance
        distance = abs(current_price - limit_price)
        if distance < self.min_limit_distance:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = f'Limit too close ({distance:.2f} USD)'
            signal['meta'] = meta
            return signal

        if distance > self.max_limit_distance:
            # Cap the distance
            if is_buy:
                limit_price = current_price - self.max_limit_distance
            else:
                limit_price = current_price + self.max_limit_distance

        # Validate against broker stops level
        limit_price = self._validate_against_stops_level(limit_price, current_price, is_buy)
        if limit_price is None:
            meta['execution_method'] = 'MARKET'
            meta['optimization_reason'] = 'Limit fails broker validation'
            signal['meta'] = meta
            return signal

        # =========================================================================
        # CHECK 4: Calculate expiration
        # =========================================================================
        expiration_minutes = self._calculate_expiration(strategy_category, momentum)

        # Update signal for limit order execution
        signal['signal'] = 'BUY_LIMIT' if is_buy else 'SELL_LIMIT'
        meta['execution_method'] = 'LIMIT'
        meta['optimized_limit_price'] = round(limit_price, 2)
        meta['limit_expiration_minutes'] = expiration_minutes
        meta['optimization_reason'] = limit_reason
        meta['limit_distance_usd'] = round(distance, 2)

        # Recalculate SL/TP based on limit entry
        original_entry = meta.get('entry_price', current_price)
        sl_price = meta.get('sl_price', 0)
        tp_price = meta.get('tp_price', 0)

        if sl_price > 0:
            # Adjust SL by the same offset
            offset = limit_price - original_entry
            meta['sl_price'] = round(sl_price + offset, 2)

        if tp_price > 0:
            offset = limit_price - original_entry
            meta['tp_price'] = round(tp_price + offset, 2)

        meta['entry_price'] = round(limit_price, 2)
        signal['meta'] = meta

        self.logger.info(
            f"[ENTRY_OPT] Converted to LIMIT | "
            f"Price: {current_price:.2f} -> {limit_price:.2f} | "
            f"Distance: {distance:.2f} USD | "
            f"Expiration: {expiration_minutes}min | "
            f"Reason: {limit_reason}"
        )

        return signal

    # =========================================================================
    # MOMENTUM CALCULATION
    # =========================================================================

    def _calculate_momentum(self, df: pd.DataFrame) -> Dict:
        """
        Calculate market momentum using ADX.
        
        Returns:
            Dict with 'adx', 'is_strong_trend'
        """
        result = {'adx': 0.0, 'is_strong_trend': False}

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 30:
                return result

            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            # Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            # Smoothed values
            period = 14
            atr = pd.Series(tr).rolling(period).mean().values
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() /
                             (pd.Series(tr).rolling(period).mean() + 1e-10)).values
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() /
                              (pd.Series(tr).rolling(period).mean() + 1e-10)).values

            # DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = pd.Series(dx).rolling(period).mean().values

            current_adx = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
            result['adx'] = current_adx
            result['is_strong_trend'] = current_adx >= self.momentum_threshold

        except Exception as e:
            self.logger.debug(f"[ENTRY_OPT] Momentum calculation error: {e}")

        return result

    # =========================================================================
    # LIMIT ORDER DECISION
    # =========================================================================

    def _should_convert_to_limit(self, df: pd.DataFrame, is_buy: bool,
                                   current_price: float,
                                   strategy_category: str) -> Tuple[bool, str]:
        """
        Determine if market order should be converted to limit order.
        
        Returns:
            Tuple of (should_convert, reason)
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 20:
                return False, 'Insufficient data'

            # =========================================================================
            # CHECK 1: RSI Pullback Detection
            # =========================================================================
            rsi = self._calculate_rsi(close, self.rsi_period)
            if rsi is not None and not np.isnan(rsi[-1]):
                current_rsi = rsi[-1]

                if is_buy and current_rsi >= self.rsi_overbought:
                    return True, f'RSI overbought ({current_rsi:.1f}) - pullback expected'
                elif not is_buy and current_rsi <= self.rsi_oversold:
                    return True, f'RSI oversold ({current_rsi:.1f}) - bounce expected'

            # =========================================================================
            # CHECK 2: Price Relative to Bollinger Bands
            # =========================================================================
            bb_position = self._calculate_bb_position(close)
            if bb_position is not None:
                if is_buy and bb_position > 0.9:
                    return True, f'Price at upper BB ({bb_position:.2f}) - pullback expected'
                elif not is_buy and bb_position < 0.1:
                    return True, f'Price at lower BB ({bb_position:.2f}) - bounce expected'

            # =========================================================================
            # CHECK 3: Wide Spread
            # =========================================================================
            tick = self._get_current_tick()
            if tick:
                symbol_info = self._get_symbol_info()
                if symbol_info:
                    spread = (tick.ask - tick.bid) / symbol_info.point
                    max_spread = config.max_spread_points
                    if spread > max_spread * 1.5:
                        return True, f'Wide spread ({spread:.0f} pts) - wait for normalization'

            # =========================================================================
            # CHECK 4: Structure-Based Entry (SMC strategies)
            # =========================================================================
            if strategy_category == 'SMC':
                structure_level = self._find_structure_level(df, is_buy)
                if structure_level and structure_level != current_price:
                    distance = abs(current_price - structure_level)
                    if self.min_limit_distance < distance < self.max_limit_distance:
                        return True, f'Structure level at {structure_level:.2f}'

            return False, 'Conditions favor market order'

        except Exception as e:
            self.logger.error(f"[ENTRY_OPT] Limit decision error: {e}")
            return False, f'Error: {str(e)}'

    # =========================================================================
    # LIMIT PRICE CALCULATION
    # =========================================================================

    def _calculate_limit_price(self, df: pd.DataFrame, is_buy: bool,
                                current_price: float,
                                strategy_category: str) -> Optional[float]:
        """
        Calculate optimal limit price.
        
        Priority:
          1. Structure level (OB, Swing High/Low, VWAP)
          2. RSI-based pullback
          3. Bollinger Band mean
          4. Fallback: Fixed distance from current price
        """
        try:
            close = df['close'].values.astype(float)

            # Priority 1: Structure level
            structure_level = self._find_structure_level(df, is_buy)
            if structure_level:
                # Adjust for spread
                symbol_info = self._get_symbol_info()
                if symbol_info:
                    tick = self._get_current_tick()
                    if tick:
                        spread = tick.ask - tick.bid
                        if is_buy:
                            # BUY LIMIT: Price below current, add spread to compensate
                            adjusted = structure_level + (spread * 0.3)
                        else:
                            # SELL LIMIT: Price above current, subtract spread
                            adjusted = structure_level - (spread * 0.3)
                        return adjusted
                return structure_level

            # Priority 2: RSI-based pullback (0.382 retracement)
            rsi = self._calculate_rsi(close, self.rsi_period)
            if rsi is not None and not np.isnan(rsi[-1]):
                if is_buy and rsi[-1] >= self.rsi_overbought:
                    # Find recent high and target 0.382 retracement
                    recent_high = np.max(close[-20:])
                    recent_low = np.min(close[-10:])
                    range_ = recent_high - recent_low
                    pullback_target = recent_high - (range_ * 0.382)
                    if current_price - pullback_target >= self.min_limit_distance:
                        return pullback_target

                elif not is_buy and rsi[-1] <= self.rsi_oversold:
                    recent_low = np.min(close[-20:])
                    recent_high = np.max(close[-10:])
                    range_ = recent_high - recent_low
                    bounce_target = recent_low + (range_ * 0.382)
                    if bounce_target - current_price >= self.min_limit_distance:
                        return bounce_target

            # Priority 3: Bollinger Band mean (middle band)
            if len(close) >= 20:
                sma = np.mean(close[-20:])
                if is_buy and current_price - sma >= self.min_limit_distance:
                    return sma
                elif not is_buy and sma - current_price >= self.min_limit_distance:
                    return sma

            # Priority 4: Fallback - fixed distance based on strategy
            fallback_distance = {
                'SCALP': 2.0,
                'TREND': 4.0,
                'SMC': 3.0,
                'MEAN_REVERSION': 2.5,
                'GENERAL': 3.0
            }.get(strategy_category, 3.0)

            if is_buy:
                return current_price - fallback_distance
            else:
                return current_price + fallback_distance

        except Exception as e:
            self.logger.error(f"[ENTRY_OPT] Limit price calculation error: {e}")
            return None

    # =========================================================================
    # STRUCTURE ANALYSIS
    # =========================================================================

    def _find_structure_level(self, df: pd.DataFrame, is_buy: bool) -> Optional[float]:
        """
        Find relevant structural level for limit entry.
        
        BUY: Look for support (swing low, bullish OB)
        SELL: Look for resistance (swing high, bearish OB)
        """
        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            if len(close) < 20:
                return None

            current_price = close[-1]

            # Look at last 30 bars for structure
            lookback = min(30, len(close))
            recent_high = high[-lookback:]
            recent_low = low[-lookback:]

            if is_buy:
                # Find recent swing lows (support levels)
                swing_lows = []
                for i in range(2, lookback - 2):
                    if (recent_low[i] < recent_low[i-1] and
                        recent_low[i] < recent_low[i-2] and
                        recent_low[i] < recent_low[i+1] and
                        recent_low[i] < recent_low[i+2]):
                        swing_lows.append(recent_low[i])

                if not swing_lows:
                    return None

                # Find closest swing low below current price
                valid_lows = [sl for sl in swing_lows if sl < current_price]
                if not valid_lows:
                    return None

                # Return the highest valid low (closest support)
                return max(valid_lows)

            else:
                # Find recent swing highs (resistance levels)
                swing_highs = []
                for i in range(2, lookback - 2):
                    if (recent_high[i] > recent_high[i-1] and
                        recent_high[i] > recent_high[i-2] and
                        recent_high[i] > recent_high[i+1] and
                        recent_high[i] > recent_high[i+2]):
                        swing_highs.append(recent_high[i])

                if not swing_highs:
                    return None

                # Find closest swing high above current price
                valid_highs = [sh for sh in swing_highs if sh > current_price]
                if not valid_highs:
                    return None

                # Return the lowest valid high (closest resistance)
                return min(valid_highs)

        except Exception as e:
            self.logger.debug(f"[ENTRY_OPT] Structure analysis error: {e}")
            return None

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        """Calculate RSI using Wilder's smoothing."""
        try:
            if len(close) < period + 1:
                return None

            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            rsi = np.zeros(len(close))
            rsi[:period] = np.nan

            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi[i + 1] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

            return np.clip(rsi, 0, 100)

        except Exception:
            return None

    def _calculate_bb_position(self, close: np.ndarray) -> Optional[float]:
        """
        Calculate Bollinger Band %B (position within bands).
        
        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        """
        try:
            if len(close) < 20:
                return None

            period = 20
            std_mult = 2.0

            sma = np.mean(close[-period:])
            std = np.std(close[-period:], ddof=1)

            upper = sma + (std_mult * std)
            lower = sma - (std_mult * std)

            if upper == lower:
                return 0.5

            current_price = close[-1]
            position = (current_price - lower) / (upper - lower)

            return max(0.0, min(1.0, position))

        except Exception:
            return None

    # =========================================================================
    # EXPIRATION CALCULATION
    # =========================================================================

    def _calculate_expiration(self, strategy_category: str,
                                momentum: Dict) -> int:
        """
        Calculate expiration time for limit order.
        
        Factors:
          - Strategy category
          - Momentum (shorter in fast market)
        """
        base_min, base_max = self.expiration_times.get(
            strategy_category, (25, 50)
        )

        # Use midpoint as base
        expiration = (base_min + base_max) / 2

        # Adjust for momentum
        if momentum.get('is_strong_trend', False):
            # Reduce expiration in trending markets
            expiration *= 0.7
        elif momentum.get('adx', 0) < 15:
            # Extend in ranging markets
            expiration *= 1.2

        # Round to nearest 5 minutes
        expiration = int(round(expiration / 5) * 5)

        # Clamp to range
        return max(base_min, min(base_max, expiration))

    # =========================================================================
    # BROKER VALIDATION
    # =========================================================================

    def _validate_against_stops_level(self, limit_price: float,
                                       current_price: float,
                                       is_buy: bool) -> Optional[float]:
        """
        Validate limit price against broker's stops level.
        
        Returns:
            Validated limit price or None if invalid
        """
        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return limit_price

        point = getattr(symbol_info, 'point', 0.01)
        stops_level = getattr(symbol_info, 'trade_stops_level', 10)

        # Minimum distance in price units
        min_distance = stops_level * point + (point * 2)  # Add buffer

        distance = abs(current_price - limit_price)
        if distance < min_distance:
            # Adjust limit price to minimum distance
            if is_buy:
                return current_price - min_distance
            else:
                return current_price + min_distance

        return limit_price

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_current_tick(self):
        """Get current tick with 0.5-second cache."""
        current_time = time.time()
        if hasattr(self, '_tick_cache') and hasattr(self, '_tick_time'):
            if current_time - self._tick_time < 0.5:
                return self._tick_cache

        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self._tick_cache = tick
            self._tick_time = current_time
        return tick

    def _get_symbol_info(self):
        """Get symbol info with 5-second cache."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_time < 5.0):
            return self._symbol_info_cache

        info = mt5.symbol_info(self.symbol)
        if info is None:
            if mt5.symbol_select(self.symbol, True):
                info = mt5.symbol_info(self.symbol)

        if info:
            self._symbol_info_cache = info
            self._symbol_info_time = current_time

        return info