"""
Dynamic Stops Manager - Intelligent Trailing Stop Management.

Manages dynamic stop loss adjustments based on profit, regime, and volatility.
Core component of Layer 2 in the 10-Layer Active Position Management system.

Critical for Micro-Account trading where:
  - Fast profit locking is essential
  - Breakeven triggers are regime-adaptive
  - Trailing increments match market volatility
  - Strategy category determines trailing method

Trailing Methods:
  1. Chandelier Exit: ATR-based (for TREND strategies)
  2. Fixed USD Increment: Regime-adaptive (for SCALP/SMC/MEAN_REVERSION)
  3. ATR Multiple: Volatility-based (fallback)

Breakeven Triggers (Regime-Adaptive):
  - Strong Trend: 14 USD profit
  - Parabolic/Panic: 18 USD profit
  - Consolidating: 10 USD profit
  - Sideways: 7 USD profit
  - Choppy: 6 USD profit
  - Reversal: 9 USD profit

Trailing Increment (Regime-Adaptive):
  - Strong Trend: 5 USD per increment
  - Parabolic/Panic: 7 USD per increment
  - Consolidating: 4 USD per increment
  - Sideways: 3 USD per increment
  - Choppy: 2 USD per increment (aggressive trailing)
  - Reversal: 4 USD per increment
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from config import config
from core.atr_cache import ATRCache


class DynamicStopsManager:
    """
    Manages dynamic stop loss adjustments for active positions.
    
    Features:
      - Regime-adaptive breakeven triggers
      - Strategy-category-based trailing method
      - Volatility-adaptive trailing increments
      - Broker stops level validation
      - Partial close coordination
      - Comprehensive statistics
    """

    # Trailing methods
    METHOD_CHANDELIER = 'CHANDELIER'
    METHOD_FIXED_USD = 'FIXED_USD'
    METHOD_ATR_MULTIPLE = 'ATR_MULTIPLE'
    METHOD_VOLATILITY_ADAPTIVE = 'VOLATILITY_ADAPTIVE'

    # Stop update reasons
    REASON_BREAKEVEN = 'BREAKEVEN'
    REASON_TRAILING = 'TRAILING'
    REASON_VOLATILITY_ADJUST = 'VOLATILITY_ADJUST'
    REASON_TIME_DECAY = 'TIME_DECAY'
    REASON_MANUAL = 'MANUAL'

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize DynamicStopsManager.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Regime-adaptive breakeven triggers (USD)
        self.be_triggers = {
            'STRONG_TREND': config.be_strong_trend_usd,      # 14.0
            'PARABOLIC': config.be_parabolic_usd,             # 18.0
            'CONSOLIDATING': config.be_consolidating_usd,     # 10.0
            'SIDEWAYS': config.be_sideways_usd,               # 7.0
            'CHOPPY': config.be_choppy_usd,                   # 6.0
            'REVERSAL': config.be_reversal_usd,               # 9.0
            'DEFAULT': 8.0
        }

        # Regime-adaptive trailing increments (USD)
        self.trail_increments = {
            'STRONG_TREND': config.trail_increment_usd,       # 5.0
            'PARABOLIC': 7.0,
            'CONSOLIDATING': 4.0,
            'SIDEWAYS': 3.0,
            'CHOPPY': 2.0,
            'REVERSAL': 4.0,
            'DEFAULT': 5.0
        }

        # Trailing method preference per strategy category
        self.method_preference = {
            'TREND': self.METHOD_CHANDELIER,
            'SCALP': self.METHOD_FIXED_USD,
            'SMC': self.METHOD_FIXED_USD,
            'MEAN_REVERSION': self.METHOD_FIXED_USD,
            'GENERAL': self.METHOD_VOLATILITY_ADAPTIVE
        }

        # Chandelier parameters
        self.chandelier_period = 22
        self.chandelier_atr_mult = 2.0

        # ATR trailing parameters
        self.atr_trail_mult = 1.5

        # Breakeven buffer (USD)
        self.breakeven_buffer = 0.5

        # Statistics
        self._stats = {
            'total_updates': 0,
            'breakeven_count': 0,
            'trailing_count': 0,
            'volatility_adjust_count': 0,
            'rejected_count': 0
        }

        # Cache
        self._symbol_info_cache = None
        self._symbol_info_time = 0

        self.logger.info(
            f"[DYNAMIC_STOPS] Initialized for {symbol} | "
            f"BE Triggers: {self.be_triggers} | "
            f"Trail Increments: {self.trail_increments}"
        )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def update_dynamic_stops(
        self,
        position: Dict,
        current_price: float,
        df: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Optional[Dict]:
        """
        Main entry point: Calculate new stop loss for a position.
        
        Args:
            position: Position dict from StateManager
            current_price: Current market price
            df: DataFrame for ATR/volatility calculations
            regime_context: Current regime information
            
        Returns:
            Dict with new stop info, or None if no update needed
        """
        # Extract position info
        entry_price = position.get('entry_price', 0)
        current_sl = position.get('sl', 0)
        position_type = position.get('position_type', 'BUY')
        meta = position.get('meta_data', {})

        if entry_price <= 0 or current_price <= 0:
            return None

        is_buy = position_type == 'BUY'

        # Calculate current profit in USD (price units)
        if is_buy:
            profit_usd = current_price - entry_price
        else:
            profit_usd = entry_price - current_price

        # Get regime name
        regime_name = 'UNKNOWN'
        if regime_context:
            regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # Get regime category
        regime_category = self._categorize_regime(regime_name)

        # Get strategy category
        strategy_category = meta.get('strategy_category', 'GENERAL')

        # Get ATR
        atr = self._calculate_atr(df) if df is not None else 5.0

        # =========================================================================
        # CHECK 1: Breakeven Trigger
        # =========================================================================
        be_result = self._check_breakeven(
            entry_price, current_sl, profit_usd, is_buy, regime_category
        )

        if be_result['should_update']:
            new_sl = be_result['new_sl']
            validated_sl = self._validate_stop(new_sl, current_price, is_buy)

            if validated_sl and validated_sl != current_sl:
                self._stats['breakeven_count'] += 1
                self._stats['total_updates'] += 1

                return {
                    'new_sl': validated_sl,
                    'old_sl': current_sl,
                    'reason': self.REASON_BREAKEVEN,
                    'method': 'BREAKEVEN',
                    'profit_usd': round(profit_usd, 2),
                    'be_trigger': be_result['be_trigger'],
                    'regime_category': regime_category
                }

        # =========================================================================
        # CHECK 2: Trailing Stop Update
        # =========================================================================
        trail_result = self._calculate_trailing_stop(
            entry_price, current_sl, current_price, profit_usd, is_buy,
            df, regime_category, strategy_category, atr
        )

        if trail_result['should_update']:
            new_sl = trail_result['new_sl']
            validated_sl = self._validate_stop(new_sl, current_price, is_buy)

            if validated_sl and validated_sl != current_sl:
                # Ensure trailing stop only moves in favorable direction
                if is_buy and validated_sl <= current_sl:
                    return None
                if not is_buy and validated_sl >= current_sl:
                    return None

                self._stats['trailing_count'] += 1
                self._stats['total_updates'] += 1

                return {
                    'new_sl': validated_sl,
                    'old_sl': current_sl,
                    'reason': self.REASON_TRAILING,
                    'method': trail_result['method'],
                    'profit_usd': round(profit_usd, 2),
                    'increment': trail_result.get('increment', 0),
                    'regime_category': regime_category,
                    'strategy_category': strategy_category
                }

        # No update needed
        return None

    # =========================================================================
    # BREAKEVEN CHECK
    # =========================================================================

    def _check_breakeven(
        self,
        entry_price: float,
        current_sl: float,
        profit_usd: float,
        is_buy: bool,
        regime_category: str
    ) -> Dict:
        """
        Check if breakeven should be triggered.
        
        Returns:
            Dict with should_update, new_sl, be_trigger
        """
        # Get regime-specific trigger
        be_trigger = self.be_triggers.get(regime_category, self.be_triggers['DEFAULT'])

        # Check if profit threshold reached
        if profit_usd < be_trigger:
            return {'should_update': False, 'new_sl': 0, 'be_trigger': be_trigger}

        # Check if SL is still behind entry (not yet at breakeven)
        is_sl_behind = (
            current_sl == 0.0 or
            (is_buy and current_sl < entry_price) or
            (not is_buy and current_sl > entry_price)
        )

        if not is_sl_behind:
            return {'should_update': False, 'new_sl': 0, 'be_trigger': be_trigger}

        # Calculate new SL (entry + buffer)
        if is_buy:
            new_sl = entry_price + self.breakeven_buffer
        else:
            new_sl = entry_price - self.breakeven_buffer

        self.logger.debug(
            f"[DYNAMIC_STOPS] Breakeven triggered | "
            f"Profit: {profit_usd:.2f} USD >= {be_trigger:.2f} USD | "
            f"Regime: {regime_category}"
        )

        return {
            'should_update': True,
            'new_sl': new_sl,
            'be_trigger': be_trigger
        }

    # =========================================================================
    # TRAILING STOP CALCULATION
    # =========================================================================

    def _calculate_trailing_stop(
        self,
        entry_price: float,
        current_sl: float,
        current_price: float,
        profit_usd: float,
        is_buy: bool,
        df: pd.DataFrame,
        regime_category: str,
        strategy_category: str,
        atr: float
    ) -> Dict:
        """
        Calculate trailing stop based on strategy and regime.
        
        Returns:
            Dict with should_update, new_sl, method, increment
        """
        # Get preferred method for strategy category
        preferred_method = self.method_preference.get(
            strategy_category, self.METHOD_VOLATILITY_ADAPTIVE
        )

        # Route to appropriate method
        if preferred_method == self.METHOD_CHANDELIER and strategy_category == 'TREND':
            return self._calculate_chandelier_exit(
                df, entry_price, current_sl, current_price, is_buy, atr
            )
        elif preferred_method == self.METHOD_FIXED_USD:
            return self._calculate_fixed_increment(
                entry_price, current_sl, profit_usd, is_buy, regime_category
            )
        elif preferred_method == self.METHOD_ATR_MULTIPLE:
            return self._calculate_atr_trailing(
                entry_price, current_sl, current_price, is_buy, atr
            )
        else:
            return self._calculate_volatility_adaptive(
                entry_price, current_sl, profit_usd, is_buy, df, regime_category
            )

    def _calculate_chandelier_exit(
        self,
        df: pd.DataFrame,
        entry_price: float,
        current_sl: float,
        current_price: float,
        is_buy: bool,
        atr: float
    ) -> Dict:
        """
        Calculate Chandelier Exit (ATR-based trailing for TREND strategies).
        
        Formula:
          BUY:  SL = Highest High - (ATR * multiplier)
          SELL: SL = Lowest Low + (ATR * multiplier)
        """
        if df is None or df.empty or len(df) < self.chandelier_period:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_CHANDELIER}

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            period = min(self.chandelier_period, len(df))

            if is_buy:
                highest_high = np.max(high[-period:])
                new_sl = highest_high - (atr * self.chandelier_atr_mult)

                # Only update if new SL is higher than current
                if new_sl > current_sl and new_sl < current_price:
                    return {
                        'should_update': True,
                        'new_sl': new_sl,
                        'method': self.METHOD_CHANDELIER,
                        'increment': atr * self.chandelier_atr_mult
                    }
            else:
                lowest_low = np.min(low[-period:])
                new_sl = lowest_low + (atr * self.chandelier_atr_mult)

                # Only update if new SL is lower than current
                if (current_sl == 0 or new_sl < current_sl) and new_sl > current_price:
                    return {
                        'should_update': True,
                        'new_sl': new_sl,
                        'method': self.METHOD_CHANDELIER,
                        'increment': atr * self.chandelier_atr_mult
                    }

        except Exception as e:
            self.logger.debug(f"[DYNAMIC_STOPS] Chandelier calculation error: {e}")

        return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_CHANDELIER}

    def _calculate_fixed_increment(
        self,
        entry_price: float,
        current_sl: float,
        profit_usd: float,
        is_buy: bool,
        regime_category: str
    ) -> Dict:
        """
        Calculate fixed USD increment trailing stop.
        
        Used by SCALP, SMC, MEAN_REVERSION strategies.
        Increment is regime-adaptive.
        """
        # Get regime-specific increment
        increment = self.trail_increments.get(regime_category, self.trail_increments['DEFAULT'])

        # Get breakeven trigger for this regime
        be_trigger = self.be_triggers.get(regime_category, self.be_triggers['DEFAULT'])

        # Only trail after breakeven is reached
        if profit_usd < be_trigger:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_FIXED_USD}

        # Calculate number of increments
        profit_beyond_be = profit_usd - be_trigger
        num_increments = int(profit_beyond_be / increment)

        if num_increments < 1:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_FIXED_USD}

        # Calculate new SL
        if is_buy:
            new_sl = entry_price + self.breakeven_buffer + (num_increments * increment)
        else:
            new_sl = entry_price - self.breakeven_buffer - (num_increments * increment)

        # Only update if better than current SL
        if is_buy:
            if new_sl > current_sl:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_FIXED_USD,
                    'increment': increment,
                    'num_increments': num_increments
                }
        else:
            if current_sl == 0 or new_sl < current_sl:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_FIXED_USD,
                    'increment': increment,
                    'num_increments': num_increments
                }

        return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_FIXED_USD}

    def _calculate_atr_trailing(
        self,
        entry_price: float,
        current_sl: float,
        current_price: float,
        is_buy: bool,
        atr: float
    ) -> Dict:
        """
        Calculate ATR multiple trailing stop.
        
        Fallback method when other methods don't apply.
        """
        if atr <= 0:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_ATR_MULTIPLE}

        trail_distance = atr * self.atr_trail_mult

        if is_buy:
            new_sl = current_price - trail_distance
            if new_sl > current_sl and new_sl < current_price:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_ATR_MULTIPLE,
                    'increment': trail_distance
                }
        else:
            new_sl = current_price + trail_distance
            if (current_sl == 0 or new_sl < current_sl) and new_sl > current_price:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_ATR_MULTIPLE,
                    'increment': trail_distance
                }

        return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_ATR_MULTIPLE}

    def _calculate_volatility_adaptive(
        self,
        entry_price: float,
        current_sl: float,
        profit_usd: float,
        is_buy: bool,
        df: pd.DataFrame,
        regime_category: str
    ) -> Dict:
        """
        Calculate volatility-adaptive trailing stop.
        
        Adjusts increment based on recent volatility.
        """
        # Get base increment for regime
        base_increment = self.trail_increments.get(regime_category, self.trail_increments['DEFAULT'])

        # Adjust for volatility
        atr = self._calculate_atr(df) if df is not None else 5.0

        # Calculate volatility ratio (current vs baseline)
        if df is not None and len(df) >= 50:
            try:
                high = df['high'].values.astype(float)
                low = df['low'].values.astype(float)
                close = df['close'].values.astype(float)

                tr = np.maximum(high[1:] - low[1:],
                                np.maximum(np.abs(high[1:] - close[:-1]),
                                           np.abs(low[1:] - close[:-1])))

                current_tr = np.mean(tr[-10:])
                baseline_tr = np.mean(tr[-50:-20])

                if baseline_tr > 0:
                    vol_ratio = current_tr / baseline_tr
                    # Adjust increment: higher volatility = larger increment
                    adjusted_increment = base_increment * min(2.0, max(0.5, vol_ratio))
                else:
                    adjusted_increment = base_increment

            except Exception:
                adjusted_increment = base_increment
        else:
            adjusted_increment = base_increment

        # Get breakeven trigger
        be_trigger = self.be_triggers.get(regime_category, self.be_triggers['DEFAULT'])

        # Only trail after breakeven
        if profit_usd < be_trigger:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_VOLATILITY_ADAPTIVE}

        # Calculate increments
        profit_beyond_be = profit_usd - be_trigger
        num_increments = int(profit_beyond_be / adjusted_increment)

        if num_increments < 1:
            return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_VOLATILITY_ADAPTIVE}

        if is_buy:
            new_sl = entry_price + self.breakeven_buffer + (num_increments * adjusted_increment)
            if new_sl > current_sl:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_VOLATILITY_ADAPTIVE,
                    'increment': adjusted_increment,
                    'num_increments': num_increments
                }
        else:
            new_sl = entry_price - self.breakeven_buffer - (num_increments * adjusted_increment)
            if current_sl == 0 or new_sl < current_sl:
                return {
                    'should_update': True,
                    'new_sl': new_sl,
                    'method': self.METHOD_VOLATILITY_ADAPTIVE,
                    'increment': adjusted_increment,
                    'num_increments': num_increments
                }

        return {'should_update': False, 'new_sl': 0, 'method': self.METHOD_VOLATILITY_ADAPTIVE}

    # =========================================================================
    # STOP VALIDATION
    # =========================================================================

    def _validate_stop(
        self, new_sl: float, current_price: float, is_buy: bool
    ) -> Optional[float]:
        """
        Validate new stop against broker's stops level.
        
        Returns:
            Validated SL price or None if invalid
        """
        if new_sl <= 0:
            return None

        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return new_sl

        point = getattr(symbol_info, 'point', 0.01)
        digits = getattr(symbol_info, 'digits', 2)
        stops_level = max(getattr(symbol_info, 'trade_stops_level', 10), 10)
        freeze_level = getattr(symbol_info, 'trade_freeze_level', 0)

        min_points = max(stops_level, freeze_level) + 2
        min_dist = min_points * point
        min_dist = max(min_dist, 0.50)  # At least $0.50 for XAUUSD

        # Validate distance from current price
        distance = abs(current_price - new_sl)
        if distance < min_dist:
            self._stats['rejected_count'] += 1
            self.logger.debug(
                f"[DYNAMIC_STOPS] Stop rejected: distance {distance:.2f} < {min_dist:.2f}"
            )
            return None

        return round(new_sl, digits)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from DataFrame."""
        try:
            atr_series = ATRCache.get_atr(df, period)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                return 5.0
            return float(atr_series.iloc[-1])
        except Exception:
            return 5.0

    def _categorize_regime(self, regime_name: str) -> str:
        """Categorize regime for trailing purposes."""
        if any(x in regime_name for x in ['HEALTHY_UPTREND', 'HEALTHY_DOWNTREND', 'QUIET_RALLY', 'SLOW_BLEED']):
            return 'STRONG_TREND'
        if any(x in regime_name for x in ['PARABOLIC_RALLY', 'PANIC_CAPITULATION']):
            return 'PARABOLIC'
        if any(x in regime_name for x in ['CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR']):
            return 'CONSOLIDATING'
        if any(x in regime_name for x in ['CLASSIC_RANGE', 'TIGHT_RANGE', 'PRE_BREAKOUT']):
            return 'SIDEWAYS'
        if any(x in regime_name for x in ['VOLATILE_CHOP', 'WHIPSAW_MARKET']):
            return 'CHOPPY'
        if any(x in regime_name for x in ['OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR', 'ANOMALY']):
            return 'REVERSAL'
        return 'DEFAULT'

    def _get_symbol_info(self):
        """Get symbol info with 5-second cache."""
        import time
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

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_trailing_stats(self) -> Dict:
        """
        Get trailing stop statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            'total_updates': self._stats['total_updates'],
            'breakeven_count': self._stats['breakeven_count'],
            'trailing_count': self._stats['trailing_count'],
            'volatility_adjust_count': self._stats['volatility_adjust_count'],
            'rejected_count': self._stats['rejected_count'],
            'success_rate': round(
                (self._stats['total_updates'] - self._stats['rejected_count']) /
                max(1, self._stats['total_updates']) * 100, 1
            )
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            'total_updates': 0,
            'breakeven_count': 0,
            'trailing_count': 0,
            'volatility_adjust_count': 0,
            'rejected_count': 0
        }
        self.logger.info("[DYNAMIC_STOPS] Statistics reset")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def format_update_log(self, result: Dict, ticket: int) -> str:
        """
        Format update result as concise log string.
        
        Args:
            result: Result from update_dynamic_stops
            ticket: Position ticket
            
        Returns:
            Formatted log string
        """
        if result is None:
            return f"[DYNAMIC_STOPS] Ticket {ticket}: No update needed"

        new_sl = result.get('new_sl', 0)
        old_sl = result.get('old_sl', 0)
        reason = result.get('reason', 'UNKNOWN')
        method = result.get('method', 'UNKNOWN')
        profit = result.get('profit_usd', 0)

        return (
            f"[DYNAMIC_STOPS] Ticket {ticket} | "
            f"SL: {old_sl:.2f} -> {new_sl:.2f} | "
            f"Reason: {reason} | "
            f"Method: {method} | "
            f"Profit: {profit:.2f} USD"
        )

    def get_regime_triggers(self) -> Dict:
        """
        Get current regime-adaptive trigger values.
        
        Returns:
            Dict with triggers and increments per regime category
        """
        return {
            'breakeven_triggers': dict(self.be_triggers),
            'trail_increments': dict(self.trail_increments),
            'method_preference': dict(self.method_preference)
        }