"""
Edge Decay Invalidation Engine.

Detects when a trading setup's edge has decayed or been invalidated.
Used by Layer 4 of the 10-Layer Active Position Management system.

Invalidation Types:
  1. Time Decay: Setup expired (position open too long without progress)
  2. Price Decay: Price moved against position beyond threshold
  3. Structure Break: Key structural level broken
  4. Volatility Spike: Abnormal volatility indicates regime change

Grace Period:
  - First 15 minutes after entry: No invalidation checks
  - Allows setup time to develop before checking for decay
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import config


class InvalidationEngine:
    """
    Detects when a trading setup's edge has decayed or been invalidated.
    
    Works with Layer 4 of the 10-Layer defense system to close positions
    when their original thesis is no longer valid.
    """

    def __init__(self):
        """Initialize InvalidationEngine with configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Grace period after entry (minutes)
        self.grace_period_minutes = 15

        # Time decay thresholds (minutes without progress)
        self.time_decay_thresholds = {
            'SCALP': 20,      # 20 minutes for scalp
            'TREND': 60,      # 60 minutes for trend
            'SMC': 45,        # 45 minutes for SMC
            'MEAN_REVERSION': 30,  # 30 minutes for mean reversion
            'GENERAL': 40     # 40 minutes default
        }

        # Price decay threshold (ATR multiple)
        self.price_decay_atr_multiple = 1.5

        # Volatility spike threshold (ATR multiple)
        self.volatility_spike_threshold = 3.0

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def check_edge_decay(self, pos: Dict, current_price: float,
                          current_time: datetime, df: pd.DataFrame = None) -> Optional[str]:
        """
        Check if position's edge has decayed.
        
        Args:
            pos: Position dict from StateManager
            current_price: Current market price
            current_time: Current timestamp
            df: DataFrame for additional analysis (optional)
            
        Returns:
            Reason string if edge decayed, None otherwise
        """
        try:
            # Get position info
            entry_price = pos.get('entry_price', 0)
            position_type = pos.get('position_type', 'BUY')
            meta = pos.get('meta_data', {})
            strategy_category = meta.get('strategy_category', 'GENERAL')
            setup_time_str = pos.get('setup_time') or pos.get('open_time')

            if entry_price <= 0 or current_price <= 0:
                return None

            # Parse setup time
            if not setup_time_str:
                return None

            setup_time = pd.to_datetime(setup_time_str)

            # Handle timezone
            if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
                current_time_cmp = current_time.replace(tzinfo=None)
            else:
                current_time_cmp = current_time

            if hasattr(setup_time, 'tzinfo') and setup_time.tzinfo is not None:
                setup_time_cmp = setup_time.replace(tzinfo=None)
            else:
                setup_time_cmp = setup_time

            # Calculate elapsed time
            elapsed_minutes = (current_time_cmp - setup_time_cmp).total_seconds() / 60.0

            # Grace period check - skip invalidation during grace period
            if elapsed_minutes < self.grace_period_minutes:
                return None

            # Calculate current profit/loss
            is_buy = position_type == 'BUY'
            if is_buy:
                profit_usd = current_price - entry_price
            else:
                profit_usd = entry_price - current_price

            # =========================================================================
            # CHECK 1: Time Decay
            # =========================================================================
            time_decay_reason = self._check_time_decay(
                pos, elapsed_minutes, profit_usd, strategy_category
            )
            if time_decay_reason:
                return time_decay_reason

            # =========================================================================
            # CHECK 2: Price Decay (only if position is losing)
            # =========================================================================
            if profit_usd < 0:
                price_decay_reason = self._check_price_decay(
                    pos, profit_usd, strategy_category, df
                )
                if price_decay_reason:
                    return price_decay_reason

            # =========================================================================
            # CHECK 3: Structure Break (optional, requires df)
            # =========================================================================
            if df is not None and not df.empty:
                structure_reason = self._check_structure_break(
                    pos, df, current_price, is_buy
                )
                if structure_reason:
                    return structure_reason

            # =========================================================================
            # CHECK 4: Volatility Spike (optional, requires df)
            # =========================================================================
            if df is not None and not df.empty:
                volatility_reason = self._check_volatility_spike(
                    pos, df, strategy_category
                )
                if volatility_reason:
                    return volatility_reason

            # No invalidation detected
            return None

        except Exception as e:
            self.logger.error(f"[INVALIDATION] Error checking edge decay: {e}")
            return None

    # =========================================================================
    # CHECK 1: TIME DECAY
    # =========================================================================

    def _check_time_decay(self, pos: Dict, elapsed_minutes: float,
                           profit_usd: float, strategy_category: str) -> Optional[str]:
        """
        Check if position has been open too long without progress.
        
        Time decay triggers when:
          - Position open > threshold AND
          - Position has no meaningful profit (< $2)
        """
        threshold = self.time_decay_thresholds.get(strategy_category, 40)

        # Extend threshold if position is profitable
        if profit_usd > 5.0:
            threshold *= 1.5
        elif profit_usd > 2.0:
            threshold *= 1.2

        if elapsed_minutes >= threshold and profit_usd < 2.0:
            return (
                f"Time decay: {elapsed_minutes:.0f}min elapsed, "
                f"threshold {threshold:.0f}min, profit ${profit_usd:.2f}"
            )

        return None

    # =========================================================================
    # CHECK 2: PRICE DECAY
    # =========================================================================

    def _check_price_decay(self, pos: Dict, profit_usd: float,
                            strategy_category: str, df: pd.DataFrame = None) -> Optional[str]:
        """
        Check if price has moved too far against position.
        
        Price decay triggers when:
          - Loss > 1.5x ATR (or fixed threshold if no df)
        """
        # Fixed threshold if no df available
        if df is None or df.empty:
            max_loss_usd = config.sl_distance_usd * 0.8  # 80% of SL distance
            if profit_usd < -max_loss_usd:
                return (
                    f"Price decay: Loss ${abs(profit_usd):.2f} exceeds "
                    f"threshold ${max_loss_usd:.2f}"
                )
            return None

        # Calculate ATR for dynamic threshold
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 14:
                return None

            # Calculate ATR
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:])

            if atr <= 0:
                return None

            # Price decay threshold
            max_loss = atr * self.price_decay_atr_multiple

            if profit_usd < -max_loss:
                return (
                    f"Price decay: Loss ${abs(profit_usd):.2f} exceeds "
                    f"{self.price_decay_atr_multiple}x ATR (${max_loss:.2f})"
                )

        except Exception as e:
            self.logger.debug(f"[INVALIDATION] Price decay check error: {e}")

        return None

    # =========================================================================
    # CHECK 3: STRUCTURE BREAK
    # =========================================================================

    def _check_structure_break(self, pos: Dict, df: pd.DataFrame,
                                current_price: float, is_buy: bool) -> Optional[str]:
        """
        Check if key structural level has been broken.
        
        Structure break triggers when:
          - BUY position: Price breaks below recent swing low
          - SELL position: Price breaks above recent swing high
        """
        try:
            if len(df) < 20:
                return None

            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Look at recent 20 bars for swing points
            recent_high = high[-20:]
            recent_low = low[-20:]

            # Find swing high/low (simplified: local extrema)
            swing_high = np.max(recent_high[:-5])  # Exclude last 5 bars
            swing_low = np.min(recent_low[:-5])

            meta = pos.get('meta_data', {})
            entry_price = pos.get('entry_price', 0)

            if is_buy:
                # BUY: Check if price broke below swing low
                if current_price < swing_low * 0.999:  # 0.1% buffer
                    return (
                        f"Structure break: Price {current_price:.2f} broke below "
                        f"swing low {swing_low:.2f}"
                    )
            else:
                # SELL: Check if price broke above swing high
                if current_price > swing_high * 1.001:  # 0.1% buffer
                    return (
                        f"Structure break: Price {current_price:.2f} broke above "
                        f"swing high {swing_high:.2f}"
                    )

        except Exception as e:
            self.logger.debug(f"[INVALIDATION] Structure break check error: {e}")

        return None

    # =========================================================================
    # CHECK 4: VOLATILITY SPIKE
    # =========================================================================

    def _check_volatility_spike(self, pos: Dict, df: pd.DataFrame,
                                 strategy_category: str) -> Optional[str]:
        """
        Check if volatility has spiked abnormally.
        
        Volatility spike triggers when:
          - Current ATR > 3x baseline ATR
          - Indicates regime change or news event
        """
        try:
            if len(df) < 50:
                return None

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Calculate True Range
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))

            if len(tr) < 50:
                return None

            # Baseline ATR (50 bars ago)
            baseline_atr = np.mean(tr[-50:-30])

            # Current ATR (last 10 bars)
            current_atr = np.mean(tr[-10:])

            if baseline_atr <= 0:
                return None

            volatility_ratio = current_atr / baseline_atr

            # Different thresholds per strategy category
            thresholds = {
                'SCALP': 2.5,      # Scalp is more sensitive
                'TREND': 3.5,      # Trend is more tolerant
                'SMC': 3.0,
                'MEAN_REVERSION': 2.5,
                'GENERAL': 3.0
            }
            threshold = thresholds.get(strategy_category, self.volatility_spike_threshold)

            if volatility_ratio >= threshold:
                return (
                    f"Volatility spike: Current ATR {current_atr:.2f} is "
                    f"{volatility_ratio:.1f}x baseline {baseline_atr:.2f}"
                )

        except Exception as e:
            self.logger.debug(f"[INVALIDATION] Volatility spike check error: {e}")

        return None

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_invalidation_summary(self, pos: Dict, current_price: float,
                                  current_time: datetime) -> Dict:
        """
        Get invalidation status summary for diagnostics.
        
        Returns:
            Dict with invalidation status details
        """
        try:
            entry_price = pos.get('entry_price', 0)
            setup_time_str = pos.get('setup_time') or pos.get('open_time')
            meta = pos.get('meta_data', {})
            strategy_category = meta.get('strategy_category', 'GENERAL')

            if not setup_time_str:
                return {'status': 'NO_SETUP_TIME'}

            setup_time = pd.to_datetime(setup_time_str)
            elapsed_minutes = (current_time - setup_time).total_seconds() / 60.0

            threshold = self.time_decay_thresholds.get(strategy_category, 40)
            grace_remaining = max(0, self.grace_period_minutes - elapsed_minutes)

            is_buy = pos.get('position_type', 'BUY') == 'BUY'
            profit_usd = (current_price - entry_price) if is_buy else (entry_price - current_price)

            return {
                'status': 'OK',
                'elapsed_minutes': round(elapsed_minutes, 1),
                'grace_remaining': round(grace_remaining, 1),
                'time_threshold': threshold,
                'profit_usd': round(profit_usd, 2),
                'strategy_category': strategy_category,
                'checks_enabled': elapsed_minutes >= self.grace_period_minutes
            }

        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}