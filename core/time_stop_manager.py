"""
Time Stop Manager - Micro-Account-Only Edition.

Manages time-based stop logic for position lifecycle.

Features:
  - Time Stop: Close positions that exceed maximum duration
  - Breakeven Stop: Move SL to entry + buffer when profit threshold reached
  - Regime-Adaptive Triggers: Different BE triggers per regime
  - Session Awareness: Adjust for session volatility

Time Limits (Micro-Account):
  - M1: 30 minutes
  - M5: 60 minutes
  - M15: 120 minutes

Breakeven Triggers (USD profit):
  - Strong Trend: 14 USD
  - Parabolic: 18 USD
  - Consolidating: 10 USD
  - Sideways: 7 USD
  - Choppy: 6 USD
  - Reversal: 9 USD
"""
import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import config


class TimeStopManager:
    """
    Manages time-based stop logic for position lifecycle.

    Two main functions:
      1. Time Stop: Close position if it exceeds maximum duration
      2. Breakeven Stop: Move SL to entry when profit threshold reached
    """

    def __init__(self):
        """Initialize TimeStopManager with config parameters."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Time limits per timeframe (minutes)
        self.time_limits = {
            'M1': config.time_stop_m1_minutes,
            'M5': config.time_stop_m5_minutes,
            'M15': config.time_stop_m15_minutes,
            'M30': 180,  # 3 hours
            'H1': 240,   # 4 hours
            'H4': 480,   # 8 hours
            'D1': 1440   # 24 hours
        }

        self.logger.info(
            f"[TIME_STOP] Initialized | M1: {self.time_limits['M1']}min | "
            f"M5: {self.time_limits['M5']}min | M15: {self.time_limits['M15']}min"
        )

    # =========================================================================
    # TIME STOP CHECK
    # =========================================================================

    def should_time_stop(self, pos: Dict, current_time: pd.Timestamp,
                          primary_tf: str, strategy_category: str,
                          current_price: float) -> bool:
        """
        Check if position should be closed due to time limit.

        Args:
            pos: Position dict from StateManager
            current_time: Current timestamp
            primary_tf: Primary timeframe of the strategy
            strategy_category: Strategy category (TREND, SCALP, etc.)
            current_price: Current market price

        Returns:
            True if time stop should trigger, False otherwise
        """
        try:
            # Get setup time
            setup_time_str = pos.get('setup_time') or pos.get('open_time')
            if not setup_time_str:
                return False

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

            # Calculate elapsed time in minutes
            elapsed_minutes = (current_time_cmp - setup_time_cmp).total_seconds() / 60.0

            # Get max time limit for this timeframe
            max_minutes = self.time_limits.get(primary_tf, 240)

            # Adjust for strategy category
            if strategy_category == 'TREND':
                # Trend strategies get 1.5x time allowance
                max_minutes *= 1.5
            elif strategy_category == 'SCALP':
                # Scalp strategies get 0.5x time allowance
                max_minutes *= 0.5

            # If position is profitable, extend time limit by 50%
            entry_price = pos.get('entry_price', 0)
            if entry_price > 0 and current_price > 0:
                is_buy = pos.get('position_type', 'BUY') == 'BUY'
                profit = (current_price - entry_price) if is_buy else (entry_price - current_price)
                if profit > 0:
                    max_minutes *= 1.5

            # Check if time limit exceeded
            if elapsed_minutes >= max_minutes:
                self.logger.info(
                    f"[TIME_STOP] Time limit exceeded | "
                    f"Elapsed: {elapsed_minutes:.0f}min >= Max: {max_minutes:.0f}min | "
                    f"TF: {primary_tf} | Category: {strategy_category}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"[TIME_STOP] Error checking time stop: {e}")
            return False

    # =========================================================================
    # BREAKEVEN STOP CHECK
    # =========================================================================

    def check_breakeven_stop(self, pos: Dict, current_price: float) -> Optional[float]:
        """
        Check if SL should be moved to breakeven.

        Breakeven is triggered when profit reaches regime-specific threshold.
        SL is moved to entry_price + buffer (0.5 USD).

        Args:
            pos: Position dict from StateManager
            current_price: Current market price

        Returns:
            New SL price if breakeven should trigger, None otherwise
        """
        try:
            entry_price = pos.get('entry_price', 0)
            current_sl = pos.get('sl', 0)
            position_type = pos.get('position_type', 'BUY')
            meta = pos.get('meta_data', {})
            regime_name = meta.get('regime_name', 'UNKNOWN')

            if entry_price <= 0 or current_price <= 0:
                return None

            is_buy = position_type == 'BUY'

            # Calculate current profit in USD (price units)
            if is_buy:
                profit_usd = current_price - entry_price
            else:
                profit_usd = entry_price - current_price

            # Get regime-specific breakeven trigger
            be_trigger = self._get_regime_breakeven_trigger(regime_name)

            # Check if profit threshold reached
            if profit_usd >= be_trigger:
                # Check if SL is still behind entry (not yet at breakeven)
                is_sl_behind = (
                    current_sl == 0.0 or
                    (is_buy and current_sl < entry_price) or
                    (not is_buy and current_sl > entry_price)
                )

                if is_sl_behind:
                    # Move SL to entry + buffer
                    be_buffer = 0.5  # 0.5 USD buffer
                    if is_buy:
                        new_sl = entry_price + be_buffer
                    else:
                        new_sl = entry_price - be_buffer

                    self.logger.info(
                        f"[BREAKEVEN] Triggered | Profit: {profit_usd:.2f} USD >= "
                        f"Trigger: {be_trigger:.2f} USD | Regime: {regime_name} | "
                        f"SL: {current_sl:.2f} -> {new_sl:.2f}"
                    )

                    return new_sl

            return None

        except Exception as e:
            self.logger.error(f"[BREAKEVEN] Error checking breakeven: {e}")
            return None

    # =========================================================================
    # REGIME-SPECIFIC TRIGGERS
    # =========================================================================

    def _get_regime_breakeven_trigger(self, regime_name: str) -> float:
        """
        Get breakeven trigger based on current regime.

        Different regimes have different volatility characteristics,
        so breakeven triggers are adjusted accordingly.

        Args:
            regime_name: Current regime name

        Returns:
            Breakeven trigger in USD
        """
        # Strong Trend: Higher trigger (let profits run)
        if regime_name in ['QUIET_RALLY', 'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND', 'SLOW_BLEED']:
            return config.be_strong_trend_usd

        # Parabolic/Panic: Highest trigger (extreme volatility)
        if regime_name in ['PARABOLIC_RALLY', 'PANIC_CAPITULATION']:
            return config.be_parabolic_usd

        # Consolidating: Medium trigger
        if regime_name in ['CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR', 'FALSE_SIDEWAY']:
            return config.be_consolidating_usd

        # Sideways: Lower trigger (mean reversion)
        if regime_name in ['CLASSIC_RANGE', 'TIGHT_RANGE', 'PRE_BREAKOUT']:
            return config.be_sideways_usd

        # Choppy: Lowest trigger (protect capital)
        if regime_name in ['VOLATILE_CHOP', 'WHIPSAW_MARKET']:
            return config.be_choppy_usd

        # Reversal: Medium-low trigger
        if regime_name in ['OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR', 'ANOMALY_BULL', 'ANOMALY_BEAR']:
            return config.be_reversal_usd

        # Default: Sideways trigger
        return config.be_sideways_usd

    # =========================================================================
    # SESSION AWARENESS
    # =========================================================================

    def _get_session_multiplier(self, session: str) -> float:
        """
        Get time limit multiplier based on session.

        More volatile sessions get longer time limits.

        Args:
            session: Current session name

        Returns:
            Time limit multiplier
        """
        multipliers = {
            'LONDON_OPEN': 1.0,
            'NY_OPEN': 1.0,
            'LONDON': 1.0,
            'NY_MIDDAY': 1.2,
            'US_CLOSE': 1.5,
            'ASIAN': 0.8,
            'OTHER': 1.0
        }
        return multipliers.get(session, 1.0)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_time_stop_summary(self, pos: Dict, current_time: pd.Timestamp) -> Dict:
        """
        Get time stop status summary for diagnostics.

        Args:
            pos: Position dict
            current_time: Current timestamp

        Returns:
            Dict with time stop status
        """
        try:
            setup_time_str = pos.get('setup_time') or pos.get('open_time')
            if not setup_time_str:
                return {'status': 'NO_SETUP_TIME', 'elapsed_minutes': 0, 'max_minutes': 0}

            setup_time = pd.to_datetime(setup_time_str)

            if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
                current_time_cmp = current_time.replace(tzinfo=None)
            else:
                current_time_cmp = current_time

            if hasattr(setup_time, 'tzinfo') and setup_time.tzinfo is not None:
                setup_time_cmp = setup_time.replace(tzinfo=None)
            else:
                setup_time_cmp = setup_time

            elapsed_minutes = (current_time_cmp - setup_time_cmp).total_seconds() / 60.0

            primary_tf = pos.get('meta_data', {}).get('timeframe', 'M15')
            max_minutes = self.time_limits.get(primary_tf, 240)

            strategy_category = pos.get('meta_data', {}).get('strategy_category', 'GENERAL')
            if strategy_category == 'TREND':
                max_minutes *= 1.5
            elif strategy_category == 'SCALP':
                max_minutes *= 0.5

            return {
                'status': 'EXCEEDED' if elapsed_minutes >= max_minutes else 'OK',
                'elapsed_minutes': round(elapsed_minutes, 1),
                'max_minutes': round(max_minutes, 1),
                'timeframe': primary_tf,
                'strategy_category': strategy_category,
                'remaining_minutes': round(max_minutes - elapsed_minutes, 1)
            }

        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}