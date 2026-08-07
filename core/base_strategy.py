"""
Base Strategy Class - Micro-Account-Only Edition.

All 30 strategies inherit from this class.
Provides common utilities: ATR, SL/TP calculation, signal building, session awareness.

Simplified for Micro-Account Mode ($500-$3000 portfolio).
No mode switching, no scalping mode, no standard mode.
"""
import pandas as pd
import numpy as np
import logging
import pytz
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from core.atr_cache import ATRCache
from config import config


class BaseStrategy(ABC):
    """
    Abstract base class for all 30 trading strategies.

    Provides common utilities:
      - Data validation
      - Stop Loss calculation (Micro-Account fixed distance)
      - Take Profit calculation (Fibonacci-based)
      - Signal building with Micro-Account validation
      - Session awareness
      - Signal logging
    """

    def __init__(self, name: str, strategy_category: str, min_risk_reward: float = 2.0):
        """
        Initialize base strategy.

        Args:
            name: Strategy name (e.g., 'S1_IOB_Rejection')
            strategy_category: Category (TREND, SCALP, SMC, MEAN_REVERSION)
            min_risk_reward: Minimum risk/reward ratio for signal validation
        """
        self.name = name
        self.strategy_category = strategy_category
        self.min_risk_reward = min_risk_reward
        self.logger = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # ABSTRACT METHOD (Must be implemented by each strategy)
    # =========================================================================

    @abstractmethod
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate market conditions and generate trading signal.

        Must be implemented by each strategy.

        Args:
            df_primary: Primary timeframe DataFrame
            df_htf: Higher timeframe DataFrame (optional confirmation)

        Returns:
            Dict with 'signal' and 'meta' keys
        """
        pass

    # =========================================================================
    # DATA VALIDATION
    # =========================================================================

    def _validate_data(self, df: pd.DataFrame, min_bars: int) -> bool:
        """
        Validate that DataFrame has sufficient data.

        Args:
            df: DataFrame to validate
            min_bars: Minimum number of bars required

        Returns:
            True if valid, False otherwise
        """
        if df is None or len(df) < min_bars:
            return False
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                return False
        return True

    # =========================================================================
    # STOP LOSS CALCULATION (Micro-Account Fixed Distance)
    # =========================================================================

    def calculate_session_sl(
        self,
        entry_price: float,
        structural_level: float,
        df: pd.DataFrame,
        is_buy: bool,
        atr_multiplier: float = 1.5,
        max_risk_atr: float = 3.0
    ) -> Dict:
        """
        Calculate Stop Loss for Micro-Account Mode.

        Uses fixed SL distance from config (16 USD for XAUUSD @ 4000).
        No mode switching, no conditional logic.

        Args:
            entry_price: Entry price
            structural_level: Key structural level (support/resistance)
            df: DataFrame for ATR calculation and session detection
            is_buy: True for BUY, False for SELL
            atr_multiplier: ATR multiplier (used for session awareness)
            max_risk_atr: Maximum risk in ATR units (safety cap)

        Returns:
            Dict with 'sl_price', 'valid', 'reason', 'session', 'risk'
        """
        # Micro-Account: Use fixed SL distance from config
        sl_distance = config.sl_distance_usd

        if is_buy:
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance

        # Get session for logging
        session = self._get_current_session(df)

        return {
            'sl_price': round(sl_price, 2),
            'valid': True,
            'reason': f'Micro-Account SL ({sl_distance:.2f} USD)',
            'session': session,
            'risk': round(sl_distance, 2)
        }

    # =========================================================================
    # TAKE PROFIT CALCULATION (Fibonacci-based)
    # =========================================================================

    def calculate_fib_tp(
        self,
        entry_price: float,
        sl_price: float,
        df: pd.DataFrame,
        is_buy: bool
    ) -> Dict:
        """
        Calculate Take Profit using Fibonacci extensions.

        Uses swing points detected by SMC engine to calculate
        A-B-C Fibonacci extension levels.

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            df: DataFrame for swing detection
            is_buy: True for BUY, False for SELL

        Returns:
            Dict with 'valid', 'tp_price', 'reason', 'risk_reward'
        """
        if df is None or len(df) < 20:
            return {'valid': False, 'tp_price': 0, 'reason': 'insufficient data'}

        try:
            from core.smc_engine import SMCStructuralEngine
            smc = SMCStructuralEngine()
            swing_highs, swing_lows = smc.detect_swings(df, order=3)

            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return {'valid': False, 'tp_price': 0, 'reason': 'insufficient swings'}

            risk = abs(entry_price - sl_price)
            if risk == 0:
                return {'valid': False, 'tp_price': 0, 'reason': 'zero risk'}

            if is_buy:
                a_price = float(df['low'].iloc[swing_lows[-1]])
                b_price = float(df['high'].iloc[swing_highs[-1]])
                c_price = entry_price
                range_ab = b_price - a_price

                if range_ab <= 0:
                    return {'valid': False, 'tp_price': 0, 'reason': 'invalid range'}

                fib_levels = [1.0, 1.272, 1.618, 2.0]
                for level in fib_levels:
                    tp_price = c_price + (range_ab * level)
                    reward = abs(tp_price - entry_price)
                    rr = reward / risk

                    if rr >= self.min_risk_reward:
                        return {
                            'valid': True,
                            'tp_price': round(tp_price, 2),
                            'reason': f'Fibonacci TP ({level:.3f} extension)',
                            'risk_reward': round(rr, 2)
                        }
            else:
                a_price = float(df['high'].iloc[swing_highs[-1]])
                b_price = float(df['low'].iloc[swing_lows[-1]])
                c_price = entry_price
                range_ab = a_price - b_price

                if range_ab <= 0:
                    return {'valid': False, 'tp_price': 0, 'reason': 'invalid range'}

                fib_levels = [1.0, 1.272, 1.618, 2.0]
                for level in fib_levels:
                    tp_price = c_price - (range_ab * level)
                    reward = abs(tp_price - entry_price)
                    rr = reward / risk

                    if rr >= self.min_risk_reward:
                        return {
                            'valid': True,
                            'tp_price': round(tp_price, 2),
                            'reason': f'Fibonacci TP ({level:.3f} extension)',
                            'risk_reward': round(rr, 2)
                        }

            return {'valid': False, 'tp_price': 0, 'reason': 'no suitable fib level'}

        except Exception as e:
            self.logger.error(f"[FIB_TP] Error: {e}")
            return {'valid': False, 'tp_price': 0, 'reason': f'error: {str(e)}'}

    # =========================================================================
    # SIGNAL BUILDING (Micro-Account Validation)
    # =========================================================================

    def build_signal(
        self,
        signal_type: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        timeframe: str,
        confidence: float,
        expiration_bars: int = 10,
        requires_dynamic_exit: bool = False,
        dynamic_exit_threshold=None,
        position_multiplier: float = 1.0,
        extra_meta: dict = None
    ) -> dict:
        """
        Build standardized signal dictionary with Micro-Account validation.

        Validates:
          - Minimum profit (config.min_profit_usd)
          - Minimum R:R ratio (config.min_rr_ratio)

        Args:
            signal_type: 'BUY_MARKET', 'SELL_MARKET', 'BUY_LIMIT', etc.
            entry_price: Entry price
            sl_price: Stop loss price
            tp_price: Take profit price
            timeframe: Timeframe (M1, M5, M15, H1)
            confidence: Signal confidence (0.0-1.0)
            expiration_bars: Bars until pending order expires
            requires_dynamic_exit: Whether to use dynamic exit
            dynamic_exit_threshold: Threshold for dynamic exit
            position_multiplier: Position size multiplier
            extra_meta: Additional metadata to merge into signal

        Returns:
            Signal dict with 'signal' and 'meta' keys
        """
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Micro-Account validation: minimum profit
        min_profit = config.min_profit_usd
        if reward < min_profit:
            self.logger.warning(
                f"[MICRO] {self.name}: Reward {reward:.2f} USD below minimum {min_profit} USD"
            )
            return {
                'signal': 'NEUTRAL',
                'meta': {'strategy': self.name, 'reason': 'insufficient_profit'}
            }

        # Micro-Account validation: minimum R:R
        min_rr = config.min_rr_ratio
        if risk_reward < min_rr:
            self.logger.warning(
                f"[MICRO] {self.name}: R:R {risk_reward:.2f} below minimum {min_rr}"
            )
            return {
                'signal': 'NEUTRAL',
                'meta': {'strategy': self.name, 'reason': 'insufficient_rr'}
            }

        meta = {
            'strategy': self.name,
            'strategy_category': self.strategy_category,
            'entry_price': round(entry_price, 2),
            'sl_price': round(sl_price, 2),
            'tp_price': round(tp_price, 2),
            'risk_reward': round(risk_reward, 2),
            'confidence': confidence,
            'timeframe': timeframe,
            'expiration_bars': expiration_bars,
            'requires_dynamic_exit': requires_dynamic_exit,
            'dynamic_exit_threshold': dynamic_exit_threshold,
            'position_multiplier': position_multiplier,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'trailing_method': 'default',
            'friction_sensitive': True
        }

        if extra_meta:
            meta.update(extra_meta)

        return {'signal': signal_type, 'meta': meta}

    # =========================================================================
    # SIGNAL LOGGING
    # =========================================================================

    def log_signal_summary(self, signal: dict):
        """
        Log a concise summary of the generated signal.

        Args:
            signal: Signal dict with 'signal' and 'meta' keys
        """
        if signal.get('signal') == 'NEUTRAL':
            return

        meta = signal.get('meta', {})
        self.logger.info(
            f"[SIGNAL] {self.name} | {signal['signal']} | "
            f"Entry: {meta.get('entry_price', 0):.2f} | "
            f"SL: {meta.get('sl_price', 0):.2f} | "
            f"TP: {meta.get('tp_price', 0):.2f} | "
            f"R:R: {meta.get('risk_reward', 0):.2f} | "
            f"Conf: {meta.get('confidence', 0):.2f}"
        )

    # =========================================================================
    # SESSION AWARENESS
    # =========================================================================

    def _get_current_session(self, df: pd.DataFrame) -> str:
        """
        Get current trading session based on time.

        Handles DST transitions safely.

        Args:
            df: DataFrame with 'time' column

        Returns:
            Session name string
        """
        if df is None or df.empty or 'time' not in df.columns:
            return 'OTHER'

        try:
            last_time = df['time'].iloc[-1]
            if not isinstance(last_time, pd.Timestamp):
                last_time = pd.to_datetime(last_time)

            if last_time.tzinfo is None:
                # Handle non-existent times during DST spring forward
                last_time = last_time.tz_localize('UTC', nonexistent='shift_forward')

            ny_tz = pytz.timezone('America/New_York')
            # Handle ambiguous times during DST fall back
            ny_time = last_time.tz_convert(ny_tz)
            hour = ny_time.hour

            if 2 <= hour < 5:
                return 'LONDON_OPEN'
            elif 8 <= hour < 11:
                return 'NY_OPEN'
            elif 5 <= hour < 8:
                return 'LONDON'
            elif 11 <= hour < 17:
                return 'NY_MIDDAY'
            elif 17 <= hour < 19:
                return 'US_CLOSE'
            elif (19 <= hour <= 23) or (0 <= hour <= 1):
                return 'ASIAN'
            else:
                return 'OTHER'

        except Exception:
            return 'OTHER'

    def _get_session_atr_multiplier(self, session: str) -> float:
        """
        Get ATR multiplier based on session volatility.

        Higher multiplier = wider SL for volatile sessions.

        Args:
            session: Session name string

        Returns:
            ATR multiplier float
        """
        multipliers = {
            'LONDON_OPEN': 1.0,
            'NY_OPEN': 1.0,
            'LONDON': 1.0,
            'NY_MIDDAY': 1.2,
            'US_CLOSE': 1.5,
            'ASIAN': 1.3,
            'OTHER': 1.2
        }
        return multipliers.get(session, 1.2)