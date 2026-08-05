"""
Base Strategy Class.
All 25 strategies inherit from this class.
"""
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.atr_cache import ATRCache

# =========================================================================
# TIMEZONE IMPORTS WITH FALLBACK
# =========================================================================
try:
    from zoneinfo import ZoneInfo
    _TZ_AVAILABLE = True
    _USE_ZONEINFO = True
except ImportError:
    try:
        import pytz
        _TZ_AVAILABLE = True
        _USE_ZONEINFO = False
    except ImportError:
        _TZ_AVAILABLE = False
        _USE_ZONEINFO = False


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Provides common utilities: ATR, SL/TP calculation, signal building.
    """
    
    def __init__(self, name: str, strategy_category: str, min_risk_reward: float = 2.0):
        """
        Initialize base strategy.
        """
        self.name = name
        self.strategy_category = strategy_category
        self.min_risk_reward = min_risk_reward
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # =========================================================================
        # TIMEZONE INITIALIZATION (Fixes AttributeError)
        # =========================================================================
        self.ny_tz = None
        self.utc_tz = None
        
        if _TZ_AVAILABLE:
            if _USE_ZONEINFO:
                self.ny_tz = ZoneInfo('America/New_York')
                self.utc_tz = ZoneInfo('UTC')
            else:
                import pytz
                self.ny_tz = pytz.timezone('America/New_York')
                self.utc_tz = pytz.UTC
    
    @abstractmethod
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate market conditions and generate trading signal.
        Must be implemented by each strategy.
        """
        pass
    
    def _validate_data(self, df: pd.DataFrame, min_bars: int) -> bool:
        """Validate that DataFrame has sufficient data."""
        if df is None or len(df) < min_bars:
            return False
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                return False
        return True
    
    def calculate_session_sl(
        self, entry_price: float, structural_level: float, df: pd.DataFrame,
        is_buy: bool, atr_multiplier: float = 1.5, max_risk_atr: float = 3.0
    ) -> Dict:
        """
        Calculate Stop Loss based on session and structural levels.
        Includes Micro-Account Mode override with fixed SL distance.
        """
        from config import config
        
        # [MICRO-ACCOUNT OVERRIDE] Use fixed SL distance scaled for current price
        # if getattr(config, 'micro_account_mode', False):
        #     sl_distance = config.micro_sl_distance_usd
            
        #     if is_buy:
        #         sl_price = entry_price - sl_distance
        #     else:
        #         sl_price = entry_price + sl_distance
            
        #     return {
        #         'sl_price': round(sl_price, 2),
        #         'valid': True,
        #         'reason': f'Micro-Account SL ({sl_distance} USD)',
        #         'session': 'MICRO_ACCOUNT',
        #         'risk': round(sl_distance, 2)
        #     }
        
        # =========================================================================
        # Normal Mode: ATR-based calculation
        # =========================================================================
        atr_series = ATRCache.get_atr(df, 14)
        if atr_series.isna().all() or len(atr_series) == 0:
            return {'sl_price': 0, 'valid': False, 'reason': 'ATR calculation failed', 'session': 'UNKNOWN'}
        
        atr = float(atr_series.iloc[-1])
        if pd.isna(atr) or atr == 0:
            return {'sl_price': 0, 'valid': False, 'reason': 'ATR is zero or NaN', 'session': 'UNKNOWN'}
        
        # Determine current session
        session = 'UNKNOWN'
        if 'time' in df.columns and self.ny_tz is not None and self.utc_tz is not None:
            try:
                last_time = df['time'].iloc[-1]
                if not isinstance(last_time, pd.Timestamp):
                    last_time = pd.to_datetime(last_time, unit='s', utc=True)
                
                # Safely handle tz-naive vs tz-aware
                if last_time.tz is None:
                    last_time = last_time.tz_localize(self.utc_tz)
                
                ny_time = last_time.tz_convert(self.ny_tz)
                hour = ny_time.hour
                
                if 2 <= hour < 5:
                    session = 'LONDON_OPEN'
                elif 9 <= hour < 11:
                    session = 'NY_OPEN'
                elif 5 <= hour < 9:
                    session = 'LONDON'
                elif 11 <= hour < 17:
                    session = 'NY_MIDDAY'
                elif (19 <= hour <= 23) or (0 <= hour <= 1):
                    session = 'ASIAN'
                elif 17 <= hour < 19:
                    session = 'US_CLOSE'
                else:
                    session = 'OTHER'
            except Exception as e:
                self.logger.debug(f"[SESSION] Timezone conversion error: {e}")
                session = 'OTHER'
        
        # Calculate SL based on structural level
        if is_buy:
            sl_price = structural_level - (atr * atr_multiplier)
        else:
            sl_price = structural_level + (atr * atr_multiplier)
        
        # Validate risk distance
        risk = abs(entry_price - sl_price)
        max_risk = atr * max_risk_atr
        
        if risk > max_risk:
            if is_buy:
                sl_price = entry_price - max_risk
            else:
                sl_price = entry_price + max_risk
            
            return {
                'sl_price': round(sl_price, 2),
                'valid': True,
                'reason': f'SL adjusted to max risk ({max_risk_atr} ATR)',
                'session': session,
                'risk': round(max_risk, 2)
            }
        
        return {
            'sl_price': round(sl_price, 2),
            'valid': True,
            'reason': 'OK',
            'session': session,
            'risk': round(risk, 2)
        }
    
    def calculate_fib_tp(
        self, entry_price: float, sl_price: float, df: pd.DataFrame, is_buy: bool
    ) -> Dict:
        """
        Calculate Take Profit using Fibonacci extensions.
        """
        if df is None or len(df) < 20:
            return {'valid': False, 'tp_price': 0, 'reason': 'insufficient data'}
        
        from core.smc_engine import SMCStructuralEngine
        smc = SMCStructuralEngine()
        swing_highs, swing_lows = smc.detect_swings(df, order=3)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'valid': False, 'tp_price': 0, 'reason': 'insufficient swings'}
        
        risk = abs(entry_price - sl_price)
        if risk == 0:
            return {'valid': False, 'tp_price': 0, 'reason': 'zero risk'}
        
        if is_buy:
            a_price = df['low'].iloc[swing_lows[-1]]
            b_price = df['high'].iloc[swing_highs[-1]]
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
            a_price = df['high'].iloc[swing_highs[-1]]
            b_price = df['low'].iloc[swing_lows[-1]]
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
    
    def build_signal(
        self, signal_type: str, entry_price: float, sl_price: float, tp_price: float,
        timeframe: str, confidence: float, expiration_bars: int = 10,
        requires_dynamic_exit: bool = False, dynamic_exit_threshold=None,
        position_multiplier: float = 1.0, extra_meta: dict = None
    ) -> dict:
        """
        Build standardized signal dictionary.
        """
        from config import config
        
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        if getattr(config, 'micro_account_mode', False):
            min_profit = getattr(config, 'micro_min_profit_usd', 6.0) if hasattr(config, 'micro_min_profit_usd') else 6.0
            min_rr = getattr(config, 'micro_min_rr_ratio', 1.5) if hasattr(config, 'micro_min_rr_ratio') else 1.5
            
            if reward < min_profit:
                self.logger.warning(
                    f"[MICRO] Reward {reward:.2f} USD below minimum {min_profit} USD, rejecting signal"
                )
                return {'signal': 'NEUTRAL', 'meta': {'strategy': self.name, 'reason': 'insufficient_profit'}}
            
            if risk_reward < min_rr:
                self.logger.warning(
                    f"[MICRO] R:R {risk_reward:.2f} below minimum {min_rr}, rejecting signal"
                )
                return {'signal': 'NEUTRAL', 'meta': {'strategy': self.name, 'reason': 'insufficient_rr'}}
        
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
            'trailing_method': 'default'
        }
        
        if extra_meta:
            meta.update(extra_meta)
        
        return {'signal': signal_type, 'meta': meta}
    
    def log_signal_summary(self, signal: dict):
        """Log a concise summary of the generated signal."""
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
    
    def _get_current_session(self, df: pd.DataFrame) -> str:
        """Get current trading session based on time."""
        if df is None or df.empty or 'time' not in df.columns:
            return 'OTHER'
        
        if self.ny_tz is None or self.utc_tz is None:
            return 'OTHER'
        
        try:
            last_time = df['time'].iloc[-1]
            if not isinstance(last_time, pd.Timestamp):
                last_time = pd.to_datetime(last_time, unit='s', utc=True)
            
            if last_time.tz is None:
                last_time = last_time.tz_localize(self.utc_tz)
            
            ny_time = last_time.tz_convert(self.ny_tz)
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
        """Get ATR multiplier based on session volatility."""
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