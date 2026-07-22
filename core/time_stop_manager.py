"""
Time Stop Manager.
Handles time-based exits and breakeven stops.
"""
import pandas as pd
import logging
from typing import Dict, Optional


class TimeStopManager:
    BAR_DURATIONS = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600, 'H4': 14400, 'D1': 86400}
    BREAKEVEN_TRIGGER = 1.0  # Trigger breakeven at 1R profit
    BREAKEVEN_BUFFER = 0.1   # Buffer above entry (10% of risk)
    TIME_STOP_PROFIT_THRESHOLD = 0.5  # Close if profit < 0.5R at time stop

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_time_stop_bars(self, timeframe: str, strategy_category: Optional[str] = None) -> int:
        base = {'M1': 60, 'M5': 48, 'M15': 32, 'M30': 24, 'H1': 18, 'H4': 12, 'D1': 10}.get(timeframe, 24)
        if strategy_category == 'SCALP': base = int(base * 0.5)
        elif strategy_category == 'MEAN_REVERSION': base = int(base * 0.75)
        elif strategy_category == 'TREND': base = int(base * 2.0)
        return base

    def check_breakeven_stop(self, position: Dict, current_price: float) -> Optional[float]:
        try:
            entry_price = self._safe_float(position.get('entry_price'))
            original_sl = self._safe_float(position.get('sl'))
            current_sl_raw = position.get('trailing_stop_level')
            current_sl = self._safe_float(current_sl_raw, default=original_sl) if current_sl_raw else original_sl
            
            if entry_price == 0 or original_sl == 0: return None
            risk = abs(entry_price - original_sl)
            if risk == 0: return None
            
            is_buy = position.get('position_type') == 'BUY'
            pnl = (current_price - entry_price) if is_buy else (entry_price - current_price)
            
            if pnl >= (self.BREAKEVEN_TRIGGER * risk):
                if is_buy:
                    new_sl = entry_price + (self.BREAKEVEN_BUFFER * risk)
                    if new_sl > current_sl: return round(new_sl, 2)
                else:
                    new_sl = entry_price - (self.BREAKEVEN_BUFFER * risk)
                    # For SELL: new_sl should be ABOVE current_sl (closer to entry)
                    if current_sl == 0 or new_sl > current_sl: return round(new_sl, 2)
            return None
        except Exception as e:
            self.logger.error(f"[FAIL] Breakeven check: {e}")
            return None

    def should_time_stop(self, position: Dict, current_time: pd.Timestamp,
                         timeframe: str, strategy_category: Optional[str] = None,
                         current_price: Optional[float] = None) -> bool:
        open_time_str = position.get('open_time')
        if not open_time_str: return False
        try:
            entry_price = self._safe_float(position.get('entry_price'))
            sl_price = self._safe_float(position.get('sl'))
            open_time = pd.to_datetime(open_time_str)
            if open_time.tzinfo is None: open_time = open_time.tz_localize('UTC')
            if current_time.tzinfo is None: current_time = current_time.tz_localize('UTC')
            else: current_time = current_time.tz_convert('UTC')
            
            seconds_elapsed = (current_time - open_time).total_seconds()
            bar_duration = self.BAR_DURATIONS.get(timeframe, 300)
            bars_elapsed = seconds_elapsed / bar_duration
            max_bars = self.get_time_stop_bars(timeframe, strategy_category)
            
            if bars_elapsed >= max_bars:
                if current_price is None:
                    trailing = position.get('trailing_stop_level')
                    current_price = self._safe_float(trailing, default=entry_price) if trailing else entry_price
                if entry_price == 0 or sl_price == 0: return True
                risk = abs(entry_price - sl_price)
                if risk == 0: return True
                is_buy = position.get('position_type') == 'BUY'
                pnl = (current_price - entry_price) if is_buy else (entry_price - current_price)
                if pnl < (self.TIME_STOP_PROFIT_THRESHOLD * risk):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def _safe_float(val, default=0.0) -> float:
        if val is None or val == '': return default
        if isinstance(val, (int, float)): return float(val)
        try: return float(val)
        except (ValueError, TypeError): return default