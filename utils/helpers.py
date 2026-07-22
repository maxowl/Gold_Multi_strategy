"""
Helper Utilities.
Provides safe type casting, mathematical helpers, and dictionary manipulation.
"""
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, Union


def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Safely cast a value to float, handling None, strings, and NaN values.
    Returns default if casting fails or value is NaN.
    """
    if val is None:
        return default
        
    if isinstance(val, (int, float, np.integer, np.floating)):
        # [FIX] Explicitly check for NaN to prevent silent math failures
        if math.isnan(val) or pd.isna(val):
            return default
        return float(val)
        
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return default
            
    try:
        f = float(val)
        # [FIX] Check for NaN after string conversion
        if math.isnan(f):
            return default
        return f
    except (ValueError, TypeError, OverflowError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely cast a value to int."""
    f = safe_float(val, float(default))
    return int(f)


def calculate_r_multiple(entry: float, current: float, sl: float, is_buy: bool) -> float:
    """
    Calculate the current R-multiple (Risk-Reward multiple) of an open position.
    
    Args:
        entry: Entry price
        current: Current market price
        sl: Stop Loss price
        is_buy: True for BUY, False for SELL
        
    Returns:
        Float representing R-multiple (e.g., 1.5 means 1.5R profit)
    """
    entry = safe_float(entry)
    current = safe_float(current)
    sl = safe_float(sl)
    
    risk = abs(entry - sl)
    if risk == 0 or math.isnan(risk):
        return 0.0
        
    if is_buy:
        pnl = current - entry
    else:
        pnl = entry - current
        
    return pnl / risk


def deep_merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    Creates a new dictionary to prevent mutation of the original base.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override if override else base
        
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a float as currency string."""
    val = safe_float(value)
    sign = "-" if val < 0 else ""
    return f"{sign}{symbol}{abs(val):,.2f}"