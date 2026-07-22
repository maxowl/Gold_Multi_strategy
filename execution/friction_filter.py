"""
Friction Filter.
Validates that the expected edge exceeds execution costs (spread + slippage + commission).
Rejects signals that would result in negative net expectancy.
"""
import MetaTrader5 as mt5
import numpy as np
import logging
from typing import Dict
from config import config


class FrictionFilter:
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine thresholds based on active mode
        if config.scalping_mode:
            self.max_spread_points = config.max_spread_points_scalp
            self.max_slippage_points = 10
            self.min_edge_to_friction_ratio = 3.5
        elif config.micro_account_mode:
            self.max_spread_points = 300
            self.max_slippage_points = 15
            self.min_edge_to_friction_ratio = 2.5
        else:
            self.max_spread_points = 300
            self.max_slippage_points = 20
            self.min_edge_to_friction_ratio = 2.0

    def validate_entry(self, signal: dict, current_atr: float, strict_mode: bool = False) -> Dict:
        """
        Validate that a signal's edge exceeds execution friction.
        [FIX] Explicit type casting prevents TypeError from JSON deserialization.
        [FIX] Guard against zero/NaN ATR in Micro-Account mode.
        """
        meta = signal.get('meta', {})
        
        # Explicit type casting to prevent TypeError from JSON deserialization
        try:
            entry_price = float(meta.get('entry_price', 0))
            sl_price = float(meta.get('sl_price', 0))
            tp_price = float(meta.get('tp_price', 0))
        except (ValueError, TypeError):
            return {'valid': False, 'reason': 'Price casting failed'}
        
        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            return {'valid': False, 'reason': 'Invalid prices'}
        
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return {'valid': False, 'reason': 'Cannot get symbol info'}
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return {'valid': False, 'reason': 'Cannot get tick'}
        
        # Current spread check
        current_spread = tick.ask - tick.bid
        spread_points = current_spread / symbol_info.point
        
        if spread_points > self.max_spread_points:
            return {'valid': False, 'reason': f'Spread too wide: {spread_points:.0f} pts (max: {self.max_spread_points})'}
        
        # Micro-Account: Spread to SL ratio check
        if config.micro_account_mode:
            # Guard against zero or NaN ATR
            if current_atr is None or np.isnan(current_atr) or current_atr <= 0:
                current_atr = 1.0
            spread_atr_ratio = current_spread / current_atr
            micro_spread_ratio = getattr(config, 'micro_spread_to_sl_ratio', 0.10)
            if spread_atr_ratio > micro_spread_ratio:
                return {
                    'valid': False,
                    'reason': f'Spread/ATR ratio too high: {spread_atr_ratio:.2f}'
                }
        
        # Net Edge calculation
        friction_cost = current_spread + (self.max_slippage_points * symbol_info.point)
        gross_risk = abs(entry_price - sl_price)
        gross_reward = abs(tp_price - entry_price)
        
        net_risk = gross_risk + friction_cost
        net_reward = gross_reward - friction_cost
        
        if net_reward <= 0:
            return {'valid': False, 'reason': f'Negative net reward: {net_reward:.2f}'}
        
        net_rr = net_reward / net_risk if net_risk > 0 else 0
        
        required_ratio = self.min_edge_to_friction_ratio
        if strict_mode or config.scalping_mode or config.micro_account_mode:
            required_ratio *= 1.3
        
        if net_rr < required_ratio:
            return {'valid': False, 'reason': f'Edge/Friction {net_rr:.2f} < {required_ratio:.2f}'}
        
        return {
            'valid': True,
            'reason': f'OK | Spread: {spread_points:.0f} pts | Net R:R: {net_rr:.2f}',
            'spread_points': spread_points,
            'net_rr': net_rr,
            'friction_cost': friction_cost
        }