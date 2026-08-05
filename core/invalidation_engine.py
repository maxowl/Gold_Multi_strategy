"""
Edge Decay & Time-Price Invalidation Engine
Cuts losing trades early if the market fails to validate the entry thesis,
drastically reducing the Average Loss.
"""
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional


class InvalidationEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def check_edge_decay(self, position: Dict, current_price: float, current_time: pd.Timestamp) -> Optional[str]:
        """
        Evaluate if a trade's edge has decayed based on Time-Price thresholds.
        
        Returns:
            - None: Edge is still valid, keep trade open.
            - str: Reason for invalidation (trigger early close).
        """
        meta = position.get('meta_data', {})
        open_time_str = position.get('open_time')
        entry_price = float(position.get('entry_price', 0))
        initial_sl = float(position.get('sl', 0))
        position_type = position.get('position_type', 'BUY')
        
        if not open_time_str or entry_price == 0 or initial_sl == 0:
            return None
            
        try:
            open_time = pd.to_datetime(open_time_str)
            if current_time.tzinfo is None and open_time.tzinfo is not None:
                current_time = current_time.tz_localize('UTC')
            elif current_time.tzinfo is not None and open_time.tzinfo is None:
                open_time = open_time.tz_localize('UTC')
                
            minutes_open = (current_time - open_time).total_seconds() / 60.0
        except Exception as e:
            self.logger.error(f"[INVALIDATION] Time parse error: {e}")
            return None
            
        # Calculate initial risk and current PnL
        initial_risk = abs(entry_price - initial_sl)
        if initial_risk == 0:
            return None
            
        is_buy = (position_type == 'BUY')
        current_pnl = (current_price - entry_price) if is_buy else (entry_price - current_price)
        pnl_r = current_pnl / initial_risk  # Negative means underwater
        
        # =========================================================================
        # INVALIDATION THRESHOLDS (Configurable per Strategy Category)
        # =========================================================================
        strategy_category = meta.get('strategy_category', 'GENERAL')
        
        # Default Thresholds
        max_stall_minutes = 60.0
        invalidation_r_threshold = -0.25  # Cut if down 25% of initial risk after stall time
        
        if strategy_category == 'SCALP':
            max_stall_minutes = 20.0
            invalidation_r_threshold = -0.15  # Scalps must work immediately
        elif strategy_category == 'TREND':
            max_stall_minutes = 120.0
            invalidation_r_threshold = -0.40  # Trends need more time to develop
            
        # =========================================================================
        # EDGE DECAY EVALUATION
        # =========================================================================
        # Condition: Trade has been open longer than max_stall AND is underwater
        if minutes_open >= max_stall_minutes and pnl_r <= invalidation_r_threshold:
            reason = (
                f"Edge Decay: Open {minutes_open:.0f}m (Max {max_stall_minutes:.0f}m), "
                f"PnL {pnl_r:.2f}R (Threshold {invalidation_r_threshold:.2f}R)"
            )
            self.logger.warning(f"[INVALIDATION] Ticket {position.get('ticket')} | {reason}")
            return reason
            
        # Condition: Immediate Rejection (Price moves against us rapidly in first 10 mins)
        if minutes_open <= 10.0 and pnl_r <= -0.50:
            reason = f"Immediate Rejection: Down {pnl_r:.2f}R within first 10 minutes"
            self.logger.warning(f"[INVALIDATION] Ticket {position.get('ticket')} | {reason}")
            return reason
            
        return None