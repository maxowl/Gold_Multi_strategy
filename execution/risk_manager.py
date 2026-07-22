"""
Risk Manager.
Calculates position sizing based on account balance, risk percentage, and Micro-Account constraints.
"""
import MetaTrader5 as mt5
import logging
from typing import Tuple
from config import config


class RiskManager:
    def __init__(self, risk_per_trade_pct: float = 1.0, max_open_positions: int = 4,
                 max_pending_orders: int = 5, max_daily_loss_pct: float = 3.0, symbol: str = "XAUUSD"):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.max_pending_orders = max_pending_orders
        self.max_daily_loss_pct = max_daily_loss_pct
        self.symbol = symbol
        self.daily_start_capital = 0.0
        self.daily_pnl = 0.0
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Micro-Account Mode adjustments
        if getattr(config, 'micro_account_mode', False):
            self.risk_per_trade_pct = getattr(config, 'micro_risk_per_trade_pct', 0.5)
            self.max_daily_loss_pct = getattr(config, 'scalp_max_daily_loss_pct', 2.0)

    def _get_symbol_info(self):
        return mt5.symbol_info(self.symbol)

    def calculate_position_size(self, entry_price: float, sl_price: float,
                                account_balance: float, position_multiplier: float = 1.0,
                                risk_pct: float = None) -> Tuple[float, str]:
        """
        Calculate position size with Micro-Account Lot Size Clamping.
        [FIX] Uses config variables instead of hardcoded values for Micro-Account bounds.
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade_pct
            
        if entry_price <= 0 or sl_price <= 0 or account_balance <= 0:
            return 0.0, "[FAIL] Invalid prices or balance"
            
        symbol_info = self._get_symbol_info()
        if not symbol_info: 
            return 0.0, f"[FAIL] Symbol {self.symbol} not found"
        
        contract_size = getattr(symbol_info, 'trade_contract_size', 100)
        volume_min = getattr(symbol_info, 'volume_min', 0.01)
        volume_max = getattr(symbol_info, 'volume_max', 100.0)
        volume_step = getattr(symbol_info, 'volume_step', 0.01)
        
        # Calculate stop distance
        stop_distance = abs(entry_price - sl_price)
        if stop_distance == 0: 
            return 0.0, "[FAIL] Stop distance is zero"
        
        # Micro-Account Mode: Override SL distance if configured
        if getattr(config, 'micro_account_mode', False):
            max_sl_distance = getattr(config, 'micro_sl_distance_usd', 8.0)
            if stop_distance > max_sl_distance:
                self.logger.info(f"[MICRO] SL distance adjusted from {stop_distance:.2f} to {max_sl_distance:.2f} USD")
                stop_distance = max_sl_distance
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100.0)
        
        # Calculate raw volume based on risk
        raw_volume = risk_amount / (stop_distance * contract_size)
        
        # Apply position multiplier
        adjusted_volume = raw_volume * max(0.1, min(position_multiplier, 3.0))
        
        # Round to volume step
        volume = round(adjusted_volume / volume_step) * volume_step
        
        # Apply broker volume constraints
        volume = max(volume_min, min(volume, volume_max))
        
        # Micro-Account Lot Size Clamping
        if getattr(config, 'micro_account_mode', False):
            # Use config variables instead of hardcoded values
            micro_min_lot = getattr(config, 'micro_min_lot_size', 0.01)
            micro_max_lot = getattr(config, 'micro_max_lot_size', 0.03)
            
            original_volume = volume
            
            if volume < micro_min_lot:
                volume = micro_min_lot
                effective_risk = (volume * stop_distance * contract_size / account_balance * 100)
                self.logger.warning(
                    f"[MICRO] Volume {original_volume:.2f} clamped to min {micro_min_lot} "
                    f"(effective risk: {effective_risk:.2f}%)"
                )
            elif volume > micro_max_lot:
                volume = micro_max_lot
                effective_risk = (volume * stop_distance * contract_size / account_balance * 100)
                self.logger.info(
                    f"[MICRO] Volume {original_volume:.2f} clamped to max {micro_max_lot} "
                    f"(reduced risk: {effective_risk:.2f}%)"
                )
            else:
                effective_risk = (volume * stop_distance * contract_size / account_balance * 100)
                self.logger.info(
                    f"[MICRO] Volume {volume:.2f} within bounds [{micro_min_lot}-{micro_max_lot}] "
                    f"(risk: {effective_risk:.2f}%)"
                )
        
        return volume, f"[OK] Vol={volume:.2f} (Risk={risk_pct:.2f}%, SL_Dist={stop_distance:.2f})"

    def check_daily_loss_limit(self, current_balance: float) -> bool:
        """Check if daily loss limit has been reached."""
        if self.daily_start_capital == 0:
            self.daily_start_capital = current_balance
            return False
        
        daily_loss = (self.daily_start_capital - current_balance) / self.daily_start_capital * 100
        
        if daily_loss >= self.max_daily_loss_pct:
            self.logger.warning(f"[RISK] Daily loss limit reached: {daily_loss:.2f}% >= {self.max_daily_loss_pct}%")
            return True
        
        return False

    def reset_daily(self, new_balance: float):
        """Reset daily counters at start of new day."""
        self.daily_start_capital = new_balance
        self.daily_pnl = 0.0