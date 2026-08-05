"""
Portfolio Risk Manager & Position Sizing Engine
Institutional-Grade Risk Control Layer.

Responsibilities:
  1. Dynamic Position Sizing (Kelly Criterion Integration)
  2. Micro-Account Lot Size Clamping (Hard Cap 0.01 - 0.03)
  3. Direction Lock (Prevent Hedging / Locking)
  4. Daily Loss Limit & Circuit Breakers
  5. Max Exposure / Position Limits
"""
import MetaTrader5 as mt5
import logging
import time
from typing import Dict, List, Tuple
from datetime import datetime
from config import config


class RiskManager:
    """
    Manages portfolio risk, exposure limits, and position sizing.
    Acts as the final gatekeeper before order execution.
    """
    
    def __init__(self, risk_per_trade_pct: float = 1.0, max_open_positions: int = 4,
                 max_pending_orders: int = 5, max_daily_loss_pct: float = 3.0, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # =========================================================================
        # MODE-SPECIFIC RISK PARAMETERS
        # =========================================================================
        if getattr(config, 'micro_account_mode', False):
            self.risk_per_trade_pct = getattr(config, 'micro_risk_per_trade_pct', 0.5)
            self.max_daily_loss_pct = getattr(config, 'scalp_max_daily_loss_pct', 2.0)
            self.mode_name = 'MICRO_ACCOUNT'
        elif getattr(config, 'scalping_mode', False):
            self.risk_per_trade_pct = getattr(config, 'scalp_risk_per_trade_pct', 0.5)
            self.max_daily_loss_pct = getattr(config, 'scalp_max_daily_loss_pct', 2.0)
            self.mode_name = 'SCALPING'
        else:
            self.risk_per_trade_pct = risk_per_trade_pct
            self.max_daily_loss_pct = max_daily_loss_pct
            self.mode_name = 'NORMAL'
            
        self.max_open_positions = max_open_positions
        self.max_pending_orders = max_pending_orders
        
        # Daily Tracking
        self.daily_start_capital = 0.0
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self._last_date = None
        
        # Caches
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        
        self.logger.info(
            f"[RISK] Initialized in {self.mode_name} mode | "
            f"Base Risk: {self.risk_per_trade_pct}% | "
            f"Max Daily Loss: {self.max_daily_loss_pct}%"
        )

    def _get_symbol_info(self):
        """Get symbol info with caching."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_time < 5.0):
            return self._symbol_info_cache
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            if mt5.symbol_select(self.symbol, True):
                symbol_info = mt5.symbol_info(self.symbol)
                
        if symbol_info:
            self._symbol_info_cache = symbol_info
            self._symbol_info_time = current_time
            
        return symbol_info

    def check_daily_reset(self):
        """Reset daily counters at the start of a new trading day."""
        today = datetime.now().date()
        if self._last_date != today:
            acc = mt5.account_info()
            if acc:
                self.daily_start_capital = acc.balance
            else:
                self.daily_start_capital = 0.0
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self._last_date = today
            self.logger.info(f"[RISK] Daily reset. Starting capital: ${self.daily_start_capital:.2f}")

    def calculate_position_size(self, entry_price: float, sl_price: float,
                                account_balance: float, position_multiplier: float = 1.0,
                                risk_pct: float = None) -> Tuple[float, str]:
        """
        Calculate position size based on fixed fractional risk or Kelly Criterion.
        Applies Micro-Account Lot Size Clamping (0.01 - 0.03).
        """
        self.check_daily_reset()
        from config import config
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
        
        # =========================================================================
        # MICRO-ACCOUNT SL OVERRIDE
        # =========================================================================
        # Micro-Account Mode: Override SL distance if configured
        if getattr(config, 'micro_account_mode', False):
            max_sl_distance = getattr(config, 'micro_sl_distance_usd', 16.0)
            if stop_distance > max_sl_distance:
                self.logger.info(f"[MICRO] SL distance adjusted from {stop_distance:.2f} to {max_sl_distance:.2f} USD")
                stop_distance = max_sl_distance
        # Calculate risk amount in USD
        risk_amount = account_balance * (risk_pct / 100.0)
                
        # Calculate raw volume based on risk
        raw_volume = risk_amount / (stop_distance * contract_size)
                
        # Apply position multiplier (from Expert Signal Scorer / Kelly)
        adjusted_volume = raw_volume * max(0.1, min(position_multiplier, 3.0))
                
        # Round to broker's volume step
        volume = round(adjusted_volume / volume_step) * volume_step
                
        # Apply broker volume constraints
        volume = max(volume_min, min(volume, volume_max))

        # if getattr(config, 'micro_account_mode', False):
        #     micro_min_lot = getattr(config, 'micro_min_lot_size', 0.01)
        #     micro_max_lot = getattr(config, 'micro_max_lot_size', 0.03)
    
        #     if volume < micro_min_lot:
        #         volume = micro_min_lot
        #     elif volume > micro_max_lot:
        #         volume = micro_max_lot
        #     max_sl_distance = getattr(config, 'micro_sl_distance_usd', 16.0)
        #     if stop_distance > max_sl_distance:
        #         self.logger.info(f"[MICRO] SL distance adjusted from {stop_distance:.2f} to {max_sl_distance:.2f} USD")
        #         stop_distance = max_sl_distance
        
        
        
        # =========================================================================
        # MICRO-ACCOUNT LOT SIZE CLAMPING (HARD CAP)
        # =========================================================================
        if getattr(config, 'micro_account_mode', False):
            micro_min_lot = getattr(config, 'micro_min_lot_size', 0.01)
            micro_max_lot = getattr(config, 'micro_max_lot_size', 0.03)
            
            original_volume = volume
            
            if volume < micro_min_lot:
                volume = micro_min_lot
                effective_risk = (volume * stop_distance * contract_size / account_balance) * 100.0
                self.logger.warning(
                    f"[MICRO] Calculated vol {original_volume:.2f} < min {micro_min_lot}. "
                    f"Forced to {micro_min_lot} (Effective risk: {effective_risk:.2f}%)"
                )
            elif volume > micro_max_lot:
                volume = micro_max_lot
                effective_risk = (volume * stop_distance * contract_size / account_balance) * 100.0
                self.logger.info(
                    f"[MICRO] Calculated vol {original_volume:.2f} > max {micro_max_lot}. "
                    f"Capped at {micro_max_lot} (Reduced risk: {effective_risk:.2f}%)"
                )
            else:
                effective_risk = (volume * stop_distance * contract_size / account_balance) * 100.0
                self.logger.debug(
                    f"[MICRO] Vol {volume:.2f} within bounds [{micro_min_lot}-{micro_max_lot}] "
                    f"(Risk: {effective_risk:.2f}%)"
                )
        
        return volume, f"[OK] Vol={volume:.2f} (Risk={risk_pct:.2f}%, SL_Dist={stop_distance:.2f})"

    def validate_new_trade(self, signal: dict, current_positions: List[Dict],
                           current_pending_orders: List[Dict], account_balance: float,
                           is_pending_order: bool = False, dynamic_risk_pct: float = None) -> Dict:
        """
        Master validation gate for all new trade signals.
        Checks duplicates, direction conflicts, daily limits, and exposure.
        """
        self.check_daily_reset()
        
        result = {'allowed': False, 'reason': 'Unknown error', 'suggested_volume': 0.0}
        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')
        signal_type = signal.get('signal', '')
        
        # =========================================================================
        # 1. DUPLICATE PREVENTION
        # =========================================================================
        for pos in current_positions:
            if pos.get('strategy') == strategy_name:
                return {'allowed': False, 'reason': f"[FAIL] Duplicate active: {strategy_name}", 'suggested_volume': 0.0}
        for order in current_pending_orders:
            if order.get('strategy') == strategy_name:
                return {'allowed': False, 'reason': f"[FAIL] Duplicate pending: {strategy_name}", 'suggested_volume': 0.0}

        # =========================================================================
        # 2. DIRECTION CONFLICT CHECK (PREVENT HEDGING)
        # =========================================================================
        if getattr(config, 'direction_lock_enabled', True):
            is_buy_signal = 'BUY' in signal_type
            
            # Check active positions
            for pos in current_positions:
                pos_type = pos.get('position_type', '')
                is_buy_pos = (pos_type == 'BUY')
                
                if is_buy_signal and not is_buy_pos:
                    return {
                        'allowed': False,
                        'reason': f"[FAIL] Direction conflict: Cannot BUY while SELL active (ticket {pos.get('ticket')})",
                        'suggested_volume': 0.0
                    }
                elif not is_buy_signal and is_buy_pos:
                    return {
                        'allowed': False,
                        'reason': f"[FAIL] Direction conflict: Cannot SELL while BUY active (ticket {pos.get('ticket')})",
                        'suggested_volume': 0.0
                    }
            
            # Check pending orders
            for order in current_pending_orders:
                order_type = order.get('order_type', '')
                is_buy_order = 'BUY' in order_type
                
                if is_buy_signal and not is_buy_order:
                    return {
                        'allowed': False,
                        'reason': f"[FAIL] Direction conflict: Cannot BUY while SELL pending (ticket {order.get('ticket')})",
                        'suggested_volume': 0.0
                    }
                elif not is_buy_signal and is_buy_order:
                    return {
                        'allowed': False,
                        'reason': f"[FAIL] Direction conflict: Cannot SELL while BUY pending (ticket {order.get('ticket')})",
                        'suggested_volume': 0.0
                    }

        # =========================================================================
        # 3. DAILY LOSS LIMIT CHECK
        # =========================================================================
        if self.daily_start_capital > 0:
            daily_loss_pct = (self.daily_pnl / self.daily_start_capital) * 100
            if daily_loss_pct <= -self.max_daily_loss_pct:
                return {
                    'allowed': False, 
                    'reason': f"[FAIL] Daily loss limit breached: {daily_loss_pct:.2f}%", 
                    'suggested_volume': 0.0
                }
                
        # =========================================================================
        # 4. MAX POSITIONS / PENDING ORDERS CHECK
        # =========================================================================
        if not is_pending_order and len(current_positions) >= self.max_open_positions:
            return {'allowed': False, 'reason': f"[FAIL] Max active positions ({self.max_open_positions})", 'suggested_volume': 0.0}
        if is_pending_order and len(current_pending_orders) >= self.max_pending_orders:
            return {'allowed': False, 'reason': f"[FAIL] Max pending orders ({self.max_pending_orders})", 'suggested_volume': 0.0}
            
        # =========================================================================
        # 5. EXTRACT SIGNAL DATA & KELLY RISK
        # =========================================================================
        entry_price = meta.get('entry_price')
        sl_price = meta.get('sl_price')
        multiplier = meta.get('position_multiplier', 1.0)
        
        if not entry_price or not sl_price or account_balance <= 0:
            return {'allowed': False, 'reason': "[FAIL] Missing entry/SL or balance", 'suggested_volume': 0.0}
            
        effective_risk_pct = dynamic_risk_pct if dynamic_risk_pct is not None else self.risk_per_trade_pct
        
        if effective_risk_pct <= 0:
            return {'allowed': False, 'reason': "[FAIL] Kelly Criterion blocked trade (Zero/Negative Risk)", 'suggested_volume': 0.0}
            
        # =========================================================================
        # 6. CALCULATE POSITION SIZE
        # =========================================================================
        volume, calc_reason = self.calculate_position_size(
            entry_price, sl_price, account_balance, multiplier, effective_risk_pct
        )
        
        if volume <= 0: 
            return {'allowed': False, 'reason': f"[FAIL] Sizing failed: {calc_reason}", 'suggested_volume': 0.0}
        
        # Success
        result['allowed'] = True
        result['suggested_volume'] = volume
        result['reason'] = f"[OK] {strategy_name} | {calc_reason}"
        
        # Increment daily trade count on successful validation
        self.daily_trade_count += 1
        
        return result

    def update_daily_pnl(self, realized_pnl: float):
        """Update daily PnL tracker when a position is closed."""
        self.daily_pnl += realized_pnl
        self.logger.debug(f"[RISK] Daily PnL updated: {realized_pnl:+.2f} | Total: {self.daily_pnl:+.2f}")
        
    def get_daily_stats(self) -> Dict:
        """Get current daily statistics for context passing."""
        self.check_daily_reset()
        daily_loss_pct = (self.daily_pnl / self.daily_start_capital) * 100 if self.daily_start_capital > 0 else 0.0
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': daily_loss_pct,
            'daily_trade_count': self.daily_trade_count,
            'start_capital': self.daily_start_capital
        }
    def reset_daily(self, new_balance: float):
        """
        Reset daily tracking variables at the start of a new trading day.
        Called by OrderManager.check_daily_reset() when date changes.
        
        Args:
            new_balance: Current account balance to use as starting capital for the day
        """
        self.daily_start_capital = new_balance
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.logger.info(
            f"[RISK] Daily reset complete | "
            f"New starting capital: ${new_balance:.2f}"
        )