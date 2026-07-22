"""
Order Manager.
Handles all MT5 order operations including placement, modification, and closure.
Includes Micro-Account trailing logic, trade history recording, and MT5 retcode handling.
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime, timedelta

from config import config
from execution.risk_manager import RiskManager
from execution.state_manager import StateManager
from execution.friction_filter import FrictionFilter
from core.atr_cache import ATRCache
from core.expert_signal_scorer import ExpertSignalScorer


class OrderManager:
    def __init__(self, symbol: str, magic_number: int, max_slippage: int,
                 risk_per_trade_pct: float, max_open_positions: int,
                 max_pending_orders: int, pending_order_timeout_minutes: int,
                 state_db_path: str):
        self.symbol = symbol
        self.magic_number = magic_number
        self.max_slippage = max_slippage
        self.risk_manager = RiskManager(risk_per_trade_pct, max_open_positions, max_pending_orders, symbol=symbol)
        self.state_manager = StateManager(state_db_path)
        self.friction_filter = FrictionFilter(symbol)
        self.scorer = ExpertSignalScorer(state_db_path)
        self.pending_order_timeout = pending_order_timeout_minutes
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_signal(self, signal: dict, account_balance: float, current_atr: float, context: dict = None) -> bool:
        """
        Process a trading signal: score it, validate, size position, and send to MT5.
        """
        if signal.get('signal') == 'NEUTRAL':
            return False
        
        # Score the signal using Expert Signal Scorer
        scoring_result = self.scorer.score_signal(signal, context or {})
        
        if not scoring_result.get('should_trade', False):
            self.logger.info(f"[SCORER] Signal rejected by scorer | Grade: {scoring_result.get('grade')} | Score: {scoring_result.get('score')}")
            return False
        
        # Inject score and regime into meta
        meta = signal.get('meta', {})
        meta['expert_score'] = scoring_result.get('score', 50.0)
        meta['expert_grade'] = scoring_result.get('grade', 'C')
        meta['regime'] = (context or {}).get('regime_name', 'UNKNOWN')
        signal['meta'] = meta
        
        # Apply position multiplier from scorer
        position_multiplier = scoring_result.get('position_multiplier', 1.0)
        
        # Validate with friction filter
        friction_result = self.friction_filter.validate_entry(signal, current_atr)
        if not friction_result.get('valid', False):
            self.logger.info(f"[FRICTION] Signal rejected: {friction_result.get('reason')}")
            return False
        
        meta = signal['meta']
        entry_price = float(meta['entry_price'])
        sl_price = float(meta['sl_price'])
        tp_price = float(meta['tp_price'])
        signal_type = signal['signal']
        
        # Calculate position size
        volume, sizing_reason = self.risk_manager.calculate_position_size(
            entry_price, sl_price, account_balance, position_multiplier
        )
        
        if volume <= 0:
            self.logger.info(f"[RISK] Position sizing failed: {sizing_reason}")
            return False
        
        # Check daily loss limit
        if self.risk_manager.check_daily_loss_limit(account_balance):
            return False
        
        # Count current positions and pending orders
        active_positions = self.state_manager.get_active_positions(self.symbol)
        pending_orders = self.state_manager.get_pending_orders(self.symbol)
        
        is_pending = 'LIMIT' in signal_type.upper() or 'STOP' in signal_type.upper()
        
        if is_pending:
            if len(pending_orders) >= self.risk_manager.max_pending_orders:
                self.logger.info("[RISK] Max pending orders reached")
                return False
        else:
            if len(active_positions) >= self.risk_manager.max_open_positions:
                self.logger.info("[RISK] Max open positions reached")
                return False
        
        # Send order to MT5
        success = self._send_order(signal, volume)
        
        if success:
            self.logger.info(f"[EXEC] Order placed | {signal_type} | Vol: {volume} | {sizing_reason}")
        
        return success

    def _send_order(self, signal: dict, volume: float) -> bool:
        """Send an order to MT5 terminal."""
        meta = signal['meta']
        signal_type = signal['signal'].upper()
        
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return False
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False
        
        point = symbol_info.point
        digits = symbol_info.digits
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "magic": self.magic_number,
            "comment": meta.get('strategy', 'BOT'),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Determine order type and price
        if 'BUY_MARKET' in signal_type:
            request['type'] = mt5.ORDER_TYPE_BUY
            request['price'] = tick.ask
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        elif 'SELL_MARKET' in signal_type:
            request['type'] = mt5.ORDER_TYPE_SELL
            request['price'] = tick.bid
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        elif 'BUY_LIMIT' in signal_type:
            request['action'] = mt5.TRADE_ACTION_PENDING
            request['type'] = mt5.ORDER_TYPE_BUY_LIMIT
            request['price'] = round(float(meta['entry_price']), digits)
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        elif 'SELL_LIMIT' in signal_type:
            request['action'] = mt5.TRADE_ACTION_PENDING
            request['type'] = mt5.ORDER_TYPE_SELL_LIMIT
            request['price'] = round(float(meta['entry_price']), digits)
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        elif 'BUY_STOP' in signal_type:
            request['action'] = mt5.TRADE_ACTION_PENDING
            request['type'] = mt5.ORDER_TYPE_BUY_STOP
            request['price'] = round(float(meta['entry_price']), digits)
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        elif 'SELL_STOP' in signal_type:
            request['action'] = mt5.TRADE_ACTION_PENDING
            request['type'] = mt5.ORDER_TYPE_SELL_STOP
            request['price'] = round(float(meta['entry_price']), digits)
            request['sl'] = round(float(meta['sl_price']), digits)
            request['tp'] = round(float(meta['tp_price']), digits)
        else:
            self.logger.error(f"[FAIL] Unknown signal type: {signal_type}")
            return False
        
        # Send the order
        result = mt5.order_send(request)
        
        if result is None:
            self.logger.error(f"[FAIL] MT5 order_send returned None")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"[FAIL] Order rejected | Retcode: {result.retcode} | Comment: {result.comment}")
            return False
        
        # Save to state database
        ticket = result.order
        
        if request['action'] == mt5.TRADE_ACTION_PENDING:
            expiration_time = datetime.now() + timedelta(minutes=meta.get('expiration_bars', 10) * 15)
            self.state_manager.save_pending_order(
                ticket, self.symbol, meta.get('strategy'), signal_type, volume,
                float(meta['entry_price']), float(meta['sl_price']), float(meta['tp_price']),
                expiration_time, meta
            )
        else:
            position_type = 'BUY' if 'BUY' in signal_type else 'SELL'
            self.state_manager.save_active_position(
                ticket, self.symbol, meta.get('strategy'), position_type, volume,
                result.price, float(meta['sl_price']), float(meta['tp_price']), meta
            )
        
        return True

    def manage_active_positions(self, current_prices: dict, data: Dict[str, pd.DataFrame] = None, regime_context: dict = None):
        """Manage trailing stops, partial closes, and dynamic exits for active positions."""
        positions = self.state_manager.get_active_positions(self.symbol)
        if not positions:
            return
        
        df_m5 = data.get('M5') if data else None
        
        for pos in positions:
            ticket = pos['ticket']
            mt5_pos_list = mt5.positions_get(ticket=ticket)
            
            if not mt5_pos_list:
                # Position closed externally, remove from local state
                self.state_manager.remove_active_position(ticket)
                continue
            
            mt5_pos = mt5_pos_list[0]
            
            # Update trailing stop
            self._update_trailing_stop(pos, mt5_pos, df_m5)
            
            # Check partial close
            self._evaluate_partial_close(pos, mt5_pos)

    def _update_trailing_stop(self, pos: dict, mt5_pos, df_m5: pd.DataFrame):
        """
        Update trailing stop based on Strategy Category and Trailing Method.
        Includes Micro-Account Fixed-Dollar Trailing Logic.
        """
        meta = pos.get('meta_data', {})
        
        if not meta.get('trailing_enabled', True):
            return
        
        is_buy = mt5_pos.type == mt5.ORDER_TYPE_BUY
        current_price = mt5_pos.price_current
        current_sl = mt5_pos.sl
        entry_price = mt5_pos.price_open
        strategy_category = meta.get('strategy_category', 'GENERAL')
        trailing_method = meta.get('trailing_method', 'default')
        
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return
        
        digits = getattr(symbol_info, 'digits', 2)
        min_distance = (max(getattr(symbol_info, 'trade_stops_level', 10), 10) + 2) * getattr(symbol_info, 'point', 0.01)
        
        new_sl = current_sl
        
        # Micro-Account Fixed-Dollar Trailing
        if getattr(config, 'micro_account_mode', False):
            profit_usd = (current_price - entry_price) if is_buy else (entry_price - current_price)
            be_trigger = getattr(config, 'micro_breakeven_trigger_usd', 3.5)
            trail_increment = getattr(config, 'micro_trail_increment_usd', 2.5)
            be_buffer = 0.5
            
            # Phase 1: Ultra-Fast Breakeven
            if profit_usd >= be_trigger and (current_sl == 0.0 or (is_buy and current_sl < entry_price) or (not is_buy and current_sl > entry_price)):
                if is_buy:
                    calculated_sl = entry_price + be_buffer
                    if calculated_sl < (current_price - min_distance):
                        new_sl = calculated_sl
                else:
                    calculated_sl = entry_price - be_buffer
                    if calculated_sl > (current_price + min_distance):
                        new_sl = calculated_sl
            
            # Phase 2: Micro-Increment Trailing
            elif profit_usd >= (be_trigger + trail_increment):
                profit_since_be = profit_usd - be_trigger
                increments_passed = int(profit_since_be / trail_increment)
                
                if is_buy:
                    calculated_sl = entry_price + be_buffer + (increments_passed * trail_increment)
                    if (current_sl <= 0.0 or calculated_sl > current_sl) and calculated_sl < (current_price - min_distance):
                        new_sl = calculated_sl
                else:
                    calculated_sl = entry_price - be_buffer - (increments_passed * trail_increment)
                    if (current_sl <= 0.0 or calculated_sl < current_sl) and calculated_sl > (current_price + min_distance):
                        new_sl = calculated_sl

        # Normal Mode: Chandelier for TREND, ATR for others
        else:
            if df_m5 is None or len(df_m5) < 22:
                return
            
            if 'atr' not in df_m5.columns:
                df_work = df_m5.copy()
                df_work['atr'] = ATRCache.get_atr(df_m5, 14).to_numpy()
            else:
                df_work = df_m5
            
            atr = df_work['atr'].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                return
            
            if strategy_category == 'TREND':
                from core.chandelier_engine import ChandelierEngine
                chandelier = ChandelierEngine()
                new_sl = chandelier.calculate_trailing_stop(
                    df_work, is_buy, current_sl, entry_price, current_price,
                    lookback=22, multiplier=3.0, min_distance=min_distance
                ) or current_sl
            else:
                trail_mult = meta.get('trail_mult', 1.5)
                trail_distance = atr * trail_mult
                
                if is_buy:
                    calculated_sl = current_price - trail_distance
                    if (current_sl <= 0.0 or calculated_sl > current_sl) and calculated_sl < (current_price - min_distance):
                        new_sl = calculated_sl
                else:
                    calculated_sl = current_price + trail_distance
                    if (current_sl <= 0.0 or calculated_sl < current_sl) and calculated_sl > (current_price + min_distance):
                        new_sl = calculated_sl
        
        # Execute modification with retcode handling
        if new_sl != current_sl and new_sl > 0:
            normalized_sl = round(new_sl, digits)
            if (is_buy and normalized_sl >= current_price) or (not is_buy and normalized_sl <= current_price):
                return
            self._modify_sl(pos['ticket'], normalized_sl)

    def _modify_sl(self, ticket: int, new_sl: float):
        """
        Modify SL on MT5 with proper retcode handling.
        [FIX] Checks MT5 retcode before updating local state to prevent Ghost Positions.
        """
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            # Position already closed externally, clean up local state
            self.state_manager.remove_active_position(ticket)
            return
        
        pos = pos_list[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            self.logger.error(f"[FAIL] MT5 order_send returned None for SL modification on ticket {ticket}")
            return
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.state_manager.update_trailing_stop(ticket, new_sl)
            self.logger.info(f"[EXEC] Modified SL for ticket {ticket} | New SL: {new_sl:.2f}")
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
            self.logger.warning(f"[WARN] MT5 rejected SL modification (Invalid Stops) for ticket {ticket}")
        elif result.retcode == mt5.TRADE_RETCODE_POSITION_NOT_FOUND:
            self.logger.info(f"[INFO] Position {ticket} not found during SL modification. Cleaning up.")
            self.state_manager.remove_active_position(ticket)
        else:
            self.logger.error(f"[FAIL] MT5 SL modification failed for ticket {ticket} | Retcode: {result.retcode}")

    def _evaluate_partial_close(self, pos: dict, mt5_pos):
        """Evaluate and execute partial close based on Micro-Account or Normal Mode."""
        meta = pos.get('meta_data', {})
        if not meta.get('partial_close_enabled', True):
            return
        
        ticket = pos['ticket']
        entry_price = mt5_pos.price_open
        current_price = mt5_pos.price_current
        sl_price = pos.get('sl', 0)
        is_buy = mt5_pos.type == mt5.ORDER_TYPE_BUY
        
        risk = abs(entry_price - sl_price)
        if risk == 0:
            return
        
        pnl_usd = (current_price - entry_price) if is_buy else (entry_price - current_price)
        
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            return
        
        vol_step = getattr(symbol_info, 'volume_step', 0.01)
        min_vol = getattr(symbol_info, 'volume_min', 0.01)
        
        # Micro-Account Partial Close
        if getattr(config, 'micro_account_mode', False):
            partial_trigger = getattr(config, 'micro_partial_close_trigger_usd', 5.0)
            partial_percent = getattr(config, 'micro_partial_close_percent', 0.50)
            
            # Check if already partially closed (by checking volume)
            original_volume = meta.get('original_volume', mt5_pos.volume)
            if mt5_pos.volume < original_volume * 0.9:
                return  # Already partially closed
            
            if pnl_usd >= partial_trigger:
                vol = max(min_vol, round((mt5_pos.volume * partial_percent) / vol_step) * vol_step)
                if vol < mt5_pos.volume:
                    self._partial_close(ticket, vol, is_buy)
                    self.logger.info(f"[MICRO] Partial close {partial_percent*100:.0f}% at {pnl_usd:.2f} USD profit")
        
        # Normal Mode: R-Multiple based
        else:
            pnl_r = pnl_usd / risk
            
            if pnl_r >= 1.5:
                vol = max(min_vol, round((mt5_pos.volume * 0.33) / vol_step) * vol_step)
                if vol < mt5_pos.volume:
                    self._partial_close(ticket, vol, is_buy)

    def _partial_close(self, ticket: int, volume: float, is_buy: bool):
        """Execute partial close on MT5."""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "position": ticket,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": self.max_slippage,
            "magic": self.magic_number,
            "comment": "Partial Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"[EXEC] Partial close executed | Ticket: {ticket} | Volume: {volume}")

    def manage_pending_orders(self, current_time: pd.Timestamp):
        """Manage pending orders: check for fills and expirations."""
        pending_orders = self.state_manager.get_pending_orders(self.symbol)
        
        for order in pending_orders:
            ticket = order['ticket']
            mt5_orders = mt5.orders_get(ticket=ticket)
            
            if not mt5_orders:
                # Order was filled or deleted
                self.state_manager.remove_pending_order(ticket)
                
                # Check if it was filled (now in positions)
                mt5_pos = mt5.positions_get(ticket=ticket)
                if mt5_pos:
                    # Convert pending to active
                    pos = mt5_pos[0]
                    meta = order.get('meta_data', {})
                    position_type = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
                    self.state_manager.save_active_position(
                        ticket, self.symbol, order.get('strategy'), position_type,
                        pos.volume, pos.price_open, pos.sl, pos.tp, meta
                    )
                continue
            
            # Check expiration
            expiration_str = order.get('expiration_time')
            if expiration_str:
                try:
                    expiration_time = pd.to_datetime(expiration_str)
                    if current_time.tzinfo is None:
                        current_time = current_time.tz_localize('UTC')
                    if expiration_time.tzinfo is None:
                        expiration_time = expiration_time.tz_localize('UTC')
                    
                    if current_time >= expiration_time:
                        # Cancel expired order
                        self._cancel_order(ticket)
                except Exception:
                    pass

    def _cancel_order(self, ticket: int):
        """Cancel a pending order."""
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.state_manager.remove_pending_order(ticket)
            self.logger.info(f"[EXEC] Cancelled expired pending order {ticket}")

    def sync_with_mt5(self):
        """
        Synchronize local state with MT5 terminal.
        Records closed positions to trade_history.
        """
        self.logger.debug("[SYNC] Synchronizing state with MT5...")
        
        mt5_pos = mt5.positions_get(symbol=self.symbol) or []
        mt5_tickets = {p.ticket for p in mt5_pos}
        
        local_pos = self.state_manager.get_active_positions(self.symbol)
        
        closed_positions = []
        for pos in local_pos:
            if pos['ticket'] not in mt5_tickets:
                closed_positions.append(pos)
        
        for closed_pos in closed_positions:
            self._record_closed_position(closed_pos)
            self.state_manager.remove_active_position(closed_pos['ticket'])
        
        if closed_positions:
            self.logger.info(f"[SYNC] Recorded {len(closed_positions)} closed positions to trade_history")

    def _record_closed_position(self, closed_pos: dict):
        """
        Record a closed position to trade_history by fetching deal information from MT5.
        """
        try:
            ticket = closed_pos['ticket']
            
            from_date = datetime.now() - timedelta(days=7)
            to_date = datetime.now()
            
            deals = mt5.history_deals_get(from_date, to_date, position=ticket)
            
            if not deals:
                self.logger.warning(f"[SYNC] No closing deal found for ticket {ticket}")
                self._record_trade_with_estimates(closed_pos)
                return
            
            closing_deal = deals[-1]
            
            profit = closing_deal.profit
            commission = closing_deal.commission
            swap = closing_deal.swap
            exit_price = closing_deal.price
            close_time = datetime.fromtimestamp(closing_deal.time).isoformat()
            
            meta_data = closed_pos.get('meta_data', {})
            strategy_name = closed_pos.get('strategy', 'Unknown')
            
            position_type = closed_pos.get('position_type', 'BUY')
            direction = 'BUY' if position_type == 'BUY' else 'SELL'
            
            # Record using StrategyPerformanceTracker
            from core.strategy_performance_tracker import StrategyPerformanceTracker
            tracker = StrategyPerformanceTracker()
            tracker.record_trade_enhanced(
                ticket=ticket,
                symbol=self.symbol,
                strategy=strategy_name,
                direction=direction,
                entry_price=closed_pos.get('entry_price', 0.0),
                exit_price=exit_price,
                sl_price=closed_pos.get('sl', 0.0),
                tp_price=closed_pos.get('tp', 0.0),
                volume=closed_pos.get('volume', 0.0),
                profit=profit,
                commission=commission,
                swap=swap,
                open_time=closed_pos.get('open_time', ''),
                close_time=close_time,
                entry_reason=meta_data.get('entry_reason', 'signal'),
                exit_reason=self._determine_exit_reason(closed_pos, exit_price),
                is_pending=False,
                order_type='MARKET',
                expected_entry=closed_pos.get('expected_entry', 0.0),
                meta_data=meta_data
            )
            
            self.logger.info(
                f"[SYNC] Recorded closed position {ticket} | "
                f"Strategy: {strategy_name} | Profit: {profit:.2f} | "
                f"Exit: {exit_price:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error recording closed position {closed_pos.get('ticket')}: {e}", exc_info=True)
            self._record_trade_with_estimates(closed_pos)

    def _record_trade_with_estimates(self, closed_pos: dict):
        """Fallback: Record trade with estimated values when deal history is not available."""
        try:
            ticket = closed_pos['ticket']
            meta_data = closed_pos.get('meta_data', {})
            strategy_name = closed_pos.get('strategy', 'Unknown')
            position_type = closed_pos.get('position_type', 'BUY')
            direction = 'BUY' if position_type == 'BUY' else 'SELL'
            
            entry_price = closed_pos.get('entry_price', 0.0)
            sl_price = closed_pos.get('sl', 0.0)
            tp_price = closed_pos.get('tp', 0.0)
            volume = closed_pos.get('volume', 0.0)
            
            trailing_stop = closed_pos.get('trailing_stop_level')
            if trailing_stop and trailing_stop > 0:
                exit_price = trailing_stop
                exit_reason = 'Trailing Stop'
            else:
                exit_price = sl_price if sl_price > 0 else entry_price
                exit_reason = 'SL Hit (Estimated)'
            
            contract_size = 100
            if position_type == 'BUY':
                profit = (exit_price - entry_price) * volume * contract_size
            else:
                profit = (entry_price - exit_price) * volume * contract_size
            
            from core.strategy_performance_tracker import StrategyPerformanceTracker
            tracker = StrategyPerformanceTracker()
            tracker.record_trade_enhanced(
                ticket=ticket,
                symbol=self.symbol,
                strategy=strategy_name,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                sl_price=sl_price,
                tp_price=tp_price,
                volume=volume,
                profit=profit,
                commission=0.0,
                swap=0.0,
                open_time=closed_pos.get('open_time', ''),
                close_time=datetime.now().isoformat(),
                entry_reason=meta_data.get('entry_reason', 'signal'),
                exit_reason=exit_reason,
                is_pending=False,
                order_type='MARKET',
                expected_entry=closed_pos.get('expected_entry', 0.0),
                meta_data=meta_data
            )
            
            self.logger.warning(
                f"[SYNC] Recorded closed position {ticket} with ESTIMATED values | "
                f"Profit: {profit:.2f} (estimated)"
            )
            
        except Exception as e:
            self.logger.error(f"[FAIL] Error in fallback recording for ticket {ticket}: {e}", exc_info=True)

    def _determine_exit_reason(self, closed_pos: dict, exit_price: float) -> str:
        """Determine the reason for position closure."""
        sl_price = closed_pos.get('sl', 0.0)
        tp_price = closed_pos.get('tp', 0.0)
        trailing_stop = closed_pos.get('trailing_stop_level', 0.0)
        
        tolerance = 0.5
        
        if trailing_stop and trailing_stop > 0:
            if abs(exit_price - trailing_stop) < tolerance:
                return 'Trailing Stop'
        
        if sl_price and sl_price > 0:
            if abs(exit_price - sl_price) < tolerance:
                return 'Stop Loss'
        
        if tp_price and tp_price > 0:
            if abs(exit_price - tp_price) < tolerance:
                return 'Take Profit'
        
        return 'Manual/Unknown'