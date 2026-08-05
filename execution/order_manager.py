"""
Order Management System - Ultimate Master Release (Final Integration)
Combines the BEST of both worlds:

ROBUST MT5 EXECUTION LAYER:
  - Retry Loop with FOK/IOC Fallback
  - Filling Mode Detection (Bitmask parsing)
  - Comprehensive MT5 Error Messages
  - Minimal Valid Request Parameters
  - Non-recoverable Error Detection

ADVANCED STRATEGY ENGINES:
  - Entry Optimizer (Market → Limit Conversion)
  - Edge Decay Invalidation (Time-Price Threshold)
  - Choppy-Specific Exit (Layer 0)
  - Choppy-Adaptive Trailing Tightening
  - Loss Attribution Engine (Categorize Loss Causes)
  - Daily PnL Tracking
  - Range Position Filter (Sideways Market Protection)
  - Multi-TF Pattern Detection (Trap Pattern Prevention)
  - Multi-TF Reversal Detection (Active Exit)
  - Adaptive Volume-Based Trailing

10-LAYER ACTIVE POSITION MANAGEMENT:
  Layer 0: Choppy-Specific Exit
  Layer 1: Breakeven Stop
  Layer 2: Trailing Stop (Chandelier / Micro-Increment / ATR / Choppy-Tightened / Adaptive Volume)
  Layer 2.5: Multi-TF Reversal Detection (Active Partial Close)
  Layer 3: Partial Close
  Layer 4: Edge Decay Invalidation
  Layer 5: Time Stop
  Layer 6: Dynamic Exit
  Layer 7: Regime-Conflict Liquidation
  Layer 8: Position Intelligence
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

from config import config

# Execution Layer
from execution.state_manager import StateManager
from execution.friction_filter import FrictionFilter
from execution.risk_manager import RiskManager
from execution.position_intelligence_manager import PositionIntelligenceManager

# Core Engines
from core.atr_cache import ATRCache
from core.time_stop_manager import TimeStopManager
from core.expert_signal_scorer import ExpertSignalScorer
from core.entry_optimizer import EntryOptimizer
from core.invalidation_engine import InvalidationEngine
from core.loss_attribution_engine import LossAttributionEngine

# Optional Engines (Graceful Degradation)
try:
    from core.choppy_detector import ChoppyDetector
    CHOPPY_AVAILABLE = True
except ImportError:
    CHOPPY_AVAILABLE = False

try:
    from core.pattern_detector import PatternDetector
    PATTERN_AVAILABLE = False
except ImportError:
    PATTERN_AVAILABLE = False

try:
    from core.reversal_detector import ReversalDetector
    REVERSAL_AVAILABLE = True
except ImportError:
    REVERSAL_AVAILABLE = False

try:
    from core.range_position_filter import RangePositionFilter
    RANGE_FILTER_AVAILABLE = True
except ImportError:
    RANGE_FILTER_AVAILABLE = False


class OrderManager:
    def __init__(self, symbol: str = "XAUUSD", magic_number: int = 888888,
                 max_slippage: int = 20, risk_per_trade_pct: float = 1.0,
                 max_open_positions: int = 4, max_pending_orders: int = 5,
                 pending_order_timeout_minutes: int = 30, state_db_path: str = "bot_state.db"):
        
        self.symbol = symbol
        self.magic_number = magic_number
        self.max_slippage = max_slippage
        self.pending_order_timeout_minutes = pending_order_timeout_minutes
        
        # Core Components
        self.state_manager = StateManager(state_db_path)
        self.friction_filter = FrictionFilter(symbol)
        self.risk_manager = RiskManager(
            risk_per_trade_pct=risk_per_trade_pct,
            max_open_positions=max_open_positions,
            max_pending_orders=max_pending_orders,
            symbol=symbol
        )
        self.time_stop_mgr = TimeStopManager()
        self.signal_scorer = ExpertSignalScorer()
        self.position_intelligence = PositionIntelligenceManager()
        
        # Advanced Engines
        self.entry_optimizer = EntryOptimizer(symbol)
        self.invalidation_engine = InvalidationEngine()
        self.loss_attribution_engine = LossAttributionEngine()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Caches
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._tick_cache = None
        self._tick_time = 0
        self._m5_cache = None
        self._m5_cache_time = 0
        
        # State
        self._partial_close_state = self._load_partial_close_state()
        self._last_date = None
        self._last_intelligence_check = 0
        self.intelligence_check_interval = 300  # 5 minutes
        
        # NEW: Periodic reconciliation
        self._last_reconciliation_time = 0
        self.reconciliation_interval = 60  # Every 60 seconds

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_symbol_info(self):
        """Get symbol info with 5-second cache."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_time < 5.0):
            return self._symbol_info_cache
            
        info = mt5.symbol_info(self.symbol)
        if info is None:
            if mt5.symbol_select(self.symbol, True):
                info = mt5.symbol_info(self.symbol)
                
        if info:
            self._symbol_info_cache = info
            self._symbol_info_time = current_time
            
        return info

    def _get_current_tick(self):
        """Get current tick with 0.5-second cache."""
        current_time = time.time()
        if self._tick_cache and (current_time - self._tick_time < 0.5):
            return self._tick_cache
            
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self._tick_cache = tick
            self._tick_time = current_time
            
        return tick

    def _get_m5_data_safe(self) -> Optional[pd.DataFrame]:
        """Safely fetch M5 data with 10-second cache."""
        current_time = time.time()
        if self._m5_cache is not None and (current_time - self._m5_cache_time < 10.0):
            return self._m5_cache
            
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                self._m5_cache = df
                self._m5_cache_time = current_time
                return df
        except Exception as e:
            self.logger.error(f"[ORDER_MGR] M5 fetch error: {e}")
            
        return None

    def _load_partial_close_state(self):
        """Load partial close state from SQLite."""
        try:
            import sqlite3
            with sqlite3.connect(self.state_manager.db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS partial_close_state "
                    "(ticket INTEGER PRIMARY KEY, tp1_hit INTEGER, tp2_hit INTEGER, reversal_close INTEGER DEFAULT 0)"
                )
                rows = conn.execute("SELECT ticket, tp1_hit, tp2_hit FROM partial_close_state").fetchall()
                return {r[0]: {'tp1_hit': bool(r[1]), 'tp2_hit': bool(r[2]), 'reversal_close': False} for r in rows}
        except Exception:
            return {}

    def _save_partial_close_state(self, ticket, state):
        """Save partial close state to SQLite."""
        try:
            import sqlite3
            with sqlite3.connect(self.state_manager.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO partial_close_state (ticket, tp1_hit, tp2_hit) VALUES (?,?,?)",
                    (ticket, int(state.get('tp1_hit', False)), int(state.get('tp2_hit', False)))
                )
        except Exception:
            pass

    def check_daily_reset(self):
        """Reset daily counters at start of new day."""
        today = datetime.now().date()
        if self._last_date != today:
            acc = mt5.account_info()
            if acc:
                self.risk_manager.reset_daily(acc.balance)
            self._last_date = today

    def _calculate_profit_usd(self, mt5_pos, is_buy: bool) -> float:
        """Calculate current profit in USD (price units)."""
        if is_buy:
            return mt5_pos.price_current - mt5_pos.price_open
        else:
            return mt5_pos.price_open - mt5_pos.price_current

    # =========================================================================
    # SIGNAL PROCESSING (10-Step Entry Pipeline)
    # =========================================================================

    def process_signal(self, signal: dict, account_balance: float, current_atr: float,
                       context: dict = None) -> bool:
        """Master signal processing pipeline with all filters and scoring."""
        if signal.get('signal') == 'NEUTRAL':
            return False

        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')
        signal_type = signal['signal']
        strategy_category = meta.get('strategy_category', 'GENERAL')

        if context is None:
            context = {'regime': 'UNKNOWN', 'session': 'OTHER', 'volatility_percentile': 50}

        # =========================================================================
        # STEP 0: MULTI-TF PATTERN DETECTION (Trap Pattern Prevention)
        # =========================================================================
        if PATTERN_AVAILABLE:
            df_primary = context.get('df_primary')
            if df_primary is not None and len(df_primary) >= 50:
                try:
                    pattern_detector = PatternDetector()
                    entry_price = meta.get('entry_price', 0)
                    sl_price = meta.get('sl_price', 0)
                    tp_price = meta.get('tp_price', 0)
                    regime_name_pat = context.get('regime_name', 'UNKNOWN')
                    
                    pattern_result = pattern_detector.detect_dangerous_patterns(
                        df=df_primary,
                        signal_type=signal_type,
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        regime_name=regime_name_pat
                    )
                    
                    if pattern_result['pattern_detected']:
                        self.logger.warning(
                            f"[PATTERN] {strategy_name} DANGEROUS PATTERN: "
                            f"{pattern_result['pattern_type']} | Severity: {pattern_result['severity']}"
                        )
                        
                        action = pattern_result.get('action', 'NONE')
                        
                        if action == 'BLOCK_TRADE':
                            self.logger.warning(
                                f"[PATTERN] BLOCKING trade: {pattern_result.get('recommendation', 'BLOCK')}"
                            )
                            return False
                        
                        elif action == 'REDUCE_POSITION':
                            current_mult = meta.get('position_multiplier', 1.0)
                            meta['position_multiplier'] = current_mult * 0.5
                            self.logger.info(f"[PATTERN] Reducing position size by 50%")
                        
                        elif action == 'WIDEN_SL':
                            current_sl_distance = abs(entry_price - sl_price) if entry_price > 0 and sl_price > 0 else 0
                            if current_sl_distance > 0:
                                new_sl_distance = current_sl_distance * 1.3
                                
                                if 'BUY' in signal_type:
                                    meta['sl_price'] = entry_price - new_sl_distance
                                else:
                                    meta['sl_price'] = entry_price + new_sl_distance
                                
                                self.logger.info(
                                    f"[PATTERN] Widening SL by 30% | "
                                    f"Old: {sl_price:.2f} | New: {meta['sl_price']:.2f}"
                                )
                except Exception as e:
                    self.logger.error(f"[PATTERN] Detection error: {e}")

        # =========================================================================
        # STEP 1: RANGE POSITION FILTER (Sideways Market Protection)
        # =========================================================================
        if RANGE_FILTER_AVAILABLE:
            regime_name = context.get('regime_name', context.get('regime', 'UNKNOWN'))
            is_sideways_regime = any(x in regime_name for x in [
                'SIDEWAY', 'RANGE', 'CONSOLIDATING', 'TIGHT', 'CHOP', 'WHIPSAW'
            ])
            
            if is_sideways_regime and strategy_category in ['MEAN_REVERSION', 'SMC', 'SCALP']:
                try:
                    range_filter = RangePositionFilter()
                    entry_price = meta.get('entry_price', 0)
                    is_buy = 'BUY' in signal_type
                    df_for_range = context.get('df_primary')
                    
                    if df_for_range is not None and entry_price > 0:
                        range_result = range_filter.evaluate_entry_position(
                            df=df_for_range,
                            entry_price=entry_price,
                            is_buy=is_buy,
                            regime_name=regime_name
                        )
                        
                        meta['range_analysis'] = range_result
                        
                        if not range_result.get('should_trade', True):
                            self.logger.info(
                                f"[RANGE FILTER] {strategy_name} REJECTED: {range_result.get('reason', 'Unknown')} "
                                f"(Percentile: {range_result.get('percentile', 50):.1f}%)"
                            )
                            return False
                        
                        if range_result.get('position_score', 50) >= 80:
                            current_mult = meta.get('position_multiplier', 1.0)
                            meta['position_multiplier'] = current_mult * 1.3
                            self.logger.info(
                                f"[RANGE FILTER] Position multiplier boosted to "
                                f"{meta['position_multiplier']:.2f}x (Strong zone)"
                            )
                except Exception as e:
                    self.logger.error(f"[RANGE FILTER] Error: {e}")
        # =========================================================================
        # STEP 1.5: ADAPTIVE TP CALCULATION (NEW)
        # =========================================================================
        from core.adaptive_tp_engine import AdaptiveTPEngine
        
        adaptive_tp_engine = AdaptiveTPEngine()
        entry_price = meta.get('entry_price', 0)
        sl_price = meta.get('sl_price', 0)
        is_buy = 'BUY' in signal_type
        df_primary = context.get('df_primary')
        range_analysis = meta.get('range_analysis', {})
        
        if df_primary is not None and entry_price > 0 and sl_price > 0:
            try:
                tp_result = adaptive_tp_engine.calculate_adaptive_tp(
                    df=df_primary,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    is_buy=is_buy,
                    regime_name=regime_name,
                    range_position=range_analysis
                )
                
                # Override TP in meta
                original_tp = meta.get('tp_price', 0)
                meta['tp_price'] = tp_result['tp_price']
                meta['tp_method'] = tp_result['tp_method']
                meta['tp_confidence'] = tp_result['confidence']
                meta['partial_targets'] = tp_result['partial_targets']
                meta['risk_reward'] = tp_result['risk_reward']
                
                self.logger.info(
                    f"[ADAPTIVE_TP] {strategy_name} | Method: {tp_result['tp_method']} | "
                    f"TP: {original_tp:.2f} -> {tp_result['tp_price']:.2f} | "
                    f"R:R: {tp_result['risk_reward']:.2f} | Conf: {tp_result['confidence']:.2f}"
                )
                
                for target in tp_result['partial_targets']:
                    self.logger.debug(
                        f"[ADAPTIVE_TP]   {target['label']}: {target['price']:.2f} ({target['percent']*100:.0f}%)"
                    )
                
            except Exception as e:
                self.logger.error(f"[ADAPTIVE_TP] Error: {e}")
                # Keep original TP if adaptive fails
        # =========================================================================
        # STEP 2: Expert Signal Scorer
        # =========================================================================
        score_result = self.signal_scorer.score_signal(signal, context)
        if not score_result['should_trade']:
            return False
            
        signal['meta']['expert_score'] = score_result['score']
        base_mult = meta.get('position_multiplier', 1.0)
        signal['meta']['position_multiplier'] = base_mult * score_result['position_multiplier']

        # =========================================================================
        # STEP 3: Friction Filter
        # =========================================================================
        strict_mode = meta.get('friction_sensitive', False)
        friction_result = self.friction_filter.validate_entry(signal, current_atr, strict_mode=strict_mode)
        if not friction_result['valid']:
            self.logger.info(f"[FRICTION] {strategy_name} rejected: {friction_result['reason']}")
            return False

        # =========================================================================
        # STEP 4: Regime Direction Filter
        # =========================================================================
        if getattr(config, 'regime_direction_filter', True):
            regime_name = context.get('regime_name', context.get('regime', 'UNKNOWN'))
            is_buy_signal = 'BUY' in signal_type
            BULL_ONLY = ['QUIET_RALLY', 'HEALTHY_UPTREND', 'PARABOLIC_RALLY', 'ANOMALY_BULL']
            BEAR_ONLY = ['SLOW_BLEED', 'HEALTHY_DOWNTREND', 'PANIC_CAPITULATION', 'ANOMALY_BEAR']
            
            if regime_name in BULL_ONLY and not is_buy_signal:
                self.logger.info(f"[REGIME FILTER] {strategy_name} SELL rejected: {regime_name} is BULL-only")
                return False
            if regime_name in BEAR_ONLY and is_buy_signal:
                self.logger.info(f"[REGIME FILTER] {strategy_name} BUY rejected: {regime_name} is BEAR-only")
                return False

        # =========================================================================
        # STEP 5: Unified Regime Mapping (Order Critical)
        # =========================================================================
        regime_name = context.get('regime_name', context.get('regime', 'UNKNOWN'))
        if any(x in regime_name for x in ['CHOP', 'WHIPSAW', 'PARABOLIC', 'PANIC']):
            unified_regime = 'HIGH_VOL'
        elif any(x in regime_name for x in ['BOUNCE', 'EXHAUSTED', 'ANOMALY']):
            unified_regime = 'REVERSAL'
        elif any(x in regime_name for x in ['UPTREND', 'DOWNTREND', 'BLEED', 'RALLY']):
            unified_regime = 'TREND'
        else:
            unified_regime = 'SIDEWAY'
            
        signal['meta']['regime'] = unified_regime
        signal['meta']['regime_name'] = regime_name

        # =========================================================================
        # STEP 6: Kelly Criterion
        # =========================================================================
        try:
            from core.kelly_criterion import KellyCriterionEngine
            kelly_engine = KellyCriterionEngine(min_trades=30, max_risk_pct=3.0)
            stats = self.signal_scorer.perf_tracker.get_strategy_stats(strategy_name, unified_regime, days=30)
            kelly_risk, _, kelly_reason = kelly_engine.calculate_kelly_risk(
                stats['winrate'], stats['avg_win'], stats['avg_loss'], 
                self.risk_manager.risk_per_trade_pct, stats['trades']
            )
            self.logger.info(f"[KELLY] {strategy_name} | {kelly_reason}")
            if kelly_risk <= 0:
                return False
        except Exception as e:
            self.logger.error(f"[KELLY] Error: {e}")
            kelly_risk = self.risk_manager.risk_per_trade_pct

        # =========================================================================
        # STEP 7: Apply Multipliers (Regime + Choppy + Killers)
        # =========================================================================
        mult = signal['meta'].get('position_multiplier', 1.0)
        mult *= context.get('kelly_multiplier', 1.0)
        mult *= context.get('killers_multiplier', 1.0)
        if context.get('choppy_score', 0) > 65:
            mult *= 0.5
        signal['meta']['position_multiplier'] = mult

        # =========================================================================
        # STEP 8: Risk Validation (with Lot Clamping in RiskManager)
        # =========================================================================
        is_pending = 'LIMIT' in signal_type or 'STOP' in signal_type
        risk_result = self.risk_manager.validate_new_trade(
            signal,
            self.state_manager.get_active_positions(self.symbol),
            self.state_manager.get_pending_orders(self.symbol),
            account_balance,
            is_pending,
            dynamic_risk_pct=kelly_risk
        )
        if not risk_result['allowed']:
            self.logger.info(f"[RISK] {strategy_name} rejected: {risk_result['reason']}")
            return False

        # =========================================================================
        # STEP 9: Execution Routing
        # =========================================================================
        volume = risk_result['suggested_volume']
        if is_pending:
            return self._place_pending_order(signal, volume)
        else:
            return self._place_market_order(signal, volume)

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def _place_market_order(self, signal: dict, volume: float) -> bool:
        """Place market order with Entry Optimizer integration."""
        df_m5 = self._get_m5_data_safe()
        signal = self.entry_optimizer.optimize_entry(signal, df_m5)
        meta = signal['meta']
        
        if 'LIMIT' in meta.get('execution_method', 'MARKET'):
            original_type = signal['signal']
            signal['signal'] = meta['execution_method']
            meta['entry_price'] = meta.get('optimized_limit_price', 0)
            exp_mins = meta.get('limit_expiration_minutes', 45)
            meta['expiration_bars'] = int(exp_mins / 15)
            
            success = self._place_pending_order(signal, volume)
            signal['signal'] = original_type
            return success

        tick = self._get_current_tick()
        if not tick:
            return False
        
        is_buy = 'BUY' in signal['signal']
        price = tick.ask if is_buy else tick.bid
        valid_sl, valid_tp = self._validate_sl_tp(
            price, meta.get('sl_price', 0), meta.get('tp_price', 0), is_buy
        )

        filling_modes = self._get_symbol_filling_modes()

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(valid_sl) if valid_sl else 0.0,
            "tp": float(valid_tp) if valid_tp else 0.0,
            "deviation": int(self.max_slippage),
            "magic": int(self.magic_number),
            "comment": str(meta.get('strategy', 'Bot'))[:31],
            "type_filling": filling_modes['primary']
        })

        if result is None:
            self.logger.error(f"[ENTRY] order_send returned None: {mt5.last_error()}")
            return False

        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
            self.state_manager.save_active_position(
                ticket=result.order,
                symbol=self.symbol,
                position_type='BUY' if is_buy else 'SELL',
                volume=result.volume,
                entry_price=result.price,
                sl=valid_sl,
                tp=valid_tp,
                strategy=meta.get('strategy', 'Unknown'),
                requires_dynamic_exit=meta.get('requires_dynamic_exit', False),
                dynamic_exit_threshold=meta.get('dynamic_exit_threshold'),
                entry_reason="Market Order",
                expected_entry=meta.get('entry_price', result.price),
                order_type="MARKET",
                is_pending=False,
                meta_data=meta
            )
            return True
        
        error_msg = self._get_mt5_error_message(result.retcode)
        self.logger.error(f"[ENTRY] Failed: {error_msg} (code {result.retcode})")
        return False

    def _place_pending_order(self, signal: dict, volume: float) -> bool:
        """Place pending order (LIMIT or STOP)."""
        meta = signal['meta']
        is_buy = 'BUY' in signal['signal']
        is_limit = 'LIMIT' in signal['signal']
        
        if is_buy and is_limit:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        elif is_limit:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
        elif is_buy:
            order_type = mt5.ORDER_TYPE_BUY_STOP
        else:
            order_type = mt5.ORDER_TYPE_SELL_STOP
        
        entry_price = meta.get('entry_price', 0)
        valid_sl, valid_tp = self._validate_sl_tp(
            entry_price, meta.get('sl_price', 0), meta.get('tp_price', 0), is_buy
        )

        filling_modes = self._get_symbol_filling_modes()

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(entry_price),
            "sl": float(valid_sl) if valid_sl else 0.0,
            "tp": float(valid_tp) if valid_tp else 0.0,
            "deviation": int(self.max_slippage),
            "magic": int(self.magic_number),
            "comment": str(meta.get('strategy', 'Bot'))[:31],
            "type_filling": filling_modes['primary']
        })

        if result is None:
            self.logger.error(f"[PENDING] order_send returned None: {mt5.last_error()}")
            return False

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.state_manager.save_pending_order(
                ticket=result.order,
                symbol=self.symbol,
                order_type=str(order_type),
                volume=volume,
                price=entry_price,
                sl=valid_sl,
                tp=valid_tp,
                strategy=meta.get('strategy', 'Unknown'),
                expiration_bars=meta.get('expiration_bars', 10),
                requires_dynamic_exit=meta.get('requires_dynamic_exit', False),
                dynamic_exit_threshold=meta.get('dynamic_exit_threshold'),
                entry_reason="Pending Order",
                meta_data=meta
            )
            return True
        
        error_msg = self._get_mt5_error_message(result.retcode)
        self.logger.error(f"[PENDING] Failed: {error_msg} (code {result.retcode})")
        return False

    def _validate_sl_tp(self, entry, sl, tp, is_buy):
        """Validate SL/TP against broker stops level."""
        info = self._get_symbol_info()
        if not info:
            return sl, tp
            
        point = getattr(info, 'point', 0.01)
        min_dist = (max(getattr(info, 'trade_stops_level', 10), 10) + 2) * point
        digits = getattr(info, 'digits', 2)
        
        if sl and ((is_buy and sl >= entry - min_dist) or (not is_buy and sl <= entry + min_dist)):
            sl = round(entry - min_dist if is_buy else entry + min_dist, digits)
        if tp and ((is_buy and tp <= entry + min_dist) or (not is_buy and tp >= entry + min_dist)):
            tp = round(entry + min_dist if is_buy else entry - min_dist, digits)
            
        return sl, tp

    # =========================================================================
    # PENDING ORDER MANAGEMENT
    # =========================================================================

    def manage_pending_orders(self, current_time: pd.Timestamp):
        """
        Cancel expired pending orders with robust timezone handling.
        FIXED: Also removes orders that no longer exist in MT5.
        """
        import MetaTrader5 as mt5
        
        # Get all pending orders from MT5 (source of truth)
        mt5_orders = mt5.orders_get(symbol=self.symbol) or []
        mt5_order_tickets = {o.ticket for o in mt5_orders}
        
        # Get all pending orders from state
        local_orders = self.state_manager.get_pending_orders(self.symbol)
        
        for order in local_orders:
            ticket = order['ticket']
            
            # =========================================================================
            # CHECK 1: Order still exists in MT5?
            # =========================================================================
            if ticket not in mt5_order_tickets:
                # Order was filled, cancelled, or expired in MT5
                self.state_manager.remove_pending_order(ticket)
                self.logger.info(
                    f"[PENDING] Removed order {ticket} (no longer in MT5 - likely filled/expired)"
                )
                continue
            
            # =========================================================================
            # CHECK 2: Order expired by time?
            # =========================================================================
            try:
                setup_time = pd.to_datetime(order['setup_time'])
                
                # Timezone alignment
                if getattr(current_time, 'tzinfo', None) is not None:
                    current_time_cmp = current_time.replace(tzinfo=None)
                else:
                    current_time_cmp = current_time
                    
                if getattr(setup_time, 'tzinfo', None) is not None:
                    setup_time_cmp = setup_time.replace(tzinfo=None)
                else:
                    setup_time_cmp = setup_time
                
                elapsed = (current_time_cmp - setup_time_cmp).total_seconds() / 60.0
                
                if elapsed >= self.pending_order_timeout_minutes:
                    # Try to cancel the order
                    result = mt5.order_send({
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": ticket
                    })
                    
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.state_manager.remove_pending_order(ticket)
                        self.logger.info(
                            f"[PENDING] Removed expired order {ticket} (Elapsed: {elapsed:.1f} min)"
                        )
                    else:
                        # Failed to cancel, but remove from state anyway
                        self.state_manager.remove_pending_order(ticket)
                        self.logger.warning(
                            f"[PENDING] Removed expired order {ticket} from state "
                            f"(MT5 cancel failed: {result.retcode if result else 'None'})"
                        )
                        
            except Exception as e:
                self.logger.error(f"[PENDING] Error managing order {ticket}: {e}")
                # Remove problematic order from state
                self.state_manager.remove_pending_order(ticket)

    # =========================================================================
    # 10-LAYER ACTIVE POSITION MANAGEMENT
    # =========================================================================

    def manage_active_positions(self, current_prices: dict, data: Dict[str, pd.DataFrame] = None,
                                 regime_context: dict = None, choppy_result: dict = None):
        """
        Manage active positions with ALL 10 layers + Periodic Reconciliation.
        
        Layers:
          0: Choppy-Specific Exit
          1: Breakeven Stop
          2: Trailing Stop (Micro-Account / Normal / Choppy-Adaptive / Adaptive Volume)
          2.5: Multi-TF Reversal Detection (Active Partial Close)
          3: Partial Close
          4: Edge Decay Invalidation
          5: Time Stop
          6: Dynamic Exit
          7: Regime-Conflict Liquidation
          8: Position Intelligence (every 5 min)
          +: Periodic State Reconciliation (every 60s)
        """
        self.check_daily_reset()
        
        # =========================================================================
        # REGIME CONTEXT SETUP
        # =========================================================================
        STRONG_BEAR = ['SLOW_BLEED', 'HEALTHY_DOWNTREND', 'PANIC_CAPITULATION', 'ANOMALY_BEAR']
        STRONG_BULL = ['QUIET_RALLY', 'HEALTHY_UPTREND', 'PARABOLIC_RALLY', 'ANOMALY_BULL']
        current_regime = regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
        choppy_severity = choppy_result.get('severity', 'NONE') if choppy_result else 'NONE'
        
        # =========================================================================
        # PERIODIC STATE RECONCILIATION (Every 60 seconds)
        # Prevents state desynchronization (stale orders/positions)
        # =========================================================================
        current_time_epoch = time.time()
        if current_time_epoch - self._last_reconciliation_time > self.reconciliation_interval:
            self._reconcile_state_with_mt5()
            self._last_reconciliation_time = current_time_epoch
        
        # =========================================================================
        # MAIN LOOP: Process each active position
        # =========================================================================
        for pos in self.state_manager.get_active_positions(self.symbol):
            mt5_pos_list = mt5.positions_get(ticket=pos['ticket'])
            
            # Position no longer exists in MT5 (closed externally or filled)
            if not mt5_pos_list:
                self.state_manager.remove_active_position(pos['ticket'])
                self.logger.debug(f"[MANAGE] Removed position {pos['ticket']} (not in MT5)")
                continue

            mt5_pos = mt5_pos_list[0]
            is_buy = (mt5_pos.type == mt5.ORDER_TYPE_BUY)
            current_price = mt5_pos.price_current
            entry_price = mt5_pos.price_open
            profit_usd = self._calculate_profit_usd(mt5_pos, is_buy)
            
            meta = pos.get('meta_data', {})
            strategy_category = meta.get('strategy_category', 'GENERAL')
            strategy_name = meta.get('strategy', 'Unknown')
            primary_tf = meta.get('timeframe', 'M15')

            # =========================================================================
            # SAFE TIME EXTRACTION (Handle timezone issues)
            # =========================================================================
            df_primary = data.get(primary_tf) if data else None
            try:
                if df_primary is not None and not df_primary.empty and 'time' in df_primary.columns:
                    current_time = df_primary['time'].iloc[-1]
                    if not isinstance(current_time, pd.Timestamp):
                        current_time = pd.Timestamp.now()
                else:
                    current_time = pd.Timestamp.now()
            except Exception:
                current_time = pd.Timestamp.now()

            # =========================================================================
            # LAYER 0: CHOPPY-SPECIFIC EXIT
            # =========================================================================
            if choppy_result and choppy_result.get('is_choppy', False):
                if choppy_severity == 'EXTREME':
                    self.logger.warning(
                        f"[CHOPPY EXIT] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"Extreme choppy detected - Force closing"
                    )
                    self._close_position_at_market(pos['ticket'], "Extreme Choppy", meta)
                    continue
                
                if choppy_severity == 'HIGH' and strategy_category == 'TREND':
                    self.logger.warning(
                        f"[CHOPPY EXIT] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"High choppy + Trend strategy - Force closing"
                    )
                    self._close_position_at_market(pos['ticket'], "High Choppy (Trend)", meta)
                    continue

            # =========================================================================
            # LAYER 1: BREAKEVEN STOP
            # =========================================================================
            new_sl = self.time_stop_mgr.check_breakeven_stop(pos, current_price)
            if new_sl and new_sl != pos.get('sl'):
                self._modify_sl(pos['ticket'], new_sl)

            # =========================================================================
            # LAYER 2: TRAILING STOP
            # (Micro-Account / Normal / Choppy-Adaptive / Adaptive Volume)
            # =========================================================================
            self._update_trailing_stop(
                pos, mt5_pos, 
                data.get('M5') if data else None, 
                choppy_result
            )

            # =========================================================================
            # LAYER 2.5: MULTI-TF REVERSAL DETECTION (Active Partial Close)
            # Only for non-TREND strategies with profit > 5 USD
            # =========================================================================
            if (REVERSAL_AVAILABLE and 
                data is not None and 
                profit_usd > 5.0 and 
                strategy_category != 'TREND'):
                
                try:
                    # Build multi-TF DataFrame dict
                    df_dict = {
                        'M1': data.get('M1'),
                        'M5': data.get('M5'),
                        'M15': data.get('M15'),
                        'H1': data.get('H1'),
                    }
                    # Filter out None/empty DataFrames
                    df_dict = {tf: df for tf, df in df_dict.items() 
                              if df is not None and len(df) >= 50}
                    
                    if df_dict:
                        reversal_detector = ReversalDetector()
                        reversal_result = reversal_detector.detect_reversal_signals(
                            df_dict=df_dict,
                            is_buy=is_buy,
                            current_profit_usd=profit_usd
                        )
                        
                        # Log reversal detection
                        if reversal_result['reversal_score'] > 0:
                            self.logger.info(
                                f"[REVERSAL] Ticket {pos['ticket']} ({strategy_name}) | "
                                f"Score: {reversal_result['reversal_score']}/3 | "
                                f"Action: {reversal_result['action']} | "
                                f"Profit: {profit_usd:.2f} USD"
                            )
                            for sig in reversal_result.get('signals', []):
                                multi_tf_flag = " (MULTI-TF)" if sig.get('multi_tf_confirmed') else ""
                                primary_flag = " [PRIMARY]" if sig.get('primary_tf') else ""
                                self.logger.debug(
                                    f"[REVERSAL]   Layer {sig.get('layer')}: {sig.get('type')} "
                                    f"on {sig.get('timeframe')}{primary_flag}{multi_tf_flag} | "
                                    f"Strength: {sig.get('strength', 0):.2f}"
                                )
                        
                        # Execute reversal action
                        if reversal_result['action'] == 'PARTIAL_CLOSE':
                            ticket = pos['ticket']
                            if ticket not in self._partial_close_state:
                                self._partial_close_state[ticket] = {
                                    'tp1_hit': False, 
                                    'tp2_hit': False, 
                                    'reversal_close': False
                                }
                            
                            state = self._partial_close_state[ticket]
                            if not state.get('reversal_close', False):
                                vol_to_close = mt5_pos.volume * 0.5
                                info = self._get_symbol_info()
                                vol_step = getattr(info, 'volume_step', 0.01)
                                min_vol = getattr(info, 'volume_min', 0.01)
                                
                                vol_to_close = max(min_vol, round(vol_to_close / vol_step) * vol_step)
                                
                                if vol_to_close < mt5_pos.volume:
                                    self._partial_close(ticket, vol_to_close, is_buy)
                                    state['reversal_close'] = True
                                    self._save_partial_close_state(ticket, state)
                                    
                                    self.logger.info(
                                        f"[REVERSAL] Partial close 50% due to multi-TF reversal | "
                                        f"Ticket: {ticket} | Profit locked: {profit_usd * 0.5:.2f} USD"
                                    )
                        
                        elif reversal_result['action'] == 'TIGHTEN_TRAIL':
                            meta['trail_mult'] = meta.get('trail_mult', 1.5) * 0.5
                            if hasattr(self.state_manager, 'update_position_meta'):
                                self.state_manager.update_position_meta(pos['ticket'], meta)
                            self.logger.info(
                                f"[REVERSAL] Tightened trailing SL by 50% | "
                                f"Ticket: {pos['ticket']} | New trail_mult: {meta['trail_mult']:.2f}"
                            )
                
                except Exception as e:
                    self.logger.error(
                        f"[REVERSAL] Detection error for ticket {pos['ticket']}: {e}"
                    )

            # =========================================================================
            # LAYER 3: PARTIAL CLOSE (TP-based)
            # =========================================================================
            self._evaluate_partial_close(pos, mt5_pos)

            # =========================================================================
            # LAYER 4: EDGE DECAY INVALIDATION
            # =========================================================================
            try:
                decay_reason = self.invalidation_engine.check_edge_decay(
                    pos, current_price, current_time
                )
                if decay_reason:
                    self.logger.info(
                        f"[EDGE DECAY] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"Reason: {decay_reason}"
                    )
                    self._close_position_at_market(
                        pos['ticket'], 
                        f"Decay: {decay_reason[:20]}", 
                        meta
                    )
                    continue
            except Exception as e:
                self.logger.error(f"[EDGE DECAY] Error for ticket {pos['ticket']}: {e}")

            # =========================================================================
            # LAYER 5: TIME STOP
            # =========================================================================
            try:
                if self.time_stop_mgr.should_time_stop(
                    pos, current_time, primary_tf, strategy_category, current_price
                ):
                    self.logger.info(
                        f"[TIME STOP] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"Time limit exceeded"
                    )
                    self._close_position_at_market(pos['ticket'], "Time Stop", meta)
                    continue
            except Exception as e:
                self.logger.error(f"[TIME STOP] Error for ticket {pos['ticket']}: {e}")

            # =========================================================================
            # LAYER 6: DYNAMIC EXIT (Strategy-Specific Indicators)
            # =========================================================================
            if meta.get('requires_dynamic_exit', False) and df_primary is not None:
                try:
                    if self._evaluate_dynamic_exit(pos, df_primary, is_buy, current_price):
                        self.logger.info(
                            f"[DYNAMIC EXIT] Ticket {pos['ticket']} ({strategy_name}) | "
                            f"Indicator reversal detected"
                        )
                        self._close_position_at_market(pos['ticket'], "Dynamic Exit", meta)
                        continue
                except Exception as e:
                    self.logger.error(f"[DYNAMIC EXIT] Error for ticket {pos['ticket']}: {e}")

            # =========================================================================
            # LAYER 7: REGIME-CONFLICT LIQUIDATION
            # Close positions that strongly conflict with current regime
            # =========================================================================
            is_conflict = (
                (is_buy and current_regime in STRONG_BEAR) or 
                (not is_buy and current_regime in STRONG_BULL)
            )
            if is_conflict:
                # Check if position is in significant profit (let trailing handle it if so)
                risk = abs(entry_price - pos.get('sl', 0))
                pnl_r = (profit_usd / risk) if risk > 0 else 0
                
                # If profit < 1.5R, thesis invalidated - close immediately
                # If profit >= 1.5R, trailing stop should protect, but still force close
                self.logger.warning(
                    f"[REGIME CONFLICT] Ticket {pos['ticket']} ({strategy_name}) | "
                    f"Position: {'BUY' if is_buy else 'SELL'} | "
                    f"Current Regime: {current_regime} | "
                    f"PnL: {profit_usd:.2f} USD ({pnl_r:.2f}R) | "
                    f"Force closing due to regime conflict"
                )
                self._close_position_at_market(
                    pos['ticket'], 
                    f"Regime Conflict ({current_regime})", 
                    meta
                )
                continue

        # =========================================================================
        # LAYER 8: POSITION INTELLIGENCE (Every 5 minutes)
        # Analyze all positions and provide recommendations
        # =========================================================================
        if time.time() - self._last_intelligence_check > self.intelligence_check_interval:
            try:
                positions = self.state_manager.get_active_positions(self.symbol)
                if positions:
                    prices = {}
                    for p in positions:
                        mt5_p = mt5.positions_get(ticket=p['ticket'])
                        if mt5_p:
                            prices[p['ticket']] = mt5_p[0].price_current
                            
                    intel = self.position_intelligence.analyze_all_positions(
                        positions, 
                        prices, 
                        data.get('M5') if data else None, 
                        regime_context
                    )
                    self.position_intelligence.log_position_intelligence(intel)
                    
                    # Execute high-priority CLOSE recommendations
                    for rec in intel.get('recommendations', []):
                        if rec.get('priority') == 1 and rec.get('action') == 'CLOSE':
                            self.logger.info(
                                f"[INTELLIGENCE] Closing ticket {rec['ticket']} | "
                                f"Reason: {rec.get('reason', 'Unknown')}"
                            )
                            self._close_position_at_market(
                                rec['ticket'], 
                                f"Intel: {rec.get('reason', 'Unknown')[:20]}", 
                                {}
                            )
            except Exception as e:
                self.logger.error(f"[INTELLIGENCE] Error: {e}")
            
            self._last_intelligence_check = time.time()

    def _reconcile_state_with_mt5(self):
        """
        Periodically reconcile local state with MT5 to prevent desynchronization.
        Removes stale positions and pending orders that no longer exist in MT5.
        """
        try:
            # Get MT5 state (source of truth)
            mt5_positions = mt5.positions_get(symbol=self.symbol) or []
            mt5_orders = mt5.orders_get(symbol=self.symbol) or []
            
            mt5_position_tickets = {p.ticket for p in mt5_positions}
            mt5_order_tickets = {o.ticket for o in mt5_orders}
            
            # Get local state
            local_positions = self.state_manager.get_active_positions(self.symbol)
            local_orders = self.state_manager.get_pending_orders(self.symbol)
            
            # Reconcile positions
            stale_positions = 0
            for pos in local_positions:
                if pos['ticket'] not in mt5_position_tickets:
                    self.state_manager.remove_active_position(pos['ticket'])
                    self.logger.debug(
                        f"[RECONCILE] Removed stale position {pos['ticket']}"
                    )
                    stale_positions += 1
            
            # Reconcile orders
            stale_orders = 0
            for order in local_orders:
                if order['ticket'] not in mt5_order_tickets:
                    self.state_manager.remove_pending_order(order['ticket'])
                    self.logger.debug(
                        f"[RECONCILE] Removed stale order {order['ticket']}"
                    )
                    stale_orders += 1
            
            # Log summary if any cleanup happened
            if stale_positions > 0 or stale_orders > 0:
                self.logger.info(
                    f"[RECONCILE] Cleaned up {stale_positions} positions, "
                    f"{stale_orders} orders"
                )
        
        except Exception as e:
            self.logger.error(f"[RECONCILE] Error: {e}")


    # =========================================================================
    # TRAILING STOP (Layer 2) - All Modes + Adaptive Volume
    # =========================================================================

    def _update_trailing_stop(self, pos: dict, mt5_pos, df_m5: pd.DataFrame, choppy_result: dict = None):
        """Update trailing stop with Micro-Account, Normal, Choppy, and Adaptive Volume logic."""
        meta = pos.get('meta_data', {})
        if not meta.get('trailing_enabled', True):
            return

        is_buy = (mt5_pos.type == mt5.ORDER_TYPE_BUY)
        current_price = mt5_pos.price_current
        current_sl = mt5_pos.sl
        entry_price = mt5_pos.price_open
        strategy_category = meta.get('strategy_category', 'GENERAL')
        
        info = self._get_symbol_info()
        if not info:
            return
            
        digits = getattr(info, 'digits', 2)
        min_dist = (max(getattr(info, 'trade_stops_level', 10), 10) + 2) * getattr(info, 'point', 0.01)
        new_sl = current_sl

        # =========================================================================
        # MICRO-ACCOUNT MODE
        # =========================================================================
        if getattr(config, 'micro_account_mode', False):
            profit_usd = (current_price - entry_price) if is_buy else (entry_price - current_price)
            regime_name = meta.get('regime_name', 'UNKNOWN')
            
            if strategy_category == 'TREND':
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
                    
                from core.chandelier_engine import ChandelierEngine
                new_sl = ChandelierEngine().calculate_trailing_stop(
                    df_work, is_buy, current_sl, entry_price, current_price, 22, 2.0, min_dist
                )
            else:
                be_trigger = self._get_regime_breakeven_trigger(regime_name)
                trail_inc = config.micro_trail_increment_usd
                be_buf = 0.5
                
                # Adaptive Volume Override (Optional)
                if getattr(config, 'trail_method_adaptive_volume', False) and df_m5 is not None:
                    if len(df_m5) >= getattr(config, 'trail_vol_lookback', 100):
                        try:
                            if 'tick_volume' in df_m5.columns:
                                vol_series = df_m5['tick_volume']
                            elif 'volume' in df_m5.columns:
                                vol_series = df_m5['volume']
                            else:
                                vol_series = None
                            
                            if vol_series is not None:
                                current_vol = vol_series.iloc[-1]
                                recent_vols = vol_series.tail(getattr(config, 'trail_vol_lookback', 100))
                                vol_percentile = (recent_vols < current_vol).sum() / len(recent_vols)
                                
                                if vol_percentile < getattr(config, 'trail_vol_low_threshold', 0.4):
                                    trail_inc = getattr(config, 'trail_low_vol_increment', 8.0)
                                elif vol_percentile < getattr(config, 'trail_vol_high_threshold', 0.7):
                                    trail_inc = getattr(config, 'trail_normal_vol_increment', 12.0)
                                else:
                                    trail_inc = getattr(config, 'trail_high_vol_increment', 15.0)
                        except Exception as e:
                            self.logger.debug(f"[TRAIL] Adaptive volume error: {e}")
                
                # Phase 1: Breakeven
                if profit_usd >= be_trigger:
                    is_sl_behind = (current_sl == 0.0 or 
                                    (is_buy and current_sl < entry_price) or 
                                    (not is_buy and current_sl > entry_price))
                    if is_sl_behind:
                        calc = entry_price + be_buf if is_buy else entry_price - be_buf
                        if (is_buy and calc < current_price - min_dist) or (not is_buy and calc > current_price + min_dist):
                            new_sl = calc
                            self.logger.info(f"[MICRO] BE at {profit_usd:.2f} USD (Regime: {regime_name})")
                            
                # Phase 2: Increment
                elif profit_usd >= (be_trigger + trail_inc):
                    inc = int((profit_usd - be_trigger) / trail_inc)
                    if is_buy:
                        calc = entry_price + be_buf + (inc * trail_inc)
                        if (current_sl <= 0.0 or calc > current_sl) and calc < current_price - min_dist:
                            new_sl = calc
                    else:
                        calc = entry_price - be_buf - (inc * trail_inc)
                        if (current_sl <= 0.0 or calc < current_sl) and calc > current_price + min_dist:
                            new_sl = calc
        
        # =========================================================================
        # NORMAL MODE
        # =========================================================================
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
                new_sl = ChandelierEngine().calculate_trailing_stop(
                    df_work, is_buy, current_sl, entry_price, current_price, 22, 3.0, min_dist
                )
            else:
                trail_mult = meta.get('trail_mult', 1.5)
                
                # Choppy-Adaptive Tightening
                if choppy_result and choppy_result.get('is_choppy', False):
                    tight_map = {'EXTREME': 0.5, 'HIGH': 0.7, 'MEDIUM': 0.85}
                    tight = tight_map.get(choppy_result.get('severity', 'NONE'), 1.0)
                    trail_mult *= tight
                    
                dist = atr * trail_mult
                if is_buy:
                    calc = current_price - dist
                    if (current_sl <= 0.0 or calc > current_sl) and calc < current_price - min_dist:
                        new_sl = calc
                else:
                    calc = current_price + dist
                    if (current_sl <= 0.0 or calc < current_sl) and calc > current_price + min_dist:
                        new_sl = calc

        if new_sl != current_sl and new_sl > 0:
            norm_sl = round(new_sl, digits)
            if (is_buy and norm_sl >= current_price) or (not is_buy and norm_sl <= current_price):
                return
            self._modify_sl(pos['ticket'], norm_sl)

    def _get_regime_breakeven_trigger(self, regime_name: str) -> float:
        """Get breakeven trigger based on current regime."""
        if regime_name in ['QUIET_RALLY', 'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND', 'SLOW_BLEED']:
            return config.micro_be_strong_trend_usd
        if regime_name in ['PARABOLIC_RALLY', 'PANIC_CAPITULATION']:
            return config.micro_be_parabolic_usd
        if regime_name in ['CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR', 'FALSE_SIDEWAY']:
            return config.micro_be_consolidating_usd
        if regime_name in ['CLASSIC_RANGE', 'TIGHT_RANGE', 'PRE_BREAKOUT']:
            return config.micro_be_sideways_usd
        if regime_name in ['VOLATILE_CHOP', 'WHIPSAW_MARKET']:
            return config.micro_be_choppy_usd
        if regime_name in ['OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR', 'ANOMALY_BULL', 'ANOMALY_BEAR']:
            return config.micro_be_reversal_usd
        return config.micro_be_sideways_usd

    # =========================================================================
    # PARTIAL CLOSE (Layer 3) - Robust Version
    # =========================================================================

    def _evaluate_partial_close(self, pos: dict, mt5_pos):
        """Evaluate partial close with Micro-Account and Normal Mode support."""
        meta = pos.get('meta_data', {})
        if not meta.get('partial_close_enabled', False):
            return
            
        ticket = pos['ticket']
        if ticket not in self._partial_close_state:
            self._partial_close_state[ticket] = {'tp1_hit': False, 'tp2_hit': False, 'reversal_close': False}
        state = self._partial_close_state[ticket]
        
        risk = abs(mt5_pos.price_open - pos.get('sl', 0))
        if risk == 0:
            return
            
        if mt5_pos.type == mt5.ORDER_TYPE_BUY:
            pnl = mt5_pos.price_current - mt5_pos.price_open
        else:
            pnl = mt5_pos.price_open - mt5_pos.price_current
            
        info = self._get_symbol_info()
        if not info:
            return
            
        step = getattr(info, 'volume_step', 0.01)
        min_v = getattr(info, 'volume_min', 0.01)
        changed = False

        if getattr(config, 'micro_account_mode', False):
            if not state['tp1_hit'] and pnl >= config.micro_partial_close_trigger_usd:
                vol = max(min_v, round((mt5_pos.volume * config.micro_partial_close_percent) / step) * step)
                if vol < mt5_pos.volume:
                    self._partial_close(ticket, vol, mt5_pos.type == mt5.ORDER_TYPE_BUY)
                    state['tp1_hit'] = True
                    changed = True
                    self.logger.info(f"[MICRO] Partial close 50% at {pnl:.2f} USD profit")
        else:
            r = pnl / risk if risk > 0 else 0
            if not state['tp1_hit'] and r >= 1.5:
                vol = max(min_v, round((mt5_pos.volume * 0.33) / step) * step)
                if vol < mt5_pos.volume:
                    self._partial_close(ticket, vol, mt5_pos.type == mt5.ORDER_TYPE_BUY)
                    state['tp1_hit'] = True
                    changed = True
            elif state['tp1_hit'] and not state['tp2_hit'] and r >= 2.5:
                vol = max(min_v, round((mt5_pos.volume * 0.5) / step) * step)
                if vol < mt5_pos.volume:
                    self._partial_close(ticket, vol, mt5_pos.type == mt5.ORDER_TYPE_BUY)
                    state['tp2_hit'] = True
                    changed = True
                    
        if changed:
            self._save_partial_close_state(ticket, state)

    # =========================================================================
    # DYNAMIC EXIT (Layer 6) - Full 5 Strategies
    # =========================================================================

    def _evaluate_dynamic_exit(self, pos: dict, df: pd.DataFrame, is_buy: bool, current_price: float) -> bool:
        """Evaluate strategy-specific dynamic exit conditions (S15, S10, S3, S24, S25)."""
        strategy = pos.get('meta_data', {}).get('strategy', '')
        try:
            if strategy == 'S15_HFT_StatArb':
                from core.stat_arb_engine import StatArbEngine
                z = StatArbEngine().calculate_z_score(df['close'], 100)
                if z is not None and not z.empty and not pd.isna(z.iloc[-1]):
                    if is_buy and z.iloc[-1] < 0.5:
                        return True
                    if not is_buy and z.iloc[-1] > -0.5:
                        return True
                        
            elif strategy == 'S10_EhlersMESA':
                from core.dsp_ehlers_engine import EhlersDSPEngine
                m, f = EhlersDSPEngine().ehlers_mesa(df['close'])
                if not m.empty and not f.empty and len(m) >= 2 and len(f) >= 2:
                    if is_buy and m.iloc[-1] < f.iloc[-1] and m.iloc[-2] >= f.iloc[-2]:
                        return True
                    if not is_buy and m.iloc[-1] > f.iloc[-1] and m.iloc[-2] <= f.iloc[-2]:
                        return True
                        
            elif strategy == 'S3_EMD_HHT':
                from core.dsp_engine import DSPEngine
                dsp = DSPEngine()
                imf1 = dsp.empirical_mode_decomposition(df['close'], max_imfs=1)
                if imf1 is not None and not imf1.isna().all():
                    phase = dsp.hilbert_phase(imf1)
                    if not phase.isna().all() and len(phase) >= 2:
                        if is_buy and phase.iloc[-2] > 0 and phase.iloc[-1] <= 0:
                            return True
                        if not is_buy and phase.iloc[-2] < 0 and phase.iloc[-1] >= 0:
                            return True
                            
            elif strategy == 'S24_KalmanMomentum':
                from core.kalman_squeeze_engine import KalmanSqueezeEngine
                kalman = KalmanSqueezeEngine()
                kalman_result = kalman.apply_kalman_filter(df['close'])
                if kalman_result is not None and len(kalman_result) >= 2:
                    if is_buy and kalman_result.iloc[-1] > current_price and kalman_result.iloc[-2] <= df['close'].iloc[-2]:
                        return True
                    if not is_buy and kalman_result.iloc[-1] < current_price and kalman_result.iloc[-2] >= df['close'].iloc[-2]:
                        return True
                        
            elif strategy == 'S25_HurstWavelet':
                from core.hurst_wavelet_engine import HurstWaveletEngine
                if HurstWaveletEngine().calculate_hurst_exponent(df['close'], 50) < 0.45:
                    return True
                    
        except Exception as e:
            self.logger.error(f"[FAIL] Dynamic exit error for {strategy}: {e}")
            
        return False

    # =========================================================================
    # EXECUTION HELPERS - Robust MT5 Layer
    # =========================================================================

    def _partial_close(self, ticket, vol, is_buy):
        """Execute partial close with robust retry logic."""
        try:
            pos_list = mt5.positions_get(ticket=ticket)
            if not pos_list:
                self.logger.error(f"[PARTIAL] Position {ticket} not found")
                return False
            
            pos = pos_list[0]
            symbol = pos.symbol
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.logger.error("[PARTIAL] Cannot get current tick")
                return False
            
            price = tick.bid if is_buy else tick.ask
            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(vol),
                "type": close_type,
                "position": int(ticket),
                "price": float(price),
                "deviation": int(self.max_slippage),
                "magic": int(self.magic_number)
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                last_error = mt5.last_error() if hasattr(mt5, 'last_error') else "Unknown"
                self.logger.error(f"[PARTIAL] order_send returned None | MT5 Error: {last_error}")
                
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(request)
                
                if result is None:
                    return False
            
            if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                self.logger.info(f"[PARTIAL] Closed {vol:.2f} lots at {price}")
                return True
            else:
                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.error(f"[PARTIAL] Failed: {error_msg} (code {result.retcode})")
                return False
                
        except Exception as e:
            self.logger.error(f"[PARTIAL] Exception: {e}", exc_info=True)
            return False

    def _close_position_at_market(self, ticket: int, reason: str, meta: dict = None, max_retries: int = 3):
        """Execute market close with robust retry logic, Loss Attribution, and Daily PnL Update."""
        pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            self.logger.warning(f"[CLOSE] Position {ticket} not found - may already be closed")
            self.state_manager.remove_active_position(ticket)
            return False
        
        pos = pos_list[0]
        symbol = pos.symbol
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        volume = pos.volume
        entry_price = pos.price_open
        spread_at_close = 0.0
        
        self.logger.info(
            f"[CLOSE] Attempting to close ticket {ticket} ({symbol}) | "
            f"{'SELL' if is_buy else 'BUY'} {volume:.2f} lots | Reason: {reason}"
        )
        
        attempt = 0
        success = False
        exit_price = 0.0
        
        for retry in range(max_retries):
            attempt += 1
            
            try:
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    self.logger.error(f"[CLOSE] Attempt {attempt}: Cannot get tick for {symbol}")
                    time.sleep(0.3)
                    continue
                
                price = tick.bid if is_buy else tick.ask
                spread_at_close = tick.ask - tick.bid
                exit_price = price
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": float(volume),
                    "type": close_type,
                    "position": int(ticket),
                    "price": float(price),
                    "deviation": int(self.max_slippage),
                    "magic": int(self.magic_number)
                }
                
                result = mt5.order_send(request)
                
                if result is None:
                    last_error = mt5.last_error() if hasattr(mt5, 'last_error') else "Unknown"
                    self.logger.error(f"[CLOSE] Attempt {attempt}: order_send returned None | {last_error}")
                    
                    terminal_info = mt5.terminal_info()
                    if terminal_info and not terminal_info.connected:
                        self.logger.critical("[CLOSE] MT5 terminal disconnected!")
                        return False
                    
                    time.sleep(0.5)
                    continue
                
                if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                    self.logger.info(f"[CLOSE] SUCCESS | Ticket {ticket} at {price} | Vol: {volume:.2f} | {reason}")
                    success = True
                    break
                
                if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    self.logger.warning(f"[CLOSE] Attempt {attempt}: INVALID_FILL, retrying with FOK")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result2 = mt5.order_send(request)
                    
                    if result2 and result2.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                        self.logger.info(f"[CLOSE] SUCCESS (with FOK) | Ticket {ticket} at {price}")
                        success = True
                        break
                
                non_recoverable = [
                    mt5.TRADE_RETCODE_INVALID,
                    mt5.TRADE_RETCODE_INVALID_VOLUME,
                    mt5.TRADE_RETCODE_INVALID_PRICE,
                    mt5.TRADE_RETCODE_POSITION_NOT_FOUND,
                    mt5.TRADE_RETCODE_POSITION_CLOSED,
                    mt5.TRADE_RETCODE_TRADE_DISABLED,
                    mt5.TRADE_RETCODE_MARKET_CLOSED
                ]
                
                if result.retcode in non_recoverable:
                    error_msg = self._get_mt5_error_message(result.retcode)
                    self.logger.error(f"[CLOSE] Non-recoverable: {error_msg} (code {result.retcode})")
                    if result.retcode in [mt5.TRADE_RETCODE_POSITION_NOT_FOUND, mt5.TRADE_RETCODE_POSITION_CLOSED]:
                        self.state_manager.remove_active_position(ticket)
                        return True
                    return False
                
                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.warning(f"[CLOSE] Attempt {attempt} failed: {error_msg} (code {result.retcode})")
                time.sleep(0.3)
                
            except Exception as e:
                self.logger.error(f"[CLOSE] Attempt {attempt} exception: {e}", exc_info=True)
                time.sleep(0.5)
        
        if success:
            if meta is None:
                meta = {}
                
            trade_data = {
                'entry_price': entry_price,
                'sl': pos.sl,
                'tp': pos.tp,
                'exit_price': exit_price,
                'position_type': 'BUY' if is_buy else 'SELL',
                'exit_reason': reason
            }
            df_at_close = self._get_m5_data_safe()
            try:
                loss_cat = self.loss_attribution_engine.analyze_loss(trade_data, df_at_close, spread_at_close)
                self.logger.info(f"[ATTRIBUTION] Ticket {ticket} Loss Category: {loss_cat}")
            except Exception as e:
                self.logger.error(f"[ATTRIBUTION] Error: {e}")
            
            if is_buy:
                pnl = (exit_price - entry_price) * volume * 100
            else:
                pnl = (entry_price - exit_price) * volume * 100
            self.risk_manager.update_daily_pnl(pnl)
            
            self.state_manager.remove_active_position(ticket)
            return True
        
        self.logger.critical(
            f"[CLOSE] CRITICAL: Failed to close ticket {ticket} after {attempt} attempts | "
            f"Manual intervention required!"
        )
        return False

    def _modify_sl(self, ticket, new_sl):
        """Modify stop loss with robust error handling."""
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                self.logger.warning(f"[MODIFY] Position {ticket} not found")
                return False
            
            pos = pos[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": str(pos.symbol),
                "position": int(ticket),
                "sl": float(new_sl),
                "tp": float(pos.tp) if pos.tp else 0.0
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                self.logger.error("[MODIFY] order_send() returned None")
                return False
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"[MODIFY] SL updated to {new_sl} for ticket {ticket}")
                self.state_manager.update_trailing_stop(ticket, new_sl)
                return True
            else:
                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.error(f"[MODIFY] Failed: {error_msg} (code {result.retcode})")
                return False
                
        except Exception as e:
            self.logger.error(f"[MODIFY] Exception: {e}", exc_info=True)
            return False

    def _get_symbol_filling_modes(self) -> Dict[str, int]:
        """Detect supported order filling modes for a symbol (bitmask parsing)."""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                return {
                    'primary': mt5.ORDER_FILLING_FOK,
                    'fok': mt5.ORDER_FILLING_FOK,
                    'ioc': mt5.ORDER_FILLING_IOC,
                    'return': mt5.ORDER_FILLING_IOC
                }
            
            filling_mode = symbol_info.filling_mode
            
            modes = {
                'fok_allowed': bool(filling_mode & 1),
                'ioc_allowed': bool(filling_mode & 2),
                'return_allowed': bool(filling_mode & 4) if hasattr(mt5, 'ORDER_FILLING_RETURN') else False
            }
            
            if modes['fok_allowed']:
                primary = mt5.ORDER_FILLING_FOK
            elif modes['ioc_allowed']:
                primary = mt5.ORDER_FILLING_IOC
            elif modes['return_allowed']:
                primary = mt5.ORDER_FILLING_RETURN if hasattr(mt5, 'ORDER_FILLING_RETURN') else mt5.ORDER_FILLING_IOC
            else:
                primary = mt5.ORDER_FILLING_FOK
            
            return {
                'primary': primary,
                'fok': mt5.ORDER_FILLING_FOK,
                'ioc': mt5.ORDER_FILLING_IOC,
                'return': mt5.ORDER_FILLING_RETURN if hasattr(mt5, 'ORDER_FILLING_RETURN') else mt5.ORDER_FILLING_IOC,
                'modes': modes
            }
            
        except Exception as e:
            self.logger.error(f"[FILL] Failed to detect filling modes: {e}")
            return {
                'primary': mt5.ORDER_FILLING_FOK,
                'fok': mt5.ORDER_FILLING_FOK,
                'ioc': mt5.ORDER_FILLING_IOC,
                'return': mt5.ORDER_FILLING_IOC
            }

    def _get_mt5_error_message(self, retcode: int) -> str:
        """Get human-readable error message for MT5 return codes."""
        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote - price changed",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled by trader",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Only part of request completed",
            mt5.TRADE_RETCODE_ERROR: "Request processing error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Not enough money",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "No quotes to process request",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid order expiration date",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too frequent requests",
            mt5.TRADE_RETCODE_NO_CHANGES: "No changes in request",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "Autotrading disabled by server",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "Autotrading disabled by client terminal",
            mt5.TRADE_RETCODE_LOCKED: "Request locked for processing",
            mt5.TRADE_RETCODE_FROZEN: "Order or position frozen",
            mt5.TRADE_RETCODE_INVALID_FILL: "Invalid order filling type",
            mt5.TRADE_RETCODE_CONNECTION: "No connection with trade server",
            mt5.TRADE_RETCODE_ONLY_REAL: "Operation allowed only for real accounts",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Orders limit reached",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Volume limit reached",
            mt5.TRADE_RETCODE_INVALID_ORDER: "Invalid or prohibited order",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Position already closed",
            mt5.TRADE_RETCODE_POSITION_NOT_FOUND: "Position not found",
        }
        
        return error_messages.get(retcode, f"Unknown error code: {retcode}")

    # =========================================================================
    # SYNC
    # =========================================================================

    def sync_with_mt5(self):
        """
        Synchronize state with MT5 terminal on startup.
        FIXED: Now calls state_manager.sync_with_mt5 for complete reconciliation.
        """
        self.logger.info("[SYNC] Reconciling state with MT5...")
        
        # Call state manager's sync method (handles both positions and orders)
        if hasattr(self.state_manager, 'sync_with_mt5'):
            self.state_manager.sync_with_mt5(self.symbol)
        else:
            # Fallback: Manual sync
            import MetaTrader5 as mt5
            
            mt5_pos = mt5.positions_get(symbol=self.symbol) or []
            local_pos = self.state_manager.get_active_positions(self.symbol)
            mt5_tickets = {p.ticket for p in mt5_pos}
            
            for p in local_pos:
                if p['ticket'] not in mt5_tickets:
                    self.state_manager.remove_active_position(p['ticket'])
            
            # Sync pending orders too
            mt5_orders = mt5.orders_get(symbol=self.symbol) or []
            local_orders = self.state_manager.get_pending_orders(self.symbol)
            mt5_order_tickets = {o.ticket for o in mt5_orders}
            
            for o in local_orders:
                if o['ticket'] not in mt5_order_tickets:
                    self.state_manager.remove_pending_order(o['ticket'])
        
        self.logger.info("[SYNC] State reconciliation complete.")