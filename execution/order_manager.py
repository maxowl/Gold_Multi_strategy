"""
Order Management System - Micro-Account-Only Edition (REVISED).
Handles all order execution, position management, and defense systems.

REVISION LOG:
  [REV-001] REMOVED Trailing Stop (Layer 2) per rule: "ห้ามมี trailing stop"
  [REV-002] REMOVED Partial Close (Layer 3 + Layer 2.5 branch) per rule:
            "Partial close = ห้ามเด็ดขาด"
  [REV-003] ADDED Regime-based TP (_apply_regime_tp) per rule:
            "ใช้ regime-based TP เท่านั้น
             (tp_trend_usd, tp_sideway_usd, tp_highvol_usd, tp_reversal_usd)"
  [REV-004] ADDED Regime-kill check (PARABOLIC/PANIC/VOLATILE_CHOP/WHIPSAW)
  [REV-005] ADDED Session kill-switch (ASIA/LONDON/NY limits)
  [REV-006] ADDED Trade history recording in close flow
  [REV-007] ADDED MT5 auto-close detection (_record_closed_trade_from_history)
  [REV-008] FIXED Kelly Criterion: singleton + min_trades=50
  [REV-009] FIXED Dynamic exit engine caching
  [REV-010] FIXED PnL calculation: use contract_size instead of hardcoded 100

Features:
  - Market/Pending order execution with retry logic
  - 8-Layer Active Position Management (reduced from 10)
  - Breakeven stop management (regime-adaptive)
  - Time stop management
  - Dynamic exit management
  - Regime conflict liquidation
  - Emergency close
  - State synchronization with MT5
  - Modification rate limiting
  - Order quality monitoring

Layer Structure (Revised):
  Layer 0: Choppy-Specific Exit
  Layer 1: Breakeven Stop (Regime-Adaptive)
  Layer 2: Multi-TF Reversal Detection (SL tightening only, NO partial close)
  Layer 3: Edge Decay Invalidation
  Layer 4: Time Stop
  Layer 5: Dynamic Exit
  Layer 6: Regime-Conflict Liquidation
  Layer 7: Position Intelligence

REMOVED per rules:
  - Trailing Stop (was Layer 2): "ห้ามมี trailing stop"
  - Partial Close (was Layer 3): "Partial close = ห้ามเด็ดขาด"
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config import config

# =========================================================================
# EXECUTION LAYER
# =========================================================================
from execution.state_manager import StateManager
from execution.friction_filter import FrictionFilter
from execution.risk_manager import RiskManager

# =========================================================================
# CORE ENGINES
# =========================================================================
from core.atr_cache import ATRCache
from core.time_stop_manager import TimeStopManager
from core.expert_signal_scorer import ExpertSignalScorer
from core.entry_optimizer import EntryOptimizer
from core.invalidation_engine import InvalidationEngine
from core.loss_attribution_engine import LossAttributionEngine

# =========================================================================
# POSITION INTELLIGENCE
# =========================================================================
try:
    from execution.position_intelligence_manager import PositionIntelligenceManager
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False

# =========================================================================
# RATE LIMITING
# =========================================================================
try:
    from execution.modification_limiter import ModificationRateLimiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# =========================================================================
# ORDER QUALITY
# =========================================================================
try:
    from execution.order_quality_monitor import OrderQualityMonitor
    QUALITY_MONITOR_AVAILABLE = True
except ImportError:
    QUALITY_MONITOR_AVAILABLE = False

# =========================================================================
# REVERSAL DETECTION
# =========================================================================
try:
    from core.reversal_detector import ReversalDetector
    REVERSAL_AVAILABLE = True
except ImportError:
    REVERSAL_AVAILABLE = False

# =========================================================================
# CHANDELIER ENGINE
# =========================================================================
try:
    from core.chandelier_engine import ChandelierEngine
    CHANDELIER_AVAILABLE = True
except ImportError:
    CHANDELIER_AVAILABLE = False

# =========================================================================
# KELLY CRITERION [REV-008]
# =========================================================================
try:
    from core.kelly_criterion import KellyCriterionEngine
    KELLY_AVAILABLE = True
except ImportError:
    KELLY_AVAILABLE = False


class OrderManager:
    """
    Manages all order execution and position lifecycle.

    8-Layer Active Position Management (Revised):
      Layer 0: Choppy-Specific Exit
      Layer 1: Breakeven Stop (Regime-Adaptive)
      Layer 2: Multi-TF Reversal Detection (SL tightening only)
      Layer 3: Edge Decay Invalidation
      Layer 4: Time Stop
      Layer 5: Dynamic Exit
      Layer 6: Regime-Conflict Liquidation
      Layer 7: Position Intelligence

    REMOVED per rules:
      - Trailing Stop (was Layer 2): "ห้ามมี trailing stop"
      - Partial Close (was Layer 3): "Partial close = ห้ามเด็ดขาด"
    """

    # =========================================================================
    # REGIME CONSTANTS [REV-004]
    # =========================================================================
    REGIME_KILL = [
        'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
        'VOLATILE_CHOP', 'WHIPSAW_MARKET'
    ]
    REGIME_HIGH_VOL = [
        'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
        'VOLATILE_CHOP', 'WHIPSAW_MARKET'
    ]
    REGIME_REVERSAL = [
        'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
        'ANOMALY_BULL', 'ANOMALY_BEAR'
    ]
    REGIME_TREND = [
        'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
        'QUIET_RALLY', 'SLOW_BLEED'
    ]
    STRONG_BEAR = [
        'SLOW_BLEED', 'HEALTHY_DOWNTREND',
        'PANIC_CAPITULATION', 'ANOMALY_BEAR'
    ]
    STRONG_BULL = [
        'QUIET_RALLY', 'HEALTHY_UPTREND',
        'PARABOLIC_RALLY', 'ANOMALY_BULL'
    ]
    BULL_ONLY = ['QUIET_RALLY', 'HEALTHY_UPTREND', 'ANOMALY_BULL']
    BEAR_ONLY = ['SLOW_BLEED', 'HEALTHY_DOWNTREND', 'ANOMALY_BEAR']

    # =========================================================================
    # SESSION CONSTANTS [REV-005]
    # =========================================================================
    SESSION_ASIA_START = 4
    SESSION_ASIA_END = 12
    SESSION_LONDON_START = 14
    SESSION_LONDON_END = 16
    SESSION_NY_START = 19
    SESSION_NY_END = 24

    def __init__(self, symbol: str = "XAUUSDm", magic_number: int = 888888,
                 max_slippage: int = 20, risk_per_trade_pct: float = 0.5,
                 max_open_positions: int = 2, max_pending_orders: int = 2,
                 pending_order_timeout_minutes: int = 15,
                 state_db_path: str = "bot_state.db"):
        """
        Initialize OrderManager.

        Args:
            symbol: Trading symbol
            magic_number: MT5 magic number
            max_slippage: Maximum slippage in points
            risk_per_trade_pct: Risk per trade as % of equity
            max_open_positions: Maximum concurrent open positions
            max_pending_orders: Maximum concurrent pending orders
            pending_order_timeout_minutes: Pending order timeout
            state_db_path: Path to SQLite state database
        """
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
        self.entry_optimizer = EntryOptimizer(symbol)
        self.invalidation_engine = InvalidationEngine()
        self.loss_attribution_engine = LossAttributionEngine()

        # [REV-008] Kelly Criterion - singleton instance, min_trades=50
        self.kelly_engine = None
        if KELLY_AVAILABLE:
            self.kelly_engine = KellyCriterionEngine(
                min_trades=50,
                max_risk_pct=3.0
            )

        # Optional Components
        self.position_intelligence = None
        if INTELLIGENCE_AVAILABLE:
            self.position_intelligence = PositionIntelligenceManager()

        self.mod_limiter = None
        if RATE_LIMITER_AVAILABLE:
            self.mod_limiter = ModificationRateLimiter()

        self.quality_monitor = None
        if QUALITY_MONITOR_AVAILABLE:
            self.quality_monitor = OrderQualityMonitor(symbol)

        self.logger = logging.getLogger(self.__class__.__name__)

        # Caches
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._tick_cache = None
        self._tick_time = 0

        # [REV-009] Dynamic exit engine cache
        self._dynamic_exit_engines = {}

        # State
        # [REV-002] REMOVED: self._partial_close_state
        self._last_date = None
        self._last_intelligence_check = 0
        self._last_reconciliation_time = 0

        # [REV-005] Session trade tracking
        self._session_trade_counts = {}

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

    def _get_contract_size(self) -> float:
        """[REV-010] Get contract size from symbol info."""
        info = self._get_symbol_info()
        if info:
            return getattr(info, 'trade_contract_size', 100)
        return 100.0

    def check_daily_reset(self):
        """Reset daily counters at start of new day."""
        today = datetime.now().date()
        if self._last_date != today:
            acc = mt5.account_info()
            if acc:
                self.risk_manager.reset_daily(acc.balance)
            self._last_date = today
            # [REV-005] Reset session trade counts
            self._session_trade_counts = {}

    def _calculate_profit_usd(self, mt5_pos, is_buy: bool) -> float:
        """
        Calculate current profit in USD.

        [REV-010] Uses contract_size instead of hardcoded 100.
        """
        contract_size = self._get_contract_size()
        if is_buy:
            return (mt5_pos.price_current - mt5_pos.price_open) * mt5_pos.volume * contract_size
        else:
            return (mt5_pos.price_open - mt5_pos.price_current) * mt5_pos.volume * contract_size

    def _get_current_session(self) -> str:
        """[REV-005] Determine current trading session."""
        hour = datetime.now().hour
        if self.SESSION_ASIA_START <= hour < self.SESSION_ASIA_END:
            return 'ASIA'
        elif self.SESSION_LONDON_START <= hour < self.SESSION_LONDON_END:
            return 'LONDON'
        elif self.SESSION_NY_START <= hour < self.SESSION_NY_END:
            return 'NY'
        else:
            return 'OTHER'

    # =========================================================================
    # [REV-005] SESSION KILL-SWITCH
    # =========================================================================

    def _check_session_limits(self) -> Dict:
        """
        [REV-005] Check session trading limits.

        Rules:
          - ASIA (04:00-12:00): Trading disabled
          - LONDON (14:00-16:00): Max 2 orders per session
          - NY (19:00-24:00): Max 1 order per session

        Returns:
            Dict with 'allowed' and 'reason'
        """
        session = self._get_current_session()

        if session == 'ASIA':
            return {
                'allowed': False,
                'reason': 'ASIA session: trading disabled (04:00-12:00)'
            }

        if session == 'LONDON':
            count = self._session_trade_counts.get('LONDON', 0)
            if count >= 2:
                return {
                    'allowed': False,
                    'reason': f'LONDON limit reached: {count}/2'
                }

        if session == 'NY':
            count = self._session_trade_counts.get('NY', 0)
            if count >= 1:
                return {
                    'allowed': False,
                    'reason': f'NY limit reached: {count}/1'
                }

        return {'allowed': True, 'reason': '', 'session': session}

    def _increment_session_count(self):
        """[REV-005] Increment session trade count after successful order."""
        session = self._get_current_session()
        if session in ('LONDON', 'NY'):
            self._session_trade_counts[session] = \
                self._session_trade_counts.get(session, 0) + 1

    # =========================================================================
    # [REV-003] REGIME-BASED TP
    # =========================================================================

    def _apply_regime_tp(self, signal: dict, regime_context: dict) -> dict:
        """
        [REV-003] Apply regime-based TP per rule:
        "ใช้ regime-based TP เท่านั้น
         (tp_trend_usd, tp_sideway_usd, tp_highvol_usd, tp_reversal_usd)"

        Maps regime_name to unified regime, then sets TP as fixed USD distance.

        Args:
            signal: Signal dict with meta containing entry_price
            regime_context: Regime information with regime_name

        Returns:
            Modified signal dict with regime-based TP
        """
        meta = signal.get('meta', {})
        entry_price = meta.get('entry_price', 0)
        is_buy = 'BUY' in signal.get('signal', '')
        regime_name = regime_context.get(
            'regime_name', regime_context.get('regime', 'UNKNOWN')
        )

        if entry_price <= 0:
            return signal

        # Map regime to unified regime and TP distance
        if regime_name in self.REGIME_HIGH_VOL:
            tp_usd = getattr(config, 'tp_highvol_usd', 25.0)
            unified = 'HIGH_VOL'
        elif regime_name in self.REGIME_REVERSAL:
            tp_usd = getattr(config, 'tp_reversal_usd', 18.0)
            unified = 'REVERSAL'
        elif regime_name in self.REGIME_TREND:
            tp_usd = getattr(config, 'tp_trend_usd', 20.0)
            unified = 'TREND'
        else:
            tp_usd = getattr(config, 'tp_sideway_usd', 12.0)
            unified = 'SIDEWAY'

        # Calculate TP price from USD distance
        if is_buy:
            tp_price = entry_price + tp_usd
        else:
            tp_price = entry_price - tp_usd

        meta['tp_price'] = tp_price
        meta['regime'] = unified
        meta['regime_name'] = regime_name
        meta['tp_usd_distance'] = tp_usd
        signal['meta'] = meta

        self.logger.info(
            f"[REGIME_TP] {meta.get('strategy', '?')} | "
            f"{regime_name} -> {unified} | "
            f"TP: {tp_price:.2f} ({tp_usd} USD)"
        )
        return signal

    # =========================================================================
    # SIGNAL PROCESSING
    # =========================================================================

    def process_signal(self, signal: dict, account_balance: float,
                       current_atr: float = 0.0,
                       regime_context: dict = None) -> bool:
        """
        Master signal processing pipeline.

        Steps:
          0. Session Kill-Switch [REV-005]
          1. Expert Signal Scorer
          2. Regime-Kill Check [REV-004]
          3. Regime Direction Filter
          4. Friction Filter
          5. Kelly Criterion [REV-008]
          6. Apply Multipliers
          7. Regime-Based TP [REV-003]
          8. Risk Validation
          9. Execution Routing

        Args:
            signal: Signal dict from strategy
            account_balance: Current account balance
            current_atr: Current ATR value
            regime_context: Current regime information

        Returns:
            True if order was placed successfully
        """
        if signal.get('signal') == 'NEUTRAL':
            return False

        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')
        signal_type = signal['signal']
        strategy_category = meta.get('strategy_category', 'GENERAL')

        if regime_context is None:
            regime_context = {
                'regime': 'UNKNOWN',
                'regime_name': 'UNKNOWN',
                'session': 'OTHER',
                'volatility_percentile': 50
            }

        # ================================================================
        # STEP 0: SESSION KILL-SWITCH [REV-005]
        # ================================================================
        session_check = self._check_session_limits()
        if not session_check['allowed']:
            self.logger.info(
                f"[SESSION] {strategy_name} rejected: {session_check['reason']}"
            )
            return False

        # ================================================================
        # STEP 1: EXPERT SIGNAL SCORER
        # ================================================================
        score_result = self.signal_scorer.score_signal(signal, regime_context)
        if not score_result['should_trade']:
            return False

        signal['meta']['expert_score'] = score_result['score']
        base_mult = meta.get('position_multiplier', 1.0)
        signal['meta']['position_multiplier'] = base_mult * score_result['position_multiplier']

        # ================================================================
        # STEP 2: REGIME-KILL CHECK [REV-004]
        # ================================================================
        regime_name = regime_context.get(
            'regime_name', regime_context.get('regime', 'UNKNOWN')
        )

        if regime_name in self.REGIME_KILL:
            self.logger.info(
                f"[REGIME_KILL] {strategy_name} rejected: "
                f"{regime_name} is kill-zone"
            )
            return False

        # ================================================================
        # STEP 3: REGIME DIRECTION FILTER
        # ================================================================
        if getattr(config, 'regime_direction_filter', True):
            is_buy_signal = 'BUY' in signal_type

            if regime_name in self.BULL_ONLY and not is_buy_signal:
                self.logger.info(
                    f"[REGIME FILTER] {strategy_name} SELL rejected: "
                    f"{regime_name} is BULL-only"
                )
                return False

            if regime_name in self.BEAR_ONLY and is_buy_signal:
                self.logger.info(
                    f"[REGIME FILTER] {strategy_name} BUY rejected: "
                    f"{regime_name} is BEAR-only"
                )
                return False

        # ================================================================
        # STEP 4: FRICTION FILTER
        # ================================================================
        strict_mode = meta.get('friction_sensitive', False)
        friction_result = self.friction_filter.validate_entry(
            signal, current_atr, strict_mode=strict_mode
        )
        if not friction_result['valid']:
            self.logger.info(
                f"[FRICTION] {strategy_name} rejected: {friction_result['reason']}"
            )
            return False

        # ================================================================
        # STEP 5: KELLY CRITERION [REV-008]
        # ================================================================
        kelly_risk = self.risk_manager.risk_per_trade_pct

        if self.kelly_engine is not None:
            try:
                # Unified regime mapping
                if regime_name in self.REGIME_HIGH_VOL:
                    unified_regime = 'HIGH_VOL'
                elif regime_name in self.REGIME_REVERSAL:
                    unified_regime = 'REVERSAL'
                elif regime_name in self.REGIME_TREND:
                    unified_regime = 'TREND'
                else:
                    unified_regime = 'SIDEWAY'

                signal['meta']['regime'] = unified_regime
                signal['meta']['regime_name'] = regime_name

                stats = self.signal_scorer.perf_tracker.get_strategy_stats(
                    strategy_name, unified_regime, days=30
                )

                kelly_risk, _, kelly_reason = self.kelly_engine.calculate_kelly_risk(
                    stats['winrate'], stats['avg_win'], stats['avg_loss'],
                    self.risk_manager.risk_per_trade_pct, stats['trades']
                )

                self.logger.info(f"[KELLY] {strategy_name} | {kelly_reason}")

                if kelly_risk <= 0:
                    return False

            except Exception as e:
                self.logger.error(f"[KELLY] Error: {e}")
                kelly_risk = self.risk_manager.risk_per_trade_pct

        # ================================================================
        # STEP 6: APPLY MULTIPLIERS
        # ================================================================
        mult = signal['meta'].get('position_multiplier', 1.0)
        mult *= regime_context.get('kelly_multiplier', 1.0)
        mult *= regime_context.get('killers_multiplier', 1.0)
        if regime_context.get('choppy_score', 0) > 65:
            mult *= 0.5
        signal['meta']['position_multiplier'] = mult

        # ================================================================
        # STEP 7: REGIME-BASED TP [REV-003]
        # ================================================================
        signal = self._apply_regime_tp(signal, regime_context)

        # ================================================================
        # STEP 8: RISK VALIDATION
        # ================================================================
        is_pending = 'LIMIT' in signal_type or 'STOP' in signal_type

        risk_result = self.risk_manager.validate_new_trade(
            signal,
            self.state_manager.get_active_positions(self.symbol),
            self.state_manager.get_pending_orders(self.symbol),
            account_balance,
            is_pending,
            dynamic_risk_pct=kelly_risk,
            regime_name=regime_name
        )

        if not risk_result['allowed']:
            self.logger.info(
                f"[RISK] {strategy_name} rejected: {risk_result['reason']}"
            )
            return False

        # ================================================================
        # STEP 9: EXECUTION ROUTING
        # ================================================================
        volume = risk_result['suggested_volume']

        # Register expected execution for quality monitoring
        if self.quality_monitor is not None:
            entry_price = meta.get('entry_price', 0)
            self.quality_monitor.register_expected_execution(
                ticket=0,
                expected_price=entry_price,
                expected_sl=meta.get('sl_price', 0),
                expected_tp=meta.get('tp_price', 0),
                direction='BUY' if 'BUY' in signal_type else 'SELL',
                strategy=strategy_name
            )

        if is_pending:
            success = self._place_pending_order(signal, volume)
        else:
            success = self._place_market_order(signal, volume)

        # [REV-005] Increment session count on success
        if success:
            self._increment_session_count()

        return success

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def _place_market_order(self, signal: dict, volume: float) -> bool:
        """Place market order with Entry Optimizer integration."""
        df_m5 = self._get_m5_data_safe()
        signal = self.entry_optimizer.optimize_entry(signal, df_m5)

        meta = signal['meta']

        # If Optimizer converted to Limit, route to pending order logic
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
            self.logger.error(
                f"[ENTRY] order_send returned None: {mt5.last_error()}"
            )
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

            # Record actual execution for quality monitoring
            if self.quality_monitor is not None:
                self.quality_monitor.record_actual_execution(
                    ticket=result.order,
                    actual_price=result.price,
                    actual_sl=valid_sl,
                    actual_tp=valid_tp
                )

            self.logger.info(
                f"[EXEC] Placed {'BUY' if is_buy else 'SELL'}_MARKET | "
                f"Vol: {volume:.2f} | Entry: {result.price:.2f} | "
                f"SL: {valid_sl:.2f} | TP: {valid_tp:.2f}"
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
            self.logger.error(
                f"[PENDING] order_send returned None: {mt5.last_error()}"
            )
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

    def _validate_sl_tp(self, entry: float, sl: float, tp: float,
                        is_buy: bool) -> Tuple[float, float]:
        """Validate SL/TP against broker stops level."""
        info = self._get_symbol_info()
        if not info:
            return sl, tp

        point = getattr(info, 'point', 0.01)
        digits = getattr(info, 'digits', 2)
        stops_level = max(getattr(info, 'trade_stops_level', 10), 10)
        freeze_level = getattr(info, 'trade_freeze_level', 0)
        min_points = max(stops_level, freeze_level) + 5
        min_dist = min_points * point
        min_dist = max(min_dist, 0.50)

        if sl and sl > 0:
            if is_buy:
                if sl >= entry - min_dist:
                    sl = round(entry - min_dist, digits)
            else:
                if sl <= entry + min_dist:
                    sl = round(entry + min_dist, digits)
        else:
            sl = 0.0

        if tp and tp > 0:
            if is_buy:
                if tp <= entry + min_dist:
                    tp = round(entry + min_dist, digits)
            else:
                if tp >= entry - min_dist:
                    tp = round(entry - min_dist, digits)
        else:
            tp = 0.0

        return sl, tp

    def _get_m5_data_safe(self) -> Optional[pd.DataFrame]:
        """Safely fetch M5 data with 30-second cache [REV-010]."""
        if not hasattr(self, '_m5_cache'):
            self._m5_cache = None
            self._m5_cache_time = 0

        current_time = time.time()
        # [REV-010] Increased cache from 10s to 30s for M5
        if self._m5_cache is not None and (current_time - self._m5_cache_time < 30.0):
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

    def _get_symbol_filling_modes(self) -> Dict[str, int]:
        """Detect supported order filling modes for a symbol."""
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
    # PENDING ORDER MANAGEMENT
    # =========================================================================

    def manage_pending_orders(self, current_time: pd.Timestamp):
        """Cancel expired pending orders with robust timezone handling."""
        mt5_orders = mt5.orders_get(symbol=self.symbol) or []
        mt5_order_tickets = {o.ticket for o in mt5_orders}
        local_orders = self.state_manager.get_pending_orders(self.symbol)

        for order in local_orders:
            ticket = order['ticket']

            # Check if order still exists in MT5
            if ticket not in mt5_order_tickets:
                self.state_manager.remove_pending_order(ticket)
                self.logger.info(f"[PENDING] Removed order {ticket} (no longer in MT5)")
                continue

            # Check if order expired by time
            try:
                setup_time = pd.to_datetime(order['setup_time'])
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
                    result = mt5.order_send({
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": ticket
                    })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.state_manager.remove_pending_order(ticket)
                        self.logger.info(
                            f"[PENDING] Removed expired order {ticket} "
                            f"(Elapsed: {elapsed:.1f} min)"
                        )
                    else:
                        self.state_manager.remove_pending_order(ticket)
                        self.logger.warning(
                            f"[PENDING] Removed expired order {ticket} "
                            f"from state (MT5 cancel failed)"
                        )
            except Exception as e:
                self.logger.error(f"[PENDING] Error managing order {ticket}: {e}")
                self.state_manager.remove_pending_order(ticket)

    # =========================================================================
    # 8-LAYER ACTIVE POSITION MANAGEMENT (REVISED)
    # =========================================================================

    def manage_active_positions(self, current_prices: dict,
                                data: Dict[str, pd.DataFrame] = None,
                                regime_context: dict = None,
                                choppy_result: dict = None):
        """
        Manage active positions with 8 layers (revised from 10).

        REMOVED:
          - Trailing Stop (was Layer 2)
          - Partial Close (was Layer 3)

        Layers:
          0: Choppy-Specific Exit
          1: Breakeven Stop (Regime-Adaptive)
          2: Multi-TF Reversal Detection (SL tightening only)
          3: Edge Decay Invalidation
          4: Time Stop
          5: Dynamic Exit
          6: Regime-Conflict Liquidation
          7: Position Intelligence
        """
        self.check_daily_reset()

        current_regime = regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
        choppy_severity = choppy_result.get('severity', 'NONE') if choppy_result else 'NONE'

        # Periodic State Reconciliation
        current_time_epoch = time.time()
        reconciliation_interval = getattr(config, 'reconciliation_interval_seconds', 60)
        if current_time_epoch - self._last_reconciliation_time > reconciliation_interval:
            self._reconcile_state_with_mt5()
            self._last_reconciliation_time = current_time_epoch

        for pos in self.state_manager.get_active_positions(self.symbol):
            mt5_pos_list = mt5.positions_get(ticket=pos['ticket'])

            if not mt5_pos_list:
                # [REV-007] Position disappeared from MT5 - record trade
                self._record_closed_trade_from_history(pos)
                self.state_manager.remove_active_position(pos['ticket'])
                if self.mod_limiter is not None:
                    self.mod_limiter.reset_position(pos['ticket'])
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

            # Safe Time Extraction
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

            # ================================================================
            # LAYER 0: CHOPPY-SPECIFIC EXIT
            # ================================================================
            if choppy_result and choppy_result.get('is_choppy', False):
                if choppy_severity == 'EXTREME':
                    self.logger.warning(
                        f"[CHOPPY EXIT] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"Extreme choppy"
                    )
                    self._close_position_at_market(
                        pos['ticket'], "Extreme Choppy", meta
                    )
                    continue

                if choppy_severity == 'HIGH' and strategy_category == 'TREND':
                    self.logger.warning(
                        f"[CHOPPY EXIT] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"High choppy + Trend"
                    )
                    self._close_position_at_market(
                        pos['ticket'], "High Choppy (Trend)", meta
                    )
                    continue

            # ================================================================
            # LAYER 1: BREAKEVEN STOP (Regime-Adaptive)
            # ================================================================
            try:
                new_sl = self.time_stop_mgr.check_breakeven_stop(pos, current_price)
                if new_sl and new_sl != pos.get('sl'):
                    if self._modify_sl(pos['ticket'], new_sl):
                        self.logger.info(
                            f"[BREAKEVEN] Ticket {pos['ticket']} | SL -> {new_sl:.2f}"
                        )
            except Exception as e:
                self.logger.error(f"[BREAKEVEN] Ticket {pos['ticket']} error: {e}")

            # ================================================================
            # LAYER 2: MULTI-TF REVERSAL DETECTION
            # [REV-002] PARTIAL_CLOSE branch REMOVED
            # Only TIGHTEN_SL action remains
            # ================================================================
            if (REVERSAL_AVAILABLE and data is not None and
                    profit_usd > 5.0 and strategy_category != 'TREND'):
                try:
                    df_dict = {
                        'M1': data.get('M1'),
                        'M5': data.get('M5'),
                        'M15': data.get('M15'),
                        'H1': data.get('H1'),
                    }
                    df_dict = {
                        tf: df for tf, df in df_dict.items()
                        if df is not None and len(df) >= 50
                    }

                    if df_dict:
                        reversal_detector = ReversalDetector()
                        reversal_result = reversal_detector.detect_reversal_signals(
                            df_dict=df_dict,
                            is_buy=is_buy,
                            current_profit_usd=profit_usd
                        )

                        if reversal_result['reversal_score'] > 0:
                            self.logger.info(
                                f"[REVERSAL] Ticket {pos['ticket']} ({strategy_name}) | "
                                f"Score: {reversal_result['reversal_score']}/3 | "
                                f"Action: {reversal_result['action']}"
                            )

                            # [REV-002] PARTIAL_CLOSE action removed
                            # Only TIGHTEN_TRAIL (SL tightening) remains
                            if reversal_result['action'] == 'TIGHTEN_TRAIL':
                                meta['trail_mult'] = meta.get('trail_mult', 1.5) * 0.5
                                if hasattr(self.state_manager, 'update_position_meta'):
                                    self.state_manager.update_position_meta(
                                        pos['ticket'], meta
                                    )
                except Exception as e:
                    self.logger.error(f"[REVERSAL] Ticket {pos['ticket']} error: {e}")

            # ================================================================
            # LAYER 3: EDGE DECAY INVALIDATION
            # ================================================================
            try:
                decay_reason = self.invalidation_engine.check_edge_decay(
                    pos, current_price, current_time
                )
                if decay_reason:
                    self.logger.info(
                        f"[EDGE DECAY] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"{decay_reason}"
                    )
                    self._close_position_at_market(
                        pos['ticket'], f"Decay: {decay_reason[:20]}", meta
                    )
                    continue
            except Exception as e:
                self.logger.error(f"[EDGE DECAY] Ticket {pos['ticket']} error: {e}")

            # ================================================================
            # LAYER 4: TIME STOP
            # ================================================================
            try:
                if self.time_stop_mgr.should_time_stop(
                    pos, current_time, primary_tf, strategy_category, current_price
                ):
                    self.logger.info(
                        f"[TIME STOP] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"Time limit exceeded"
                    )
                    self._close_position_at_market(
                        pos['ticket'], "Time Stop", meta
                    )
                    continue
            except Exception as e:
                self.logger.error(f"[TIME STOP] Ticket {pos['ticket']} error: {e}")

            # ================================================================
            # LAYER 5: DYNAMIC EXIT
            # ================================================================
            if meta.get('requires_dynamic_exit', False) and df_primary is not None:
                try:
                    if self._evaluate_dynamic_exit(pos, df_primary, is_buy, current_price):
                        self.logger.info(
                            f"[DYNAMIC EXIT] Ticket {pos['ticket']} ({strategy_name})"
                        )
                        self._close_position_at_market(
                            pos['ticket'], "Dynamic Exit", meta
                        )
                        continue
                except Exception as e:
                    self.logger.error(f"[DYNAMIC EXIT] Ticket {pos['ticket']} error: {e}")

            # ================================================================
            # LAYER 6: REGIME-CONFLICT LIQUIDATION
            # ================================================================
            try:
                is_conflict = (
                    (is_buy and current_regime in self.STRONG_BEAR) or
                    (not is_buy and current_regime in self.STRONG_BULL)
                )
                if is_conflict:
                    self.logger.warning(
                        f"[REGIME CONFLICT] Ticket {pos['ticket']} ({strategy_name}) | "
                        f"{'BUY' if is_buy else 'SELL'} vs {current_regime}"
                    )
                    self._close_position_at_market(
                        pos['ticket'], f"Regime Conflict ({current_regime})", meta
                    )
                    continue
            except Exception as e:
                self.logger.error(f"[REGIME CONFLICT] Ticket {pos['ticket']} error: {e}")

        # ================================================================
        # LAYER 7: POSITION INTELLIGENCE (Every N minutes)
        # ================================================================
        intelligence_interval = getattr(config, 'intelligence_check_interval_seconds', 300)
        if (self.position_intelligence is not None and
                time.time() - self._last_intelligence_check > intelligence_interval):
            try:
                positions = self.state_manager.get_active_positions(self.symbol)
                if positions:
                    prices = {}
                    for p in positions:
                        mt5_p = mt5.positions_get(ticket=p['ticket'])
                        if mt5_p:
                            prices[p['ticket']] = mt5_p[0].price_current

                    intel = self.position_intelligence.analyze_all_positions(
                        positions, prices,
                        data.get('M5') if data else None,
                        regime_context
                    )
                    self.position_intelligence.log_position_intelligence(intel)

                    for rec in intel.get('recommendations', []):
                        if rec.get('priority') == 1 and rec.get('action') == 'CLOSE':
                            self.logger.info(
                                f"[INTELLIGENCE] Closing ticket {rec['ticket']}"
                            )
                            self._close_position_at_market(
                                rec['ticket'],
                                f"Intel: {rec.get('reason', 'Unknown')[:20]}",
                                {}
                            )
            except Exception as e:
                self.logger.error(f"[INTELLIGENCE] Error: {e}")

            self._last_intelligence_check = time.time()

    # =========================================================================
    # DYNAMIC EXIT [REV-009: Engine Caching]
    # =========================================================================

    def _get_exit_engine(self, engine_class):
        """[REV-009] Lazy-init and cache engine instances."""
        name = engine_class.__name__
        if name not in self._dynamic_exit_engines:
            self._dynamic_exit_engines[name] = engine_class()
        return self._dynamic_exit_engines[name]

    def _evaluate_dynamic_exit(self, pos: dict, df: pd.DataFrame,
                               is_buy: bool, current_price: float) -> bool:
        """Evaluate strategy-specific dynamic exit conditions."""
        strategy = pos.get('meta_data', {}).get('strategy', '')

        try:
            if strategy == 'S15_HFT_StatArb':
                from core.stat_arb_engine import StatArbEngine
                engine = self._get_exit_engine(StatArbEngine)
                z = engine.calculate_z_score(df['close'], 100)
                if z is not None and not z.empty and not pd.isna(z.iloc[-1]):
                    if is_buy and z.iloc[-1] < 0.5:
                        return True
                    if not is_buy and z.iloc[-1] > -0.5:
                        return True

            elif strategy == 'S10_EhlersMESA':
                from core.dsp_ehlers_engine import EhlersDSPEngine
                engine = self._get_exit_engine(EhlersDSPEngine)
                m, f = engine.ehlers_mesa(df['close'])
                if not m.empty and not f.empty and len(m) >= 2 and len(f) >= 2:
                    if is_buy and m.iloc[-1] < f.iloc[-1] and m.iloc[-2] >= f.iloc[-2]:
                        return True
                    if not is_buy and m.iloc[-1] > f.iloc[-1] and m.iloc[-2] <= f.iloc[-2]:
                        return True

            elif strategy == 'S3_EMD_HHT':
                from core.dsp_engine import DSPEngine
                engine = self._get_exit_engine(DSPEngine)
                imf1 = engine.empirical_mode_decomposition(df['close'], max_imfs=1)
                if imf1 is not None and not imf1.isna().all():
                    phase = engine.hilbert_phase(imf1)
                    if not phase.isna().all() and len(phase) >= 2:
                        if is_buy and phase.iloc[-2] > 0 and phase.iloc[-1] <= 0:
                            return True
                        if not is_buy and phase.iloc[-2] < 0 and phase.iloc[-1] >= 0:
                            return True

            elif strategy == 'S24_KalmanMomentum':
                from core.kalman_squeeze_engine import KalmanSqueezeEngine
                engine = self._get_exit_engine(KalmanSqueezeEngine)
                kalman_result = engine.apply_kalman_filter(df['close'])
                if kalman_result is not None and len(kalman_result) >= 2:
                    if is_buy and kalman_result.iloc[-1] > current_price and kalman_result.iloc[-2] <= df['close'].iloc[-2]:
                        return True
                    if not is_buy and kalman_result.iloc[-1] < current_price and kalman_result.iloc[-2] >= df['close'].iloc[-2]:
                        return True

            elif strategy == 'S25_HurstWavelet':
                from core.hurst_wavelet_engine import HurstWaveletEngine
                engine = self._get_exit_engine(HurstWaveletEngine)
                if engine.calculate_hurst_exponent(df['close'], 50) < 0.45:
                    return True

        except Exception as e:
            self.logger.error(f"[DYNAMIC EXIT] Error for {strategy}: {e}")

        return False

    # =========================================================================
    # EXECUTION HELPERS
    # =========================================================================

    def _close_position_at_market(self, ticket: int, reason: str,
                                  meta: dict = None, max_retries: int = 3) -> bool:
        """
        Execute market close with robust retry logic.

        [REV-006] Added trade history recording.
        [REV-010] Fixed PnL calculation with contract_size.
        """
        pos_list = mt5.positions_get(ticket=ticket)

        if not pos_list:
            self.logger.warning(
                f"[CLOSE] Position {ticket} not found - may already be closed"
            )
            self.state_manager.remove_active_position(ticket)
            if self.mod_limiter is not None:
                self.mod_limiter.reset_position(ticket)
            return False

        pos = pos_list[0]
        symbol = pos.symbol
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        volume = pos.volume
        entry_price = pos.price_open
        open_time = datetime.fromtimestamp(pos.time).isoformat()

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
                    self.logger.error(
                        f"[CLOSE] Attempt {attempt}: Cannot get tick for {symbol}"
                    )
                    time.sleep(0.3)
                    continue

                price = tick.bid if is_buy else tick.ask
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
                    self.logger.error(
                        f"[CLOSE] Attempt {attempt}: order_send returned None | {last_error}"
                    )
                    terminal_info = mt5.terminal_info()
                    if terminal_info and not terminal_info.connected:
                        self.logger.critical("[CLOSE] MT5 terminal disconnected!")
                        return False
                    time.sleep(0.5)
                    continue

                if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                    self.logger.info(
                        f"[CLOSE] SUCCESS | Ticket {ticket} at {price} | "
                        f"Vol: {volume:.2f} | {reason}"
                    )
                    success = True
                    break

                if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    self.logger.warning(
                        f"[CLOSE] Attempt {attempt}: INVALID_FILL, retrying with FOK"
                    )
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result2 = mt5.order_send(request)
                    if result2 and result2.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                        self.logger.info(
                            f"[CLOSE] SUCCESS (with FOK) | Ticket {ticket} at {price}"
                        )
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
                    self.logger.error(
                        f"[CLOSE] Non-recoverable: {error_msg} (code {result.retcode})"
                    )
                    if result.retcode in [mt5.TRADE_RETCODE_POSITION_NOT_FOUND,
                                          mt5.TRADE_RETCODE_POSITION_CLOSED]:
                        self.state_manager.remove_active_position(ticket)
                        if self.mod_limiter is not None:
                            self.mod_limiter.reset_position(ticket)
                        return True
                    return False

                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.warning(
                    f"[CLOSE] Attempt {attempt} failed: {error_msg} (code {result.retcode})"
                )
                time.sleep(0.3)

            except Exception as e:
                self.logger.error(
                    f"[CLOSE] Attempt {attempt} exception: {e}", exc_info=True
                )
                time.sleep(0.5)

        if success:
            # [REV-010] Calculate PnL with contract_size
            contract_size = self._get_contract_size()
            if is_buy:
                pnl = (exit_price - entry_price) * volume * contract_size
            else:
                pnl = (entry_price - exit_price) * volume * contract_size

            # [REV-006] Record trade history
            if meta is None:
                meta = {}

            self.state_manager.save_trade_history(
                ticket=ticket,
                symbol=symbol,
                direction='BUY' if is_buy else 'SELL',
                entry_price=entry_price,
                exit_price=exit_price,
                volume=volume,
                profit=pnl,
                open_time=open_time,
                close_time=datetime.now().isoformat(),
                strategy=meta.get('strategy', 'Unknown'),
                meta_data={
                    'regime': meta.get('regime', 'UNKNOWN'),
                    'regime_name': meta.get('regime_name', 'UNKNOWN'),
                    'session': self._get_current_session(),
                    'exit_reason': reason,
                    'expected_entry': meta.get('entry_price', 0),
                    'tp_usd_distance': meta.get('tp_usd_distance', 0),
                }
            )

            # Update daily PnL
            self.risk_manager.update_daily_pnl(pnl)

            # Remove from state
            self.state_manager.remove_active_position(ticket)
            if self.mod_limiter is not None:
                self.mod_limiter.reset_position(ticket)

            return True

        self.logger.critical(
            f"[CLOSE] CRITICAL: Failed to close ticket {ticket} "
            f"after {attempt} attempts | Manual intervention required!"
        )
        return False

    def _modify_sl(self, ticket: int, new_sl: float) -> bool:
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
                self.logger.info(
                    f"[MODIFY] SL updated to {new_sl} for ticket {ticket}"
                )
                self.state_manager.update_trailing_stop(ticket, new_sl)
                return True
            else:
                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.error(f"[MODIFY] Failed: {error_msg} (code {result.retcode})")
                return False

        except Exception as e:
            self.logger.error(f"[MODIFY] Exception: {e}", exc_info=True)
            return False

    def _modify_tp(self, ticket: int, new_tp: float) -> bool:
        """Modify take profit with robust error handling."""
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            pos = pos[0]

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": str(pos.symbol),
                "position": int(ticket),
                "sl": float(pos.sl) if pos.sl else 0.0,
                "tp": float(new_tp)
            }

            result = mt5.order_send(request)

            if result is None:
                return False

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"[MODIFY] TP updated to {new_tp} for ticket {ticket}"
                )
                return True
            else:
                error_msg = self._get_mt5_error_message(result.retcode)
                self.logger.error(f"[MODIFY] TP Failed: {error_msg}")
                return False

        except Exception as e:
            self.logger.error(f"[MODIFY] TP Exception: {e}")
            return False

    # =========================================================================
    # STATE SYNCHRONIZATION [REV-007]
    # =========================================================================

    def _record_closed_trade_from_history(self, local_pos: dict):
        """
        [REV-007] Record trade that was closed by MT5 (SL/TP hit).

        When a position disappears from MT5 but exists in local state,
        query MT5 history to get the actual close details.
        """
        try:
            now = datetime.now()
            deals = mt5.history_deals_get(
                now - timedelta(days=1), now,
                symbol=self.symbol,
                position=local_pos['ticket']
            )

            if not deals:
                self.logger.warning(
                    f"[RECONCILE] No history deals found for ticket {local_pos['ticket']}"
                )
                return

            # Find the DEAL_ENTRY_OUT deal
            out_deal = None
            for d in deals:
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    out_deal = d
                    break

            if out_deal is None:
                self.logger.warning(
                    f"[RECONCILE] No OUT deal found for ticket {local_pos['ticket']}"
                )
                return

            meta = local_pos.get('meta_data', {})

            self.state_manager.save_trade_history(
                ticket=local_pos['ticket'],
                symbol=self.symbol,
                direction=local_pos.get('position_type', 'BUY'),
                entry_price=local_pos.get('entry_price', 0),
                exit_price=out_deal.price,
                volume=out_deal.volume,
                profit=out_deal.profit,
                open_time=local_pos.get('open_time', ''),
                close_time=datetime.fromtimestamp(out_deal.time).isoformat(),
                strategy=meta.get('strategy', 'Unknown'),
                meta_data={
                    'regime': meta.get('regime', 'UNKNOWN'),
                    'regime_name': meta.get('regime_name', 'UNKNOWN'),
                    'session': self._get_current_session(),
                    'exit_reason': 'SL/TP Hit (MT5)',
                    'commission': out_deal.commission,
                    'swap': out_deal.swap,
                }
            )

            self.risk_manager.update_daily_pnl(out_deal.profit)

            self.logger.info(
                f"[RECONCILE] Recorded closed trade {local_pos['ticket']} | "
                f"PnL: ${out_deal.profit:.2f} | Reason: SL/TP Hit"
            )

        except Exception as e:
            self.logger.error(
                f"[RECONCILE] Record closed trade error for "
                f"ticket {local_pos.get('ticket', '?')}: {e}"
            )

    def _reconcile_state_with_mt5(self):
        """
        Periodically reconcile local state with MT5.

        [REV-007] Added trade history recording for positions
        that were closed by MT5 (SL/TP hit).
        """
        try:
            mt5_positions = mt5.positions_get(symbol=self.symbol) or []
            mt5_orders = mt5.orders_get(symbol=self.symbol) or []

            mt5_position_tickets = {p.ticket for p in mt5_positions}
            mt5_order_tickets = {o.ticket for o in mt5_orders}

            local_positions = self.state_manager.get_active_positions(self.symbol)
            local_orders = self.state_manager.get_pending_orders(self.symbol)

            stale_positions = 0
            for pos in local_positions:
                if pos['ticket'] not in mt5_position_tickets:
                    # [REV-007] Record the trade before removing
                    self._record_closed_trade_from_history(pos)
                    self.state_manager.remove_active_position(pos['ticket'])
                    if self.mod_limiter is not None:
                        self.mod_limiter.reset_position(pos['ticket'])
                    stale_positions += 1

            stale_orders = 0
            for order in local_orders:
                if order['ticket'] not in mt5_order_tickets:
                    self.state_manager.remove_pending_order(order['ticket'])
                    stale_orders += 1

            if stale_positions > 0 or stale_orders > 0:
                self.logger.info(
                    f"[RECONCILE] Cleaned up {stale_positions} positions, "
                    f"{stale_orders} orders"
                )

        except Exception as e:
            self.logger.error(f"[RECONCILE] Error: {e}")

    def sync_with_mt5(self):
        """Synchronize state with MT5 terminal on startup."""
        self.logger.info("[SYNC] Reconciling state with MT5...")

        mt5_pos = mt5.positions_get(symbol=self.symbol) or []
        local_pos = self.state_manager.get_active_positions(self.symbol)
        mt5_tickets = {p.ticket for p in mt5_pos}

        for p in local_pos:
            if p['ticket'] not in mt5_tickets:
                self.state_manager.remove_active_position(p['ticket'])

        mt5_orders = mt5.orders_get(symbol=self.symbol) or []
        local_orders = self.state_manager.get_pending_orders(self.symbol)
        mt5_order_tickets = {o.ticket for o in mt5_orders}

        for o in local_orders:
            if o['ticket'] not in mt5_order_tickets:
                self.state_manager.remove_pending_order(o['ticket'])

        self.logger.info("[SYNC] State reconciliation complete.")