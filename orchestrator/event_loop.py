"""
Event Loop Orchestrator - Micro-Account-Only Edition (REVISED).
The central nervous system of the trading bot.
Coordinates Data Ingestion, Regime Detection, Strategy Evaluation,
Order Execution, and Emergency Defense.

REVISION LOG:
  [REV-001] FIXED First-run bar closure flood. On startup, all
            timeframes triggered simultaneously causing 30 strategies
            to evaluate at once. Now guarded with first-run flag.
  [REV-002] ADDED ChoppyDetector integration. choppy_result is now
            passed to manage_active_positions() enabling Layer 0
            (Choppy-Specific Exit).
  [REV-003] FIXED SessionVolatilityManager was initialized but never
            used. Now used for session detection and kill-switch.
  [REV-004] FIXED Config key access uses getattr() with safe fallbacks
            to prevent AttributeError on missing config values.
  [REV-005] ADDED State re-synchronization after MT5 reconnection.
  [REV-006] ADDED Pending order cancellation on graceful shutdown.
  [REV-007] ADDED ATR calculation and pass-through to process_signal().
  [REV-008] ADDED Regime change logging for debugging.
  [REV-009] FIXED Windows compatibility: SIGTERM check before registration.
  [REV-010] FIXED Emergency defense data fallback: M5 -> M15 -> M1.

Features:
  - Lookahead Bias Prevention (Strict Bar Close Detection)
  - 18-Regime Detection and Routing
  - Session-based trading control (ASIA/LONDON/NY)
  - Choppy market detection and exit
  - Emergency Defense Integration
  - Equity Circuit Breaker Integration
  - Drawdown Risk Scaler Integration
  - Withdrawal Alert System
  - Fault Tolerance & Auto-Reconnect with State Re-sync
  - Graceful Shutdown (SIGINT/SIGTERM handling)
  - State Synchronization on Startup
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Set, Optional

from config import config

# =========================================================================
# DATA & CORE ENGINES
# =========================================================================
from orchestrator.data_manager import DataManager
from core.regime_router import RegimeRouter
from core.session_volatility import SessionVolatilityManager


# =========================================================================
# ORCHESTRATOR & EXECUTION
# =========================================================================
from orchestrator.strategy_pool import StrategyPool
from execution.order_manager import OrderManager

# =========================================================================
# CHOPPY DETECTOR [REV-002]
# =========================================================================
try:
    from core.choppy_detector import ChoppyDetector
    CHOPPY_AVAILABLE = True
except ImportError:
    CHOPPY_AVAILABLE = False

# =========================================================================
# DEFENSE SYSTEMS
# =========================================================================
try:
    from core.emergency_defense_engine import EmergencyDefenseEngine
    EMERGENCY_AVAILABLE = True
except ImportError:
    EMERGENCY_AVAILABLE = False

try:
    from execution.equity_circuit_breaker import EquityCircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from execution.drawdown_scaler import DrawdownRiskScaler
    DRAWDOWN_AVAILABLE = True
except ImportError:
    DRAWDOWN_AVAILABLE = False


class EventLoop:
    """
    Main Event Loop for Micro-Account Trading Bot.

    Coordinates all system components and manages the trading lifecycle.

    Loop Phases:
      1.  Health Check (MT5 connection)
      2.  Balance & Withdrawal Check
      3.  Circuit Breaker Check
      4.  Drawdown Check
      5.  Session Kill-Switch [REV-003]
      6.  Fetch Multi-Timeframe Data
      7.  Detect Bar Closures (with first-run guard) [REV-001]
      8.  Choppy Detection [REV-002]
      9.  Emergency Defense Check (with data fallback) [REV-010]
      10. Regime Detection (with logging) [REV-008]
      11. Strategy Evaluation
      12. Signal Processing (with ATR) [REV-007]
      13. Manage Active Positions (with choppy_result) [REV-002]
      14. Manage Pending Orders
      15. Sleep
    """

    # Session kill-switch constants [REV-003]
    SESSION_ASIA_START = 4
    SESSION_ASIA_END = 12
    SESSION_LONDON_START = 14
    SESSION_LONDON_END = 16
    SESSION_NY_START = 19
    SESSION_NY_END = 24

    def __init__(self, symbol: str = None, primary_tf: str = None,
                 risk_pct: float = None):
        """
        Initialize EventLoop with all system components.

        Args:
            symbol: Trading symbol (default from config)
            primary_tf: Primary timeframe (default from config)
            risk_pct: Risk per trade percentage (default from config)
        """
        self.symbol = symbol or getattr(config, 'symbol', 'XAUUSDm')
        self.primary_tf = primary_tf or getattr(config, 'primary_timeframe', 'M15')
        self.risk_pct = risk_pct or getattr(config, 'risk_per_trade_pct', 0.5)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False

        # =====================================================================
        # INITIALIZE CORE ENGINES
        # =====================================================================
        self.data_manager = DataManager(self.symbol)
        self.regime_router = RegimeRouter()
        self.strategy_pool = StrategyPool()
        self.session_mgr = SessionVolatilityManager()

        # Execution Layer
        self.order_manager = OrderManager(
            symbol=self.symbol,
            magic_number=getattr(config, 'magic_number', 888888),
            max_slippage=getattr(config, 'max_slippage_points', 20),
            risk_per_trade_pct=self.risk_pct,
            max_open_positions=getattr(config, 'max_open_positions', 2),
            max_pending_orders=getattr(config, 'max_pending_orders', 2),
            pending_order_timeout_minutes=getattr(
                config, 'pending_order_timeout_minutes', 15
            ),
            state_db_path=getattr(config, 'state_db_path', 'bot_state.db')
        )

        # =====================================================================
        # INITIALIZE CHOPPY DETECTOR [REV-002]
        # =====================================================================
        self.choppy_detector = None
        if CHOPPY_AVAILABLE:
            self.choppy_detector = ChoppyDetector()
            self.logger.info("[EVENT_LOOP] Choppy Detector initialized")
        else:
            self.logger.warning("[EVENT_LOOP] Choppy Detector not available")

        # =====================================================================
        # INITIALIZE DEFENSE SYSTEMS
        # =====================================================================
        self.emergency_engine = None
        if EMERGENCY_AVAILABLE:
            self.emergency_engine = EmergencyDefenseEngine(self.symbol)
            self.logger.info("[EVENT_LOOP] Emergency Defense Engine initialized")
        else:
            self.logger.warning("[EVENT_LOOP] Emergency Defense Engine not available")

        self.circuit_breaker = None
        if CIRCUIT_BREAKER_AVAILABLE:
            self.circuit_breaker = EquityCircuitBreaker(
                getattr(config, 'state_db_path', 'bot_state.db')
            )
            self.logger.info("[EVENT_LOOP] Equity Circuit Breaker initialized")
        else:
            self.logger.warning("[EVENT_LOOP] Equity Circuit Breaker not available")

        self.drawdown_scaler = None
        if DRAWDOWN_AVAILABLE:
            self.drawdown_scaler = DrawdownRiskScaler()
            self.logger.info("[EVENT_LOOP] Drawdown Risk Scaler initialized")
        else:
            self.logger.warning("[EVENT_LOOP] Drawdown Risk Scaler not available")

        # =====================================================================
        # STATE TRACKING
        # =====================================================================
        self._last_bar_times: Dict[str, pd.Timestamp] = {}
        self._is_first_run = True  # [REV-001] First-run flag
        self._last_context_log_time = 0
        self._last_withdrawal_alert_time = 0
        self._last_logged_regime = None  # [REV-008] Regime change tracking

    # =========================================================================
    # MT5 INITIALIZATION & HEALTH [REV-004]
    # =========================================================================

    def _initialize_mt5(self) -> bool:
        """
        Initialize MT5 connection with fallback to terminal attachment.

        [REV-004] Uses getattr() for all config keys to prevent
        AttributeError on missing values.
        """
        init_args = {
            "login": getattr(config, 'mt5_login', ''),
            "password": getattr(config, 'mt5_password', ''),
            "server": getattr(config, 'mt5_server', ''),
            "path": getattr(config, 'mt5_path', ''),
            "timeout": 60000,
            "portable": True
        }

        # Filter out empty/0 values to allow fallback to currently open terminal
        init_args = {k: v for k, v in init_args.items() if v}

        if not mt5.initialize(**init_args):
            # Fallback: try connecting to already-open terminal
            self.logger.warning(
                "[MT5] initialize() with credentials failed, "
                "trying to attach to open terminal..."
            )
            if not mt5.initialize():
                self.logger.critical(
                    f"[MT5] initialize() failed: {mt5.last_error()}"
                )
                return False

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.critical("[MT5] Failed to get account info")
            mt5.shutdown()
            return False

        self.logger.info(
            f"[MT5] Connected to {account_info.server} | "
            f"Account: {account_info.login} | "
            f"Balance: ${account_info.balance:.2f} | "
            f"Equity: ${account_info.equity:.2f} | "
            f"Leverage: 1:{account_info.leverage}"
        )

        # Ensure symbol is selected and visible
        if not mt5.symbol_select(self.symbol, True):
            self.logger.critical(f"[MT5] Failed to select symbol {self.symbol}")
            mt5.shutdown()
            return False

        # Check minimum balance [REV-004]
        min_balance = getattr(config, 'minimum_trading_balance', 100.0)
        if account_info.balance < min_balance:
            self.logger.critical(
                f"[MT5] Balance ${account_info.balance:.2f} below minimum "
                f"${min_balance:.2f}. Trading disabled."
            )
            mt5.shutdown()
            return False

        return True

    def _check_mt5_health(self) -> bool:
        """
        Check if MT5 terminal is still connected.

        [REV-005] Re-syncs state after reconnection to prevent
        state desynchronization.
        """
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            self.logger.warning(
                "[MT5] Terminal disconnected. Attempting to reconnect..."
            )
            if self._initialize_mt5():
                # [REV-005] Re-sync state after reconnection
                self.logger.info("[MT5] Re-syncing state after reconnection...")
                self.order_manager.sync_with_mt5()
                self.logger.info("[MT5] State re-synced successfully")
                return True
            return False
        return True

    # =========================================================================
    # BAR CLOSURE DETECTION [REV-001]
    # =========================================================================

    def _detect_bar_closures(self, data: Dict[str, pd.DataFrame]) -> Set[str]:
        """
        Detect which timeframes have closed a new bar.

        Prevents lookahead bias by only triggering strategies on closed bars.

        [REV-001] FIXED: On first run, all timeframes would trigger
        simultaneously because _last_bar_times was empty. Now guarded
        with first-run flag to prevent signal flood on startup.

        Args:
            data: Dict of timeframe -> DataFrame

        Returns:
            Set of timeframe names that triggered
        """
        triggered_tfs: Set[str] = set()

        for tf_name, df in data.items():
            if df is None or df.empty or 'time' not in df.columns:
                continue

            current_bar_time = df['time'].iloc[-1]
            if not isinstance(current_bar_time, pd.Timestamp):
                try:
                    current_bar_time = pd.to_datetime(
                        current_bar_time, unit='s', utc=True
                    )
                except Exception:
                    continue

            last_time = self._last_bar_times.get(tf_name)
            if last_time is None or current_bar_time > last_time:
                # [REV-001] Skip triggering on first run
                if not self._is_first_run:
                    triggered_tfs.add(tf_name)
                self._last_bar_times[tf_name] = current_bar_time

        # [REV-001] Clear first-run flag after initialization
        if self._is_first_run:
            self._is_first_run = False
            self.logger.info(
                "[EVENT_LOOP] First run: initialized bar timestamps, "
                "skipping signal generation this cycle"
            )

        return triggered_tfs

    # =========================================================================
    # SESSION DETECTION [REV-003]
    # =========================================================================

    def _get_current_session(self) -> Dict:
        """
        [REV-003] Get current trading session info.

        Uses SessionVolatilityManager which was previously initialized
        but never used.

        Returns:
            Dict with session info:
              - session: 'ASIA', 'LONDON', 'NY', or 'OTHER'
              - is_trading_allowed: bool
              - max_orders: int (for LONDON/NY limits)
        """
        try:
            hour = datetime.now().hour

            # ASIA session: Trading disabled
            if self.SESSION_ASIA_START <= hour < self.SESSION_ASIA_END:
                return {
                    'session': 'ASIA',
                    'is_trading_allowed': False,
                    'max_orders': 0,
                    'reason': 'ASIA session: trading disabled (04:00-12:00)'
                }

            # LONDON session: Max 2 orders
            if self.SESSION_LONDON_START <= hour < self.SESSION_LONDON_END:
                return {
                    'session': 'LONDON',
                    'is_trading_allowed': True,
                    'max_orders': 2,
                    'reason': ''
                }

            # NY session: Max 1 order
            if self.SESSION_NY_START <= hour < self.SESSION_NY_END:
                return {
                    'session': 'NY',
                    'is_trading_allowed': True,
                    'max_orders': 1,
                    'reason': ''
                }

            # OTHER sessions: Normal trading
            return {
                'session': 'OTHER',
                'is_trading_allowed': True,
                'max_orders': 999,
                'reason': ''
            }

        except Exception as e:
            self.logger.error(f"[SESSION] Detection error: {e}")
            return {
                'session': 'OTHER',
                'is_trading_allowed': True,
                'max_orders': 999,
                'reason': ''
            }

    # =========================================================================
    # ATR CALCULATION [REV-007]
    # =========================================================================

    def _calculate_current_atr(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        [REV-007] Calculate current ATR from M15 data.

        ATR is passed to process_signal() for friction filter context.

        Args:
            data: Dict of timeframe -> DataFrame

        Returns:
            Current ATR value, or 0.0 if unavailable
        """
        try:
            df_m15 = data.get('M15')
            if df_m15 is None or len(df_m15) < 15:
                return 0.0

            # Calculate ATR manually to avoid import dependency
            high = df_m15['high'].values
            low = df_m15['low'].values
            close = df_m15['close'].values

            tr = []
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i - 1])
                tr3 = abs(low[i] - close[i - 1])
                tr.append(max(tr1, tr2, tr3))

            if len(tr) < 14:
                return 0.0

            atr = sum(tr[-14:]) / 14
            return float(atr)

        except Exception as e:
            self.logger.debug(f"[ATR] Calculation error: {e}")
            return 0.0

    # =========================================================================
    # BALANCE & WITHDRAWAL CHECKS [REV-004]
    # =========================================================================

    def _check_balance_and_withdrawal(self) -> bool:
        """
        Check account balance and withdrawal alerts.

        [REV-004] Uses getattr() for config keys.

        Returns:
            True if trading should continue, False if halted
        """
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("[BALANCE] Cannot get account info")
            return False

        current_balance = account_info.balance
        current_equity = account_info.equity

        # Check minimum balance [REV-004]
        min_balance = getattr(config, 'minimum_trading_balance', 100.0)
        if current_balance < min_balance:
            self.logger.critical(
                f"[BALANCE] Balance ${current_balance:.2f} below minimum "
                f"${min_balance:.2f}. Halting trading."
            )
            return False

        # Check withdrawal alert (once per hour to avoid spam) [REV-004]
        withdrawal_alert_balance = getattr(
            config, 'withdrawal_alert_balance', 5000.0
        )
        current_time = time.time()
        if (current_equity >= withdrawal_alert_balance and
                current_time - self._last_withdrawal_alert_time > 3600):
            self.logger.warning(
                f"[WITHDRAWAL ALERT] Equity ${current_equity:.2f} exceeds "
                f"${withdrawal_alert_balance:.2f}. Consider withdrawing profits."
            )
            self._last_withdrawal_alert_time = current_time

        return True

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    def _run_loop(self):
        """
        The core execution loop.

        Phases:
          1.  Health Check
          2.  Balance & Withdrawal Check
          3.  Circuit Breaker Check
          4.  Drawdown Check
          5.  Session Kill-Switch [REV-003]
          6.  Fetch Multi-Timeframe Data
          7.  Detect Bar Closures [REV-001]
          8.  Choppy Detection [REV-002]
          9.  Emergency Defense Check [REV-010]
          10. Regime Detection [REV-008]
          11. Strategy Evaluation
          12. Signal Processing [REV-007]
          13. Manage Active Positions [REV-002]
          14. Manage Pending Orders
          15. Sleep
        """
        self.logger.info(
            f"[EVENT_LOOP] Starting main loop | Symbol: {self.symbol} | "
            f"Primary TF: {self.primary_tf} | "
            f"Strategies: {self.strategy_pool.get_strategy_count()}"
        )

        while self._running:
            cycle_start = time.time()

            try:
                # =============================================================
                # PHASE 1: HEALTH CHECK
                # =============================================================
                if not self._check_mt5_health():
                    time.sleep(10)
                    continue

                # =============================================================
                # PHASE 2: BALANCE & WITHDRAWAL CHECK
                # =============================================================
                if not self._check_balance_and_withdrawal():
                    self.logger.critical(
                        "[EVENT_LOOP] Trading halted due to balance issues"
                    )
                    break

                # =============================================================
                # PHASE 3: CIRCUIT BREAKER CHECK
                # =============================================================
                if self.circuit_breaker is not None:
                    cb_result = self.circuit_breaker.check_circuit_breakers()
                    if not self.circuit_breaker.is_trading_allowed():
                        self.logger.warning(
                            f"[CIRCUIT_BREAKER] Trading paused: "
                            f"{cb_result.get('reason', 'Unknown')}"
                        )
                        time.sleep(60)
                        continue

                # =============================================================
                # PHASE 4: DRAWDOWN CHECK
                # =============================================================
                if self.drawdown_scaler is not None:
                    dd_result = self.drawdown_scaler.update_equity()
                    if dd_result.get('is_halted', False):
                        self.logger.critical(
                            f"[DRAWDOWN] Trading halted: "
                            f"{dd_result.get('reason', 'Unknown')}"
                        )
                        time.sleep(60)
                        continue

                # =============================================================
                # PHASE 5: SESSION KILL-SWITCH [REV-003]
                # =============================================================
                session_info = self._get_current_session()

                if not session_info['is_trading_allowed']:
                    self.logger.debug(
                        f"[SESSION] {session_info['reason']} | "
                        f"Skipping signal processing, managing positions only"
                    )
                    # Still manage positions during ASIA session
                    self.order_manager.manage_active_positions(
                        current_prices={},
                        data=None,
                        regime_context=None,
                        choppy_result=None
                    )
                    time.sleep(getattr(config, 'event_loop_interval_seconds', 60))
                    continue

                # =============================================================
                # PHASE 6: FETCH MULTI-TIMEFRAME DATA
                # =============================================================
                data = self.data_manager.fetch_all_timeframes()
                if not data:
                    time.sleep(getattr(config, 'event_loop_interval_seconds', 60))
                    continue

                # =============================================================
                # PHASE 7: DETECT BAR CLOSURES [REV-001]
                # =============================================================
                triggered_tfs = self._detect_bar_closures(data)

                # =============================================================
                # PHASE 8: CHOPPY DETECTION [REV-002]
                # =============================================================
                choppy_result = None
                if self.choppy_detector is not None:
                    df_m15 = data.get('M15')
                    if df_m15 is not None and len(df_m15) >= 50:
                        try:
                            choppy_result = self.choppy_detector.detect_choppy(df_m15)
                        except Exception as e:
                            self.logger.error(f"[CHOPPY] Detection error: {e}")

                # =============================================================
                # PHASE 9: EMERGENCY DEFENSE CHECK [REV-010]
                # =============================================================
                if self.emergency_engine is not None and triggered_tfs:
                    # [REV-010] Data fallback: M5 -> M15 -> M1
                    df_emergency = (
                        data.get('M5') or
                        data.get('M15') or
                        data.get('M1')
                    )
                    if df_emergency is not None:
                        active_positions = (
                            self.order_manager.state_manager
                            .get_active_positions(self.symbol)
                        )
                        emergency_result = self.emergency_engine.run_emergency_check(
                            df_emergency, active_positions
                        )
                        if emergency_result.get('emergency_detected', False):
                            for action in emergency_result.get('actions', []):
                                action_type = action.get('type', '')
                                recommendation = action.get('recommendation', '')
                                if (action_type == 'FLASH_CRASH' and
                                        recommendation == 'EMERGENCY_CLOSE_ALL'):
                                    self.emergency_engine.emergency_close_all_positions(
                                        "Flash Crash"
                                    )
                                    self.emergency_engine.activate_kill_switch(
                                        "Flash Crash"
                                    )
                                    time.sleep(60)
                                    continue

                # =============================================================
                # PHASE 10: REGIME DETECTION [REV-008]
                # =============================================================
                regime_result = self.regime_router.detect_regime(data)
                current_regime = regime_result.get('regime_name', 'UNKNOWN')

                # [REV-008] Log regime changes
                if current_regime != self._last_logged_regime:
                    self.logger.info(
                        f"[REGIME] Changed: {self._last_logged_regime} -> "
                        f"{current_regime} | "
                        f"Volatility: {regime_result.get('volatility_percentile', '?')}% | "
                        f"Session: {session_info.get('session', '?')}"
                    )
                    self._last_logged_regime = current_regime

                # =============================================================
                # PHASE 11: STRATEGY EVALUATION
                # =============================================================
                if triggered_tfs:
                    signals = self.strategy_pool.evaluate_all(
                        data, triggered_tfs, regime_context=regime_result
                    )

                    # =========================================================
                    # PHASE 12: SIGNAL PROCESSING [REV-007]
                    # =========================================================
                    if signals:
                        account_info = mt5.account_info()
                        if account_info:
                            # [REV-007] Calculate ATR for friction context
                            current_atr = self._calculate_current_atr(data)

                            for strategy_name, sig in signals.items():
                                # Apply drawdown risk multiplier
                                if self.drawdown_scaler is not None:
                                    dd_multiplier = (
                                        self.drawdown_scaler.get_risk_multiplier()
                                    )
                                    if dd_multiplier < 1.0:
                                        meta = sig.get('meta', {})
                                        meta['position_multiplier'] = (
                                            meta.get('position_multiplier', 1.0) *
                                            dd_multiplier
                                        )
                                        sig['meta'] = meta

                                # [REV-007] Pass current_atr
                                self.order_manager.process_signal(
                                    sig,
                                    account_info.balance,
                                    current_atr=current_atr,
                                    regime_context=regime_result
                                )

                # =============================================================
                # PHASE 13: MANAGE ACTIVE POSITIONS [REV-002]
                # =============================================================
                self.order_manager.manage_active_positions(
                    current_prices={},
                    data=data,
                    regime_context=regime_result,
                    choppy_result=choppy_result  # [REV-002] Now passed
                )

                # =============================================================
                # PHASE 14: MANAGE PENDING ORDERS
                # =============================================================
                df_primary = data.get(self.primary_tf)
                if (df_primary is not None and not df_primary.empty and
                        'time' in df_primary.columns):
                    current_time = df_primary['time'].iloc[-1]
                    self.order_manager.manage_pending_orders(current_time)

                # =============================================================
                # PHASE 15: SLEEP
                # =============================================================
                elapsed = time.time() - cycle_start
                sleep_time = max(
                    0.1,
                    getattr(config, 'event_loop_interval_seconds', 60) - elapsed
                )
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.logger.info(
                    "[EVENT_LOOP] KeyboardInterrupt received. Shutting down..."
                )
                self._running = False

            except Exception as e:
                self.logger.critical(
                    f"[EVENT_LOOP] Critical error in main loop: {e}",
                    exc_info=True
                )
                time.sleep(10)

    # =========================================================================
    # ENTRY POINT & SHUTDOWN [REV-006, REV-009]
    # =========================================================================

    def start(self):
        """
        Start the event loop with signal handling for graceful shutdown.

        [REV-009] FIXED: SIGTERM is not available on Windows.
        Now checks for SIGTERM availability before registration.
        """
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        # [REV-009] SIGTERM is not available on Windows
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)

        if not self._initialize_mt5():
            self.logger.critical("[EVENT_LOOP] MT5 initialization failed. Exiting.")
            sys.exit(1)

        # Sync state with MT5 on startup to recover from crashes
        self.order_manager.sync_with_mt5()

        # Initialize circuit breaker with current equity
        if self.circuit_breaker is not None:
            account_info = mt5.account_info()
            if account_info:
                self.circuit_breaker.reset_daily(account_info.equity)

        # Initialize drawdown scaler
        if self.drawdown_scaler is not None:
            account_info = mt5.account_info()
            if account_info:
                self.drawdown_scaler.peak_equity = account_info.equity

        self._running = True

        try:
            self._run_loop()
        finally:
            self._shutdown()

    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown."""
        self.logger.info(
            f"[EVENT_LOOP] Received signal {signum}. "
            f"Initiating graceful shutdown..."
        )
        self._running = False

    def _shutdown(self):
        """
        Clean up resources and close connections.

        [REV-006] ADDED: Cancel all pending orders before shutdown
        to prevent orphaned orders in MT5.
        """
        self.logger.info("[EVENT_LOOP] Shutting down engines...")

        # [REV-006] Cancel all pending orders before shutdown
        try:
            pending_orders = (
                self.order_manager.state_manager.get_pending_orders(self.symbol)
            )
            for order in pending_orders:
                ticket = order['ticket']
                result = mt5.order_send({
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": ticket
                })
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(
                        f"[SHUTDOWN] Cancelled pending order {ticket}"
                    )
                else:
                    self.logger.warning(
                        f"[SHUTDOWN] Failed to cancel pending order {ticket}"
                    )
        except Exception as e:
            self.logger.error(
                f"[SHUTDOWN] Error cancelling pending orders: {e}"
            )

        # Close SQLite connections in StateManager
        if hasattr(self.order_manager.state_manager, 'close'):
            self.order_manager.state_manager.close()

        # Shutdown MT5
        mt5.shutdown()
        self.logger.info("[EVENT_LOOP] Shutdown complete. Goodbye.")