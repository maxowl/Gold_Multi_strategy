"""
Event Loop Orchestrator - Institutional Grade (Upgraded)
The central nervous system of the trading bot.
Coordinates Data Ingestion, 18-Regime Detection, Strategy Evaluation, and Order Execution.

Features:
  - 18-Regime Detection (EnhancedRegimeRouter with HMM + LightGBM + Rules)
  - Multi-TF Context Propagation (M1, M5, M15, H1 for Pattern & Reversal Detection)
  - Scalping Mode (M1 primary, faster loop, stricter filters)
  - Micro-Account Mode (tighter risk management, lot clamping)
  - Active Regime-Conflict Liquidation (close positions when regime flips)
  - Lookahead Bias Prevention (Strict Bar Close Detection)
  - Choppy & Market Killers Detection (Graceful Degradation)
  - Signal Deduplication (MD5 hash of strategy + bar time)
  - Fault Tolerance & Auto-Reconnect
  - Graceful Shutdown (SIGINT/SIGTERM handling)
  - State Synchronization on Startup
"""
import MetaTrader5 as mt5
import pandas as pd
import logging
import time
import signal
import sys
import hashlib
import json
import pytz
from datetime import datetime
from typing import Dict, Set, Optional

from config import config

# Data & Core Engines
from orchestrator.data_manager import DataManager
from core.regime_router import EnhancedRegimeRouter  # [UPGRADE] 18-Regime Router
from core.choppy_detector import ChoppyDetector
from core.market_killers_detector import MarketKillersDetector
from core.session_volatility import SessionVolatilityManager
from core.atr_cache import ATRCache

# Orchestrator & Execution
from orchestrator.strategy_pool import StrategyPool
from execution.order_manager import OrderManager


class EventLoop:
    def __init__(self, symbol: str = None, primary_tf: str = None, risk_pct: float = None):
        self.symbol = symbol or config.symbol
        self.primary_tf = primary_tf or config.primary_timeframe
        self.risk_pct = risk_pct or config.risk_per_trade_pct
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        
        # =========================================================================
        # INITIALIZE ENGINES
        # =========================================================================
        self.data_manager = DataManager(self.symbol)
        
        # [UPGRADE] Use EnhancedRegimeRouter with named parameters
        self.regime_router = EnhancedRegimeRouter(
            rule_model_path="regime_model.pkl",
            hmm_model_path=config.regime_model_path,
            hybrid_model_path=config.lightgbm_models_path
        )
        
        self.strategy_pool = StrategyPool()
        self.session_mgr = SessionVolatilityManager()
        
        # Optional Detectors (Graceful Degradation)
        try:
            self.choppy_detector = ChoppyDetector()
        except Exception:
            self.choppy_detector = None
            self.logger.warning("[EVENT_LOOP] ChoppyDetector not available.")
            
        try:
            self.killers_detector = MarketKillersDetector(self.symbol)
        except Exception:
            self.killers_detector = None
            self.logger.warning("[EVENT_LOOP] MarketKillersDetector not available.")

        # Execution Layer
        self.order_manager = OrderManager(
            symbol=self.symbol,
            magic_number=config.magic_number,
            max_slippage=config.max_slippage_points,
            risk_per_trade_pct=self.risk_pct,
            max_open_positions=config.max_open_positions,
            max_pending_orders=config.max_pending_orders,
            pending_order_timeout_minutes=config.pending_order_timeout_minutes,
            state_db_path=config.state_db_path
        )
        
        # NEW: Add symbol to state_manager for sync_with_mt5
        self.order_manager.state_manager.symbol = self.symbol
        
        # State Tracking
        self._last_bar_times: Dict[str, pd.Timestamp] = {}
        self._last_context_log_time = 0
        self.context_log_interval = 300  # Log context summary every 5 minutes
        
        # [NEW] Signal Deduplication
        self._processed_signal_hashes: Set[str] = set()

    # =========================================================================
    # MT5 INITIALIZATION & HEALTH
    # =========================================================================

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection with fallback to terminal attachment."""
        init_args = {
            "login": config.mt5_login,
            "password": config.mt5_password,
            "server": config.mt5_server,
            "path": config.mt5_path,
            "timeout": 60000,
            "portable": True
        }
        
        # Filter out empty/0 values to allow fallback to currently open terminal
        init_args = {k: v for k, v in init_args.items() if v}
        
        if not mt5.initialize(**init_args):
            self.logger.critical(f"[MT5] initialize() failed, error code: {mt5.last_error()}")
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
            f"Leverage: 1:{account_info.leverage}"
        )
        
        # Ensure symbol is selected and visible
        if not mt5.symbol_select(self.symbol, True):
            self.logger.critical(f"[MT5] Failed to select symbol {self.symbol}")
            mt5.shutdown()
            return False
            
        return True

    def _check_mt5_health(self) -> bool:
        """Check if MT5 terminal is still connected."""
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            self.logger.warning("[MT5] Terminal disconnected. Attempting to reconnect...")
            return self._initialize_mt5()
        return True

    # =========================================================================
    # BAR CLOSURE DETECTION (Lookahead Bias Prevention)
    # =========================================================================

    def _detect_bar_closures(self, data: Dict[str, pd.DataFrame]) -> Set[str]:
        """
        Detect which timeframes have closed a new bar.
        Prevents lookahead bias by only triggering strategies on closed bars.
        """
        triggered_tfs: Set[str] = set()
        
        for tf_name, df in data.items():
            if df is None or df.empty or 'time' not in df.columns:
                continue
                
            current_bar_time = df['time'].iloc[-1]
            if not isinstance(current_bar_time, pd.Timestamp):
                try:
                    current_bar_time = pd.to_datetime(current_bar_time, unit='s', utc=True)
                except Exception:
                    continue
                    
            last_time = self._last_bar_times.get(tf_name)
            
            if last_time is None or current_bar_time > last_time:
                triggered_tfs.add(tf_name)
                self._last_bar_times[tf_name] = current_bar_time
                
        return triggered_tfs

    # =========================================================================
    # CONTEXT BUILDING (Multi-TF + Regime + Choppy + Killers)
    # =========================================================================

    def _build_context(self, data: Dict[str, pd.DataFrame], regime_result: Dict, 
                       choppy_result: Optional[Dict], killers_report: Optional[Dict], 
                       session_info, daily_stats: Dict) -> Dict:
        """
        Aggregate all market conditions into a unified context dictionary.
        Includes Multi-TF DataFrames for Pattern & Reversal Detection.
        
        Note: session_info can be either:
          - str: session name (e.g., 'LONDON_OPEN')
          - dict: {'session': 'LONDON_OPEN', 'volatility_percentile': 50}
        """
        # =========================================================================
        # Handle session_info (can be str or dict)
        # =========================================================================
        if isinstance(session_info, str):
            session_name = session_info
            volatility_percentile = 50  # Default
        elif isinstance(session_info, dict):
            session_name = session_info.get('session', 'OTHER')
            volatility_percentile = session_info.get('volatility_percentile', 50)
        else:
            session_name = 'OTHER'
            volatility_percentile = 50
        
        # Get primary DataFrame for Range Position Filter
        df_primary = data.get(self.primary_tf)
        
        context = {
            # =========================================================================
            # Regime Data (18-Regime System)
            # =========================================================================
            'regime': regime_result.get('regime_name', 'UNKNOWN'),  # [UPGRADE] Use regime_name directly
            'regime_name': regime_result.get('regime_name', 'UNKNOWN'),
            'trend': regime_result.get('trend', 'UNKNOWN'),
            'volatility': regime_result.get('volatility', 'NORMAL'),
            'fractal': regime_result.get('fractal', 'TRENDING'),
            'kelly_multiplier': regime_result.get('kelly_multiplier', 1.0),
            'regime_confidence': regime_result.get('trend_confidence', 0.5),
            'regime_stability': regime_result.get('regime_stability', 0.5),
            'hurst_value': regime_result.get('hurst_value', 0.5),
            'vol_percentile': regime_result.get('vol_percentile', 0.5),
            
            # =========================================================================
            # Session & Volatility
            # =========================================================================
            'session': session_name,
            'volatility_percentile': volatility_percentile,
            
            # =========================================================================
            # Choppy Data
            # =========================================================================
            'choppy_score': choppy_result.get('choppy_score', 0) if choppy_result else 0,
            'choppy_severity': choppy_result.get('severity', 'NONE') if choppy_result else 'NONE',
            'choppy_result': choppy_result,
            
            # =========================================================================
            # Market Killers Data
            # =========================================================================
            'killers_multiplier': killers_report.get('killers_multiplier', 1.0) if killers_report else 1.0,
            'active_killers': killers_report.get('active_killers', []) if killers_report else [],
            
            # =========================================================================
            # Portfolio Health
            # =========================================================================
            'daily_pnl_pct': daily_stats.get('daily_pnl_pct', 0.0),
            'daily_pnl_percent': daily_stats.get('daily_pnl_pct', 0.0),
            'account_drawdown': 0.0,
            
            # =========================================================================
            # Performance Tracking
            # =========================================================================
            'recent_strategy_performance': {},
            'bars_since_signal': 0,
            'concurrent_signals': len(self.order_manager.state_manager.get_active_positions(self.symbol)),
            'mtf_alignment': regime_result.get('trend_confidence', 0.5),
            
            # =========================================================================
            # Multi-TimeFrame DataFrames (for Pattern & Reversal Detection)
            # =========================================================================
            'df_primary': df_primary,
            'df_m1': data.get('M1'),
            'df_m5': data.get('M5'),
            'df_m15': data.get('M15'),
            'df_h1': data.get('H1'),
        }
        return context

    def _log_context_summary(self, context: Dict, current_atr: float):
        """Periodically log the current market context for monitoring."""
        current_time = time.time()
        if current_time - self._last_context_log_time > self.context_log_interval:
            self.logger.info(
                f"[CONTEXT] Regime: {context['regime_name']} ({context['regime']}) | "
                f"Session: {context['session']} | "
                f"Choppy: {context['choppy_score']:.0f} ({context['choppy_severity']}) | "
                f"Killers: {len(context['active_killers'])} | "
                f"Kelly Mult: {context['kelly_multiplier']:.2f}x | "
                f"Daily PnL: {context['daily_pnl_pct']:.2f}% | "
                f"ATR: {current_atr:.2f}"
            )
            self._last_context_log_time = current_time

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    def _run_loop(self):
        """The core execution loop with Scalping Mode support."""
        self.logger.info(f"[EVENT_LOOP] Starting main loop | Symbol: {self.symbol} | Primary TF: {self.primary_tf}")
        
        # Log active modes
        if getattr(config, 'scalping_mode', False):
            self.logger.info(f"[MODE] Scalping Mode ENABLED (Primary TF: M1, Loop: 0.1s)")
        if getattr(config, 'micro_account_mode', False):
            sl_distance = getattr(config, 'micro_sl_distance_usd', 16.0)
            self.logger.info(
                f"[MODE] Micro-Account Mode ENABLED | "
                f"SL: {sl_distance} USD | Lot: 0.01-0.03 | Risk: {self.risk_pct:.2f}%"
            )
        
        while self._running:
            cycle_start = time.time()
            
            try:
                # 1. Health Check
                if not self._check_mt5_health():
                    time.sleep(10)
                    continue
                
                # 2. Fetch Multi-Timeframe Data
                data = self.data_manager.fetch_all_timeframes()
                if not data:
                    time.sleep(config.event_loop_interval_seconds)
                    continue
                
                # =========================================================================
                # Scalping Mode: Trigger all TFs every loop (faster reaction)
                # =========================================================================
                if getattr(config, 'scalping_mode', False):
                    triggered_tfs = set()
                    for tf_name, df in data.items():
                        if df is not None and not df.empty and 'time' in df.columns:
                            triggered_tfs.add(tf_name)
                # =========================================================================
                # Normal Mode: Trigger only on bar close (Lookahead Bias Prevention)
                # =========================================================================
                else:
                    triggered_tfs = self._detect_bar_closures(data)
                
                # 3. Get Current Time & Session
                df_primary = data.get(self.primary_tf)
                if df_primary is not None and not df_primary.empty and 'time' in df_primary.columns:
                    current_time = df_primary['time'].iloc[-1]
                else:
                    current_time = pd.Timestamp.now()
                    
                session_info = self.session_mgr.get_current_session(current_time)
                
                # 4. Detect Market Conditions (18-Regime, Choppy, Killers)
                # [UPGRADE] Use analyze_and_route with signals for weight allocation
                regime_result = self.regime_router.analyze_and_route(
                    data.get('M5'), data.get('M15'), data.get('H1'), {}
                )
                
                choppy_result = None
                if self.choppy_detector is not None:
                    try:
                        choppy_result = self.choppy_detector.detect_choppy(
                            data.get('M5'), regime_result.get('hurst_value')
                        )
                    except Exception as e:
                        self.logger.error(f"[EVENT_LOOP] Choppy detection error: {e}")
                        
                killers_report = None
                if self.killers_detector is not None:
                    try:
                        killers_report = self.killers_detector.detect_all_killers(data.get('M5'), None)
                    except Exception as e:
                        self.logger.error(f"[EVENT_LOOP] Killers detection error: {e}")
                        
                # 5. Build Context & Get Daily Stats
                daily_stats = self.order_manager.risk_manager.get_daily_stats() if hasattr(self.order_manager.risk_manager, 'get_daily_stats') else {'daily_pnl_pct': 0.0}
                context = self._build_context(data, regime_result, choppy_result, killers_report, session_info, daily_stats)
                
                # Calculate Current ATR for Friction Filter
                current_atr = 0.0
                if df_primary is not None and len(df_primary) >= 14:
                    atr_series = ATRCache.get_atr(df_primary, 14)
                    atr_val = atr_series.iloc[-1]
                    if not pd.isna(atr_val):
                        current_atr = float(atr_val)
                
                # Log Context Periodically
                self._log_context_summary(context, current_atr)
                
                # 6. Evaluate Strategies (Only on Bar Close or First Run, or Scalping Mode)
                if triggered_tfs or not self._last_bar_times or getattr(config, 'scalping_mode', False):
                    signals = self.strategy_pool.evaluate_all(data, triggered_tfs, regime_context=context)
                    
                    account_info = mt5.account_info()
                    if account_info and signals:
                        for strategy_name, sig in signals.items():
                            # Signal Deduplication
                            tf_primary = self.strategy_pool.ROUTE_MAP.get(strategy_name, ('M15', None))[0]
                            df_strat = data.get(tf_primary)
                            strat_bar_time = df_strat['time'].iloc[-1] if df_strat is not None and not df_strat.empty else pd.Timestamp.now()
                            
                            signal_hash = hashlib.md5(
                                json.dumps({'s': strategy_name, 't': str(strat_bar_time)}, sort_keys=True).encode()
                            ).hexdigest()
                            
                            if signal_hash in self._processed_signal_hashes:
                                continue
                            
                            if self.order_manager.process_signal(
                                sig, account_info.balance, current_atr, context=context
                            ):
                                self._processed_signal_hashes.add(signal_hash)
                        
                        # Keep only last 500 hashes to prevent memory leak
                        if len(self._processed_signal_hashes) > 1000:
                            self._processed_signal_hashes = set(list(self._processed_signal_hashes)[-500:])
                
                # 7. Manage Pending Orders
                self.order_manager.manage_pending_orders(current_time)
                
                # 8. Manage Active Positions (Multi-Layer Defense)
                # [UPGRADE] Pass regime_context for Active Liquidation
                self.order_manager.manage_active_positions(
                    current_prices={}, 
                    data=data, 
                    regime_context=regime_result,
                    choppy_result=choppy_result
                )
                
                # 9. Sleep (faster in scalping mode)
                elapsed = time.time() - cycle_start
                sleep_base = 0.1 if getattr(config, 'scalping_mode', False) else config.event_loop_interval_seconds
                sleep_time = max(0.1, sleep_base - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("[EVENT_LOOP] KeyboardInterrupt received. Shutting down...")
                self._running = False
            except Exception as e:
                self.logger.critical(f"[EVENT_LOOP] Critical error in main loop: {e}", exc_info=True)
                time.sleep(10)  # Prevent tight loop on persistent errors

    # =========================================================================
    # ENTRY POINT & SHUTDOWN
    # =========================================================================

    def start(self):
        """Start the event loop with signal handling for graceful shutdown."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        if not self._initialize_mt5():
            self.logger.critical("[EVENT_LOOP] MT5 initialization failed. Exiting.")
            sys.exit(1)
            
        # Sync state with MT5 on startup to recover from crashes
        self.order_manager.sync_with_mt5()
        
        self._running = True
        try:
            self._run_loop()
        finally:
            self._shutdown()

    def _signal_handler(self, signum, frame):
        """Handle OS signals for graceful shutdown."""
        self.logger.info(f"[EVENT_LOOP] Received signal {signum}. Initiating graceful shutdown...")
        self._running = False

    def _shutdown(self):
        """Clean up resources and close connections."""
        self.logger.info("[EVENT_LOOP] Shutting down engines...")
        
        # Close SQLite connections in StateManager
        if hasattr(self.order_manager.state_manager, 'close'):
            self.order_manager.state_manager.close()
            
        # Shutdown MT5
        mt5.shutdown()
        self.logger.info("[EVENT_LOOP] Shutdown complete. Goodbye.")