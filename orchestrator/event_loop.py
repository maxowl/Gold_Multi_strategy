"""
Main Event Loop Controller.
Orchestrates data fetching, strategy evaluation, regime routing, and order management.
"""
import time
import signal
import logging
import hashlib
import json
import pytz
from typing import Dict, Set
import pandas as pd
import MetaTrader5 as mt5

from core.regime_router import EnhancedRegimeRouter
from execution.order_manager import OrderManager
from execution.state_manager import StateManager
from core.session_volatility import SessionVolatilityManager
from orchestrator.data_manager import DataManager
from orchestrator.strategy_pool import StrategyPool
from config import config


class EventLoop:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = DataManager(config.symbol)
        self.strategy_pool = StrategyPool()
        
        # [FIX] Explicit parameter mapping for EnhancedRegimeRouter
        self.regime_router = EnhancedRegimeRouter(
            rule_model_path="regime_model.pkl",
            hmm_model_path=config.regime_model_path,
            hybrid_model_path=config.lightgbm_models_path
        )
        
        self.order_manager = OrderManager(
            symbol=config.symbol, 
            magic_number=config.magic_number,
            max_slippage=config.max_slippage_points, 
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_open_positions=config.max_open_positions, 
            max_pending_orders=config.max_pending_orders,
            pending_order_timeout_minutes=config.pending_order_timeout_minutes, 
            state_db_path=config.state_db_path
        )
        
        self.state_manager = StateManager(config.state_db_path)
        self.session_mgr = SessionVolatilityManager()
        self._running = False
        self._last_bar_times = {}
        self._processed_signal_hashes: Set[str] = set()
        signal.signal(signal.SIGINT, self._signal_handler)

    def start(self):
        if not self.data_manager.connect(): 
            return
        self.order_manager.sync_with_mt5()
        self._running = True
        self._run_loop()
        self._shutdown()

    def _run_loop(self):
        while self._running:
            try:
                cycle_start = time.time()
                if not self.data_manager.health_check()['mt5_connected']:
                    self.logger.warning("[WARN] MT5 disconnected, attempting reconnect...")
                    self.data_manager.connect()
                    time.sleep(10)
                    continue
                    
                data = self.data_manager.fetch_all_timeframes()
                if not data: 
                    time.sleep(config.event_loop_interval_seconds)
                    continue
                
                triggered_tfs = set()
                for tf_name, df in data.items():
                    if df is not None and not df.empty and 'time' in df.columns:
                        current_bar_time = df['time'].iloc[-1]
                        if self._is_bar_closed(df, current_bar_time, tf_name):
                            last_time = self._last_bar_times.get(tf_name)
                            if last_time is None or current_bar_time > last_time:
                                triggered_tfs.add(tf_name)
                                self._last_bar_times[tf_name] = current_bar_time
                                
                if triggered_tfs: 
                    self._on_new_bar(data, triggered_tfs)
                    
                if 'M1' in triggered_tfs or not self._last_bar_times: 
                    current_regime_summary = self.regime_router.get_regime_summary()
                    self._manage_orders(data, regime_context=current_regime_summary)
                    
                sleep_time = 0.1 if config.scalping_mode else config.event_loop_interval_seconds
                time.sleep(max(0.1, sleep_time - (time.time() - cycle_start)))
            except KeyboardInterrupt: 
                self._running = False
            except Exception as e: 
                self.logger.critical(f"[FAIL] Loop: {e}", exc_info=True)
                time.sleep(10)

    def _is_bar_closed(self, df: pd.DataFrame, bar_time: pd.Timestamp, tf_name: str) -> bool:
        try:
            # [FIX] Safe timezone handling
            if bar_time.tzinfo is None:
                bar_epoch = bar_time.tz_localize(pytz.utc).timestamp()
            else:
                bar_epoch = bar_time.tz_convert(pytz.utc).timestamp()
                
            tf_seconds = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600, 'H4': 14400, 'D1': 86400}.get(tf_name, 300)
            return (time.time() - bar_epoch) >= (tf_seconds - 5)
        except Exception: 
            return False

    def _on_new_bar(self, data: Dict[str, pd.DataFrame], triggered_tfs: set):
        signals = self.strategy_pool.evaluate_all(data, triggered_tfs)
        
        regime_result = self.regime_router.analyze_and_route(
            data.get('M5'), data.get('M15'), data.get('H1'), signals
        )
        
        active_signals = self.strategy_pool.get_active_signals(signals)
        if not active_signals: 
            return
        
        account_info = self.data_manager.get_account_info()
        if not account_info: 
            return
        
        current_atr = 0.0
        df_m15 = data.get('M15')
        if df_m15 is not None and 'atr' in df_m15.columns:
            atr_val = df_m15['atr'].iloc[-1]
            if not pd.isna(atr_val): 
                current_atr = float(atr_val)
            
        context = self._build_scoring_context(data, regime_result)
        
        for strategy_name, sig in active_signals:
            tf_primary = self.strategy_pool.ROUTE_MAP.get(strategy_name, ('M15', None))[0]
            df_strat = data.get(tf_primary)
            strat_bar_time = df_strat['time'].iloc[-1] if df_strat is not None and not df_strat.empty else pd.Timestamp.now()
            
            # [FIX] Ensure strat_bar_time is timezone-aware for consistent hashing
            if strat_bar_time.tzinfo is None:
                strat_bar_time = strat_bar_time.tz_localize(pytz.utc)
                
            signal_hash = hashlib.md5(json.dumps({'s': strategy_name, 't': str(strat_bar_time)}, sort_keys=True).encode()).hexdigest()
            if signal_hash in self._processed_signal_hashes: 
                continue
            
            if self.order_manager.process_signal(sig, account_info['balance'], current_atr, context=context):
                self._processed_signal_hashes.add(signal_hash)
                
        if len(self._processed_signal_hashes) > 1000:
            self._processed_signal_hashes = set(list(self._processed_signal_hashes)[-500:])

    def _manage_orders(self, data: Dict[str, pd.DataFrame], regime_context: dict = None):
        tick = mt5.symbol_info_tick(config.symbol)
        if tick: 
            self.order_manager.manage_active_positions(
                {'bid': tick.bid, 'ask': tick.ask}, 
                data, 
                regime_context=regime_context
            )
            # Sync to detect closed positions and record trade history
            self.order_manager.sync_with_mt5()
            
        df_m1 = data.get('M1')
        if df_m1 is not None and not df_m1.empty:
            self.order_manager.manage_pending_orders(df_m1['time'].iloc[-1])

    def _build_scoring_context(self, data: Dict[str, pd.DataFrame], regime_result: Dict) -> dict:
        df_primary = data.get(config.primary_timeframe)
        if df_primary is None: 
            return {}
        
        # [FIX] Use .get() with defaults to prevent KeyError if router fails
        regime_name = regime_result.get('regime_name', 'UNKNOWN')
        trend_name = regime_result.get('trend', 'UNKNOWN')
        vol_name = regime_result.get('volatility', 'NORMAL')
        fractal_name = regime_result.get('fractal', 'TRENDING')
        kelly_mult = regime_result.get('kelly_multiplier', 1.0)
        trend_conf = regime_result.get('trend_confidence', 0.5)
        
        # Map 18-regime to unified regime for Kelly historical stats
        if any(x in regime_name for x in ['CHOP', 'WHIPSAW', 'PARABOLIC', 'PANIC']):
            unified_regime = 'HIGH_VOL'
        elif any(x in regime_name for x in ['BOUNCE', 'EXHAUSTED', 'ANOMALY']):
            unified_regime = 'REVERSAL'
        elif any(x in regime_name for x in ['UPTREND', 'DOWNTREND', 'BLEED', 'RALLY']):
            unified_regime = 'TREND'
        else:
            unified_regime = 'SIDEWAY'
        
        return {
            'regime': unified_regime, 
            'regime_name': regime_name,
            'trend': trend_name,
            'volatility': vol_name,
            'fractal': fractal_name,
            'kelly_multiplier': kelly_mult,
            'regime_confidence': trend_conf,
            'session': self.session_mgr.get_current_session(df_primary['time'].iloc[-1] if 'time' in df_primary.columns else None),
            'volatility_percentile': regime_result.get('vol_percentile', 0.5) * 100.0, 
            'mtf_alignment': trend_conf, 
            'recent_strategy_performance': {},
            'bars_since_signal': 0, 
            'concurrent_signals': len(self.state_manager.get_active_positions(config.symbol)),
            'daily_pnl_percent': 0.0, 
            'account_drawdown': 0.0
        }

    def _signal_handler(self, signum, frame): 
        self._running = False
        
    def _shutdown(self):
        self.data_manager.disconnect()
        self.state_manager.close()