#!/usr/bin/env python3
"""
Gold Multi-Strategy Trading Bot - Main Entry Point.

Institutional-grade multi-strategy trading system for XAUUSD (Gold).
Designed for micro-account trading ($500-$3000).

Features:
  - 30 trading strategies (SMC, TREND, SCALP, MEAN_REVERSION)
  - 18-regime detection system
  - 10-layer active position management
  - Regime-adaptive risk management
  - Multi-timeframe analysis
  - Machine learning integration (HMM, LightGBM)

Author: maxowl
Version: 1.0
"""
import sys
import logging
import argparse
import os
from datetime import datetime
from pathlib import Path

from orchestrator.event_loop import EventLoop
from orchestrator.data_manager import DataManager
from orchestrator.strategy_pool import StrategyPool
from execution.order_manager import OrderManager
from execution.state_manager import StateManager
from execution.risk_manager import RiskManager
from execution.friction_filter import FrictionFilter
from execution.position_intelligence_manager import PositionIntelligenceManager
from execution.order_quality_monitor import OrderQualityMonitor
from execution.drawdown_scaler import DrawdownRiskScaler
from execution.modification_limiter import ModificationRateLimiter
from execution.equity_circuit_breaker import EquityCircuitBreaker
from execution.trade_recorder import TradeRecorder
from core.regime_router import RegimeRouter
from core.expert_signal_scorer import ExpertSignalScorer
from core.kelly_criterion import KellyCriterionEngine
from core.time_stop_manager import TimeStopManager
from core.emergency_defense_engine import EmergencyDefenseEngine
from core.dynamic_stops_manager import DynamicStopsManager
from core.strategy_performance_tracker import StrategyPerformanceTracker
from core.session_volatility import SessionVolatilityManager
from config import config


def setup_logging(log_dir: str = "logs"):
    """Configure logging with rotation and ASCII-safe formatting."""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress noisy third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('lightgbm').setLevel(logging.WARNING)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Gold Multi-Strategy Trading Bot')
    parser.add_argument('--symbol', type=str, default=config.symbol, help='Trading symbol')
    parser.add_argument('--tf', type=str, default=config.primary_timeframe, help='Primary timeframe')
    parser.add_argument('--risk', type=float, default=config.risk_per_trade_pct, help='Risk per trade %%')
    return parser.parse_args()


def log_startup_banner():
    """Log comprehensive startup banner with current configuration."""
    logger = logging.getLogger("Main")

    logger.info("=" * 78)
    logger.info("GOLD MULTI-STRATEGY TRADING BOT INITIALIZING")
    logger.info("MICRO-ACCOUNT MODE (XAUUSD @ 4000 USD baseline)")
    logger.info("=" * 78)

    # Basic Configuration
    logger.info(f"[CONFIG] Symbol: {config.symbol}")
    logger.info(f"[CONFIG] Primary Timeframe: {config.primary_timeframe}")
    logger.info(f"[CONFIG] Magic Number: {config.magic_number}")
    logger.info(f"[CONFIG] Max Slippage: {config.max_slippage_points} points")

    # Micro-Account Risk Management
    logger.info(f"[RISK] Risk Per Trade: {config.risk_per_trade_pct}%")
    logger.info(f"[RISK] SL Distance: {config.sl_distance_usd} USD (Fixed)")
    logger.info(f"[RISK] Lot Size Range: {config.min_lot_size} - {config.max_lot_size}")
    logger.info(f"[RISK] Max Open Positions: {config.max_open_positions}")
    logger.info(f"[RISK] Max Pending Orders: {config.max_pending_orders}")
    logger.info(f"[RISK] Max Daily Loss: {config.max_daily_loss_pct}%")

    # Regime-Adaptive Breakeven Triggers
    logger.info(f"[BREAKEVEN] Regime-Adaptive Triggers:")
    logger.info(f"  - Strong Trend: {config.be_strong_trend_usd} USD")
    logger.info(f"  - Parabolic: {config.be_parabolic_usd} USD")
    logger.info(f"  - Consolidating: {config.be_consolidating_usd} USD")
    logger.info(f"  - Sideways: {config.be_sideways_usd} USD")
    logger.info(f"  - Choppy: {config.be_choppy_usd} USD")
    logger.info(f"  - Reversal: {config.be_reversal_usd} USD")

    # Trailing & Partial Close
    logger.info(f"[TRAILING] Increment: {config.trail_increment_usd} USD (after breakeven)")
    logger.info(f"[PARTIAL] Close {config.partial_close_percent*100:.0f}% at {config.partial_close_trigger_usd} USD profit")

    # Friction Filter
    logger.info(f"[FRICTION] Max Spread: {config.max_spread_points} points")
    logger.info(f"[FRICTION] Min R:R: {config.min_rr_ratio}")
    logger.info(f"[FRICTION] Min Profit: ${config.min_profit_usd}")
    logger.info(f"[FRICTION] Min Edge/Friction: {config.min_edge_to_friction}")

    # Adaptive TP
    logger.info(f"[TP] Trend R:R: {config.tp_trend_rr}")
    logger.info(f"[TP] Sideway R:R: {config.tp_sideway_rr}")
    logger.info(f"[TP] High Vol R:R: {config.tp_highvol_rr}")
    logger.info(f"[TP] Reversal R:R: {config.tp_reversal_rr}")

    # Event Loop
    logger.info(f"[LOOP] Event Loop Interval: {config.event_loop_interval_seconds}s")
    logger.info(f"[LOOP] Pending Order Timeout: {config.pending_order_timeout_minutes} min")
    logger.info(f"[LOOP] Reconciliation Interval: {config.reconciliation_interval_seconds}s")
    logger.info(f"[LOOP] Intelligence Check Interval: {config.intelligence_check_interval_seconds}s")

    # File Paths
    logger.info(f"[PATHS] State DB: {config.state_db_path}")
    logger.info(f"[PATHS] HMM Model: {config.regime_model_path}")
    logger.info(f"[PATHS] LightGBM Models: {config.lightgbm_models_path}")
    logger.info(f"[PATHS] Rule Model: {config.rule_model_path}")
    logger.info(f"[PATHS] Log Directory: {config.log_directory}")

    logger.info("=" * 78)


def initialize_components():
    """Initialize all system components."""
    logger = logging.getLogger("Main")

    logger.info("[INIT] Initializing components...")

    # =========================================================================
    # State and Persistence
    # =========================================================================
    state_manager = StateManager(db_path=config.state_db_path)

    # =========================================================================
    # Risk Management
    # =========================================================================
    performance_tracker = StrategyPerformanceTracker(db_path=config.state_db_path)

    # Kelly Criterion - ไม่รับ performance_tracker
    kelly_criterion = KellyCriterionEngine(
        min_trades=50,
        max_risk_pct=3.0,
        use_half_kelly=True,
        min_winrate=0.4,
        min_profit_factor=1.2
    )

    # RiskManager - รับเฉพาะ 5 parameters
    risk_manager = RiskManager(
        risk_per_trade_pct=config.risk_per_trade_pct,
        max_open_positions=config.max_open_positions,
        max_pending_orders=config.max_pending_orders,
        max_daily_loss_pct=config.max_daily_loss_pct,
        symbol=config.symbol
    )

    # FrictionFilter - รับเฉพาะ symbol
    friction_filter = FrictionFilter(symbol=config.symbol)

    # =========================================================================
    # Order Manager - สร้าง components ภายในเอง
    # =========================================================================
    order_manager = OrderManager(
        symbol=config.symbol,
        magic_number=config.magic_number,
        max_slippage=config.max_slippage_points,
        risk_per_trade_pct=config.risk_per_trade_pct,
        max_open_positions=config.max_open_positions,
        max_pending_orders=config.max_pending_orders,
        pending_order_timeout_minutes=config.pending_order_timeout_minutes,
        state_db_path=config.state_db_path
    )

    # =========================================================================
    # Data Manager
    # =========================================================================
    data_manager = DataManager(symbol=config.symbol)

    # =========================================================================
    # Regime Router
    # =========================================================================
    regime_router = RegimeRouter()

    # =========================================================================
    # Strategy Pool
    # =========================================================================
    strategy_pool = StrategyPool()

    # =========================================================================
    # Event Loop
    # =========================================================================
    event_loop = EventLoop(
        symbol=config.symbol,
        primary_tf=config.primary_timeframe,
        risk_pct=config.risk_per_trade_pct
    )

    logger.info("[INIT] All components initialized successfully")

    return event_loop


def main():
    """Main entry point."""
    args = parse_args()

    # Override config with CLI args
    config.symbol = args.symbol
    config.primary_timeframe = args.tf
    config.risk_per_trade_pct = args.risk

    setup_logging(config.log_directory)
    logger = logging.getLogger("Main")

    # Log startup banner
    log_startup_banner()

    try:
        # Initialize MT5
        import MetaTrader5 as mt5
        if not mt5.initialize():
            logger.error("[FAIL] MT5 initialization failed")
            return

        logger.info("[INIT] MT5 initialized successfully")

        # Initialize components
        event_loop = initialize_components()

        # Start event loop
        event_loop.start()

    except KeyboardInterrupt:
        logger.info("[SYSTEM] Shutdown requested by user")
    except Exception as e:
        logger.critical(f"[FAIL] Fatal error in main execution: {e}", exc_info=True)
    finally:
        # Cleanup
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass

        logger.info("[SYSTEM] Bot shutdown complete.")


if __name__ == "__main__":
    main()