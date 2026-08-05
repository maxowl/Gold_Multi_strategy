#!/usr/bin/env python3
"""
Institutional Multi-Strategy Trading Bot - Main Entry Point.
Updated for Micro-Account Mode with Regime-Adaptive Breakeven.
"""
import sys
import logging
import argparse
import os
from datetime import datetime
from pathlib import Path

from orchestrator.event_loop import EventLoop
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
    parser = argparse.ArgumentParser(description='Institutional Trading Bot')
    parser.add_argument('--symbol', type=str, default=config.symbol, help='Trading symbol')
    parser.add_argument('--tf', type=str, default=config.primary_timeframe, help='Primary timeframe')
    parser.add_argument('--risk', type=float, default=config.risk_per_trade_pct, help='Risk per trade %')
    return parser.parse_args()


def log_startup_banner():
    """Log comprehensive startup banner with current configuration."""
    logger = logging.getLogger("Main")
    
    logger.info("=" * 78)
    logger.info("INSTITUTIONAL TRADING BOT INITIALIZING")
    logger.info("=" * 78)
    
    # Basic Configuration
    logger.info(f"[CONFIG] Symbol: {config.symbol}")
    logger.info(f"[CONFIG] Primary Timeframe: {config.primary_timeframe}")
    logger.info(f"[CONFIG] Magic Number: {config.magic_number}")
    logger.info(f"[CONFIG] Max Slippage: {config.max_slippage_points} points")
    
    # Operating Mode
    if getattr(config, 'micro_account_mode', False):
        logger.info(f"[MODE] Micro-Account Mode: ENABLED (for XAUUSD @ 4000 USD baseline)")
        logger.info(f"    Risk Per Trade: {config.micro_risk_per_trade_pct}%")
        logger.info(f"    SL Distance: {config.micro_sl_distance_usd} USD (Fixed)")
        logger.info(f"    Lot Size Range: {config.micro_min_lot_size} - {config.micro_max_lot_size}")
        
        # Regime-Adaptive Breakeven Triggers
        logger.info(f"    Breakeven Triggers (Regime-Adaptive):")
        logger.info(f"      - Strong Trend:   {config.micro_be_strong_trend_usd} USD  (HEALTHY_UPTREND, DOWNTREND, QUIET_RALLY, SLOW_BLEED)")
        logger.info(f"      - Parabolic:      {config.micro_be_parabolic_usd} USD  (PARABOLIC_RALLY, PANIC_CAPITULATION)")
        logger.info(f"      - Consolidating:  {config.micro_be_consolidating_usd} USD  (CONSOLIDATING_*, FALSE_SIDEWAY)")
        logger.info(f"      - Sideways:       {config.micro_be_sideways_usd} USD  (CLASSIC_RANGE, TIGHT_RANGE, PRE_BREAKOUT)")
        logger.info(f"      - Choppy:         {config.micro_be_choppy_usd} USD  (VOLATILE_CHOP, WHIPSAW_MARKET)")
        logger.info(f"      - Reversal:       {config.micro_be_reversal_usd} USD  (OVERSOLD_BOUNCE, EXHAUSTED_*, ANOMALY_*)")
        
        # Trailing & Partial Close
        logger.info(f"    Trail Increment: {config.micro_trail_increment_usd} USD (after breakeven)")
        logger.info(f"    Partial Close: {config.micro_partial_close_percent*100:.0f}% at {config.micro_partial_close_trigger_usd} USD profit")
    
    elif getattr(config, 'scalping_mode', False):
        logger.info(f"[MODE] Scalping Mode: ENABLED")
        logger.info(f"    Primary TF: {config.scalping_primary_tf}")
        logger.info(f"    Risk Per Trade: {config.scalp_risk_per_trade_pct}%")
        logger.info(f"    Max Spread: {config.max_spread_points_scalp} points")
        logger.info(f"    Max Trades/Day: {config.max_trades_per_day_scalp}")
        logger.info(f"    Allowed Regimes: {', '.join(config.scalping_regimes_allowed)}")
    
    else:
        logger.info(f"[MODE] Normal Multi-Strategy Mode")
        logger.info(f"    Risk Per Trade: {config.risk_per_trade_pct}%")
        logger.info(f"    Max Daily Loss: {config.max_daily_loss_pct}%")
        logger.info(f"    Max Open Positions: {config.max_open_positions}")
        logger.info(f"    Max Pending Orders: {config.max_pending_orders}")
    
    # Event Loop
    logger.info(f"[LOOP] Event Loop Interval: {config.event_loop_interval_seconds}s")
    logger.info(f"[LOOP] Pending Order Timeout: {config.pending_order_timeout_minutes} min")
    
    # File Paths
    logger.info(f"[PATHS] State DB: {config.state_db_path}")
    logger.info(f"[PATHS] HMM Model: {config.regime_model_path}")
    logger.info(f"[PATHS] LightGBM Models: {config.lightgbm_models_path}")
    logger.info(f"[PATHS] Log Directory: {config.log_directory}")
    
    logger.info("=" * 78)


def main():
    args = parse_args()
    
    # Override config with CLI args
    config.symbol = args.symbol
    config.primary_timeframe = args.tf
    config.risk_per_trade_pct = args.risk
    
    setup_logging(config.log_directory)
    logger = logging.getLogger("Main")
    
    # Log startup banner with current configuration
    log_startup_banner()
    
    try:
        event_loop = EventLoop()
        event_loop.start()
    except KeyboardInterrupt:
        logger.info("[SYSTEM] Shutdown requested by user")
    except Exception as e:
        logger.critical(f"[FAIL] Fatal error in main execution: {e}", exc_info=True)
    finally:
        logger.info("[SYSTEM] Bot shutdown complete.")


if __name__ == "__main__":
    main()