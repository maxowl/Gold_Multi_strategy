#!/usr/bin/env python3
"""
Institutional Multi-Strategy Trading Bot - Main Entry Point.
"""
import sys
import logging
import argparse
import os
import signal
from datetime import datetime
from pathlib import Path

from orchestrator.event_loop import EventLoop
from config import config


def setup_logging(log_dir: str = "logs"):
    """Configure logging with rotation and ASCII-safe formatting."""
    Path(log_dir).mkdir(exist_ok=True)
    
    log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Check for debug mode via environment variable
    log_level = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
    level = logging.DEBUG if log_level == "DEBUG" else logging.INFO
    
    logging.basicConfig(
        level=level,
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
    logging.getLogger('hmmlearn').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Institutional Trading Bot')
    parser.add_argument('--symbol', type=str, default=config.symbol, help='Trading symbol')
    parser.add_argument('--tf', type=str, default=config.primary_timeframe, help='Primary timeframe')
    parser.add_argument('--risk', type=float, default=config.risk_per_trade_pct, help='Risk per trade %')
    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger = logging.getLogger("Main")
    logger.info(f"[SYSTEM] Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


def main():
    """Main entry point for the trading bot."""
    args = parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Override config with CLI args
    config.symbol = args.symbol
    config.primary_timeframe = args.tf
    config.risk_per_trade_pct = args.risk
    
    setup_logging(config.log_directory)
    logger = logging.getLogger("Main")
    
    logger.info("=" * 70)
    logger.info("INSTITUTIONAL TRADING BOT INITIALIZING")
    logger.info(f"Symbol: {config.symbol} | Timeframe: {config.primary_timeframe} | Risk: {config.risk_per_trade_pct}%")
    logger.info("=" * 70)
    
    # Log configuration summary
    logger.info("[CONFIG] Mode Settings:")
    logger.info(f"  Scalping Mode: {config.scalping_mode}")
    logger.info(f"  Micro-Account Mode: {config.micro_account_mode}")
    if config.micro_account_mode:
        logger.info(f"    SL Distance: {config.micro_sl_distance_usd} USD")
        logger.info(f"    Breakeven Trigger: {config.micro_breakeven_trigger_usd} USD")
        logger.info(f"    Trail Increment: {config.micro_trail_increment_usd} USD")
        logger.info(f"    Lot Range: {config.micro_min_lot_size}-{config.micro_max_lot_size}")
    logger.info("")
    
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