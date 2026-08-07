#!/usr/bin/env python3
"""
Signal Scorer Monitor.

Monitors and displays signal scoring statistics in real-time.
Useful for debugging and optimizing signal quality.

Usage:
    python monitor_signal_scorer.py
"""
import time
import logging
import sqlite3
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignalScorerMonitor")


def monitor_signal_scorer(db_path: str = "bot_state.db", interval: int = 60):
    """Monitor signal scorer statistics."""
    logger.info("[MONITOR] Starting signal scorer monitor...")
    logger.info(f"[MONITOR] Database: {db_path}")
    logger.info(f"[MONITOR] Interval: {interval}s")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        while True:
            try:
                with sqlite3.connect(db_path, timeout=10) as conn:
                    # Get recent signals
                    cursor = conn.execute("""
                        SELECT strategy, COUNT(*) as count,
                               AVG(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winrate,
                               SUM(profit) as total_profit
                        FROM trade_history
                        WHERE close_time >= datetime('now', '-24 hours')
                        GROUP BY strategy
                        ORDER BY count DESC
                    """)
                    rows = cursor.fetchall()

                    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Last 24 hours:")
                    logger.info("-" * 60)

                    if not rows:
                        logger.info("  No trades in the last 24 hours")
                    else:
                        for row in rows:
                            strategy, count, winrate, profit = row
                            logger.info(
                                f"  {strategy}: {count} trades, "
                                f"Winrate: {winrate*100:.1f}%, "
                                f"Profit: {profit:.2f}"
                            )

                    logger.info("=" * 60)

            except Exception as e:
                logger.error(f"[MONITOR] Error: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("[MONITOR] Stopped by user")


if __name__ == "__main__":
    monitor_signal_scorer()