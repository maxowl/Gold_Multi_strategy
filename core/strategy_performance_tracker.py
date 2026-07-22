"""
Strategy Performance Tracker.
Records trade history to SQLite and calculates performance metrics for Kelly Criterion and Expert Scorer.
"""
import sqlite3
import json
import logging
import numpy as np
from typing import Dict
from datetime import datetime, timedelta


class StrategyPerformanceTracker:
    def __init__(self, db_path: str = "bot_state.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and create the trade_history table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_history (
                        ticket INTEGER PRIMARY KEY,
                        symbol TEXT, strategy TEXT, direction TEXT,
                        entry_price REAL, exit_price REAL, sl_price REAL, tp_price REAL,
                        volume REAL, profit REAL, commission REAL, swap REAL,
                        open_time TEXT, close_time TEXT,
                        entry_reason TEXT, exit_reason TEXT,
                        is_pending INTEGER, order_type TEXT, expected_entry REAL,
                        meta_data TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] DB init error: {e}")

    def record_trade_enhanced(self, ticket, symbol, strategy, direction, entry_price, exit_price, 
                              sl_price, tp_price, volume, profit, commission, swap, 
                              open_time, close_time, entry_reason, exit_reason, 
                              is_pending, order_type, expected_entry, meta_data):
        """
        Record a trade to the SQLite database.
        [FIX] Serializes meta_data dict to JSON string to prevent sqlite3.InterfaceError.
        """
        try:
            # Serialize meta_data dict to JSON string safely
            meta_str = json.dumps(meta_data) if isinstance(meta_data, dict) else str(meta_data)
            
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trade_history 
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (ticket, symbol, strategy, direction, entry_price, exit_price, 
                      sl_price, tp_price, volume, profit, commission, swap, 
                      str(open_time), str(close_time), entry_reason, exit_reason, 
                      int(is_pending), order_type, expected_entry, meta_str))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Record trade error: {e}")

    def get_strategy_stats(self, strategy: str, regime: str = 'UNKNOWN', days: int = 30) -> Dict:
        """
        Calculate historical performance metrics for a specific strategy and regime.
        [FIX] Prevents ZeroDivisionError in Profit Factor and caps it at 10.0.
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.execute("""
                    SELECT profit, meta_data FROM trade_history 
                    WHERE strategy = ? AND close_time >= ? AND is_pending = 0
                """, (strategy, cutoff_date))
                rows = cursor.fetchall()
                
            if not rows:
                return {'trades': 0, 'winrate': 0.5, 'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 1.0}
            
            # Filter by regime from meta_data JSON
            filtered_profits = []
            for profit, meta_str in rows:
                try:
                    meta = json.loads(meta_str) if meta_str else {}
                    trade_regime = meta.get('regime', 'UNKNOWN')
                    if regime == 'UNKNOWN' or trade_regime == regime:
                        filtered_profits.append(float(profit))
                except Exception:
                    filtered_profits.append(float(profit))
            
            if not filtered_profits:
                return {'trades': 0, 'winrate': 0.5, 'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 1.0}
            
            profits = np.array(filtered_profits)
            wins = profits[profits > 0]
            losses = profits[profits < 0]
            
            winrate = len(wins) / len(profits)
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
            
            total_win = np.sum(wins)
            total_loss = abs(np.sum(losses))
            
            # Prevent ZeroDivisionError and cap Profit Factor at 10.0 for Kelly stability
            if total_loss == 0:
                profit_factor = 10.0 if total_win > 0 else 1.0
            else:
                profit_factor = min(10.0, total_win / total_loss)
            
            return {
                'trades': len(profits),
                'winrate': float(winrate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor)
            }
        except Exception as e:
            self.logger.error(f"[FAIL] Get stats error: {e}")
            return {'trades': 0, 'winrate': 0.5, 'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 1.0}