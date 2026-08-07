"""
Trade Attribution Engine - Post-Trade Analysis.
Analyzes every closed trade to determine what worked and what didn't.
"""
import pandas as pd
import numpy as np
import logging
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional


class TradeAttributionEngine:
    """
    Post-trade analysis and attribution engine.
    
    Tracks:
      - Strategy performance by regime
      - Execution quality impact
      - Session performance
      - Exit reason analysis
      - MAE/MFE tracking
    """
    
    def __init__(self, db_path: str = "bot_state.db"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = db_path
        self._create_attribution_table()
    
    def _create_attribution_table(self):
        """Create attribution table if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_attribution (
                    ticket INTEGER PRIMARY KEY,
                    strategy TEXT,
                    regime TEXT,
                    session TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    volume REAL,
                    profit REAL,
                    expected_rr REAL,
                    actual_rr REAL,
                    slippage REAL,
                    execution_cost REAL,
                    exit_reason TEXT,
                    duration_minutes REAL,
                    mae REAL,
                    mfe REAL,
                    attribution_data TEXT,
                    close_time TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"[ATTRIBUTION] Table creation error: {e}")
    
    def record_trade_attribution(self, trade_data: Dict):
        """Record complete trade attribution."""
        try:
            # Calculate derived metrics
            entry = trade_data.get('entry_price', 0)
            exit_p = trade_data.get('exit_price', 0)
            sl = trade_data.get('sl_price', 0)
            tp = trade_data.get('tp_price', 0)
            direction = trade_data.get('direction', 'BUY')
            
            # Expected R:R
            if direction == 'BUY':
                risk = entry - sl if sl > 0 else 0
                reward = tp - entry if tp > 0 else 0
            else:
                risk = sl - entry if sl > 0 else 0
                reward = entry - tp if tp > 0 else 0
            
            expected_rr = reward / risk if risk > 0 else 0
            
            # Actual R:R
            if direction == 'BUY':
                actual_profit = exit_p - entry
            else:
                actual_profit = entry - exit_p
            
            actual_rr = actual_profit / risk if risk > 0 else 0
            
            # MAE/MFE
            mae = trade_data.get('mae', 0)
            mfe = trade_data.get('mfe', 0)
            
            # Attribution data
            attribution = {
                'regime_at_entry': trade_data.get('regime', 'UNKNOWN'),
                'session_at_entry': trade_data.get('session', 'UNKNOWN'),
                'exit_reason': trade_data.get('exit_reason', 'UNKNOWN'),
                'slippage': trade_data.get('slippage', 0),
                'spread_at_entry': trade_data.get('spread_at_entry', 0),
                'commission': trade_data.get('commission', 0),
                'swap': trade_data.get('swap', 0)
            }
            
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO trade_attribution 
                (ticket, strategy, regime, session, direction, entry_price, exit_price,
                 sl_price, tp_price, volume, profit, expected_rr, actual_rr,
                 slippage, execution_cost, exit_reason, duration_minutes,
                 mae, mfe, attribution_data, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('ticket'),
                trade_data.get('strategy'),
                trade_data.get('regime'),
                trade_data.get('session'),
                direction,
                entry,
                exit_p,
                sl,
                tp,
                trade_data.get('volume', 0),
                trade_data.get('profit', 0),
                expected_rr,
                actual_rr,
                trade_data.get('slippage', 0),
                trade_data.get('execution_cost', 0),
                trade_data.get('exit_reason'),
                trade_data.get('duration_minutes', 0),
                mae,
                mfe,
                json.dumps(attribution),
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            
            self.logger.info(
                f"[ATTRIBUTION] Ticket {trade_data.get('ticket')} recorded | "
                f"Strategy: {trade_data.get('strategy')} | "
                f"Exit: {trade_data.get('exit_reason')} | "
                f"R:R Expected: {expected_rr:.2f} Actual: {actual_rr:.2f}"
            )
        
        except Exception as e:
            self.logger.error(f"[ATTRIBUTION] Recording error: {e}")
    
    def get_strategy_performance(self, strategy: str, days: int = 30) -> Dict:
        """Get performance breakdown for a specific strategy."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(profit) as total_profit,
                    AVG(profit) as avg_profit,
                    AVG(expected_rr) as avg_expected_rr,
                    AVG(actual_rr) as avg_actual_rr,
                    AVG(slippage) as avg_slippage,
                    AVG(duration_minutes) as avg_duration
                FROM trade_attribution
                WHERE strategy = ? AND close_time >= datetime('now', ?)
            """, (strategy, f"-{days} days"))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None or row['total_trades'] == 0:
                return {'total_trades': 0}
            
            win_rate = row['wins'] / row['total_trades'] if row['total_trades'] > 0 else 0
            
            return {
                'total_trades': row['total_trades'],
                'win_rate': round(win_rate, 3),
                'total_profit': round(row['total_profit'], 2),
                'avg_profit': round(row['avg_profit'], 2),
                'avg_expected_rr': round(row['avg_expected_rr'], 2),
                'avg_actual_rr': round(row['avg_actual_rr'], 2),
                'rr_efficiency': round(row['avg_actual_rr'] / row['avg_expected_rr'], 3) if row['avg_expected_rr'] > 0 else 0,
                'avg_slippage': round(row['avg_slippage'], 2),
                'avg_duration_min': round(row['avg_duration'], 1)
            }
        
        except Exception as e:
            self.logger.error(f"[ATTRIBUTION] Performance query error: {e}")
            return {'total_trades': 0}
    
    def get_exit_reason_analysis(self, days: int = 30) -> Dict:
        """Analyze exit reasons to understand what's closing positions."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    exit_reason,
                    COUNT(*) as count,
                    SUM(profit) as total_profit,
                    AVG(profit) as avg_profit
                FROM trade_attribution
                WHERE close_time >= datetime('now', ?)
                GROUP BY exit_reason
                ORDER BY count DESC
            """, (f"-{days} days",))
            
            results = {}
            for row in cursor.fetchall():
                results[row['exit_reason']] = {
                    'count': row['count'],
                    'total_profit': round(row['total_profit'], 2),
                    'avg_profit': round(row['avg_profit'], 2)
                }
            
            conn.close()
            return results
        
        except Exception as e:
            self.logger.error(f"[ATTRIBUTION] Exit reason query error: {e}")
            return {}