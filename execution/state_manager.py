"""
State Manager.
Handles SQLite persistence for active positions and pending orders.
Ensures state survives bot restarts.
"""
import sqlite3
import json
import logging
from typing import List, Dict
from datetime import datetime


class StateManager:
    def __init__(self, db_path: str = "bot_state.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS active_positions (
                        ticket INTEGER PRIMARY KEY, symbol TEXT, strategy TEXT,
                        position_type TEXT, volume REAL, entry_price REAL,
                        sl REAL, tp REAL, open_time TEXT, meta_data TEXT,
                        trailing_stop_level REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pending_orders (
                        ticket INTEGER PRIMARY KEY, symbol TEXT, strategy TEXT,
                        order_type TEXT, volume REAL, expected_entry REAL,
                        sl REAL, tp REAL, placed_time TEXT, meta_data TEXT,
                        expiration_time TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] State DB init error: {e}")

    def save_active_position(self, ticket, symbol, strategy, position_type, volume,
                             entry_price, sl, tp, meta_data):
        """Save or update an active position."""
        try:
            meta_str = json.dumps(meta_data) if isinstance(meta_data, dict) else str(meta_data)
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO active_positions 
                    (ticket, symbol, strategy, position_type, volume, entry_price, sl, tp, open_time, meta_data, trailing_stop_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticket, symbol, strategy, position_type, volume, entry_price, sl, tp,
                      datetime.now().isoformat(), meta_str, sl))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Save active position error: {e}")

    def save_pending_order(self, ticket, symbol, strategy, order_type, volume,
                           expected_entry, sl, tp, expiration_time, meta_data):
        """Save a pending order."""
        try:
            meta_str = json.dumps(meta_data) if isinstance(meta_data, dict) else str(meta_data)
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pending_orders
                    (ticket, symbol, strategy, order_type, volume, expected_entry, sl, tp, placed_time, meta_data, expiration_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticket, symbol, strategy, order_type, volume, expected_entry, sl, tp,
                      datetime.now().isoformat(), meta_str, expiration_time.isoformat() if expiration_time else None))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Save pending order error: {e}")

    def get_active_positions(self, symbol: str) -> List[Dict]:
        """
        Retrieve all active positions for a symbol.
        [FIX] Parses meta_data JSON string back to dict to prevent AttributeError.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM active_positions WHERE symbol = ?", (symbol,))
                rows = cursor.fetchall()
                
                positions = []
                for row in rows:
                    pos_dict = dict(row)
                    # Parse meta_data JSON string back to dict
                    meta_str = pos_dict.get('meta_data', '{}')
                    try:
                        pos_dict['meta_data'] = json.loads(meta_str) if meta_str else {}
                    except json.JSONDecodeError:
                        pos_dict['meta_data'] = {}
                    positions.append(pos_dict)
                    
                return positions
        except Exception as e:
            self.logger.error(f"[FAIL] Get active positions error: {e}")
            return []

    def get_pending_orders(self, symbol: str) -> List[Dict]:
        """
        Retrieve all pending orders for a symbol.
        [FIX] Parses meta_data JSON string back to dict to prevent AttributeError.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM pending_orders WHERE symbol = ?", (symbol,))
                rows = cursor.fetchall()
                
                orders = []
                for row in rows:
                    order_dict = dict(row)
                    # Parse meta_data JSON string back to dict
                    meta_str = order_dict.get('meta_data', '{}')
                    try:
                        order_dict['meta_data'] = json.loads(meta_str) if meta_str else {}
                    except json.JSONDecodeError:
                        order_dict['meta_data'] = {}
                    orders.append(order_dict)
                    
                return orders
        except Exception as e:
            self.logger.error(f"[FAIL] Get pending orders error: {e}")
            return []

    def remove_active_position(self, ticket: int):
        """Remove an active position from the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("DELETE FROM active_positions WHERE ticket = ?", (ticket,))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Remove active position error: {e}")

    def remove_pending_order(self, ticket: int):
        """Remove a pending order from the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("DELETE FROM pending_orders WHERE ticket = ?", (ticket,))
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Remove pending order error: {e}")

    def update_trailing_stop(self, ticket: int, new_sl: float):
        """Update the trailing stop level for an active position."""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute(
                    "UPDATE active_positions SET sl = ?, trailing_stop_level = ? WHERE ticket = ?",
                    (new_sl, new_sl, ticket)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Update trailing stop error: {e}")

    def close(self):
        """Close database connection (SQLite auto-closes but kept for interface consistency)."""
        pass