"""
State Manager for SQLite Database Persistence - Ultimate Master Release.
Manages active positions, pending orders, trade history, and partial close states.
Includes robust connection handling, JSON meta_data parsing, and daily PnL tracking.
"""
import sqlite3
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import contextmanager


class StateManager:
    """
    SQLite-based state persistence for the trading bot.
    
    Features:
    - Active positions tracking with meta_data JSON support
    - Pending orders management with expiration tracking
    - Trade history for Kelly Criterion and Performance Analysis
    - Partial close state persistence (TP1, TP2, Reversal Close)
    - Daily PnL calculation for Portfolio Health scoring
    - Thread-safe connection handling with check_same_thread=False
    """
    
    def __init__(self, db_path: str = "bot_state.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # =========================================================================
        # Initialize SQLite Connection
        # =========================================================================
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            self._create_tables()
            self.logger.info(f"[OK] StateManager initialized with database: {db_path}")
        except Exception as e:
            self.logger.critical(f"[FAIL] Could not initialize database: {e}")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Active Positions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_positions (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT,
                position_type TEXT,
                volume REAL,
                entry_price REAL,
                sl REAL,
                tp REAL,
                strategy TEXT,
                open_time TEXT,
                requires_dynamic_exit INTEGER,
                dynamic_exit_threshold TEXT,
                entry_reason TEXT,
                expected_entry REAL,
                order_type TEXT,
                is_pending INTEGER,
                trailing_stop_level REAL,
                meta_data TEXT
            )
        """)
        
        # Pending Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_orders (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT,
                order_type TEXT,
                volume REAL,
                price REAL,
                sl REAL,
                tp REAL,
                strategy TEXT,
                expiration_bars INTEGER,
                requires_dynamic_exit INTEGER,
                dynamic_exit_threshold TEXT,
                setup_time TEXT,
                entry_reason TEXT,
                meta_data TEXT
            )
        """)
        
        # Trade History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                profit REAL,
                open_time TEXT,
                close_time TEXT,
                strategy TEXT,
                meta_data TEXT
            )
        """)
        
        # Partial Close State Table (with reversal_close column)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partial_close_state (
                ticket INTEGER PRIMARY KEY,
                tp1_hit INTEGER DEFAULT 0,
                tp2_hit INTEGER DEFAULT 0,
                reversal_close INTEGER DEFAULT 0
            )
        """)
        
        # Migrate partial_close_state table if reversal_close column is missing
        try:
            cursor.execute("PRAGMA table_info(partial_close_state)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'reversal_close' not in columns:
                cursor.execute("ALTER TABLE partial_close_state ADD COLUMN reversal_close INTEGER DEFAULT 0")
                self.logger.info("[OK] Migrated partial_close_state: added reversal_close column")
        except Exception as e:
            self.logger.debug(f"[MIGRATE] partial_close_state migration check: {e}")
        
        self.conn.commit()

    # =========================================================================
    # ACTIVE POSITIONS
    # =========================================================================

    def save_active_position(self, ticket, symbol, position_type, volume, entry_price, sl, tp, strategy,
                              requires_dynamic_exit=False, dynamic_exit_threshold=None, entry_reason="",
                              expected_entry=0.0, order_type="MARKET", is_pending=False, meta_data=None):
        """Save active position to database."""
        try:
            meta_json = json.dumps(meta_data or {}, default=str)
            
            values = (
                int(ticket),
                str(symbol),
                str(position_type),
                float(volume),
                float(entry_price),
                float(sl) if sl else 0.0,
                float(tp) if tp else 0.0,
                str(strategy),
                datetime.now().isoformat(),
                int(requires_dynamic_exit),
                str(dynamic_exit_threshold) if dynamic_exit_threshold else "",
                str(entry_reason),
                float(expected_entry),
                str(order_type),
                int(is_pending),
                None,  # trailing_stop_level
                meta_json
            )
            
            self.conn.cursor().execute(
                "INSERT OR REPLACE INTO active_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values
            )
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"[FAIL] Save active position error: {e}")
            raise

    def get_active_positions(self, symbol: str = None) -> List[Dict]:
        """Get all active positions, optionally filtered by symbol."""
        try:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute("SELECT * FROM active_positions WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("SELECT * FROM active_positions")
            
            rows = cursor.fetchall()
            positions = []
            
            for row in rows:
                pos = dict(row)
                # Parse meta_data JSON
                if pos.get('meta_data'):
                    try:
                        pos['meta_data'] = json.loads(pos['meta_data'])
                    except (json.JSONDecodeError, TypeError):
                        pos['meta_data'] = {}
                else:
                    pos['meta_data'] = {}
                positions.append(pos)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get active positions error: {e}")
            return []

    def remove_active_position(self, ticket: int):
        """Remove active position from database."""
        try:
            self.conn.cursor().execute(
                "DELETE FROM active_positions WHERE ticket = ?",
                (ticket,)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Remove active position error: {e}")

    def update_trailing_stop(self, ticket: int, new_sl: float):
        """Update trailing stop level for active position."""
        try:
            self.conn.cursor().execute(
                "UPDATE active_positions SET trailing_stop_level = ? WHERE ticket = ?",
                (new_sl, ticket)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Update trailing stop error: {e}")

    def update_position_meta(self, ticket: int, meta: dict):
        """
        Update meta_data for an active position.
        Used by Reversal Detection to persist trail_mult changes.
        """
        try:
            meta_json = json.dumps(meta, default=str)
            self.conn.cursor().execute(
                "UPDATE active_positions SET meta_data = ? WHERE ticket = ?",
                (meta_json, ticket)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Update position meta error: {e}")

    def update_position_sl(self, ticket: int, new_sl: float):
        """Update stop loss for active position."""
        try:
            self.conn.cursor().execute(
                "UPDATE active_positions SET sl = ? WHERE ticket = ?",
                (new_sl, ticket)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Update position SL error: {e}")

    # =========================================================================
    # PENDING ORDERS
    # =========================================================================

    def save_pending_order(self, ticket, symbol, order_type, volume, price, sl, tp, strategy,
                           expiration_bars=10, requires_dynamic_exit=False, dynamic_exit_threshold=None,
                           entry_reason="", meta_data=None):
        """Save pending order to database."""
        try:
            meta_json = json.dumps(meta_data or {}, default=str)
            
            values = (
                int(ticket),
                str(symbol),
                str(order_type),
                float(volume),
                float(price),
                float(sl) if sl else 0.0,
                float(tp) if tp else 0.0,
                str(strategy),
                int(expiration_bars),
                int(requires_dynamic_exit),
                str(dynamic_exit_threshold) if dynamic_exit_threshold else "",
                datetime.now().isoformat(),
                str(entry_reason),
                meta_json
            )
            
            self.conn.cursor().execute(
                "INSERT OR REPLACE INTO pending_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values
            )
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"[FAIL] Save pending order error: {e}")
            raise

    def get_pending_orders(self, symbol: str = None) -> List[Dict]:
        """Get all pending orders, optionally filtered by symbol."""
        try:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute("SELECT * FROM pending_orders WHERE symbol = ?", (symbol,))
            else:
                cursor.execute("SELECT * FROM pending_orders")
            
            rows = cursor.fetchall()
            orders = []
            
            for row in rows:
                order = dict(row)
                # Parse meta_data JSON
                if order.get('meta_data'):
                    try:
                        order['meta_data'] = json.loads(order['meta_data'])
                    except (json.JSONDecodeError, TypeError):
                        order['meta_data'] = {}
                else:
                    order['meta_data'] = {}
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get pending orders error: {e}")
            return []

    def remove_pending_order(self, ticket: int):
        """Remove pending order from database."""
        try:
            self.conn.cursor().execute(
                "DELETE FROM pending_orders WHERE ticket = ?",
                (ticket,)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Remove pending order error: {e}")

    def update_pending_order_meta(self, ticket: int, meta: dict):
        """Update meta_data for a pending order."""
        try:
            meta_json = json.dumps(meta, default=str)
            self.conn.cursor().execute(
                "UPDATE pending_orders SET meta_data = ? WHERE ticket = ?",
                (meta_json, ticket)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Update pending order meta error: {e}")

    # =========================================================================
    # TRADE HISTORY
    # =========================================================================

    def save_trade_history(self, ticket, symbol, direction, entry_price, exit_price, volume, profit,
                           open_time, close_time, strategy, meta_data=None):
        """Save trade to history."""
        try:
            meta_json = json.dumps(meta_data or {}, default=str)
            
            values = (
                int(ticket),
                str(symbol),
                str(direction),
                float(entry_price),
                float(exit_price),
                float(volume),
                float(profit),
                str(open_time),
                str(close_time),
                str(strategy),
                meta_json
            )
            
            self.conn.cursor().execute(
                "INSERT OR REPLACE INTO trade_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                values
            )
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"[FAIL] Save trade history error: {e}")

    def get_trade_history(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """Get trade history, optionally filtered by symbol and date range."""
        try:
            cursor = self.conn.cursor()
            
            if symbol:
                cursor.execute(
                    "SELECT * FROM trade_history WHERE symbol = ? AND close_time >= datetime('now', ?) ORDER BY close_time DESC",
                    (symbol, f"-{days} days")
                )
            else:
                cursor.execute(
                    "SELECT * FROM trade_history WHERE close_time >= datetime('now', ?) ORDER BY close_time DESC",
                    (f"-{days} days",)
                )
            
            rows = cursor.fetchall()
            trades = []
            
            for row in rows:
                trade = dict(row)
                # Parse meta_data JSON
                if trade.get('meta_data'):
                    try:
                        trade['meta_data'] = json.loads(trade['meta_data'])
                    except (json.JSONDecodeError, TypeError):
                        trade['meta_data'] = {}
                else:
                    trade['meta_data'] = {}
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get trade history error: {e}")
            return []

    def get_strategy_stats_from_history(self, strategy_name: str, unified_regime: str = None, days: int = 30) -> Dict:
        """
        Get aggregated stats for a strategy from trade history.
        Used by Kelly Criterion and Expert Signal Scorer.
        """
        default_stats = {
            'trades': 0,
            'winrate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        try:
            trades = self.get_trade_history(days=days)
            
            # Filter by strategy
            strategy_trades = [t for t in trades if t.get('strategy') == strategy_name]
            
            # Filter by regime if specified
            if unified_regime and unified_regime not in ['UNKNOWN', 'SIDEWAY']:
                filtered_trades = []
                for t in strategy_trades:
                    meta = t.get('meta_data', {})
                    if meta.get('regime') == unified_regime:
                        filtered_trades.append(t)
                
                # Fallback to all if filtered is empty
                if filtered_trades:
                    strategy_trades = filtered_trades
            
            if not strategy_trades:
                return default_stats
            
            trades_count = len(strategy_trades)
            winners = [t for t in strategy_trades if t.get('profit', 0) > 0]
            losers = [t for t in strategy_trades if t.get('profit', 0) <= 0]
            
            winrate = len(winners) / trades_count if trades_count > 0 else 0.0
            avg_win = sum(t.get('profit', 0) for t in winners) / len(winners) if winners else 0.0
            avg_loss = abs(sum(t.get('profit', 0) for t in losers) / len(losers)) if losers else 0.0
            
            gross_profit = sum(t.get('profit', 0) for t in winners) if winners else 0.0
            gross_loss = abs(sum(t.get('profit', 0) for t in losers)) if losers else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            return {
                'trades': trades_count,
                'winrate': winrate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get strategy stats error: {e}")
            return default_stats

    # =========================================================================
    # PARTIAL CLOSE STATE
    # =========================================================================

    def save_partial_close_state(self, ticket: int, tp1_hit: bool, tp2_hit: bool, reversal_close: bool = False):
        """Save partial close state."""
        try:
            self.conn.cursor().execute(
                "INSERT OR REPLACE INTO partial_close_state VALUES (?,?,?,?)",
                (ticket, int(tp1_hit), int(tp2_hit), int(reversal_close))
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Save partial close state error: {e}")

    def get_partial_close_state(self, ticket: int) -> Optional[Dict]:
        """Get partial close state for a ticket."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM partial_close_state WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get partial close state error: {e}")
            return None

    def remove_partial_close_state(self, ticket: int):
        """Remove partial close state when position is fully closed."""
        try:
            self.conn.cursor().execute(
                "DELETE FROM partial_close_state WHERE ticket = ?",
                (ticket,)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"[FAIL] Remove partial close state error: {e}")

    # =========================================================================
    # DAILY PNL
    # =========================================================================

    def get_daily_pnl_percent(self) -> float:
        """
        Calculate daily PnL percentage.
        Used by Expert Signal Scorer for Portfolio Health factor.
        """
        try:
            cursor = self.conn.cursor()
            
            # Get today's trades
            cursor.execute("""
                SELECT SUM(profit) as daily_pnl
                FROM trade_history
                WHERE DATE(close_time) = DATE('now')
            """)
            
            result = cursor.fetchone()
            if result and result['daily_pnl'] is not None:
                daily_pnl = float(result['daily_pnl'])
                
                # Get account balance from MT5
                try:
                    import MetaTrader5 as mt5
                    acc_info = mt5.account_info()
                    if acc_info:
                        balance = acc_info.balance
                        if balance > 0:
                            return (daily_pnl / balance) * 100.0
                except ImportError:
                    # MT5 not available (e.g., in testing)
                    pass
                except Exception as mt5_err:
                    self.logger.debug(f"[DAILY_PNL] MT5 error: {mt5_err}")
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get daily PnL error: {e}")
            return 0.0

    def get_daily_trade_count(self) -> int:
        """Get the number of trades executed today."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as trade_count
                FROM trade_history
                WHERE DATE(close_time) = DATE('now')
            """)
            
            result = cursor.fetchone()
            return result['trade_count'] if result else 0
            
        except Exception as e:
            self.logger.error(f"[FAIL] Get daily trade count error: {e}")
            return 0

    # =========================================================================
    # CLEANUP & MAINTENANCE
    # =========================================================================

    def cleanup_old_history(self, days_to_keep: int = 90):
        """Remove trade history older than specified days to prevent DB bloat."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "DELETE FROM trade_history WHERE close_time < datetime('now', ?)",
                (f"-{days_to_keep} days",)
            )
            deleted = cursor.rowcount
            self.conn.commit()
            
            if deleted > 0:
                self.logger.info(f"[CLEANUP] Removed {deleted} old trade history records (>{days_to_keep} days)")
                
        except Exception as e:
            self.logger.error(f"[FAIL] Cleanup old history error: {e}")

    def vacuum_database(self):
        """Vacuum the database to reclaim space and optimize performance."""
        try:
            self.conn.execute("VACUUM")
            self.logger.info("[OK] Database vacuumed successfully")
        except Exception as e:
            self.logger.error(f"[FAIL] Vacuum database error: {e}")

    def close(self):
        """Close database connection."""
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.logger.info("[OK] StateManager database connection closed")
        except Exception as e:
            self.logger.error(f"[FAIL] Close database connection error: {e}")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
        
    def sync_with_mt5(self, symbol: str = None):
        """
        Synchronize state with MT5 terminal.
        FIXED: Now syncs BOTH active positions AND pending orders.
        """
        import MetaTrader5 as mt5
        
        if symbol is None:
            symbol = self.symbol if hasattr(self, 'symbol') else "XAUUSDm"
        
        self.logger.info(f"[SYNC] Reconciling state with MT5 for {symbol}...")
        
        # =========================================================================
        # 1. Sync Active Positions
        # =========================================================================
        mt5_positions = mt5.positions_get(symbol=symbol) or []
        local_positions = self.get_active_positions(symbol)
        
        mt5_position_tickets = {p.ticket for p in mt5_positions}
        local_position_tickets = {p['ticket'] for p in local_positions}
        
        # Remove positions that exist in state but not in MT5
        stale_positions = local_position_tickets - mt5_position_tickets
        for ticket in stale_positions:
            self.remove_active_position(ticket)
            self.logger.info(f"[SYNC] Removed stale active position: {ticket}")
        
        # =========================================================================
        # 2. Sync Pending Orders (NEW)
        # =========================================================================
        mt5_orders = mt5.orders_get(symbol=symbol) or []
        local_orders = self.get_pending_orders(symbol)
        
        mt5_order_tickets = {o.ticket for o in mt5_orders}
        local_order_tickets = {o['ticket'] for o in local_orders}
        
        # Remove orders that exist in state but not in MT5
        stale_orders = local_order_tickets - mt5_order_tickets
        for ticket in stale_orders:
            self.remove_pending_order(ticket)
            self.logger.info(f"[SYNC] Removed stale pending order: {ticket}")
        
        # Log summary
        self.logger.info(
            f"[SYNC] Complete | "
            f"Active: {len(mt5_positions)} MT5 / {len(local_positions) - len(stale_positions)} local | "
            f"Pending: {len(mt5_orders)} MT5 / {len(local_orders) - len(stale_orders)} local | "
            f"Removed: {len(stale_positions)} positions, {len(stale_orders)} orders"
        )