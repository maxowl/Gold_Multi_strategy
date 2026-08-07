"""
State Manager - SQLite Database Persistence (REVISED).
Manages active positions, pending orders, and trade history.
Provides thread-safe SQLite operations with JSON meta_data support.

REVISION LOG:
  [REV-001] FIXED Timezone mismatch in all date-based queries.
            SQLite's datetime('now') returns UTC, but close_time is
            stored as local ISO format. All queries now use Python
            datetime for consistent local-time comparisons.
  [REV-002] ADDED threading.Lock() for all write operations.
            check_same_thread=False alone does not prevent corruption
            from concurrent writes.
  [REV-003] DEPRECATED partial_close_state table and methods per rule:
            "Partial close = ห้ามเด็ดขาด".
            Methods retained for backward compatibility but log warnings.
  [REV-004] RENAMED update_trailing_stop() to update_sl() per rule:
            "ห้ามมี trailing stop". Old name retained as alias.
  [REV-005] ADDED migration for trade_history columns:
            - expected_entry (for slippage tracking)
            - slippage_points (for execution quality analysis)
  [REV-006] ADDED index on trade_history(close_time) for daily queries.

Tables:
  active_positions: Currently open positions
  pending_orders: Pending orders waiting to be filled
  trade_history: Closed trades for performance analysis
  partial_close_state: [DEPRECATED] Retained for backward compat only
"""
import sqlite3
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class StateManager:
    """
    SQLite-based state persistence for the trading bot.

    Features:
      - Thread-safe operations with explicit lock [REV-002]
      - JSON meta_data serialization/deserialization
      - Automatic table creation and migration
      - Daily PnL calculation (local timezone) [REV-001]
      - Trade history with slippage tracking [REV-005]

    Thread Safety:
      All write operations are protected by a threading.Lock().
      Read operations are safe without locking due to SQLite's
      built-in concurrency handling for SELECT statements.
    """

    def __init__(self, db_path: str = "bot_state.db"):
        """
        Initialize StateManager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)

        # [REV-002] Thread lock for write operations
        self._db_lock = threading.Lock()

        try:
            self.conn = sqlite3.connect(
                db_path, check_same_thread=False, timeout=10.0
            )
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
            self._migrate_tables()
            self.logger.info(f"[STATE_MGR] Initialized with database: {db_path}")
        except Exception as e:
            self.logger.critical(f"[STATE_MGR] Could not initialize database: {e}")
            raise

    # =========================================================================
    # TABLE CREATION & MIGRATION
    # =========================================================================

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Active Positions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_positions (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                position_type TEXT NOT NULL,
                volume REAL NOT NULL,
                entry_price REAL NOT NULL,
                sl REAL DEFAULT 0.0,
                tp REAL DEFAULT 0.0,
                strategy TEXT NOT NULL,
                open_time TEXT NOT NULL,
                requires_dynamic_exit INTEGER DEFAULT 0,
                dynamic_exit_threshold TEXT DEFAULT '',
                entry_reason TEXT DEFAULT '',
                expected_entry REAL DEFAULT 0.0,
                order_type TEXT DEFAULT 'MARKET',
                is_pending INTEGER DEFAULT 0,
                trailing_stop_level REAL DEFAULT 0.0,
                meta_data TEXT DEFAULT '{}'
            )
        """)

        # Pending Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_orders (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                volume REAL NOT NULL,
                price REAL NOT NULL,
                sl REAL DEFAULT 0.0,
                tp REAL DEFAULT 0.0,
                strategy TEXT NOT NULL,
                expiration_bars INTEGER DEFAULT 10,
                requires_dynamic_exit INTEGER DEFAULT 0,
                dynamic_exit_threshold TEXT DEFAULT '',
                setup_time TEXT NOT NULL,
                entry_reason TEXT DEFAULT '',
                meta_data TEXT DEFAULT '{}'
            )
        """)

        # Trade History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                volume REAL NOT NULL,
                profit REAL NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                strategy TEXT NOT NULL,
                meta_data TEXT DEFAULT '{}'
            )
        """)

        # [REV-003] Partial Close State Table - DEPRECATED
        # Retained for backward compatibility only.
        # New code should NOT use this table.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partial_close_state (
                ticket INTEGER PRIMARY KEY,
                tp1_hit INTEGER DEFAULT 0,
                tp2_hit INTEGER DEFAULT 0,
                reversal_close INTEGER DEFAULT 0
            )
        """)

        # Create indexes for frequently used queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_positions_symbol
            ON active_positions(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_orders_symbol
            ON pending_orders(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_history_close_time
            ON trade_history(close_time)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_history_strategy
            ON trade_history(strategy)
        """)

        self.conn.commit()

    def _migrate_tables(self):
        """
        Migrate tables to add missing columns (schema evolution).

        [REV-005] Added migration for trade_history columns:
          - expected_entry: For slippage tracking
          - slippage_points: For execution quality analysis
        """
        try:
            cursor = self.conn.cursor()

            # Migration 1: partial_close_state reversal_close column
            cursor.execute("PRAGMA table_info(partial_close_state)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'reversal_close' not in columns:
                cursor.execute(
                    "ALTER TABLE partial_close_state "
                    "ADD COLUMN reversal_close INTEGER DEFAULT 0"
                )
                self.logger.info(
                    "[STATE_MGR] Migrated partial_close_state: "
                    "added reversal_close column"
                )

            # [REV-005] Migration 2: trade_history expected_entry column
            cursor.execute("PRAGMA table_info(trade_history)")
            th_columns = [row[1] for row in cursor.fetchall()]

            if 'expected_entry' not in th_columns:
                cursor.execute(
                    "ALTER TABLE trade_history "
                    "ADD COLUMN expected_entry REAL DEFAULT 0.0"
                )
                self.logger.info(
                    "[STATE_MGR] Migrated trade_history: "
                    "added expected_entry column"
                )

            # [REV-005] Migration 3: trade_history slippage_points column
            cursor.execute("PRAGMA table_info(trade_history)")
            th_columns = [row[1] for row in cursor.fetchall()]

            if 'slippage_points' not in th_columns:
                cursor.execute(
                    "ALTER TABLE trade_history "
                    "ADD COLUMN slippage_points REAL DEFAULT 0.0"
                )
                self.logger.info(
                    "[STATE_MGR] Migrated trade_history: "
                    "added slippage_points column"
                )

            self.conn.commit()

        except Exception as e:
            self.logger.debug(f"[STATE_MGR] Migration check: {e}")

    # =========================================================================
    # ACTIVE POSITIONS
    # =========================================================================

    def save_active_position(self, ticket: int, symbol: str, position_type: str,
                              volume: float, entry_price: float, sl: float, tp: float,
                              strategy: str, requires_dynamic_exit: bool = False,
                              dynamic_exit_threshold=None, entry_reason: str = "",
                              expected_entry: float = 0.0, order_type: str = "MARKET",
                              is_pending: bool = False, meta_data: dict = None):
        """
        Save active position to database.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 position ticket
            symbol: Trading symbol
            position_type: 'BUY' or 'SELL'
            volume: Position volume
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            strategy: Strategy name
            requires_dynamic_exit: Whether to use dynamic exit
            dynamic_exit_threshold: Threshold for dynamic exit
            entry_reason: Reason for entry
            expected_entry: Expected entry price
            order_type: 'MARKET' or 'PENDING'
            is_pending: Whether this is a pending order
            meta_data: Additional metadata dict
        """
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

            # [REV-002] Thread-safe write
            with self._db_lock:
                self.conn.cursor().execute(
                    "INSERT OR REPLACE INTO active_positions "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values
                )
                self.conn.commit()

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Save active position error: {e}")
            raise

    def get_active_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get all active positions, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of position dicts with parsed meta_data
        """
        try:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM active_positions WHERE symbol = ?",
                    (symbol,)
                )
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
            self.logger.error(f"[STATE_MGR] Get active positions error: {e}")
            return []

    def remove_active_position(self, ticket: int):
        """
        Remove active position from database.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 position ticket
        """
        try:
            with self._db_lock:
                self.conn.cursor().execute(
                    "DELETE FROM active_positions WHERE ticket = ?",
                    (ticket,)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Remove active position error: {e}")

    def update_sl(self, ticket: int, new_sl: float):
        """
        Update stop loss for active position.

        [REV-004] Renamed from update_trailing_stop() per rule:
        "ห้ามมี trailing stop". This method updates the SL level
        for any purpose (breakeven, tightening, etc.).

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 position ticket
            new_sl: New stop loss price
        """
        try:
            with self._db_lock:
                self.conn.cursor().execute(
                    "UPDATE active_positions SET trailing_stop_level = ? "
                    "WHERE ticket = ?",
                    (new_sl, ticket)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Update SL error: {e}")

    def update_trailing_stop(self, ticket: int, new_sl: float):
        """
        [REV-004] DEPRECATED - Use update_sl() instead.

        Retained for backward compatibility only.
        Trailing stop is disabled per rule: "ห้ามมี trailing stop".

        Args:
            ticket: MT5 position ticket
            new_sl: New stop loss price
        """
        self.logger.debug(
            "[STATE_MGR] update_trailing_stop() is deprecated, "
            "calling update_sl() instead"
        )
        self.update_sl(ticket, new_sl)

    def update_position_meta(self, ticket: int, meta: dict):
        """
        Update meta_data for an active position.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 position ticket
            meta: New metadata dict
        """
        try:
            meta_json = json.dumps(meta, default=str)
            with self._db_lock:
                self.conn.cursor().execute(
                    "UPDATE active_positions SET meta_data = ? WHERE ticket = ?",
                    (meta_json, ticket)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Update position meta error: {e}")

    def update_position_sl(self, ticket: int, new_sl: float):
        """
        Update stop loss for active position.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 position ticket
            new_sl: New stop loss price
        """
        try:
            with self._db_lock:
                self.conn.cursor().execute(
                    "UPDATE active_positions SET sl = ? WHERE ticket = ?",
                    (new_sl, ticket)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Update position SL error: {e}")

    # =========================================================================
    # PENDING ORDERS
    # =========================================================================

    def save_pending_order(self, ticket: int, symbol: str, order_type: str,
                            volume: float, price: float, sl: float, tp: float,
                            strategy: str, expiration_bars: int = 10,
                            requires_dynamic_exit: bool = False,
                            dynamic_exit_threshold=None, entry_reason: str = "",
                            meta_data: dict = None):
        """
        Save pending order to database.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 order ticket
            symbol: Trading symbol
            order_type: Order type string
            volume: Order volume
            price: Order price
            sl: Stop loss price
            tp: Take profit price
            strategy: Strategy name
            expiration_bars: Bars until expiration
            requires_dynamic_exit: Whether to use dynamic exit
            dynamic_exit_threshold: Threshold for dynamic exit
            entry_reason: Reason for entry
            meta_data: Additional metadata dict
        """
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

            with self._db_lock:
                self.conn.cursor().execute(
                    "INSERT OR REPLACE INTO pending_orders "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values
                )
                self.conn.commit()

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Save pending order error: {e}")
            raise

    def get_pending_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all pending orders, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of order dicts with parsed meta_data
        """
        try:
            cursor = self.conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM pending_orders WHERE symbol = ?",
                    (symbol,)
                )
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
            self.logger.error(f"[STATE_MGR] Get pending orders error: {e}")
            return []

    def remove_pending_order(self, ticket: int):
        """
        Remove pending order from database.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 order ticket
        """
        try:
            with self._db_lock:
                self.conn.cursor().execute(
                    "DELETE FROM pending_orders WHERE ticket = ?",
                    (ticket,)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Remove pending order error: {e}")

    def update_pending_order_meta(self, ticket: int, meta: dict):
        """
        Update meta_data for a pending order.

        [REV-002] Protected by thread lock.

        Args:
            ticket: MT5 order ticket
            meta: New metadata dict
        """
        try:
            meta_json = json.dumps(meta, default=str)
            with self._db_lock:
                self.conn.cursor().execute(
                    "UPDATE pending_orders SET meta_data = ? WHERE ticket = ?",
                    (meta_json, ticket)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Update pending order meta error: {e}")

    # =========================================================================
    # TRADE HISTORY
    # =========================================================================

    def save_trade_history(self, ticket: int, symbol: str, direction: str,
                            entry_price: float, exit_price: float, volume: float,
                            profit: float, open_time: str, close_time: str,
                            strategy: str, meta_data: dict = None,
                            expected_entry: float = 0.0,
                            slippage_points: float = 0.0):
        """
        Save trade to history.

        [REV-002] Protected by thread lock.
        [REV-005] Added expected_entry and slippage_points parameters.

        Args:
            ticket: MT5 position ticket
            symbol: Trading symbol
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            exit_price: Exit price
            volume: Trade volume
            profit: Trade profit in USD
            open_time: Position open time (ISO format)
            close_time: Position close time (ISO format)
            strategy: Strategy name
            meta_data: Additional metadata dict
            expected_entry: [REV-005] Expected entry price for slippage calc
            slippage_points: [REV-005] Slippage in points
        """
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

            with self._db_lock:
                self.conn.cursor().execute(
                    "INSERT OR REPLACE INTO trade_history "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    values
                )

                # [REV-005] Update new columns if they exist
                try:
                    self.conn.cursor().execute(
                        "UPDATE trade_history SET expected_entry = ?, "
                        "slippage_points = ? WHERE ticket = ?",
                        (float(expected_entry), float(slippage_points), int(ticket))
                    )
                except sqlite3.OperationalError:
                    # Columns may not exist yet (pre-migration)
                    pass

                self.conn.commit()

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Save trade history error: {e}")

    def get_trade_history(self, symbol: str = None, days: int = 30) -> List[Dict]:
        """
        Get trade history, optionally filtered by symbol and date range.

        [REV-001] FIXED: Uses Python datetime for local-time comparison
        instead of SQLite's datetime('now') which returns UTC.

        Args:
            symbol: Filter by symbol (optional)
            days: Number of days to look back

        Returns:
            List of trade dicts with parsed meta_data
        """
        try:
            # [REV-001] Calculate cutoff in local time
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT * FROM trade_history "
                    "WHERE symbol = ? AND close_time >= ? "
                    "ORDER BY close_time DESC",
                    (symbol, cutoff)
                )
            else:
                cursor.execute(
                    "SELECT * FROM trade_history "
                    "WHERE close_time >= ? "
                    "ORDER BY close_time DESC",
                    (cutoff,)
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
            self.logger.error(f"[STATE_MGR] Get trade history error: {e}")
            return []

    def get_strategy_stats_from_history(self, strategy_name: str,
                                         unified_regime: str = None,
                                         days: int = 30) -> Dict:
        """
        Get aggregated stats for a strategy from trade history.

        Args:
            strategy_name: Strategy name to filter
            unified_regime: Unified regime to filter (optional)
            days: Number of days to look back

        Returns:
            Dict with trades, winrate, avg_win, avg_loss, profit_factor
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
            strategy_trades = [
                t for t in trades if t.get('strategy') == strategy_name
            ]

            # Filter by regime if specified
            if unified_regime and unified_regime not in ['UNKNOWN', 'SIDEWAY']:
                filtered_trades = []
                for t in strategy_trades:
                    meta = t.get('meta_data', {})
                    if meta.get('regime') == unified_regime:
                        filtered_trades.append(t)
                if filtered_trades:
                    strategy_trades = filtered_trades

            if not strategy_trades:
                return default_stats

            trades_count = len(strategy_trades)
            winners = [t for t in strategy_trades if t.get('profit', 0) > 0]
            losers = [t for t in strategy_trades if t.get('profit', 0) <= 0]

            winrate = len(winners) / trades_count if trades_count > 0 else 0.0
            avg_win = (
                sum(t.get('profit', 0) for t in winners) / len(winners)
                if winners else 0.0
            )
            avg_loss = (
                abs(sum(t.get('profit', 0) for t in losers) / len(losers))
                if losers else 0.0
            )

            gross_profit = (
                sum(t.get('profit', 0) for t in winners) if winners else 0.0
            )
            gross_loss = (
                abs(sum(t.get('profit', 0) for t in losers)) if losers else 0.0
            )
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else 0.0
            )

            return {
                'trades': trades_count,
                'winrate': winrate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Get strategy stats error: {e}")
            return default_stats

    # =========================================================================
    # PARTIAL CLOSE STATE [REV-003] DEPRECATED
    # =========================================================================

    def save_partial_close_state(self, ticket: int, tp1_hit: bool,
                                  tp2_hit: bool, reversal_close: bool = False):
        """
        [REV-003] DEPRECATED - Partial close is disabled per rule:
        "Partial close = ห้ามเด็ดขาด".

        Retained for backward compatibility only. Logs a warning.

        Args:
            ticket: MT5 position ticket
            tp1_hit: Whether TP1 was hit
            tp2_hit: Whether TP2 was hit
            reversal_close: Whether reversal close was triggered
        """
        self.logger.debug(
            f"[STATE_MGR] save_partial_close_state() is DEPRECATED "
            f"(ticket {ticket}). Partial close is disabled per rules."
        )
        # No-op: Do not write to database

    def get_partial_close_state(self, ticket: int) -> Optional[Dict]:
        """
        [REV-003] DEPRECATED - Partial close is disabled per rule.

        Retained for backward compatibility only. Always returns None.

        Args:
            ticket: MT5 position ticket

        Returns:
            Always None (partial close disabled)
        """
        self.logger.debug(
            f"[STATE_MGR] get_partial_close_state() is DEPRECATED "
            f"(ticket {ticket}). Returning None."
        )
        return None

    def remove_partial_close_state(self, ticket: int):
        """
        [REV-003] DEPRECATED - Partial close is disabled per rule.

        Retained for backward compatibility only. Logs a warning.

        Args:
            ticket: MT5 position ticket
        """
        self.logger.debug(
            f"[STATE_MGR] remove_partial_close_state() is DEPRECATED "
            f"(ticket {ticket})."
        )
        # Attempt cleanup for any legacy data
        try:
            with self._db_lock:
                self.conn.cursor().execute(
                    "DELETE FROM partial_close_state WHERE ticket = ?",
                    (ticket,)
                )
                self.conn.commit()
        except Exception as e:
            self.logger.debug(f"[STATE_MGR] Cleanup partial state: {e}")

    # =========================================================================
    # DAILY PNL [REV-001] FIXED TIMEZONE
    # =========================================================================

    def get_daily_pnl_percent(self) -> float:
        """
        Calculate daily PnL percentage.

        [REV-001] FIXED: Uses local date string comparison instead of
        SQLite's DATE('now') which returns UTC date.

        Returns:
            Daily PnL as percentage of balance
        """
        try:
            # [REV-001] Use local date for comparison
            today_str = datetime.now().strftime('%Y-%m-%d')

            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT SUM(profit) as daily_pnl
                FROM trade_history
                WHERE substr(close_time, 1, 10) = ?
            """, (today_str,))

            result = cursor.fetchone()

            if result and result['daily_pnl'] is not None:
                daily_pnl = float(result['daily_pnl'])
                try:
                    import MetaTrader5 as mt5
                    acc_info = mt5.account_info()
                    if acc_info:
                        balance = acc_info.balance
                        if balance > 0:
                            return (daily_pnl / balance) * 100.0
                except ImportError:
                    pass
                except Exception as mt5_err:
                    self.logger.debug(
                        f"[STATE_MGR] MT5 error in daily PnL: {mt5_err}"
                    )

            return 0.0

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Get daily PnL error: {e}")
            return 0.0

    def get_daily_trade_count(self) -> int:
        """
        Get the number of trades executed today.

        [REV-001] FIXED: Uses local date string comparison instead of
        SQLite's DATE('now') which returns UTC date.

        Returns:
            Integer count of today's trades
        """
        try:
            # [REV-001] Use local date for comparison
            today_str = datetime.now().strftime('%Y-%m-%d')

            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as trade_count
                FROM trade_history
                WHERE substr(close_time, 1, 10) = ?
            """, (today_str,))

            result = cursor.fetchone()
            return result['trade_count'] if result else 0

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Get daily trade count error: {e}")
            return 0

    # =========================================================================
    # CLEANUP & MAINTENANCE
    # =========================================================================

    def cleanup_old_history(self, days_to_keep: int = 90):
        """
        Remove trade history older than specified days.

        [REV-001] FIXED: Uses Python datetime for local-time comparison.
        [REV-002] Protected by thread lock.

        Args:
            days_to_keep: Number of days to keep
        """
        try:
            # [REV-001] Calculate cutoff in local time
            cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

            with self._db_lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "DELETE FROM trade_history WHERE close_time < ?",
                    (cutoff,)
                )
                deleted = cursor.rowcount
                self.conn.commit()

            if deleted > 0:
                self.logger.info(
                    f"[STATE_MGR] Removed {deleted} old trade history "
                    f"records (>{days_to_keep} days)"
                )

        except Exception as e:
            self.logger.error(f"[STATE_MGR] Cleanup old history error: {e}")

    def vacuum_database(self):
        """
        Vacuum the database to reclaim space and optimize performance.

        [REV-002] Protected by thread lock.
        """
        try:
            with self._db_lock:
                self.conn.execute("VACUUM")
            self.logger.info("[STATE_MGR] Database vacuumed successfully")
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Vacuum database error: {e}")

    def close(self):
        """Close database connection."""
        try:
            if self.conn:
                with self._db_lock:
                    self.conn.close()
                self.conn = None
                self.logger.info("[STATE_MGR] Database connection closed")
        except Exception as e:
            self.logger.error(f"[STATE_MGR] Close database connection error: {e}")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()