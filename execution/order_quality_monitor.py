"""
Order Quality Monitor - Execution Quality Tracking.

Tracks and analyzes the quality of order executions to identify
broker performance issues and optimize execution strategy.

Critical for Micro-Account trading where friction costs significantly
impact profitability.

Tracked Metrics:
  1. Slippage: Expected vs Actual entry/exit price
  2. Spread: Spread at time of execution
  3. Latency: Time from signal to fill
  4. Fill Rate: % of orders that get filled
  5. Requote Rate: % of orders that get requoted

Quality Score (0-100):
  Based on:
    - Slippage (lower is better)
    - Spread (tighter is better)
    - Latency (faster is better)
    - Fill success rate

Actions Based on Quality:
  - Quality >= 80: Excellent, continue current strategy
  - Quality 60-79: Good, monitor closely
  - Quality 40-59: Poor, consider adjusting execution method
  - Quality < 40: Critical, review broker or strategy
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import config


class OrderQualityMonitor:
    """
    Monitors and analyzes order execution quality.
    
    Features:
      - Slippage tracking (entry and exit)
      - Spread tracking at execution time
      - Latency measurement
      - Quality scoring (0-100)
      - Historical statistics
      - Comprehensive reporting
      - SQLite persistence
    """

    def __init__(self, symbol: str = "XAUUSDm", db_path: str = None):
        """
        Initialize OrderQualityMonitor.
        
        Args:
            symbol: Trading symbol
            db_path: Path to SQLite database (defaults to config.state_db_path)
        """
        self.symbol = symbol
        self.db_path = db_path or config.state_db_path
        self.logger = logging.getLogger(self.__class__.__name__)

        # Pending executions (ticket -> expected data)
        self._pending_executions: Dict[int, Dict] = {}

        # In-memory cache for recent executions (last 100)
        self._recent_executions = deque(maxlen=100)

        # Thresholds
        self.max_acceptable_slippage_points = 10  # 10 points max
        self.max_acceptable_latency_ms = 500  # 500ms max
        self.max_acceptable_spread_points = config.max_spread_points

        # Create table
        self._create_tables()

        self.logger.info(
            f"[ORDER_QUALITY] Initialized for {symbol} | "
            f"Max Slippage: {self.max_acceptable_slippage_points} pts"
        )

    # =========================================================================
    # TABLE CREATION
    # =========================================================================

    def _create_tables(self):
        """Create execution_quality table if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    direction TEXT,
                    order_type TEXT,
                    expected_price REAL,
                    actual_price REAL,
                    slippage_points REAL,
                    slippage_usd REAL,
                    spread_at_send REAL,
                    spread_at_fill REAL,
                    latency_ms REAL,
                    quality_score REAL,
                    fill_success INTEGER,
                    requote_count INTEGER DEFAULT 0,
                    execution_time TEXT NOT NULL,
                    meta_data TEXT DEFAULT '{}'
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_quality_time
                ON execution_quality(execution_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_quality_strategy
                ON execution_quality(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exec_quality_score
                ON execution_quality(quality_score)
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Table creation error: {e}")

    # =========================================================================
    # EXPECTED EXECUTION REGISTRATION
    # =========================================================================

    def register_expected_execution(self, ticket: int, expected_price: float,
                                      expected_sl: float, expected_tp: float,
                                      direction: str, strategy: str,
                                      order_type: str = "MARKET",
                                      meta: Dict = None):
        """
        Register expected execution parameters before sending order.
        
        Args:
            ticket: Order ticket (0 for new orders, will be updated)
            expected_price: Expected entry price
            expected_sl: Expected stop loss
            expected_tp: Expected take profit
            direction: 'BUY' or 'SELL'
            strategy: Strategy name
            order_type: 'MARKET', 'LIMIT', 'STOP'
            meta: Additional metadata
        """
        # Get current spread
        tick = mt5.symbol_info_tick(self.symbol)
        symbol_info = mt5.symbol_info(self.symbol)

        spread_points = 0
        if tick and symbol_info:
            spread_points = (tick.ask - tick.bid) / symbol_info.point

        self._pending_executions[ticket] = {
            'expected_price': expected_price,
            'expected_sl': expected_sl,
            'expected_tp': expected_tp,
            'direction': direction,
            'strategy': strategy,
            'order_type': order_type,
            'send_time': time.time(),
            'spread_at_send': spread_points,
            'meta': meta or {}
        }

        self.logger.debug(
            f"[ORDER_QUALITY] Registered expected execution | "
            f"Ticket: {ticket} | Price: {expected_price:.2f} | "
            f"Spread: {spread_points:.1f} pts"
        )

    # =========================================================================
    # ACTUAL EXECUTION RECORDING
    # =========================================================================

    def record_actual_execution(self, ticket: int, actual_price: float,
                                  actual_sl: float, actual_tp: float,
                                  fill_success: bool = True,
                                  requote_count: int = 0):
        """
        Record actual execution after order is filled.
        
        Args:
            ticket: Order ticket
            actual_price: Actual fill price
            actual_sl: Actual stop loss
            actual_tp: Actual take profit
            fill_success: Whether order was filled
            requote_count: Number of requotes received
        """
        # Check if we have expected data
        if ticket not in self._pending_executions:
            self.logger.warning(
                f"[ORDER_QUALITY] No expected data for ticket {ticket}"
            )
            return

        expected = self._pending_executions.pop(ticket)

        # Calculate metrics
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            self.logger.error("[ORDER_QUALITY] Cannot get symbol info")
            return

        point = symbol_info.point
        contract_size = getattr(symbol_info, 'trade_contract_size', 100)

        # Slippage calculation
        expected_price = expected['expected_price']
        direction = expected['direction']

        if direction == 'BUY':
            slippage_points = (actual_price - expected_price) / point
        else:
            slippage_points = (expected_price - actual_price) / point

        slippage_usd = abs(actual_price - expected_price)

        # Latency calculation
        latency_ms = (time.time() - expected['send_time']) * 1000

        # Get spread at fill time
        tick = mt5.symbol_info_tick(self.symbol)
        spread_at_fill = 0
        if tick:
            spread_at_fill = (tick.ask - tick.bid) / point

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            slippage_points=abs(slippage_points),
            spread_at_fill=spread_at_fill,
            latency_ms=latency_ms,
            fill_success=fill_success,
            requote_count=requote_count
        )

        # Build execution record
        record = {
            'ticket': ticket,
            'symbol': self.symbol,
            'strategy': expected['strategy'],
            'direction': direction,
            'order_type': expected['order_type'],
            'expected_price': expected_price,
            'actual_price': actual_price,
            'slippage_points': round(slippage_points, 1),
            'slippage_usd': round(slippage_usd, 2),
            'spread_at_send': round(expected['spread_at_send'], 1),
            'spread_at_fill': round(spread_at_fill, 1),
            'latency_ms': round(latency_ms, 1),
            'quality_score': round(quality_score, 1),
            'fill_success': int(fill_success),
            'requote_count': requote_count,
            'execution_time': datetime.now().isoformat(),
            'meta_data': json.dumps(expected.get('meta', {}))
        }

        # Save to database
        self._save_execution(record)

        # Add to recent cache
        self._recent_executions.append(record)

        # Log result
        quality_level = self._get_quality_level(quality_score)
        self.logger.info(
            f"[ORDER_QUALITY] Ticket {ticket} ({expected['strategy']}) | "
            f"Quality: {quality_score:.0f} ({quality_level}) | "
            f"Slippage: {slippage_points:.1f} pts | "
            f"Latency: {latency_ms:.0f}ms | "
            f"Spread: {spread_at_fill:.1f} pts"
        )

        # Warn if quality is poor
        if quality_score < 60:
            self.logger.warning(
                f"[ORDER_QUALITY] Poor execution quality for ticket {ticket}: "
                f"{quality_score:.0f}/100"
            )

    # =========================================================================
    # QUALITY SCORE CALCULATION
    # =========================================================================

    def _calculate_quality_score(self, slippage_points: float,
                                    spread_at_fill: float,
                                    latency_ms: float,
                                    fill_success: bool,
                                    requote_count: int) -> float:
        """
        Calculate execution quality score (0-100).
        
        Factors:
          - Slippage: 40% weight
          - Spread: 25% weight
          - Latency: 20% weight
          - Fill success: 15% weight
        """
        # If fill failed, score is 0
        if not fill_success:
            return 0.0

        # Slippage score (0-40 points)
        # 0 slippage = 40, 10+ slippage = 0
        slippage_score = max(0, 40 - (abs(slippage_points) / self.max_acceptable_slippage_points * 40))

        # Spread score (0-25 points)
        # Spread <= max = 25, Spread > 2x max = 0
        if spread_at_fill <= self.max_acceptable_spread_points:
            spread_score = 25
        elif spread_at_fill <= self.max_acceptable_spread_points * 2:
            ratio = (spread_at_fill - self.max_acceptable_spread_points) / self.max_acceptable_spread_points
            spread_score = 25 * (1 - ratio)
        else:
            spread_score = 0

        # Latency score (0-20 points)
        # Latency <= 200ms = 20, Latency >= 1000ms = 0
        if latency_ms <= 200:
            latency_score = 20
        elif latency_ms <= 1000:
            ratio = (latency_ms - 200) / 800
            latency_score = 20 * (1 - ratio)
        else:
            latency_score = 0

        # Fill success score (0-15 points)
        fill_score = 15 if fill_success else 0

        # Requote penalty
        requote_penalty = min(10, requote_count * 3)

        # Total score
        total_score = slippage_score + spread_score + latency_score + fill_score - requote_penalty

        return max(0, min(100, total_score))

    def _get_quality_level(self, score: float) -> str:
        """Map quality score to level."""
        if score >= 80:
            return 'EXCELLENT'
        elif score >= 60:
            return 'GOOD'
        elif score >= 40:
            return 'POOR'
        else:
            return 'CRITICAL'

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    def _save_execution(self, record: Dict):
        """Save execution record to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO execution_quality
                (ticket, symbol, strategy, direction, order_type,
                 expected_price, actual_price, slippage_points, slippage_usd,
                 spread_at_send, spread_at_fill, latency_ms, quality_score,
                 fill_success, requote_count, execution_time, meta_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['ticket'],
                record['symbol'],
                record['strategy'],
                record['direction'],
                record['order_type'],
                record['expected_price'],
                record['actual_price'],
                record['slippage_points'],
                record['slippage_usd'],
                record['spread_at_send'],
                record['spread_at_fill'],
                record['latency_ms'],
                record['quality_score'],
                record['fill_success'],
                record['requote_count'],
                record['execution_time'],
                record['meta_data']
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Save execution error: {e}")

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_execution_stats(self, days: int = 7) -> Dict:
        """
        Get execution quality statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total_executions,
                    AVG(slippage_points) as avg_slippage,
                    AVG(spread_at_fill) as avg_spread,
                    AVG(latency_ms) as avg_latency,
                    AVG(quality_score) as avg_quality,
                    SUM(fill_success) as successful_fills,
                    SUM(requote_count) as total_requotes
                FROM execution_quality
                WHERE execution_time >= datetime('now', ?)
            """, (f"-{days} days",))

            row = cursor.fetchone()
            conn.close()

            if not row or row[0] == 0:
                return {
                    'total_executions': 0,
                    'avg_slippage': 0,
                    'avg_spread': 0,
                    'avg_latency': 0,
                    'avg_quality': 0,
                    'fill_rate': 0,
                    'requote_rate': 0
                }

            total = row[0]
            avg_slippage = row[1] or 0
            avg_spread = row[2] or 0
            avg_latency = row[3] or 0
            avg_quality = row[4] or 0
            successful = row[5] or 0
            requotes = row[6] or 0

            fill_rate = (successful / total * 100) if total > 0 else 0
            requote_rate = (requotes / total) if total > 0 else 0

            return {
                'total_executions': total,
                'avg_slippage': round(avg_slippage, 2),
                'avg_spread': round(avg_spread, 1),
                'avg_latency': round(avg_latency, 1),
                'avg_quality': round(avg_quality, 1),
                'fill_rate': round(fill_rate, 1),
                'requote_rate': round(requote_rate, 2),
                'quality_level': self._get_quality_level(avg_quality),
                'analysis_period_days': days
            }

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Stats error: {e}")
            return {
                'total_executions': 0,
                'error': str(e)
            }

    def get_strategy_execution_stats(self, days: int = 7) -> Dict:
        """
        Get execution quality statistics per strategy.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with per-strategy statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    strategy,
                    COUNT(*) as executions,
                    AVG(slippage_points) as avg_slippage,
                    AVG(quality_score) as avg_quality,
                    SUM(fill_success) as successful
                FROM execution_quality
                WHERE execution_time >= datetime('now', ?)
                GROUP BY strategy
                ORDER BY avg_quality DESC
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            conn.close()

            stats = {}
            for row in rows:
                strategy = row[0]
                executions = row[1]
                avg_slippage = row[2] or 0
                avg_quality = row[3] or 0
                successful = row[4] or 0

                fill_rate = (successful / executions * 100) if executions > 0 else 0

                stats[strategy] = {
                    'executions': executions,
                    'avg_slippage': round(avg_slippage, 2),
                    'avg_quality': round(avg_quality, 1),
                    'fill_rate': round(fill_rate, 1),
                    'quality_level': self._get_quality_level(avg_quality)
                }

            return stats

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Strategy stats error: {e}")
            return {}

    # =========================================================================
    # SLIPPAGE ANALYSIS
    # =========================================================================

    def _analyze_slippage(self, days: int = 7) -> Dict:
        """
        Analyze slippage patterns.
        
        Returns:
            Dict with slippage analysis
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    direction,
                    COUNT(*) as count,
                    AVG(slippage_points) as avg_slippage,
                    MIN(slippage_points) as min_slippage,
                    MAX(slippage_points) as max_slippage,
                    AVG(slippage_usd) as avg_slippage_usd
                FROM execution_quality
                WHERE execution_time >= datetime('now', ?)
                GROUP BY direction
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            conn.close()

            analysis = {}
            for row in rows:
                direction = row[0]
                analysis[direction] = {
                    'count': row[1],
                    'avg_slippage_points': round(row[2], 2),
                    'min_slippage_points': round(row[3], 2),
                    'max_slippage_points': round(row[4], 2),
                    'avg_slippage_usd': round(row[5], 2)
                }

            return analysis

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Slippage analysis error: {e}")
            return {}

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    def generate_quality_report(self, days: int = 7) -> str:
        """
        Generate comprehensive execution quality report.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Formatted report string
        """
        stats = self.get_execution_stats(days)
        strategy_stats = self.get_strategy_execution_stats(days)
        slippage_analysis = self._analyze_slippage(days)

        report_lines = [
            "=" * 70,
            f"EXECUTION QUALITY REPORT (Last {days} days)",
            "=" * 70,
            "",
            "OVERALL STATISTICS:",
            "-" * 40,
            f"  Total Executions: {stats.get('total_executions', 0)}",
            f"  Average Quality Score: {stats.get('avg_quality', 0):.1f}/100 ({stats.get('quality_level', 'N/A')})",
            f"  Average Slippage: {stats.get('avg_slippage', 0):.2f} points",
            f"  Average Spread at Fill: {stats.get('avg_spread', 0):.1f} points",
            f"  Average Latency: {stats.get('avg_latency', 0):.0f} ms",
            f"  Fill Rate: {stats.get('fill_rate', 0):.1f}%",
            f"  Requote Rate: {stats.get('requote_rate', 0):.2f} per execution",
            "",
            "SLIPPAGE BY DIRECTION:",
            "-" * 40
        ]

        for direction, analysis in slippage_analysis.items():
            report_lines.append(
                f"  {direction}: {analysis['count']} trades | "
                f"Avg: {analysis['avg_slippage_points']:.2f} pts (${analysis['avg_slippage_usd']:.2f}) | "
                f"Range: {analysis['min_slippage_points']:.2f} to {analysis['max_slippage_points']:.2f} pts"
            )

        report_lines.extend([
            "",
            "QUALITY BY STRATEGY:",
            "-" * 40
        ])

        for strategy, strat_stats in sorted(strategy_stats.items(),
                                            key=lambda x: x[1]['avg_quality'],
                                            reverse=True):
            report_lines.append(
                f"  {strategy}: {strat_stats['executions']} trades | "
                f"Quality: {strat_stats['avg_quality']:.1f} ({strat_stats['quality_level']}) | "
                f"Slippage: {strat_stats['avg_slippage']:.2f} pts | "
                f"Fill Rate: {strat_stats['fill_rate']:.1f}%"
            )

        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40
        ])

        avg_quality = stats.get('avg_quality', 0)
        if avg_quality >= 80:
            report_lines.append("  ✓ Execution quality is EXCELLENT - continue current strategy")
        elif avg_quality >= 60:
            report_lines.append("  ✓ Execution quality is GOOD - monitor closely")
        elif avg_quality >= 40:
            report_lines.append("  ⚠ Execution quality is POOR - consider:")
            report_lines.append("    - Using limit orders instead of market orders")
            report_lines.append("    - Trading during high-liquidity sessions")
            report_lines.append("    - Reviewing broker execution quality")
        else:
            report_lines.append("  ✗ Execution quality is CRITICAL - immediate action needed:")
            report_lines.append("    - Switch to limit orders")
            report_lines.append("    - Avoid trading during low-liquidity periods")
            report_lines.append("    - Consider changing broker")
            report_lines.append("    - Increase profit targets to cover friction")

        report_lines.extend([
            "",
            "=" * 70
        ])

        return "\n".join(report_lines)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def cleanup_old_records(self, days_to_keep: int = 30):
        """
        Remove old execution records.
        
        Args:
            days_to_keep: Number of days to keep
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM execution_quality WHERE execution_time < datetime('now', ?)",
                (f"-{days_to_keep} days",)
            )

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted > 0:
                self.logger.info(
                    f"[ORDER_QUALITY] Cleaned up {deleted} old execution records"
                )

        except Exception as e:
            self.logger.error(f"[ORDER_QUALITY] Cleanup error: {e}")

    def get_recent_executions(self, limit: int = 20) -> List[Dict]:
        """
        Get recent execution records from cache.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        return list(self._recent_executions)[-limit:]

    def format_quality_log(self, ticket: int, quality_score: float,
                            slippage_points: float, latency_ms: float) -> str:
        """
        Format a concise log string for execution quality.
        
        Args:
            ticket: Order ticket
            quality_score: Quality score (0-100)
            slippage_points: Slippage in points
            latency_ms: Latency in milliseconds
            
        Returns:
            Formatted log string
        """
        level = self._get_quality_level(quality_score)

        return (
            f"[ORDER_QUALITY] Ticket {ticket} | "
            f"Quality: {quality_score:.0f} ({level}) | "
            f"Slippage: {slippage_points:.1f} pts | "
            f"Latency: {latency_ms:.0f}ms"
        )