"""
Equity Circuit Breaker - System-Level Performance Protection.

Monitors equity curve performance and automatically pauses trading
when performance degrades below critical thresholds.

Acts as the final safety net to prevent catastrophic losses during
periods of poor strategy performance.

Circuit Breaker Triggers (configurable):
  1. Win Rate: < 40% in last 20 trades → Pause 24 hours
  2. Profit Factor: < 0.8 in last 20 trades → Pause 12 hours
  3. Consecutive Losses: >= 5 losses → Pause 4 hours
  4. Daily Loss: > 2% of equity → Stop for the day

Features:
  - Multi-condition circuit breaking
  - Automatic pause with duration
  - Auto-resume after pause expires
  - Manual override capability
  - Grace period after resume
  - Pause history tracking
  - Comprehensive statistics
  - SQLite trade history integration
"""
import MetaTrader5 as mt5
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import config


class EquityCircuitBreaker:
    """
    Monitors performance and halts trading when conditions deteriorate.
    
    Features:
      - Win rate monitoring
      - Profit factor monitoring
      - Consecutive loss tracking
      - Daily loss limit
      - Automatic pause/resume
      - Manual override
      - Grace period after resume
      - Pause history tracking
    """

    # Trigger types
    TRIGGER_WIN_RATE = 'WIN_RATE'
    TRIGGER_PROFIT_FACTOR = 'PROFIT_FACTOR'
    TRIGGER_CONSECUTIVE_LOSSES = 'CONSECUTIVE_LOSSES'
    TRIGGER_DAILY_LOSS = 'DAILY_LOSS'
    TRIGGER_MANUAL = 'MANUAL'

    # Status
    STATUS_ACTIVE = 'ACTIVE'
    STATUS_PAUSED = 'PAUSED'
    STATUS_GRACE_PERIOD = 'GRACE_PERIOD'

    def __init__(self, db_path: str = None):
        """
        Initialize EquityCircuitBreaker.
        
        Args:
            db_path: Path to SQLite database (defaults to config.state_db_path)
        """
        self.db_path = db_path or config.state_db_path
        self.logger = logging.getLogger(self.__class__.__name__)

        # Thresholds from config
        self.min_win_rate = config.circuit_breaker_min_winrate  # 0.40
        self.min_profit_factor = config.circuit_breaker_min_pf  # 0.80
        self.max_consecutive_losses = config.circuit_breaker_max_consec_loss  # 5
        self.daily_loss_limit_pct = config.circuit_breaker_daily_loss_limit  # 2.0
        self.rolling_window = config.circuit_breaker_rolling_window  # 20

        # Pause durations from config (hours)
        self.winrate_pause_hours = config.circuit_breaker_winrate_pause_hours  # 24.0
        self.pf_pause_hours = config.circuit_breaker_pf_pause_hours  # 12.0
        self.consec_pause_hours = config.circuit_breaker_consec_pause_hours  # 4.0

        # State
        self.is_paused = False
        self.pause_until = None
        self.pause_reason = ""
        self.pause_trigger = None
        self.status = self.STATUS_ACTIVE

        # Consecutive loss tracking (in-memory, also tracked in DB)
        self.consecutive_losses = 0

        # Daily tracking
        self.daily_start_equity = 0.0
        self.daily_start_date = None

        # Grace period after resume (1 hour)
        self.grace_period_hours = 1.0
        self.grace_until = None

        # Pause history (last 20 pauses)
        self._pause_history = deque(maxlen=20)

        # Trigger statistics
        self._trigger_stats: Dict[str, int] = {
            self.TRIGGER_WIN_RATE: 0,
            self.TRIGGER_PROFIT_FACTOR: 0,
            self.TRIGGER_CONSECUTIVE_LOSSES: 0,
            self.TRIGGER_DAILY_LOSS: 0,
            self.TRIGGER_MANUAL: 0
        }

        self.logger.info(
            f"[CIRCUIT_BREAKER] Initialized | "
            f"Win Rate < {self.min_win_rate:.0%} | "
            f"PF < {self.min_profit_factor:.2f} | "
            f"Consec Losses >= {self.max_consecutive_losses} | "
            f"Daily Loss > {self.daily_loss_limit_pct}%"
        )

    # =========================================================================
    # MAIN CHECK
    # =========================================================================

    def check_circuit_breakers(self) -> Dict:
        """
        Check all circuit breaker conditions.
        
        Returns:
            Dict with status and actions
        """
        result = {
            'status': self.status,
            'is_paused': self.is_paused,
            'pause_until': self.pause_until,
            'pause_reason': self.pause_reason,
            'pause_trigger': self.pause_trigger,
            'triggers_checked': [],
            'actions': []
        }

        # Check if currently paused
        if self.is_paused:
            if self.pause_until and datetime.now() >= self.pause_until:
                self._resume_trading()
                result['actions'].append("Auto-resumed: Pause duration expired")
            else:
                remaining = (self.pause_until - datetime.now()).total_seconds() / 3600
                result['actions'].append(
                    f"Paused: {self.pause_reason} ({remaining:.1f} hours remaining)"
                )
                return result

        # Check if in grace period
        if self.grace_until and datetime.now() < self.grace_until:
            self.status = self.STATUS_GRACE_PERIOD
            result['status'] = self.STATUS_GRACE_PERIOD
            result['actions'].append("In grace period after resume")
            return result
        elif self.grace_until:
            self.grace_until = None

        self.status = self.STATUS_ACTIVE
        result['status'] = self.STATUS_ACTIVE

        # =========================================================================
        # CHECK 1: Win Rate
        # =========================================================================
        win_rate_result = self._check_win_rate()
        result['triggers_checked'].append(self.TRIGGER_WIN_RATE)
        if win_rate_result['triggered']:
            self._pause_trading(
                hours=self.winrate_pause_hours,
                reason=win_rate_result['reason'],
                trigger=self.TRIGGER_WIN_RATE
            )
            result['actions'].append(f"PAUSED: {win_rate_result['reason']}")
            result['is_paused'] = True
            result['pause_until'] = self.pause_until
            result['pause_reason'] = self.pause_reason
            result['pause_trigger'] = self.pause_trigger
            return result

        # =========================================================================
        # CHECK 2: Profit Factor
        # =========================================================================
        pf_result = self._check_profit_factor()
        result['triggers_checked'].append(self.TRIGGER_PROFIT_FACTOR)
        if pf_result['triggered']:
            self._pause_trading(
                hours=self.pf_pause_hours,
                reason=pf_result['reason'],
                trigger=self.TRIGGER_PROFIT_FACTOR
            )
            result['actions'].append(f"PAUSED: {pf_result['reason']}")
            result['is_paused'] = True
            result['pause_until'] = self.pause_until
            result['pause_reason'] = self.pause_reason
            result['pause_trigger'] = self.pause_trigger
            return result

        # =========================================================================
        # CHECK 3: Consecutive Losses
        # =========================================================================
        consec_result = self._check_consecutive_losses()
        result['triggers_checked'].append(self.TRIGGER_CONSECUTIVE_LOSSES)
        if consec_result['triggered']:
            self._pause_trading(
                hours=self.consec_pause_hours,
                reason=consec_result['reason'],
                trigger=self.TRIGGER_CONSECUTIVE_LOSSES
            )
            result['actions'].append(f"PAUSED: {consec_result['reason']}")
            result['is_paused'] = True
            result['pause_until'] = self.pause_until
            result['pause_reason'] = self.pause_reason
            result['pause_trigger'] = self.pause_trigger
            return result

        # =========================================================================
        # CHECK 4: Daily Loss Limit
        # =========================================================================
        daily_result = self._check_daily_loss()
        result['triggers_checked'].append(self.TRIGGER_DAILY_LOSS)
        if daily_result['triggered']:
            self._pause_until_end_of_day(reason=daily_result['reason'])
            result['actions'].append(f"PAUSED: {daily_result['reason']}")
            result['is_paused'] = True
            result['pause_until'] = self.pause_until
            result['pause_reason'] = self.pause_reason
            result['pause_trigger'] = self.pause_trigger
            return result

        return result

    # =========================================================================
    # TRIGGER 1: WIN RATE CHECK
    # =========================================================================

    def _check_win_rate(self) -> Dict:
        """
        Check if win rate has fallen below threshold.
        
        Returns:
            Dict with triggered status and reason
        """
        result = {
            'triggered': False,
            'reason': '',
            'current_value': 0.0,
            'threshold': self.min_win_rate
        }

        trades = self._get_recent_trades(self.rolling_window)

        if len(trades) < 5:
            return result  # Not enough data

        wins = sum(1 for t in trades if t['profit'] > 0)
        win_rate = wins / len(trades)

        result['current_value'] = win_rate

        if win_rate < self.min_win_rate:
            result['triggered'] = True
            result['reason'] = (
                f"Win Rate {win_rate:.1%} < {self.min_win_rate:.0%} "
                f"({wins}/{len(trades)} trades)"
            )

        return result

    # =========================================================================
    # TRIGGER 2: PROFIT FACTOR CHECK
    # =========================================================================

    def _check_profit_factor(self) -> Dict:
        """
        Check if profit factor has fallen below threshold.
        
        Returns:
            Dict with triggered status and reason
        """
        result = {
            'triggered': False,
            'reason': '',
            'current_value': 0.0,
            'threshold': self.min_profit_factor
        }

        trades = self._get_recent_trades(self.rolling_window)

        if len(trades) < 5:
            return result  # Not enough data

        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] <= 0))

        if gross_loss == 0:
            if gross_profit > 0:
                return result  # All wins, no issue
            else:
                return result  # No trades with P&L

        profit_factor = gross_profit / gross_loss
        result['current_value'] = profit_factor

        if profit_factor < self.min_profit_factor:
            result['triggered'] = True
            result['reason'] = (
                f"Profit Factor {profit_factor:.2f} < {self.min_profit_factor:.2f} "
                f"(GP: ${gross_profit:.2f}, GL: ${gross_loss:.2f})"
            )

        return result

    # =========================================================================
    # TRIGGER 3: CONSECUTIVE LOSSES CHECK
    # =========================================================================

    def _check_consecutive_losses(self) -> Dict:
        """
        Check if consecutive losses have exceeded threshold.
        
        Returns:
            Dict with triggered status and reason
        """
        result = {
            'triggered': False,
            'reason': '',
            'current_value': self.consecutive_losses,
            'threshold': self.max_consecutive_losses
        }

        if self.consecutive_losses >= self.max_consecutive_losses:
            result['triggered'] = True
            result['reason'] = (
                f"Consecutive losses: {self.consecutive_losses} >= {self.max_consecutive_losses}"
            )

        return result

    # =========================================================================
    # TRIGGER 4: DAILY LOSS CHECK
    # =========================================================================

    def _check_daily_loss(self) -> Dict:
        """
        Check if daily loss has exceeded threshold.
        
        Returns:
            Dict with triggered status and reason
        """
        result = {
            'triggered': False,
            'reason': '',
            'current_value': 0.0,
            'threshold': self.daily_loss_limit_pct
        }

        # Ensure daily tracking initialized
        if self.daily_start_equity == 0:
            account_info = mt5.account_info()
            if account_info:
                self.daily_start_equity = account_info.equity
                self.daily_start_date = datetime.now().date()

        account_info = mt5.account_info()
        if not account_info or self.daily_start_equity <= 0:
            return result

        current_equity = account_info.equity
        daily_pnl = current_equity - self.daily_start_equity
        daily_pnl_pct = (daily_pnl / self.daily_start_equity) * 100

        result['current_value'] = daily_pnl_pct

        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            result['triggered'] = True
            result['reason'] = (
                f"Daily loss {daily_pnl_pct:.2f}% <= -{self.daily_loss_limit_pct}% "
                f"(${daily_pnl:.2f} from ${self.daily_start_equity:.2f})"
            )

        return result

    # =========================================================================
    # PAUSE MANAGEMENT
    # =========================================================================

    def _pause_trading(self, hours: float, reason: str, trigger: str):
        """
        Pause trading for specified hours.
        
        Args:
            hours: Hours to pause
            reason: Reason for pause
            trigger: Trigger type
        """
        self.is_paused = True
        self.pause_until = datetime.now() + timedelta(hours=hours)
        self.pause_reason = reason
        self.pause_trigger = trigger
        self.status = self.STATUS_PAUSED

        # Record to history
        self._pause_history.append({
            'timestamp': datetime.now().isoformat(),
            'trigger': trigger,
            'reason': reason,
            'duration_hours': hours,
            'resume_at': self.pause_until.isoformat()
        })

        # Update trigger stats
        self._trigger_stats[trigger] = self._trigger_stats.get(trigger, 0) + 1

        self.logger.warning(
            f"[CIRCUIT_BREAKER] TRADING PAUSED for {hours:.1f} hours | "
            f"Trigger: {trigger} | Reason: {reason}"
        )

    def _pause_until_end_of_day(self, reason: str):
        """Pause trading until end of day."""
        now = datetime.now()
        end_of_day = now.replace(hour=23, minute=59, second=59)
        hours = (end_of_day - now).total_seconds() / 3600

        self._pause_trading(
            hours=hours,
            reason=reason,
            trigger=self.TRIGGER_DAILY_LOSS
        )

    def _resume_trading(self):
        """Resume trading after pause expires."""
        self.is_paused = False
        self.pause_until = None
        self.pause_reason = ""
        self.pause_trigger = None
        self.status = self.STATUS_ACTIVE

        # Start grace period
        self.grace_until = datetime.now() + timedelta(hours=self.grace_period_hours)

        # Reset consecutive losses on resume
        self.consecutive_losses = 0

        self.logger.info(
            f"[CIRCUIT_BREAKER] Trading resumed | "
            f"Grace period: {self.grace_period_hours} hours"
        )

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    def record_trade_result(self, profit: float):
        """
        Record trade result for consecutive loss tracking.
        
        Args:
            profit: Trade profit/loss in USD
        """
        if profit > 0:
            if self.consecutive_losses > 0:
                self.logger.info(
                    f"[CIRCUIT_BREAKER] Win breaks streak of {self.consecutive_losses} losses"
                )
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.logger.info(
                f"[CIRCUIT_BREAKER] Loss recorded | "
                f"Consecutive losses: {self.consecutive_losses}"
            )

    # =========================================================================
    # MANUAL OVERRIDE
    # =========================================================================

    def manual_pause(self, hours: float, reason: str = "Manual pause"):
        """
        Manually pause trading.
        
        Args:
            hours: Hours to pause
            reason: Reason for pause
        """
        self._pause_trading(
            hours=hours,
            reason=reason,
            trigger=self.TRIGGER_MANUAL
        )

    def force_resume(self):
        """
        Force resume trading (bypass grace period check).
        Use with caution.
        """
        if self.is_paused:
            self.logger.warning(
                f"[CIRCUIT_BREAKER] FORCE RESUME | "
                f"Was paused for: {self.pause_reason}"
            )
            self._resume_trading()
            self.grace_until = None  # Skip grace period

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.
        
        Returns:
            True if trading is allowed
        """
        if self.is_paused:
            if self.pause_until and datetime.now() >= self.pause_until:
                self._resume_trading()
                return True
            return False

        # In grace period, trading is allowed but monitored
        return True

    # =========================================================================
    # DAILY RESET
    # =========================================================================

    def reset_daily(self, current_equity: float = None):
        """
        Reset daily tracking at start of new day.
        
        Args:
            current_equity: Current equity (if None, fetch from MT5)
        """
        today = datetime.now().date()

        if self.daily_start_date != today:
            if current_equity is None:
                account_info = mt5.account_info()
                if account_info:
                    current_equity = account_info.equity
                else:
                    return

            self.daily_start_equity = current_equity
            self.daily_start_date = today

            self.logger.info(
                f"[CIRCUIT_BREAKER] Daily reset | "
                f"Start equity: ${current_equity:.2f}"
            )

    # =========================================================================
    # DATABASE QUERIES
    # =========================================================================

    def _get_recent_trades(self, count: int) -> List[Dict]:
        """
        Get recent trade history from SQLite.
        
        Args:
            count: Number of recent trades to fetch
            
        Returns:
            List of trade dicts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute(
                "SELECT profit, close_time FROM trade_history ORDER BY close_time DESC LIMIT ?",
                (count,)
            )

            trades = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return trades

        except sqlite3.OperationalError as e:
            self.logger.debug(f"[CIRCUIT_BREAKER] DB not ready: {e}")
            return []
        except Exception as e:
            self.logger.error(f"[CIRCUIT_BREAKER] DB query error: {e}")
            return []

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_circuit_breaker_stats(self) -> Dict:
        """
        Get comprehensive circuit breaker statistics.
        
        Returns:
            Dict with statistics
        """
        # Get recent trades for current metrics
        recent_trades = self._get_recent_trades(self.rolling_window)

        current_win_rate = 0.0
        current_pf = 0.0

        if len(recent_trades) >= 5:
            wins = sum(1 for t in recent_trades if t['profit'] > 0)
            current_win_rate = wins / len(recent_trades)

            gross_profit = sum(t['profit'] for t in recent_trades if t['profit'] > 0)
            gross_loss = abs(sum(t['profit'] for t in recent_trades if t['profit'] <= 0))
            if gross_loss > 0:
                current_pf = gross_profit / gross_loss

        # Daily PnL
        daily_pnl_pct = 0.0
        if self.daily_start_equity > 0:
            account_info = mt5.account_info()
            if account_info:
                daily_pnl_pct = ((account_info.equity - self.daily_start_equity) / 
                                 self.daily_start_equity) * 100

        return {
            'status': self.status,
            'is_paused': self.is_paused,
            'pause_until': self.pause_until.isoformat() if self.pause_until else None,
            'pause_reason': self.pause_reason,
            'pause_trigger': self.pause_trigger,
            'grace_until': self.grace_until.isoformat() if self.grace_until else None,
            'current_metrics': {
                'win_rate': round(current_win_rate, 3),
                'profit_factor': round(current_pf, 2),
                'consecutive_losses': self.consecutive_losses,
                'daily_pnl_pct': round(daily_pnl_pct, 2)
            },
            'thresholds': {
                'min_win_rate': self.min_win_rate,
                'min_profit_factor': self.min_profit_factor,
                'max_consecutive_losses': self.max_consecutive_losses,
                'daily_loss_limit_pct': self.daily_loss_limit_pct
            },
            'trigger_history': dict(self._trigger_stats),
            'total_pauses': len(self._pause_history),
            'recent_pauses': list(self._pause_history)[-5:]
        }

    def get_pause_history(self, limit: int = 10) -> List[Dict]:
        """
        Get pause history.
        
        Args:
            limit: Maximum records to return
            
        Returns:
            List of pause records
        """
        return list(self._pause_history)[-limit:]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_status_summary(self) -> Dict:
        """
        Get concise status summary.
        
        Returns:
            Dict with summary
        """
        stats = self.get_circuit_breaker_stats()
        metrics = stats['current_metrics']

        return {
            'status': stats['status'],
            'is_paused': stats['is_paused'],
            'pause_remaining_hours': None,
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'consecutive_losses': metrics['consecutive_losses'],
            'daily_pnl_pct': metrics['daily_pnl_pct']
        }

    def format_status_log(self) -> str:
        """
        Format status as concise log string.
        
        Returns:
            Formatted log string
        """
        stats = self.get_circuit_breaker_stats()
        metrics = stats['current_metrics']

        if stats['is_paused']:
            if stats['pause_until']:
                remaining = (datetime.fromisoformat(stats['pause_until']) - 
                           datetime.now()).total_seconds() / 3600
                return (
                    f"[CIRCUIT_BREAKER] PAUSED ({stats['pause_trigger']}) | "
                    f"{remaining:.1f}h remaining | "
                    f"Reason: {stats['pause_reason'][:50]}"
                )
            return f"[CIRCUIT_BREAKER] PAUSED | {stats['pause_reason'][:50]}"

        if stats['status'] == self.STATUS_GRACE_PERIOD:
            return "[CIRCUIT_BREAKER] GRACE PERIOD | Monitoring after resume"

        return (
            f"[CIRCUIT_BREAKER] ACTIVE | "
            f"WR: {metrics['win_rate']:.0%} | "
            f"PF: {metrics['profit_factor']:.2f} | "
            f"Consec: {metrics['consecutive_losses']} | "
            f"Daily: {metrics['daily_pnl_pct']:+.2f}%"
        )

    def check_health(self) -> Dict:
        """
        Check health of circuit breaker.
        
        Returns:
            Dict with health status
        """
        stats = self.get_circuit_breaker_stats()
        metrics = stats['current_metrics']
        issues = []

        # Check if close to triggers
        if metrics['win_rate'] < self.min_win_rate * 1.2:
            issues.append(f"Win rate {metrics['win_rate']:.0%} approaching threshold")

        if metrics['profit_factor'] < self.min_profit_factor * 1.2:
            issues.append(f"PF {metrics['profit_factor']:.2f} approaching threshold")

        if metrics['consecutive_losses'] >= self.max_consecutive_losses - 1:
            issues.append(f"Consecutive losses {metrics['consecutive_losses']} near threshold")

        if metrics['daily_pnl_pct'] < -self.daily_loss_limit_pct * 0.7:
            issues.append(f"Daily loss {metrics['daily_pnl_pct']:.2f}% approaching limit")

        # Check pause frequency
        total_pauses = sum(stats['trigger_history'].values())
        if total_pauses > 10:
            issues.append(f"High pause frequency: {total_pauses} total pauses")

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }