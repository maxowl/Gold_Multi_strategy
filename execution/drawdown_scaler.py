"""
Drawdown Risk Scaler - Progressive Risk Reduction.

Monitors account drawdown from peak equity and progressively reduces
position sizing to protect capital during losing periods.

Critical for Micro-Account trading where a few consecutive losses
can significantly impact the account.

Drawdown Levels (configurable):
  Level 1: 1.0% drawdown -> Risk multiplier 0.75 (reduce 25%)
  Level 2: 2.0% drawdown -> Risk multiplier 0.50 (reduce 50%)
  Level 3: 3.0% drawdown -> Risk multiplier 0.00 (HALT trading)

Features:
  - Peak equity tracking
  - Drawdown calculation
  - Progressive risk scaling
  - Trading halt detection
  - Recovery mechanism
  - Drawdown history tracking
  - Consecutive loss tracking
  - Alert generation
"""
import MetaTrader5 as mt5
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import config


class DrawdownRiskScaler:
    """
    Monitors drawdown and scales position size accordingly.
    
    Features:
      - Peak equity tracking
      - Drawdown calculation
      - Progressive risk reduction
      - Trading halt at critical drawdown
      - Recovery detection
      - Drawdown history
      - Consecutive loss tracking
    """

    # Drawdown levels
    LEVEL_NORMAL = 'NORMAL'
    LEVEL_MODERATE = 'MODERATE'
    LEVEL_HIGH = 'HIGH'
    LEVEL_CRITICAL = 'CRITICAL'
    LEVEL_HALTED = 'HALTED'

    def __init__(self):
        """Initialize DrawdownRiskScaler with config parameters."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Drawdown thresholds from config
        self.level1_threshold = config.drawdown_level1_pct  # 1.0%
        self.level2_threshold = config.drawdown_level2_pct  # 2.0%
        self.level3_threshold = config.drawdown_level3_pct  # 3.0%

        # Risk multipliers from config
        self.level1_multiplier = config.drawdown_level1_multiplier  # 0.75
        self.level2_multiplier = config.drawdown_level2_multiplier  # 0.50
        self.level3_multiplier = config.drawdown_level3_multiplier  # 0.00

        # State
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown_pct = 0.0
        self.risk_multiplier = 1.0
        self.is_halted = False
        self.halt_reason = ""

        # Consecutive loss tracking
        self.consecutive_losses = 0
        self.consecutive_loss_threshold = 5  # Halt after 5 consecutive losses

        # Drawdown history (last 100 records)
        self._drawdown_history = deque(maxlen=100)

        # Peak history for analysis
        self._peak_history = deque(maxlen=50)

        # Time tracking
        self._last_update_time = 0
        self._halt_start_time = None
        self._recovery_threshold_pct = 0.5  # Recover if drawdown < 0.5%

        self.logger.info(
            f"[DRAWDOWN] Initialized | "
            f"Level 1: {self.level1_threshold}% (x{self.level1_multiplier}) | "
            f"Level 2: {self.level2_threshold}% (x{self.level2_multiplier}) | "
            f"Level 3: {self.level3_threshold}% (x{self.level3_multiplier})"
        )

    # =========================================================================
    # EQUITY TRACKING
    # =========================================================================

    def update_equity(self, current_equity: float = None) -> Dict:
        """
        Update equity tracking and calculate risk multiplier.
        
        Args:
            current_equity: Current account equity (if None, fetch from MT5)
            
        Returns:
            Dict with drawdown status and risk multiplier
        """
        # Get equity from MT5 if not provided
        if current_equity is None:
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("[DRAWDOWN] Cannot get account info")
                return {
                    'current_equity': self.current_equity,
                    'peak_equity': self.peak_equity,
                    'drawdown_pct': self.current_drawdown_pct,
                    'risk_multiplier': self.risk_multiplier,
                    'is_halted': self.is_halted,
                    'reason': 'Cannot get account info'
                }
            current_equity = account_info.equity

        self.current_equity = current_equity

        # Initialize peak if first run
        if self.peak_equity == 0:
            self.peak_equity = current_equity
            self._record_peak()
            self.logger.info(f"[DRAWDOWN] Initial peak set: ${current_equity:.2f}")

        # Update peak if new high
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self._record_peak()
            self.logger.info(f"[DRAWDOWN] New peak: ${current_equity:.2f}")

        # Calculate drawdown
        self._calculate_drawdown()

        # Determine risk multiplier
        self._determine_risk_multiplier()

        # Check for halt
        self._check_halt_conditions()

        # Check for recovery
        self._check_recovery()

        # Record to history
        self._record_drawdown()

        # Update timestamp
        self._last_update_time = time.time()

        # Log if drawdown significant
        if self.current_drawdown_pct >= self.level1_threshold:
            level = self._get_drawdown_level()
            self.logger.warning(
                f"[DRAWDOWN] {level} | "
                f"Equity: ${self.current_equity:.2f} | "
                f"Peak: ${self.peak_equity:.2f} | "
                f"Drawdown: {self.current_drawdown_pct:.2f}% | "
                f"Multiplier: {self.risk_multiplier:.2f}x | "
                f"Halted: {self.is_halted}"
            )

        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': round(self.current_drawdown_pct, 2),
            'risk_multiplier': self.risk_multiplier,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'drawdown_level': self._get_drawdown_level(),
            'consecutive_losses': self.consecutive_losses
        }

    # =========================================================================
    # DRAWDOWN CALCULATION
    # =========================================================================

    def _calculate_drawdown(self):
        """Calculate current drawdown percentage."""
        if self.peak_equity <= 0:
            self.current_drawdown_pct = 0.0
            return

        drawdown = self.peak_equity - self.current_equity
        self.current_drawdown_pct = (drawdown / self.peak_equity) * 100

        # Clamp to 0-100
        self.current_drawdown_pct = max(0.0, min(100.0, self.current_drawdown_pct))

    def calculate_drawdown(self, equity: float = None) -> float:
        """
        Public method to calculate drawdown.
        
        Args:
            equity: Equity to calculate drawdown for (if None, use current)
            
        Returns:
            Drawdown percentage
        """
        if equity is not None:
            if self.peak_equity <= 0:
                return 0.0
            drawdown = self.peak_equity - equity
            return max(0.0, min(100.0, (drawdown / self.peak_equity) * 100))

        return self.current_drawdown_pct

    # =========================================================================
    # RISK MULTIPLIER DETERMINATION
    # =========================================================================

    def _determine_risk_multiplier(self):
        """Determine risk multiplier based on drawdown level."""
        if self.is_halted:
            self.risk_multiplier = 0.0
            return

        if self.current_drawdown_pct >= self.level3_threshold:
            self.risk_multiplier = self.level3_multiplier
        elif self.current_drawdown_pct >= self.level2_threshold:
            self.risk_multiplier = self.level2_multiplier
        elif self.current_drawdown_pct >= self.level1_threshold:
            self.risk_multiplier = self.level1_multiplier
        else:
            self.risk_multiplier = 1.0

    def get_risk_multiplier(self) -> float:
        """
        Get current risk multiplier for position sizing.
        
        Returns:
            Risk multiplier (0.0 to 1.0)
        """
        return self.risk_multiplier

    # =========================================================================
    # HALT DETECTION
    # =========================================================================

    def _check_halt_conditions(self):
        """Check if trading should be halted."""
        halt_reasons = []

        # Check drawdown level 3
        if self.current_drawdown_pct >= self.level3_threshold:
            halt_reasons.append(
                f"Drawdown {self.current_drawdown_pct:.2f}% >= {self.level3_threshold}%"
            )

        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_threshold:
            halt_reasons.append(
                f"Consecutive losses: {self.consecutive_losses} >= {self.consecutive_loss_threshold}"
            )

        # Determine halt status
        if halt_reasons and not self.is_halted:
            self.is_halted = True
            self.halt_reason = "; ".join(halt_reasons)
            self._halt_start_time = datetime.now()
            self.logger.critical(
                f"[DRAWDOWN] TRADING HALTED | Reason: {self.halt_reason}"
            )
        elif not halt_reasons and self.is_halted:
            # Check if recovery conditions met
            if self.current_drawdown_pct < self._recovery_threshold_pct:
                self.is_halted = False
                self.halt_reason = ""
                self._halt_start_time = None
                self.logger.info("[DRAWDOWN] Trading resumed - drawdown recovered")

    def is_trading_halted(self) -> bool:
        """
        Check if trading is currently halted.
        
        Returns:
            True if trading is halted
        """
        return self.is_halted

    def get_halt_reason(self) -> str:
        """
        Get the reason for trading halt.
        
        Returns:
            Halt reason string
        """
        return self.halt_reason

    # =========================================================================
    # RECOVERY DETECTION
    # =========================================================================

    def _check_recovery(self):
        """Check for recovery conditions."""
        if self.is_halted and self._halt_start_time:
            # Check time-based recovery (minimum 1 hour halt)
            halt_duration = (datetime.now() - self._halt_start_time).total_seconds() / 60.0

            if halt_duration >= 60 and self.current_drawdown_pct < self._recovery_threshold_pct:
                self.is_halted = False
                self.halt_reason = ""
                self._halt_start_time = None
                self.consecutive_losses = 0
                self.logger.info(
                    f"[DRAWDOWN] Trading resumed after {halt_duration:.0f} minutes halt"
                )

    def reset_peak(self, new_peak: float = None):
        """
        Reset peak equity (use after deposit or manual reset).
        
        Args:
            new_peak: New peak value (if None, use current equity)
        """
        if new_peak is not None:
            self.peak_equity = new_peak
        else:
            account_info = mt5.account_info()
            if account_info:
                self.peak_equity = account_info.equity

        self.is_halted = False
        self.halt_reason = ""
        self.risk_multiplier = 1.0
        self.consecutive_losses = 0
        self._record_peak()

        self.logger.info(f"[DRAWDOWN] Peak reset to ${self.peak_equity:.2f}")

    # =========================================================================
    # CONSECUTIVE LOSS TRACKING
    # =========================================================================

    def record_trade_result(self, profit: float):
        """
        Record trade result for consecutive loss tracking.
        
        Args:
            profit: Trade profit/loss in USD
        """
        if profit > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.logger.info(
                f"[DRAWDOWN] Consecutive losses: {self.consecutive_losses}"
            )

        # Check if halt triggered
        if self.consecutive_losses >= self.consecutive_loss_threshold:
            if not self.is_halted:
                self.is_halted = True
                self.halt_reason = f"Consecutive losses: {self.consecutive_losses}"
                self._halt_start_time = datetime.now()
                self.logger.critical(
                    f"[DRAWDOWN] TRADING HALTED | Reason: {self.halt_reason}"
                )

    def get_consecutive_losses(self) -> int:
        """
        Get current consecutive loss count.
        
        Returns:
            Number of consecutive losses
        """
        return self.consecutive_losses

    def reset_consecutive_losses(self):
        """Reset consecutive loss counter."""
        self.consecutive_losses = 0

    # =========================================================================
    # HISTORY TRACKING
    # =========================================================================

    def _record_drawdown(self):
        """Record drawdown to history."""
        self._drawdown_history.append({
            'timestamp': datetime.now().isoformat(),
            'equity': self.current_equity,
            'peak': self.peak_equity,
            'drawdown_pct': self.current_drawdown_pct,
            'risk_multiplier': self.risk_multiplier,
            'is_halted': self.is_halted
        })

    def _record_peak(self):
        """Record peak to history."""
        self._peak_history.append({
            'timestamp': datetime.now().isoformat(),
            'peak_equity': self.peak_equity
        })

    def get_drawdown_history(self, limit: int = 50) -> List[Dict]:
        """
        Get drawdown history.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of drawdown records
        """
        return list(self._drawdown_history)[-limit:]

    def get_peak_history(self, limit: int = 20) -> List[Dict]:
        """
        Get peak history.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of peak records
        """
        return list(self._peak_history)[-limit:]

    def get_drawdown_stats(self) -> Dict:
        """
        Get drawdown statistics.
        
        Returns:
            Dict with statistics
        """
        if not self._drawdown_history:
            return {
                'max_drawdown_pct': 0,
                'avg_drawdown_pct': 0,
                'current_drawdown_pct': self.current_drawdown_pct,
                'halt_count': 0
            }

        drawdowns = [r['drawdown_pct'] for r in self._drawdown_history]
        halts = [r for r in self._drawdown_history if r['is_halted']]

        return {
            'max_drawdown_pct': round(max(drawdowns), 2),
            'avg_drawdown_pct': round(sum(drawdowns) / len(drawdowns), 2),
            'current_drawdown_pct': round(self.current_drawdown_pct, 2),
            'halt_count': len(halts),
            'total_records': len(self._drawdown_history)
        }

    # =========================================================================
    # ALERTS
    # =========================================================================

    def generate_drawdown_alert(self) -> Optional[Dict]:
        """
        Generate drawdown alert if conditions warrant.
        
        Returns:
            Alert dict or None
        """
        level = self._get_drawdown_level()

        if level == self.LEVEL_HALTED:
            return {
                'level': 'CRITICAL',
                'message': f"TRADING HALTED: Drawdown {self.current_drawdown_pct:.2f}% | Reason: {self.halt_reason}",
                'action': 'STOP_TRADING',
                'drawdown_pct': self.current_drawdown_pct,
                'risk_multiplier': 0.0
            }
        elif level == self.LEVEL_CRITICAL:
            return {
                'level': 'CRITICAL',
                'message': f"CRITICAL: Drawdown {self.current_drawdown_pct:.2f}% - Trading will halt at {self.level3_threshold}%",
                'action': 'REDUCE_RISK',
                'drawdown_pct': self.current_drawdown_pct,
                'risk_multiplier': self.risk_multiplier
            }
        elif level == self.LEVEL_HIGH:
            return {
                'level': 'HIGH',
                'message': f"HIGH: Drawdown {self.current_drawdown_pct:.2f}% - Risk reduced to {self.risk_multiplier:.0%}",
                'action': 'REDUCE_RISK',
                'drawdown_pct': self.current_drawdown_pct,
                'risk_multiplier': self.risk_multiplier
            }
        elif level == self.LEVEL_MODERATE:
            return {
                'level': 'MEDIUM',
                'message': f"MODERATE: Drawdown {self.current_drawdown_pct:.2f}% - Risk reduced to {self.risk_multiplier:.0%}",
                'action': 'MONITOR',
                'drawdown_pct': self.current_drawdown_pct,
                'risk_multiplier': self.risk_multiplier
            }

        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_drawdown_level(self) -> str:
        """Get current drawdown level."""
        if self.is_halted:
            return self.LEVEL_HALTED
        elif self.current_drawdown_pct >= self.level3_threshold:
            return self.LEVEL_CRITICAL
        elif self.current_drawdown_pct >= self.level2_threshold:
            return self.LEVEL_HIGH
        elif self.current_drawdown_pct >= self.level1_threshold:
            return self.LEVEL_MODERATE
        else:
            return self.LEVEL_NORMAL

    def get_drawdown_summary(self) -> Dict:
        """
        Get comprehensive drawdown summary.
        
        Returns:
            Dict with summary
        """
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': round(self.current_drawdown_pct, 2),
            'drawdown_usd': round(self.peak_equity - self.current_equity, 2),
            'drawdown_level': self._get_drawdown_level(),
            'risk_multiplier': self.risk_multiplier,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'consecutive_losses': self.consecutive_losses,
            'stats': self.get_drawdown_stats()
        }

    def format_drawdown_log(self) -> str:
        """
        Format a concise log string for drawdown status.
        
        Returns:
            Formatted log string
        """
        level = self._get_drawdown_level()
        halt_str = " | HALTED" if self.is_halted else ""

        return (
            f"[DRAWDOWN] {level}{halt_str} | "
            f"Equity: ${self.current_equity:.2f} | "
            f"Peak: ${self.peak_equity:.2f} | "
            f"DD: {self.current_drawdown_pct:.2f}% | "
            f"Multiplier: {self.risk_multiplier:.2f}x"
        )