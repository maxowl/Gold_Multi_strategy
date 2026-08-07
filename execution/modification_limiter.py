"""
Modification Rate Limiter - Order Modification Throttling.

Prevents excessive SL/TP modifications that could trigger broker
rate limiting, throttling, or account restrictions.

Critical for systems with Dynamic Stops that frequently adjust
SL/TP based on market conditions.

Rate Limits (configurable):
  - Per Position: Max 10 modifications per position lifetime
  - Global Rate: Max 5 modifications per minute (all positions)
  - Cooldown: 3 seconds between modifications for same position

Features:
  - Per-position modification tracking
  - Global rate limiting (sliding window)
  - Cooldown enforcement
  - Burst detection and handling
  - Adaptive rate based on broker response
  - Statistics tracking
  - Position cleanup on close
"""
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import config


class ModificationRateLimiter:
    """
    Rate limiter for order modifications (SL/TP changes).
    
    Features:
      - Per-position modification limit
      - Global rate limiting (sliding window)
      - Cooldown between modifications
      - Burst detection
      - Statistics tracking
      - Position cleanup
    """

    # Block reasons
    REASON_OK = 'OK'
    REASON_POSITION_LIMIT = 'POSITION_LIMIT_REACHED'
    REASON_GLOBAL_RATE = 'GLOBAL_RATE_LIMIT'
    REASON_COOLDOWN = 'COOLDOWN_ACTIVE'
    REASON_BURST_DETECTED = 'BURST_DETECTED'
    REASON_BROKER_THROTTLE = 'BROKER_THROTTLE'

    def __init__(self):
        """Initialize ModificationRateLimiter with config parameters."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Limits from config
        self.max_per_position = config.mod_max_per_position  # 10
        self.max_per_minute = config.mod_max_per_minute  # 5
        self.cooldown_seconds = config.mod_cooldown_seconds  # 3.0

        # Per-position tracking: ticket -> {count, last_time, timestamps}
        self._position_tracking: Dict[int, Dict] = {}

        # Global modification timestamps (sliding window)
        self._global_timestamps = deque(maxlen=100)

        # Burst detection
        self._burst_window_seconds = 10  # 10 second burst window
        self._burst_threshold = 8  # 8 modifications in burst window = burst
        self._burst_cooldown_seconds = 30  # 30 second cooldown after burst

        # Burst state
        self._burst_detected = False
        self._burst_cooldown_until = 0

        # Statistics
        self._total_modifications = 0
        self._total_blocked = 0
        self._block_reasons: Dict[str, int] = {}

        self.logger.info(
            f"[MOD_LIMIT] Initialized | "
            f"Per Position: {self.max_per_position} | "
            f"Per Minute: {self.max_per_minute} | "
            f"Cooldown: {self.cooldown_seconds}s"
        )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def can_modify(self, ticket: int, modification_type: str = "SL") -> Tuple[bool, str]:
        """
        Check if modification is allowed for this ticket.
        
        Args:
            ticket: Position ticket
            modification_type: Type of modification ('SL', 'TP', 'BOTH')
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check burst cooldown first
        if self._burst_detected:
            if time.time() < self._burst_cooldown_until:
                remaining = self._burst_cooldown_until - time.time()
                self._record_block(self.REASON_BURST_DETECTED)
                return False, f"Burst cooldown active ({remaining:.1f}s remaining)"
            else:
                self._burst_detected = False

        # Initialize position tracking if new
        if ticket not in self._position_tracking:
            self._position_tracking[ticket] = {
                'count': 0,
                'last_time': 0,
                'timestamps': deque(maxlen=50),
                'first_time': time.time()
            }

        tracking = self._position_tracking[ticket]

        # =========================================================================
        # CHECK 1: Per-Position Limit
        # =========================================================================
        if tracking['count'] >= self.max_per_position:
            self._record_block(self.REASON_POSITION_LIMIT)
            return False, f"Position {ticket} reached max modifications ({tracking['count']}/{self.max_per_position})"

        # =========================================================================
        # CHECK 2: Cooldown for this position
        # =========================================================================
        current_time = time.time()
        if tracking['last_time'] > 0:
            elapsed = current_time - tracking['last_time']
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                self._record_block(self.REASON_COOLDOWN)
                return False, f"Cooldown active for position {ticket} ({remaining:.1f}s remaining)"

        # =========================================================================
        # CHECK 3: Global Rate Limit (sliding window)
        # =========================================================================
        # Clean old timestamps (older than 60 seconds)
        cutoff_time = current_time - 60
        while self._global_timestamps and self._global_timestamps[0] < cutoff_time:
            self._global_timestamps.popleft()

        # Check if at limit
        if len(self._global_timestamps) >= self.max_per_minute:
            oldest_in_window = self._global_timestamps[0]
            wait_time = 60 - (current_time - oldest_in_window)
            self._record_block(self.REASON_GLOBAL_RATE)
            return False, f"Global rate limit reached ({len(self._global_timestamps)}/{self.max_per_minute} in last 60s, wait {wait_time:.1f}s)"

        # =========================================================================
        # CHECK 4: Burst Detection
        # =========================================================================
        burst_cutoff = current_time - self._burst_window_seconds
        recent_mods = sum(1 for ts in self._global_timestamps if ts > burst_cutoff)

        if recent_mods >= self._burst_threshold:
            self._burst_detected = True
            self._burst_cooldown_until = current_time + self._burst_cooldown_seconds
            self.logger.warning(
                f"[MOD_LIMIT] Burst detected: {recent_mods} modifications in {self._burst_window_seconds}s"
            )
            self._record_block(self.REASON_BURST_DETECTED)
            return False, f"Burst detected ({recent_mods} mods in {self._burst_window_seconds}s), cooling down for {self._burst_cooldown_seconds}s"

        # All checks passed
        return True, self.REASON_OK

    # =========================================================================
    # RECORD MODIFICATION
    # =========================================================================

    def record_modification(self, ticket: int, modification_type: str = "SL"):
        """
        Record that a modification was successfully made.
        
        Args:
            ticket: Position ticket
            modification_type: Type of modification ('SL', 'TP', 'BOTH')
        """
        current_time = time.time()

        # Initialize if new
        if ticket not in self._position_tracking:
            self._position_tracking[ticket] = {
                'count': 0,
                'last_time': 0,
                'timestamps': deque(maxlen=50),
                'first_time': current_time
            }

        tracking = self._position_tracking[ticket]

        # Update tracking
        tracking['count'] += 1
        tracking['last_time'] = current_time
        tracking['timestamps'].append(current_time)

        # Update global tracking
        self._global_timestamps.append(current_time)

        # Update statistics
        self._total_modifications += 1

        self.logger.debug(
            f"[MOD_LIMIT] Recorded modification for ticket {ticket} | "
            f"Type: {modification_type} | "
            f"Count: {tracking['count']}/{self.max_per_position}"
        )

    # =========================================================================
    # POSITION CLEANUP
    # =========================================================================

    def reset_position(self, ticket: int):
        """
        Reset tracking for a closed position.
        
        Args:
            ticket: Position ticket
        """
        if ticket in self._position_tracking:
            tracking = self._position_tracking[ticket]
            self.logger.debug(
                f"[MOD_LIMIT] Reset position {ticket} | "
                f"Total modifications: {tracking['count']}"
            )
            del self._position_tracking[ticket]

    def cleanup_stale_positions(self, active_tickets: List[int]):
        """
        Remove tracking for positions that are no longer active.
        
        Args:
            active_tickets: List of currently active position tickets
        """
        stale_tickets = [
            ticket for ticket in self._position_tracking.keys()
            if ticket not in active_tickets
        ]

        for ticket in stale_tickets:
            self.reset_position(ticket)

        if stale_tickets:
            self.logger.debug(
                f"[MOD_LIMIT] Cleaned up {len(stale_tickets)} stale position(s)"
            )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dict with statistics
        """
        current_time = time.time()

        # Calculate modifications in last minute
        cutoff = current_time - 60
        mods_last_minute = sum(1 for ts in self._global_timestamps if ts > cutoff)

        # Calculate modifications in last hour
        hour_cutoff = current_time - 3600
        mods_last_hour = sum(1 for ts in self._global_timestamps if ts > hour_cutoff)

        # Position statistics
        active_positions = len(self._position_tracking)
        total_position_mods = sum(t['count'] for t in self._position_tracking.values())

        return {
            'total_modifications': self._total_modifications,
            'total_blocked': self._total_blocked,
            'block_rate': round(self._total_blocked / max(1, self._total_modifications + self._total_blocked) * 100, 1),
            'mods_last_minute': mods_last_minute,
            'mods_last_hour': mods_last_hour,
            'active_positions_tracked': active_positions,
            'total_position_mods': total_position_mods,
            'burst_detected': self._burst_detected,
            'burst_cooldown_remaining': max(0, self._burst_cooldown_until - current_time) if self._burst_detected else 0,
            'block_reasons': dict(self._block_reasons),
            'limits': {
                'max_per_position': self.max_per_position,
                'max_per_minute': self.max_per_minute,
                'cooldown_seconds': self.cooldown_seconds
            }
        }

    def get_position_stats(self, ticket: int) -> Optional[Dict]:
        """
        Get statistics for a specific position.
        
        Args:
            ticket: Position ticket
            
        Returns:
            Dict with position stats or None
        """
        if ticket not in self._position_tracking:
            return None

        tracking = self._position_tracking[ticket]
        current_time = time.time()

        return {
            'ticket': ticket,
            'modification_count': tracking['count'],
            'max_allowed': self.max_per_position,
            'remaining': self.max_per_position - tracking['count'],
            'last_modification_ago': current_time - tracking['last_time'] if tracking['last_time'] > 0 else None,
            'tracking_duration': current_time - tracking['first_time'],
            'cooldown_active': (current_time - tracking['last_time']) < self.cooldown_seconds if tracking['last_time'] > 0 else False
        }

    # =========================================================================
    # ADAPTIVE RATE LIMITING
    # =========================================================================

    def report_broker_response(self, success: bool, error_code: int = None):
        """
        Report broker response to adapt rate limiting.
        
        Args:
            success: Whether modification succeeded
            error_code: MT5 error code if failed
        """
        if not success:
            # Check for throttle-related errors
            throttle_errors = [
                10004,  # TRADE_RETCODE_TOO_MANY_REQUESTS
                10005,  # TRADE_RETCODE_TIMEOUT
            ]

            if error_code in throttle_errors:
                self.logger.warning(
                    f"[MOD_LIMIT] Broker throttle detected (error {error_code}), increasing cooldown"
                )
                # Temporarily increase cooldown
                self.cooldown_seconds = min(10.0, self.cooldown_seconds * 1.5)
                self._record_block(self.REASON_BROKER_THROTTLE)

    def reset_to_defaults(self):
        """Reset rate limits to config defaults."""
        self.max_per_position = config.mod_max_per_position
        self.max_per_minute = config.mod_max_per_minute
        self.cooldown_seconds = config.mod_cooldown_seconds

        self.logger.info("[MOD_LIMIT] Reset to default limits")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _record_block(self, reason: str):
        """Record a blocked modification."""
        self._total_blocked += 1
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    def get_active_tracking(self) -> Dict[int, Dict]:
        """
        Get all active position tracking data.
        
        Returns:
            Dict of ticket -> tracking info
        """
        result = {}
        current_time = time.time()

        for ticket, tracking in self._position_tracking.items():
            result[ticket] = {
                'count': tracking['count'],
                'remaining': self.max_per_position - tracking['count'],
                'last_mod_ago': current_time - tracking['last_time'] if tracking['last_time'] > 0 else None,
                'cooldown_active': (current_time - tracking['last_time']) < self.cooldown_seconds if tracking['last_time'] > 0 else False
            }

        return result

    def format_stats_log(self) -> str:
        """
        Format statistics as concise log string.
        
        Returns:
            Formatted log string
        """
        stats = self.get_stats()

        return (
            f"[MOD_LIMIT] Total: {stats['total_modifications']} mods, "
            f"{stats['total_blocked']} blocked ({stats['block_rate']:.1f}%) | "
            f"Last min: {stats['mods_last_minute']}/{self.max_per_minute} | "
            f"Positions: {stats['active_positions_tracked']}"
        )

    def check_health(self) -> Dict:
        """
        Check health of rate limiter.
        
        Returns:
            Dict with health status
        """
        stats = self.get_stats()
        issues = []

        # Check block rate
        if stats['block_rate'] > 50:
            issues.append(f"High block rate: {stats['block_rate']:.1f}%")

        # Check if at global limit
        if stats['mods_last_minute'] >= self.max_per_minute:
            issues.append("At global rate limit")

        # Check burst state
        if stats['burst_detected']:
            issues.append(f"Burst cooldown active: {stats['burst_cooldown_remaining']:.1f}s")

        # Check for specific block reasons
        if stats['block_reasons'].get(self.REASON_BROKER_THROTTLE, 0) > 5:
            issues.append("Multiple broker throttle events")

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }