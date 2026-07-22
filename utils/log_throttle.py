"""
Log Throttling Utility.
Prevents log spam by limiting message frequency and deduplicating repetitive logs.
Thread-safe implementation for production environments.
"""
import logging
import time
import threading
from typing import Dict, Optional


class LogThrottler:
    """
    Throttles log messages to prevent spam.
    Thread-safe singleton-style class.
    """
    
    def __init__(self):
        self._last_log_times: Dict[str, float] = {}
        self._last_messages: Dict[str, str] = {}
        # [FIX] Add threading lock to prevent Race Conditions in multi-threaded environments
        self._lock = threading.Lock()
    
    def should_log(self, key: str, message: str, min_interval: float = 60.0, 
                   deduplicate: bool = True) -> bool:
        """
        Check if a log message should be emitted.
        """
        # [FIX] Wrap state mutations in a thread lock
        with self._lock:
            current_time = time.time()
            
            # Check time interval
            last_time = self._last_log_times.get(key, 0.0)
            if current_time - last_time < min_interval:
                return False
            
            # Check message deduplication
            if deduplicate:
                last_message = self._last_messages.get(key)
                if last_message == message:
                    return False
            
            # Update tracking
            self._last_log_times[key] = current_time
            self._last_messages[key] = message
            
        return True
    
    def reset(self, key: Optional[str] = None):
        """Reset throttling for a specific key or all keys."""
        with self._lock:
            if key:
                self._last_log_times.pop(key, None)
                self._last_messages.pop(key, None)
            else:
                self._last_log_times.clear()
                self._last_messages.clear()


# Global throttler instance
_log_throttler = LogThrottler()


def throttled_log(logger: logging.Logger, level: int, message: str, 
                  key: str, min_interval: float = 60.0, deduplicate: bool = True):
    """Log a message with throttling."""
    if _log_throttler.should_log(key, message, min_interval, deduplicate):
        logger.log(level, message)


def throttled_info(logger: logging.Logger, message: str, key: str, 
                   min_interval: float = 60.0, deduplicate: bool = True):
    """Convenience function for throttled INFO logs."""
    throttled_log(logger, logging.INFO, message, key, min_interval, deduplicate)


def throttled_debug(logger: logging.Logger, message: str, key: str,
                    min_interval: float = 60.0, deduplicate: bool = True):
    """Convenience function for throttled DEBUG logs."""
    throttled_log(logger, logging.DEBUG, message, key, min_interval, deduplicate)


def throttled_warning(logger: logging.Logger, message: str, key: str,
                      min_interval: float = 300.0, deduplicate: bool = True):
    """Convenience function for throttled WARNING logs."""
    throttled_log(logger, logging.WARNING, message, key, min_interval, deduplicate)