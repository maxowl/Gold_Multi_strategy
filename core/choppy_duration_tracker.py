"""
core/choppy_duration_tracker.py
Tracks how long market has been choppy to prevent over-trading.
"""
import time
from typing import Dict
import logging


class ChoppyDurationTracker:
    """
    Tracks choppy duration to implement fatigue-based trading pause.
    """
    
    def __init__(self, max_consecutive_choppy_hours: int = 4):
        self.max_hours = max_consecutive_choppy_hours
        self.choppy_start_time = None
        self.consecutive_choppy_checks = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update(self, is_choppy: bool) -> Dict:
        """
        Update choppy duration tracking.
        
        Returns:
            {
                'is_fatigued': bool,
                'choppy_hours': float,
                'should_pause': bool,
                'pause_reason': str
            }
        """
        current_time = time.time()
        
        if is_choppy:
            if self.choppy_start_time is None:
                self.choppy_start_time = current_time
                self.logger.info("[CHOPPY] Choppy period started")
            
            self.consecutive_choppy_checks += 1
            
            choppy_hours = (current_time - self.choppy_start_time) / 3600
            
            is_fatigued = choppy_hours >= self.max_hours
            
            return {
                'is_fatigued': is_fatigued,
                'choppy_hours': choppy_hours,
                'should_pause': is_fatigued,
                'pause_reason': f'Choppy for {choppy_hours:.1f}h (max: {self.max_hours}h)' if is_fatigued else ''
            }
        
        else:
            if self.choppy_start_time is not None:
                choppy_hours = (current_time - self.choppy_start_time) / 3600
                self.logger.info(f"[CHOPPY] Choppy period ended after {choppy_hours:.1f}h")
            
            self.choppy_start_time = None
            self.consecutive_choppy_checks = 0
            
            return {
                'is_fatigued': False,
                'choppy_hours': 0,
                'should_pause': False,
                'pause_reason': ''
            }