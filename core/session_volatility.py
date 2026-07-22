"""
Session Volatility Manager.
Identifies current trading session and volatility regime.
"""
import pandas as pd
import pytz
import logging


class SessionVolatilityManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ny_tz = pytz.timezone('America/New_York')
        self.utc_tz = pytz.utc

    def get_current_session(self, timestamp: pd.Timestamp) -> str:
        if timestamp is None: return 'OTHER'
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        # Use tzinfo instead of tz to prevent AttributeError
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(self.utc_tz)
        else:
            timestamp = timestamp.tz_convert(self.utc_tz)
        
        hour = timestamp.tz_convert(self.ny_tz).hour
        
        if 2 <= hour < 5: return 'LONDON_OPEN'
        elif 9 <= hour < 11: return 'NY_OPEN'
        elif 5 <= hour < 9: return 'LONDON'
        elif 11 <= hour < 17: return 'NY_MIDDAY'
        elif (19 <= hour <= 23) or (0 <= hour <= 1): return 'ASIAN'
        elif 17 <= hour < 19: return 'US_CLOSE'
        else: return 'OTHER'