"""
Session Volatility Manager - Micro-Account-Only Edition.

Manages trading session detection and session-specific volatility analysis.
Critical for Micro-Account trading where spread and liquidity vary significantly
across sessions.

Trading Sessions (UTC):
  1. LONDON_OPEN (07:00-09:00): High liquidity, tight spread, good for all strategies
  2. LONDON (09:00-12:00): Good liquidity, moderate spread
  3. NY_OPEN (12:00-15:00): High liquidity, tight spread, good for all strategies
  4. NY_MIDDAY (15:00-18:00): Moderate liquidity, wider spread
  5. ASIAN (18:00-06:00): Low liquidity, wide spread, scalp only
  6. TRANSITION (22:00-23:00): Very low liquidity, no trading recommended

Session Quality Scoring (0-100):
  - Liquidity: Higher = better
  - Spread: Lower = better
  - Volatility: Optimal range preferred
  - Historical Win Rate: Higher = better
"""
import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime, time
from typing import Dict, Optional, Tuple, List

from config import config


class SessionVolatilityManager:
    """
    Manages trading session detection and session-specific volatility analysis.
    
    Provides:
      - Current session detection with DST safety
      - Session quality scoring (0-100)
      - Expected spread per session
      - Volatility percentile per session
      - Trading recommendations per session
    """

    # Session definitions (UTC hours)
    SESSIONS = {
        'LONDON_OPEN': {'start': 7, 'end': 9, 'quality': 95, 'spread_mult': 1.0},
        'LONDON': {'start': 9, 'end': 12, 'quality': 85, 'spread_mult': 1.2},
        'NY_OPEN': {'start': 12, 'end': 15, 'quality': 95, 'spread_mult': 1.0},
        'NY_MIDDAY': {'start': 15, 'end': 18, 'quality': 70, 'spread_mult': 1.5},
        'ASIAN': {'start': 18, 'end': 6, 'quality': 40, 'spread_mult': 2.5},
        'TRANSITION': {'start': 22, 'end': 23, 'quality': 10, 'spread_mult': 4.0},
        'OTHER': {'start': 0, 'end': 0, 'quality': 50, 'spread_mult': 1.5}
    }

    # Optimal volatility range (ATR in USD for XAUUSD)
    OPTIMAL_VOLATILITY_MIN = 3.0
    OPTIMAL_VOLATILITY_MAX = 15.0

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize SessionVolatilityManager.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Timezone setup
        self.utc_tz = pytz.UTC
        self.ny_tz = pytz.timezone('America/New_York')
        self.bkk_tz = pytz.timezone('Asia/Bangkok')

        # Volatility cache per session
        self._session_volatility_cache: Dict[str, List[float]] = {
            'LONDON_OPEN': [],
            'LONDON': [],
            'NY_OPEN': [],
            'NY_MIDDAY': [],
            'ASIAN': [],
            'TRANSITION': [],
            'OTHER': []
        }

        self.logger.info("[SESSION] Initialized with DST-safe timezone handling")

    # =========================================================================
    # SESSION DETECTION
    # =========================================================================

    def get_current_session(self, timestamp: datetime = None) -> str:
        """
        Get current trading session based on UTC time.
        
        Handles DST transitions safely.
        
        Args:
            timestamp: Timestamp to check (defaults to now)
            
        Returns:
            Session name string
        """
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        elif timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = pytz.UTC.localize(timestamp)

        try:
            # Convert to UTC
            utc_time = timestamp.astimezone(self.utc_tz)
            hour = utc_time.hour

            # Check TRANSITION first (22:00-23:00)
            if self.SESSIONS['TRANSITION']['start'] <= hour < self.SESSIONS['TRANSITION']['end']:
                return 'TRANSITION'

            # Check LONDON_OPEN (07:00-09:00)
            if self.SESSIONS['LONDON_OPEN']['start'] <= hour < self.SESSIONS['LONDON_OPEN']['end']:
                return 'LONDON_OPEN'

            # Check LONDON (09:00-12:00)
            if self.SESSIONS['LONDON']['start'] <= hour < self.SESSIONS['LONDON']['end']:
                return 'LONDON'

            # Check NY_OPEN (12:00-15:00)
            if self.SESSIONS['NY_OPEN']['start'] <= hour < self.SESSIONS['NY_OPEN']['end']:
                return 'NY_OPEN'

            # Check NY_MIDDAY (15:00-18:00)
            if self.SESSIONS['NY_MIDDAY']['start'] <= hour < self.SESSIONS['NY_MIDDAY']['end']:
                return 'NY_MIDDAY'

            # Check ASIAN (18:00-06:00, wraps midnight)
            if hour >= self.SESSIONS['ASIAN']['start'] or hour < self.SESSIONS['ASIAN']['end']:
                return 'ASIAN'

            return 'OTHER'

        except Exception as e:
            self.logger.error(f"[SESSION] Error detecting session: {e}")
            return 'OTHER'

    def get_session_from_dataframe(self, df: pd.DataFrame) -> str:
        """
        Get session from DataFrame's last timestamp.
        
        Args:
            df: DataFrame with 'time' column
            
        Returns:
            Session name string
        """
        if df is None or df.empty or 'time' not in df.columns:
            return 'OTHER'

        try:
            last_time = df['time'].iloc[-1]

            if not isinstance(last_time, pd.Timestamp):
                last_time = pd.to_datetime(last_time)

            if last_time.tzinfo is None:
                # Handle DST spring forward
                last_time = last_time.tz_localize('UTC', nonexistent='shift_forward')

            return self.get_current_session(last_time)

        except Exception as e:
            self.logger.debug(f"[SESSION] Error getting session from df: {e}")
            return 'OTHER'

    # =========================================================================
    # SESSION QUALITY SCORING
    # =========================================================================

    def get_session_quality_score(self, session: str = None,
                                    df: pd.DataFrame = None) -> Dict:
        """
        Calculate session quality score (0-100).
        
        Factors:
          - Base quality (from SESSIONS config)
          - Current volatility vs optimal range
          - Historical performance (if available)
        
        Args:
            session: Session name (if None, detect from df or now)
            df: DataFrame for volatility calculation
            
        Returns:
            Dict with quality score and breakdown
        """
        # Determine session
        if session is None:
            if df is not None:
                session = self.get_session_from_dataframe(df)
            else:
                session = self.get_current_session()

        # Base quality from config
        base_quality = self.SESSIONS.get(session, {}).get('quality', 50)

        # Volatility adjustment
        volatility_score = 100.0
        volatility_adjustment = 0.0

        if df is not None and len(df) >= 20:
            try:
                atr = self._calculate_atr(df)
                if atr > 0:
                    # Record volatility for this session
                    self._record_session_volatility(session, atr)

                    # Score based on optimal range
                    if self.OPTIMAL_VOLATILITY_MIN <= atr <= self.OPTIMAL_VOLATILITY_MAX:
                        volatility_score = 100.0
                    elif atr < self.OPTIMAL_VOLATILITY_MIN:
                        # Too low volatility
                        volatility_score = max(50.0, 100.0 - (self.OPTIMAL_VOLATILITY_MIN - atr) * 10)
                    else:
                        # Too high volatility
                        volatility_score = max(50.0, 100.0 - (atr - self.OPTIMAL_VOLATILITY_MAX) * 5)

                    volatility_adjustment = (volatility_score - 100.0) * 0.2  # 20% weight

            except Exception as e:
                self.logger.debug(f"[SESSION] Volatility calculation error: {e}")

        # Final score
        final_score = max(0, min(100, base_quality + volatility_adjustment))

        return {
            'session': session,
            'quality_score': round(final_score, 1),
            'base_quality': base_quality,
            'volatility_score': round(volatility_score, 1),
            'volatility_adjustment': round(volatility_adjustment, 1),
            'is_prime_session': session in ['LONDON_OPEN', 'NY_OPEN'],
            'is_avoid_session': session in ['TRANSITION']
        }

    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================

    def calculate_session_volatility(self, df: pd.DataFrame,
                                      session: str = None) -> Dict:
        """
        Calculate volatility metrics for current session.
        
        Args:
            df: DataFrame with OHLCV data
            session: Session name (if None, detect from df)
            
        Returns:
            Dict with volatility metrics
        """
        if session is None:
            session = self.get_session_from_dataframe(df)

        result = {
            'session': session,
            'current_atr': 0.0,
            'session_avg_atr': 0.0,
            'volatility_percentile': 50.0,
            'is_high_volatility': False,
            'is_low_volatility': False
        }

        if df is None or len(df) < 20:
            return result

        try:
            # Current ATR
            current_atr = self._calculate_atr(df)
            result['current_atr'] = round(current_atr, 2)

            # Session average ATR (from cache)
            session_vols = self._session_volatility_cache.get(session, [])
            if session_vols:
                session_avg = np.mean(session_vols)
                result['session_avg_atr'] = round(session_avg, 2)

                # Calculate percentile
                if len(session_vols) >= 10:
                    percentile = np.sum(np.array(session_vols) < current_atr) / len(session_vols) * 100
                    result['volatility_percentile'] = round(percentile, 1)

            # Classification
            result['is_high_volatility'] = current_atr > self.OPTIMAL_VOLATILITY_MAX
            result['is_low_volatility'] = current_atr < self.OPTIMAL_VOLATILITY_MIN

        except Exception as e:
            self.logger.debug(f"[SESSION] Volatility calculation error: {e}")

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from DataFrame."""
        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            if len(high) < period + 1:
                return 0.0

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            atr = np.mean(tr[-period:])
            return float(atr)

        except Exception:
            return 0.0

    def _record_session_volatility(self, session: str, atr: float):
        """Record volatility for session (keep last 100 samples)."""
        if session not in self._session_volatility_cache:
            self._session_volatility_cache[session] = []

        cache = self._session_volatility_cache[session]
        cache.append(atr)

        # Keep only last 100 samples
        if len(cache) > 100:
            self._session_volatility_cache[session] = cache[-100:]

    # =========================================================================
    # SPREAD EXPECTATION
    # =========================================================================

    def get_expected_spread(self, session: str = None) -> Dict:
        """
        Get expected spread for current session.
        
        Args:
            session: Session name (if None, detect current)
            
        Returns:
            Dict with expected spread info
        """
        if session is None:
            session = self.get_current_session()

        spread_mult = self.SESSIONS.get(session, {}).get('spread_mult', 1.5)
        base_spread = config.max_spread_points

        expected_spread = base_spread * spread_mult

        return {
            'session': session,
            'expected_spread_points': round(expected_spread, 0),
            'spread_multiplier': spread_mult,
            'base_spread_points': base_spread,
            'is_wide_spread_session': spread_mult >= 2.0
        }

    # =========================================================================
    # TRADING RECOMMENDATION
    # =========================================================================

    def should_trade_this_session(self, session: str = None,
                                    df: pd.DataFrame = None,
                                    strategy_category: str = 'GENERAL') -> Dict:
        """
        Determine if current session is suitable for trading.
        
        Args:
            session: Session name (if None, detect from df or now)
            df: DataFrame for quality calculation
            strategy_category: Strategy category (SCALP, TREND, SMC, etc.)
            
        Returns:
            Dict with trading recommendation
        """
        # Get session quality
        quality_info = self.get_session_quality_score(session, df)
        session = quality_info['session']
        quality_score = quality_info['quality_score']

        # Get volatility info
        volatility_info = self.calculate_session_volatility(df, session)

        # Get spread info
        spread_info = self.get_expected_spread(session)

        # Determine recommendation
        should_trade = True
        reason = "Session suitable for trading"
        risk_level = "NORMAL"

        # Block TRANSITION session
        if session == 'TRANSITION':
            should_trade = False
            reason = "TRANSITION session: Very low liquidity, high spread"
            risk_level = "EXTREME"

        # Block low quality sessions for most strategies
        elif quality_score < 40 and strategy_category != 'SCALP':
            should_trade = False
            reason = f"Low session quality ({quality_score:.0f}) for {strategy_category}"
            risk_level = "HIGH"

        # Warn about wide spread sessions
        elif spread_info['is_wide_spread_session']:
            reason = f"Wide spread expected ({spread_info['expected_spread_points']:.0f} pts)"
            risk_level = "ELEVATED"

            # SCALP strategies should avoid wide spread
            if strategy_category == 'SCALP':
                should_trade = False
                reason += " - Not suitable for SCALP"

        # Warn about low volatility
        elif volatility_info['is_low_volatility'] and strategy_category in ['TREND', 'SMC']:
            reason = "Low volatility - trend strategies may underperform"
            risk_level = "MODERATE"

        # Warn about high volatility
        elif volatility_info['is_high_volatility'] and strategy_category == 'MEAN_REVERSION':
            reason = "High volatility - mean reversion may be risky"
            risk_level = "ELEVATED"

        return {
            'session': session,
            'should_trade': should_trade,
            'reason': reason,
            'risk_level': risk_level,
            'quality_score': quality_score,
            'volatility_percentile': volatility_info['volatility_percentile'],
            'expected_spread_points': spread_info['expected_spread_points'],
            'is_prime_session': quality_info['is_prime_session'],
            'strategy_category': strategy_category
        }

    # =========================================================================
    # SESSION HOURS & INFO
    # =========================================================================

    def get_session_hours(self, timezone: str = 'UTC') -> Dict:
        """
        Get session hours in specified timezone.
        
        Args:
            timezone: Target timezone ('UTC', 'NY', 'BKK')
            
        Returns:
            Dict with session hours
        """
        tz_map = {
            'UTC': self.utc_tz,
            'NY': self.ny_tz,
            'BKK': self.bkk_tz
        }
        target_tz = tz_map.get(timezone, self.utc_tz)

        sessions_info = {}
        for session_name, session_data in self.SESSIONS.items():
            if session_name == 'OTHER':
                continue

            start_utc = session_data['start']
            end_utc = session_data['end']

            # Convert to target timezone
            start_dt = datetime.now(self.utc_tz).replace(hour=start_utc, minute=0, second=0)
            end_dt = datetime.now(self.utc_tz).replace(hour=end_utc % 24, minute=0, second=0)

            start_local = start_dt.astimezone(target_tz)
            end_local = end_dt.astimezone(target_tz)

            sessions_info[session_name] = {
                'start_utc': start_utc,
                'end_utc': end_utc,
                'start_local': start_local.strftime('%H:%M'),
                'end_local': end_local.strftime('%H:%M'),
                'quality': session_data['quality'],
                'spread_multiplier': session_data['spread_mult']
            }

        return {
            'timezone': timezone,
            'sessions': sessions_info
        }

    def get_all_sessions_summary(self, df: pd.DataFrame = None) -> Dict:
        """
        Get summary of all sessions with current conditions.
        
        Args:
            df: DataFrame for volatility calculation
            
        Returns:
            Dict with all sessions summary
        """
        current_session = self.get_current_session()
        summary = {
            'current_session': current_session,
            'current_time_utc': datetime.now(pytz.UTC).strftime('%H:%M:%S'),
            'sessions': {}
        }

        for session_name in self.SESSIONS.keys():
            if session_name == 'OTHER':
                continue

            quality = self.get_session_quality_score(session_name, df)
            volatility = self.calculate_session_volatility(df, session_name)
            spread = self.get_expected_spread(session_name)

            summary['sessions'][session_name] = {
                'is_current': session_name == current_session,
                'quality_score': quality['quality_score'],
                'is_prime': quality['is_prime_session'],
                'is_avoid': quality['is_avoid_session'],
                'current_atr': volatility['current_atr'],
                'volatility_percentile': volatility['volatility_percentile'],
                'expected_spread_points': spread['expected_spread_points'],
                'spread_multiplier': spread['spread_multiplier']
            }

        return summary

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def format_session_log(self, df: pd.DataFrame = None) -> str:
        """
        Format a concise log string for current session status.
        
        Args:
            df: DataFrame for calculations
            
        Returns:
            Formatted log string
        """
        quality_info = self.get_session_quality_score(df=df)
        volatility_info = self.calculate_session_volatility(df)
        spread_info = self.get_expected_spread(quality_info['session'])

        session = quality_info['session']
        quality = quality_info['quality_score']
        atr = volatility_info['current_atr']
        spread = spread_info['expected_spread_points']
        is_prime = quality_info['is_prime_session']

        prime_flag = " [PRIME]" if is_prime else ""

        return (
            f"[SESSION] {session}{prime_flag} | "
            f"Quality: {quality:.0f}/100 | "
            f"ATR: {atr:.2f} USD | "
            f"Spread: {spread:.0f} pts"
        )

    def get_volatility_cache_stats(self) -> Dict:
        """
        Get statistics on volatility cache.
        
        Returns:
            Dict with cache statistics
        """
        stats = {}
        for session, vols in self._session_volatility_cache.items():
            if vols:
                stats[session] = {
                    'sample_count': len(vols),
                    'avg_atr': round(np.mean(vols), 2),
                    'min_atr': round(np.min(vols), 2),
                    'max_atr': round(np.max(vols), 2),
                    'std_atr': round(np.std(vols), 2)
                }
            else:
                stats[session] = {
                    'sample_count': 0,
                    'avg_atr': 0,
                    'min_atr': 0,
                    'max_atr': 0,
                    'std_atr': 0
                }

        return stats

SessionVolatilityEngine = SessionVolatilityManager