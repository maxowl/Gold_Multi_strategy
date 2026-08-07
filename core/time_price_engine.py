"""
Time-Price Engine.

Provides time-based price analysis:
  - VWAP (Volume Weighted Average Price) calculation
  - Anchored VWAP
  - Session VWAP
  - Time-based price levels
  - VWAP deviation analysis

Used by:
  - S27_VWAP_MeanReversion (VWAP-based strategy)
  - Session-based trading
  - Time-weighted price analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time


class TimePriceEngine:
    """
    Time-Price Analysis engine.
    
    Features:
      - VWAP calculation
      - Anchored VWAP
      - Session VWAP
      - Time-based price levels
      - VWAP deviation analysis
    """

    def __init__(self):
        """Initialize TimePriceEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # VWAP parameters
        self.vwap_std_bands = [1.0, 2.0, 3.0]  # Standard deviation bands

        # Session definitions (UTC hours)
        self.sessions = {
            'ASIAN': (0, 7),      # 00:00 - 07:00 UTC
            'LONDON': (7, 12),    # 07:00 - 12:00 UTC
            'NY': (12, 21),       # 12:00 - 21:00 UTC
            'LATE': (21, 24)      # 21:00 - 24:00 UTC
        }

    # =========================================================================
    # VWAP CALCULATION
    # =========================================================================

    def calculate_vwap(
        self, df: pd.DataFrame, reset_daily: bool = False
    ) -> Optional[Dict]:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        VWAP = Sum(Price * Volume) / Sum(Volume)
        
        Args:
            df: DataFrame with OHLCV data
            reset_daily: Whether to reset VWAP daily
            
        Returns:
            Dict with VWAP data, or None on failure
        """
        if df is None or df.empty or len(df) < 10:
            return None

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Calculate typical price
            typical_price = (high + low + close) / 3

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            # Calculate VWAP
            if reset_daily and 'time' in df.columns:
                vwap = self._calculate_daily_vwap(typical_price, volume, df['time'])
            else:
                vwap = self._calculate_cumulative_vwap(typical_price, volume)

            if vwap is None:
                return None

            # Calculate VWAP standard deviation bands
            vwap_std = self._calculate_vwap_std(typical_price, volume, vwap)

            # Calculate current deviation
            current_price = close[-1]
            current_vwap = vwap[-1]
            deviation = (current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0

            return {
                'vwap': vwap,
                'vwap_std': vwap_std,
                'current_vwap': float(current_vwap),
                'current_price': float(current_price),
                'deviation_pct': float(deviation),
                'price_above_vwap': current_price > current_vwap,
                'bands': self._calculate_vwap_bands(vwap, vwap_std)
            }

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] VWAP calculation error: {e}")
            return None

    def _calculate_cumulative_vwap(
        self, typical_price: np.ndarray, volume: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculate cumulative VWAP."""
        try:
            n = len(typical_price)
            vwap = np.zeros(n)

            cum_price_volume = 0.0
            cum_volume = 0.0

            for i in range(n):
                cum_price_volume += typical_price[i] * volume[i]
                cum_volume += volume[i]

                if cum_volume > 0:
                    vwap[i] = cum_price_volume / cum_volume
                else:
                    vwap[i] = typical_price[i]

            return vwap

        except Exception:
            return None

    def _calculate_daily_vwap(
        self, typical_price: np.ndarray, volume: np.ndarray, time_series: pd.Series
    ) -> Optional[np.ndarray]:
        """Calculate daily-reset VWAP."""
        try:
            n = len(typical_price)
            vwap = np.zeros(n)

            cum_price_volume = 0.0
            cum_volume = 0.0
            current_date = None

            for i in range(n):
                # Get date from time
                try:
                    if isinstance(time_series.iloc[i], pd.Timestamp):
                        bar_date = time_series.iloc[i].date()
                    else:
                        bar_date = pd.to_datetime(time_series.iloc[i]).date()
                except Exception:
                    bar_date = None

                # Reset on new day
                if bar_date != current_date:
                    cum_price_volume = 0.0
                    cum_volume = 0.0
                    current_date = bar_date

                cum_price_volume += typical_price[i] * volume[i]
                cum_volume += volume[i]

                if cum_volume > 0:
                    vwap[i] = cum_price_volume / cum_volume
                else:
                    vwap[i] = typical_price[i]

            return vwap

        except Exception:
            return None

    def _calculate_vwap_std(
        self, typical_price: np.ndarray, volume: np.ndarray, vwap: np.ndarray
    ) -> np.ndarray:
        """Calculate VWAP standard deviation."""
        try:
            n = len(typical_price)
            vwap_std = np.zeros(n)

            cum_volume = 0.0
            cum_variance = 0.0

            for i in range(n):
                cum_volume += volume[i]

                if cum_volume > 0:
                    diff = typical_price[i] - vwap[i]
                    cum_variance += volume[i] * diff ** 2
                    vwap_std[i] = np.sqrt(cum_variance / cum_volume)
                else:
                    vwap_std[i] = 0.0

            return vwap_std

        except Exception:
            return np.zeros_like(vwap)

    def _calculate_vwap_bands(self, vwap: np.ndarray, vwap_std: np.ndarray) -> Dict:
        """Calculate VWAP bands."""
        bands = {}

        for std_mult in self.vwap_std_bands:
            bands[f'upper_{std_mult}'] = vwap + std_mult * vwap_std
            bands[f'lower_{std_mult}'] = vwap - std_mult * vwap_std

        return bands

    # =========================================================================
    # ANCHORED VWAP
    # =========================================================================

    def calculate_anchored_vwap(
        self, df: pd.DataFrame, anchor_index: int
    ) -> Optional[Dict]:
        """
        Calculate Anchored VWAP from a specific point.
        
        Args:
            df: DataFrame with OHLCV data
            anchor_index: Index to anchor VWAP from
            
        Returns:
            Dict with anchored VWAP data, or None on failure
        """
        if df is None or df.empty or len(df) < anchor_index + 10:
            return None

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            # Calculate typical price
            typical_price = (high + low + close) / 3

            # Calculate anchored VWAP (from anchor_index onwards)
            n = len(typical_price)
            anchored_vwap = np.zeros(n)

            cum_price_volume = 0.0
            cum_volume = 0.0

            for i in range(anchor_index, n):
                cum_price_volume += typical_price[i] * volume[i]
                cum_volume += volume[i]

                if cum_volume > 0:
                    anchored_vwap[i] = cum_price_volume / cum_volume
                else:
                    anchored_vwap[i] = typical_price[i]

            # Copy values before anchor
            anchored_vwap[:anchor_index] = anchored_vwap[anchor_index]

            # Calculate current deviation
            current_price = close[-1]
            current_vwap = anchored_vwap[-1]
            deviation = (current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0

            return {
                'anchored_vwap': anchored_vwap,
                'anchor_index': anchor_index,
                'anchor_price': float(close[anchor_index]),
                'current_vwap': float(current_vwap),
                'current_price': float(current_price),
                'deviation_pct': float(deviation),
                'price_above_vwap': current_price > current_vwap
            }

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] Anchored VWAP error: {e}")
            return None

    # =========================================================================
    # SESSION VWAP
    # =========================================================================

    def calculate_session_vwap(
        self, df: pd.DataFrame, session: str = 'LONDON'
    ) -> Optional[Dict]:
        """
        Calculate VWAP for a specific trading session.
        
        Args:
            df: DataFrame with OHLCV data
            session: Session name ('ASIAN', 'LONDON', 'NY', 'LATE')
            
        Returns:
            Dict with session VWAP data, or None on failure
        """
        if df is None or df.empty or len(df) < 10:
            return None

        if session not in self.sessions:
            return None

        try:
            session_start, session_end = self.sessions[session]

            # Filter data for session
            if 'time' not in df.columns:
                return None

            # Get session data
            session_mask = []
            for t in df['time']:
                try:
                    if isinstance(t, pd.Timestamp):
                        hour = t.hour
                    else:
                        hour = pd.to_datetime(t).hour

                    session_mask.append(session_start <= hour < session_end)
                except Exception:
                    session_mask.append(False)

            session_df = df[session_mask]

            if len(session_df) < 5:
                return None

            # Calculate VWAP for session
            return self.calculate_vwap(session_df, reset_daily=False)

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] Session VWAP error: {e}")
            return None

    # =========================================================================
    # TIME-BASED PRICE LEVELS
    # =========================================================================

    def calculate_time_levels(
        self, df: pd.DataFrame, level_type: str = 'session'
    ) -> Dict:
        """
        Calculate time-based price levels.
        
        Args:
            df: DataFrame with OHLCV data
            level_type: Type of levels ('session', 'daily', 'weekly')
            
        Returns:
            Dict with time-based levels
        """
        if df is None or df.empty or len(df) < 10:
            return {}

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            levels = {}

            if level_type == 'session':
                # Session high/low/close
                levels['session_high'] = float(np.max(high[-20:]))
                levels['session_low'] = float(np.min(low[-20:]))
                levels['session_close'] = float(close[-1])

            elif level_type == 'daily':
                # Daily high/low/close
                levels['daily_high'] = float(np.max(high[-50:]))
                levels['daily_low'] = float(np.min(low[-50:]))
                levels['daily_close'] = float(close[-1])

            elif level_type == 'weekly':
                # Weekly high/low/close
                levels['weekly_high'] = float(np.max(high[-200:]))
                levels['weekly_low'] = float(np.min(low[-200:]))
                levels['weekly_close'] = float(close[-1])

            # Calculate pivot points
            if levels:
                h = levels.get('session_high', levels.get('daily_high', np.max(high)))
                l = levels.get('session_low', levels.get('daily_low', np.min(low)))
                c = levels.get('session_close', levels.get('daily_close', close[-1]))

                # Classic pivot points
                pivot = (h + l + c) / 3
                levels['pivot'] = float(pivot)
                levels['r1'] = float(2 * pivot - l)
                levels['s1'] = float(2 * pivot - h)
                levels['r2'] = float(pivot + (h - l))
                levels['s2'] = float(pivot - (h - l))

            return levels

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] Time levels error: {e}")
            return {}

    # =========================================================================
    # VWAP DEVIATION
    # =========================================================================

    def calculate_vwap_deviation(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> Dict:
        """
        Calculate deviation from VWAP.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with deviation analysis
        """
        if df is None or df.empty or len(df) < lookback:
            return {
                'deviation_pct': 0.0,
                'deviation_atr': 0.0,
                'is_extended': False
            }

        try:
            # Calculate VWAP
            vwap_result = self.calculate_vwap(df)

            if vwap_result is None:
                return {
                    'deviation_pct': 0.0,
                    'deviation_atr': 0.0,
                    'is_extended': False
                }

            current_price = vwap_result['current_price']
            current_vwap = vwap_result['current_vwap']
            deviation_pct = vwap_result['deviation_pct']

            # Calculate ATR for normalization
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # ATR
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Deviation in ATR units
            deviation_atr = (current_price - current_vwap) / atr if atr > 0 else 0

            # Determine if price is extended
            is_extended = abs(deviation_atr) > 2.0

            return {
                'deviation_pct': float(deviation_pct),
                'deviation_atr': float(deviation_atr),
                'is_extended': is_extended,
                'current_price': float(current_price),
                'current_vwap': float(current_vwap),
                'atr': float(atr)
            }

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] Deviation calculation error: {e}")
            return {
                'deviation_pct': 0.0,
                'deviation_atr': 0.0,
                'is_extended': False
            }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_volume(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get volume array from DataFrame."""
        try:
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
            else:
                return None

            return np.nan_to_num(volume, nan=1.0)
        except Exception:
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_time_price_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive time-price analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete time-price analysis
        """
        result = {
            'vwap': None,
            'deviation': None,
            'levels': None,
            'session_vwap': None
        }

        if df is None or df.empty or len(df) < 20:
            return result

        try:
            # Calculate VWAP
            result['vwap'] = self.calculate_vwap(df)

            # Calculate deviation
            result['deviation'] = self.calculate_vwap_deviation(df)

            # Calculate time levels
            result['levels'] = self.calculate_time_levels(df, 'session')

            # Calculate session VWAP
            result['session_vwap'] = self.calculate_session_vwap(df, 'LONDON')

            return result

        except Exception as e:
            self.logger.error(f"[TIMEPRICE] Analysis error: {e}")
            return result

    def format_time_price_log(self, analysis_result: Dict) -> str:
        """
        Format time-price analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_time_price_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[TIMEPRICE] Analysis failed"

        vwap = analysis_result.get('vwap', {})
        deviation = analysis_result.get('deviation', {})

        vwap_str = f"{vwap.get('current_vwap', 0):.2f}" if vwap else "N/A"
        deviation_str = f"{deviation.get('deviation_pct', 0):.2f}%" if deviation else "N/A"
        extended_str = "YES" if deviation.get('is_extended', False) else "NO"

        return (
            f"[TIMEPRICE] VWAP: {vwap_str} | "
            f"Deviation: {deviation_str} | "
            f"Extended: {extended_str}"
        )

    def is_vwap_reversion_signal(self, analysis_result: Dict, threshold: float = 1.5) -> Dict:
        """
        Check for VWAP mean reversion signal.
        
        Args:
            analysis_result: Result from get_time_price_analysis
            threshold: Deviation threshold for signal
            
        Returns:
            Dict with reversion signal
        """
        if analysis_result is None:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        deviation = analysis_result.get('deviation', {})
        deviation_atr = deviation.get('deviation_atr', 0)

        if deviation_atr > threshold:
            return {
                'signal': 'SELL_REVERSION',
                'reason': f'Price {deviation_atr:.2f} ATR above VWAP',
                'strength': min(1.0, deviation_atr / (threshold * 2))
            }
        elif deviation_atr < -threshold:
            return {
                'signal': 'BUY_REVERSION',
                'reason': f'Price {abs(deviation_atr):.2f} ATR below VWAP',
                'strength': min(1.0, abs(deviation_atr) / (threshold * 2))
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'reason': 'Price near VWAP',
                'strength': 0.0
            }