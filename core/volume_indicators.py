"""
Volume Indicators Engine.

Provides comprehensive volume-based technical indicators:
  - True Money Flow (TMF)
  - Ease of Movement (EOM)
  - Chaikin Money Flow (CMF)
  - Volume Weighted Average Price (VWAP)
  - Price Volume Trend (PVT)
  - Accumulation/Distribution Line (ADL)

Used by:
  - S13_TMF_EOM (TMF + EOM strategy)
  - Volume-based strategies
  - Technical analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class VolumeIndicatorsEngine:
    """
    Volume Indicators Analysis engine.
    
    Features:
      - True Money Flow (TMF)
      - Ease of Movement (EOM)
      - Chaikin Money Flow (CMF)
      - Volume Weighted Average Price (VWAP)
      - Price Volume Trend (PVT)
      - Accumulation/Distribution Line (ADL)
    """

    def __init__(self):
        """Initialize VolumeIndicatorsEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default periods
        self.tmf_period = 21  # TMF lookback
        self.eom_period = 14  # EOM lookback
        self.cmf_period = 20  # CMF lookback
        self.pvt_smoothing = 5  # PVT smoothing period

    # =========================================================================
    # TRUE MONEY FLOW (TMF)
    # =========================================================================

    def calculate_tmf(
        self, df: pd.DataFrame, period: int = None
    ) -> Optional[Dict]:
        """
        Calculate True Money Flow (TMF).
        
        TMF measures the flow of money into and out of a security,
        considering the true range of price movement.
        
        Formula:
          TMF = ((Close - Low) - (High - Close)) / (High - Low) * Volume
        
        Args:
            df: DataFrame with OHLCV data
            period: TMF lookback period
            
        Returns:
            Dict with TMF data, or None on failure
        """
        if period is None:
            period = self.tmf_period

        if df is None or df.empty or len(df) < period + 5:
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

            n = len(close)
            tmf = np.zeros(n)

            for i in range(n):
                # Calculate true range components
                high_low = high[i] - low[i]

                if high_low > 0:
                    # Money flow multiplier
                    mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / high_low
                    tmf[i] = mf_multiplier * volume[i]
                else:
                    tmf[i] = 0.0

            # Calculate TMF sum over period
            tmf_sum = pd.Series(tmf).rolling(period, min_periods=1).sum().values
            tmf_sum = np.nan_to_num(tmf_sum, nan=0.0)

            # Calculate volume sum over period
            volume_sum = pd.Series(volume).rolling(period, min_periods=1).sum().values
            volume_sum = np.nan_to_num(volume_sum, nan=0.0)

            # Normalize TMF
            tmf_normalized = np.zeros(n)
            for i in range(n):
                if volume_sum[i] > 0:
                    tmf_normalized[i] = tmf_sum[i] / volume_sum[i]
                else:
                    tmf_normalized[i] = 0.0

            # Determine TMF trend
            current_tmf = tmf_normalized[-1]
            if current_tmf > 0.1:
                tmf_trend = 'STRONG_BULLISH'
            elif current_tmf > 0:
                tmf_trend = 'BULLISH'
            elif current_tmf < -0.1:
                tmf_trend = 'STRONG_BEARISH'
            elif current_tmf < 0:
                tmf_trend = 'BEARISH'
            else:
                tmf_trend = 'NEUTRAL'

            return {
                'tmf': tmf_normalized,
                'tmf_raw': tmf,
                'current_tmf': float(current_tmf),
                'tmf_trend': tmf_trend,
                'period': period
            }

        except Exception as e:
            self.logger.error(f"[VOLIND] TMF calculation error: {e}")
            return None

    # =========================================================================
    # EASE OF MOVEMENT (EOM)
    # =========================================================================

    def calculate_eom(
        self, df: pd.DataFrame, period: int = None, divisor: float = 100000000
    ) -> Optional[Dict]:
        """
        Calculate Ease of Movement (EOM).
        
        EOM measures the relationship between volume and price change,
        indicating how easily price can move.
        
        Formula:
          Distance Moved = ((High + Low)/2 - (Prev_High + Prev_Low)/2)
          Box Ratio = Volume / (High - Low)
          EOM = Distance Moved / Box Ratio
        
        Args:
            df: DataFrame with OHLCV data
            period: EOM lookback period
            divisor: Divisor for scaling
            
        Returns:
            Dict with EOM data, or None on failure
        """
        if period is None:
            period = self.eom_period

        if df is None or df.empty or len(df) < period + 5:
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

            n = len(close)
            eom = np.zeros(n)

            for i in range(1, n):
                # Distance moved (midpoint change)
                distance_moved = ((high[i] + low[i]) / 2) - ((high[i-1] + low[i-1]) / 2)

                # Box ratio
                high_low = high[i] - low[i]
                if high_low > 0:
                    box_ratio = volume[i] / high_low
                else:
                    box_ratio = 0

                # EOM
                if box_ratio > 0:
                    eom[i] = distance_moved / box_ratio * divisor
                else:
                    eom[i] = 0.0

            # Smooth EOM
            eom_smoothed = pd.Series(eom).rolling(period, min_periods=1).mean().values
            eom_smoothed = np.nan_to_num(eom_smoothed, nan=0.0)

            # Determine EOM trend
            current_eom = eom_smoothed[-1]
            if current_eom > 0:
                eom_trend = 'BULLISH'
            elif current_eom < 0:
                eom_trend = 'BEARISH'
            else:
                eom_trend = 'NEUTRAL'

            return {
                'eom': eom_smoothed,
                'eom_raw': eom,
                'current_eom': float(current_eom),
                'eom_trend': eom_trend,
                'period': period
            }

        except Exception as e:
            self.logger.error(f"[VOLIND] EOM calculation error: {e}")
            return None

    # =========================================================================
    # CHAIKIN MONEY FLOW (CMF)
    # =========================================================================

    def calculate_cmf(
        self, df: pd.DataFrame, period: int = None
    ) -> Optional[Dict]:
        """
        Calculate Chaikin Money Flow (CMF).
        
        CMF measures the amount of money flow volume over a specific period.
        
        Formula:
          Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
          Money Flow Volume = MFM * Volume
          CMF = Sum(MFV, period) / Sum(Volume, period)
        
        Args:
            df: DataFrame with OHLCV data
            period: CMF lookback period
            
        Returns:
            Dict with CMF data, or None on failure
        """
        if period is None:
            period = self.cmf_period

        if df is None or df.empty or len(df) < period + 5:
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

            n = len(close)
            mfv = np.zeros(n)  # Money Flow Volume

            for i in range(n):
                high_low = high[i] - low[i]

                if high_low > 0:
                    # Money Flow Multiplier
                    mfm = ((close[i] - low[i]) - (high[i] - close[i])) / high_low
                    mfv[i] = mfm * volume[i]
                else:
                    mfv[i] = 0.0

            # Calculate CMF
            mfv_sum = pd.Series(mfv).rolling(period, min_periods=1).sum().values
            volume_sum = pd.Series(volume).rolling(period, min_periods=1).sum().values

            mfv_sum = np.nan_to_num(mfv_sum, nan=0.0)
            volume_sum = np.nan_to_num(volume_sum, nan=0.0)

            cmf = np.zeros(n)
            for i in range(n):
                if volume_sum[i] > 0:
                    cmf[i] = mfv_sum[i] / volume_sum[i]
                else:
                    cmf[i] = 0.0

            # Determine CMF trend
            current_cmf = cmf[-1]
            if current_cmf > 0.05:
                cmf_trend = 'STRONG_BULLISH'
            elif current_cmf > 0:
                cmf_trend = 'BULLISH'
            elif current_cmf < -0.05:
                cmf_trend = 'STRONG_BEARISH'
            elif current_cmf < 0:
                cmf_trend = 'BEARISH'
            else:
                cmf_trend = 'NEUTRAL'

            return {
                'cmf': cmf,
                'mfv': mfv,
                'current_cmf': float(current_cmf),
                'cmf_trend': cmf_trend,
                'period': period
            }

        except Exception as e:
            self.logger.error(f"[VOLIND] CMF calculation error: {e}")
            return None

    # =========================================================================
    # VOLUME WEIGHTED AVERAGE PRICE (VWAP)
    # =========================================================================

    def calculate_vwap(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame with OHLCV data
            
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

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            # Calculate typical price
            typical_price = (high + low + close) / 3

            # Calculate cumulative VWAP
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

            # Calculate current deviation
            current_price = close[-1]
            current_vwap = vwap[-1]
            deviation = (current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0

            return {
                'vwap': vwap,
                'current_vwap': float(current_vwap),
                'current_price': float(current_price),
                'deviation_pct': float(deviation),
                'price_above_vwap': current_price > current_vwap
            }

        except Exception as e:
            self.logger.error(f"[VOLIND] VWAP calculation error: {e}")
            return None

    # =========================================================================
    # PRICE VOLUME TREND (PVT)
    # =========================================================================

    def calculate_pvt(
        self, df: pd.DataFrame, smoothing: int = None
    ) -> Optional[Dict]:
        """
        Calculate Price Volume Trend (PVT).
        
        PVT is similar to OBV but uses percentage price change.
        
        Formula:
          PVT = PVT_prev + (Price_Change_% * Volume)
        
        Args:
            df: DataFrame with OHLCV data
            smoothing: Smoothing period
            
        Returns:
            Dict with PVT data, or None on failure
        """
        if smoothing is None:
            smoothing = self.pvt_smoothing

        if df is None or df.empty or len(df) < 20:
            return None

        try:
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            n = len(close)
            pvt = np.zeros(n)

            for i in range(1, n):
                # Price change percentage
                if close[i-1] > 0:
                    price_change_pct = (close[i] - close[i-1]) / close[i-1]
                else:
                    price_change_pct = 0.0

                # PVT
                pvt[i] = pvt[i-1] + (price_change_pct * volume[i])

            # Smooth PVT
            pvt_smoothed = pd.Series(pvt).rolling(smoothing, min_periods=1).mean().values
            pvt_smoothed = np.nan_to_num(pvt_smoothed, nan=0.0)

            # Determine PVT trend
            current_pvt = pvt_smoothed[-1]
            prev_pvt = pvt_smoothed[-2] if len(pvt_smoothed) > 1 else 0

            if current_pvt > prev_pvt:
                pvt_trend = 'BULLISH'
            elif current_pvt < prev_pvt:
                pvt_trend = 'BEARISH'
            else:
                pvt_trend = 'NEUTRAL'

            return {
                'pvt': pvt_smoothed,
                'pvt_raw': pvt,
                'current_pvt': float(current_pvt),
                'pvt_trend': pvt_trend,
                'smoothing': smoothing
            }

        except Exception as e:
            self.logger.error(f"[VOLIND] PVT calculation error: {e}")
            return None

    # =========================================================================
    # ACCUMULATION/DISTRIBUTION LINE (ADL)
    # =========================================================================

    def calculate_adl(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate Accumulation/Distribution Line (ADL).
        
        ADL measures cumulative money flow volume.
        
        Formula:
          MFM = ((Close - Low) - (High - Close)) / (High - Low)
          ADL = ADL_prev + (MFM * Volume)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            ADL array, or None on failure
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

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            n = len(close)
            adl = np.zeros(n)

            for i in range(n):
                high_low = high[i] - low[i]

                if high_low > 0:
                    mfm = ((close[i] - low[i]) - (high[i] - close[i])) / high_low
                    adl[i] = adl[i-1] + (mfm * volume[i]) if i > 0 else mfm * volume[i]
                else:
                    adl[i] = adl[i-1] if i > 0 else 0.0

            return adl

        except Exception as e:
            self.logger.error(f"[VOLIND] ADL calculation error: {e}")
            return None

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

    def get_volume_indicators_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive volume indicators analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete volume indicators analysis
        """
        result = {
            'tmf': None,
            'eom': None,
            'cmf': None,
            'vwap': None,
            'pvt': None,
            'adl': None
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            # Calculate all indicators
            result['tmf'] = self.calculate_tmf(df)
            result['eom'] = self.calculate_eom(df)
            result['cmf'] = self.calculate_cmf(df)
            result['vwap'] = self.calculate_vwap(df)
            result['pvt'] = self.calculate_pvt(df)
            result['adl'] = self.calculate_adl(df)

            return result

        except Exception as e:
            self.logger.error(f"[VOLIND] Analysis error: {e}")
            return result

    def format_volume_indicators_log(self, analysis_result: Dict) -> str:
        """
        Format volume indicators analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_volume_indicators_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[VOLIND] Analysis failed"

        tmf = analysis_result.get('tmf', {})
        eom = analysis_result.get('eom', {})
        cmf = analysis_result.get('cmf', {})

        tmf_str = f"{tmf.get('current_tmf', 0):.3f}" if tmf else "N/A"
        eom_str = f"{eom.get('current_eom', 0):.2f}" if eom else "N/A"
        cmf_str = f"{cmf.get('current_cmf', 0):.3f}" if cmf else "N/A"

        return (
            f"[VOLIND] TMF: {tmf_str} | "
            f"EOM: {eom_str} | "
            f"CMF: {cmf_str}"
        )

    def get_volume_signal(self, analysis_result: Dict) -> Dict:
        """
        Get volume-based trading signal.
        
        Args:
            analysis_result: Result from get_volume_indicators_analysis
            
        Returns:
            Dict with volume signal
        """
        if analysis_result is None:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        tmf = analysis_result.get('tmf', {})
        eom = analysis_result.get('eom', {})
        cmf = analysis_result.get('cmf', {})

        tmf_trend = tmf.get('tmf_trend', 'NEUTRAL') if tmf else 'NEUTRAL'
        eom_trend = eom.get('eom_trend', 'NEUTRAL') if eom else 'NEUTRAL'
        cmf_trend = cmf.get('cmf_trend', 'NEUTRAL') if cmf else 'NEUTRAL'

        # Count bullish/bearish signals
        bullish_count = sum([
            tmf_trend in ['BULLISH', 'STRONG_BULLISH'],
            eom_trend == 'BULLISH',
            cmf_trend in ['BULLISH', 'STRONG_BULLISH']
        ])

        bearish_count = sum([
            tmf_trend in ['BEARISH', 'STRONG_BEARISH'],
            eom_trend == 'BEARISH',
            cmf_trend in ['BEARISH', 'STRONG_BEARISH']
        ])

        if bullish_count >= 2:
            return {
                'signal': 'VOLUME_BULLISH',
                'reason': f'{bullish_count}/3 volume indicators bullish',
                'strength': bullish_count / 3
            }
        elif bearish_count >= 2:
            return {
                'signal': 'VOLUME_BEARISH',
                'reason': f'{bearish_count}/3 volume indicators bearish',
                'strength': bearish_count / 3
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'reason': 'Mixed volume signals',
                'strength': 0.0
            }