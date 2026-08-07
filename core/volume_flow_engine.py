"""
Volume Flow Engine.

Provides volume flow analysis for accumulation/distribution detection:
  - Volume Flow Indicator (VFI)
  - Accumulation/Distribution detection
  - Volume trend analysis
  - Money Flow Index (MFI)
  - On Balance Volume (OBV)

Used by:
  - S20_VFIAccumulation (VFI-based accumulation strategy)
  - Volume-based analysis
  - Smart money detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class VolumeFlowEngine:
    """
    Volume Flow Analysis engine.
    
    Features:
      - Volume Flow Indicator (VFI)
      - Accumulation detection
      - Distribution detection
      - Volume trend analysis
      - Money Flow Index (MFI)
      - On Balance Volume (OBV)
    """

    def __init__(self):
        """Initialize VolumeFlowEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # VFI parameters
        self.vfi_period = 130  # VFI lookback period
        self.vfi_ma_period = 5  # VFI moving average period

        # Accumulation/Distribution parameters
        self.ad_lookback = 50  # Lookback for A/D detection
        self.ad_threshold = 0.6  # Threshold for A/D detection

    # =========================================================================
    # VOLUME FLOW INDICATOR (VFI)
    # =========================================================================

    def calculate_vfi(
        self, df: pd.DataFrame, period: int = None, ma_period: int = None
    ) -> Optional[Dict]:
        """
        Calculate Volume Flow Indicator (VFI).
        
        VFI measures the flow of volume into and out of a security.
        Positive VFI = volume flowing in (accumulation)
        Negative VFI = volume flowing out (distribution)
        
        Args:
            df: DataFrame with OHLCV data
            period: VFI lookback period
            ma_period: Moving average period for VFI
            
        Returns:
            Dict with VFI data, or None on failure
        """
        if period is None:
            period = self.vfi_period
        if ma_period is None:
            ma_period = self.vfi_ma_period

        if df is None or df.empty or len(df) < period + 10:
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
            vfi = np.zeros(n)

            # Calculate typical price
            typical_price = (high + low + close) / 3

            # Calculate inter-bar price change
            price_change = np.diff(typical_price)
            price_change = np.insert(price_change, 0, 0)

            # Calculate VFI
            for i in range(1, n):
                # Volume direction based on price change
                if price_change[i] > 0:
                    volume_direction = volume[i]
                elif price_change[i] < 0:
                    volume_direction = -volume[i]
                else:
                    volume_direction = 0

                # Accumulate VFI
                if i >= period:
                    vfi[i] = vfi[i-1] + volume_direction - volume_direction * (period / n)
                else:
                    vfi[i] = vfi[i-1] + volume_direction

            # Normalize VFI by average volume
            avg_volume = np.mean(volume[-period:])
            if avg_volume > 0:
                vfi_normalized = vfi / avg_volume
            else:
                vfi_normalized = vfi

            # Calculate VFI moving average
            vfi_ma = pd.Series(vfi_normalized).rolling(ma_period, min_periods=1).mean().values
            vfi_ma = np.nan_to_num(vfi_ma, nan=0.0)

            # Determine VFI trend
            current_vfi = vfi_normalized[-1]
            current_ma = vfi_ma[-1]

            if current_vfi > current_ma:
                vfi_trend = 'BULLISH'
            elif current_vfi < current_ma:
                vfi_trend = 'BEARISH'
            else:
                vfi_trend = 'NEUTRAL'

            return {
                'vfi': vfi_normalized,
                'vfi_ma': vfi_ma,
                'current_vfi': float(current_vfi),
                'current_ma': float(current_ma),
                'vfi_trend': vfi_trend,
                'vfi_above_ma': current_vfi > current_ma
            }

        except Exception as e:
            self.logger.error(f"[VOLFLOW] VFI calculation error: {e}")
            return None

    # =========================================================================
    # ACCUMULATION DETECTION
    # =========================================================================

    def detect_accumulation(
        self, df: pd.DataFrame, lookback: int = None
    ) -> Dict:
        """
        Detect accumulation patterns.
        
        Accumulation occurs when:
          - Price is stable or declining
          - Volume is increasing
          - VFI is positive
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with accumulation analysis
        """
        if lookback is None:
            lookback = self.ad_lookback

        if df is None or df.empty or len(df) < lookback:
            return {
                'is_accumulating': False,
                'accumulation_strength': 0.0,
                'reason': 'Insufficient data'
            }

        try:
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return {
                    'is_accumulating': False,
                    'accumulation_strength': 0.0,
                    'reason': 'No volume data'
                }

            # Use lookback window
            recent_close = close[-lookback:]
            recent_volume = volume[-lookback:]

            # Calculate price trend
            price_change = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100

            # Calculate volume trend
            avg_volume_first_half = np.mean(recent_volume[:lookback//2])
            avg_volume_second_half = np.mean(recent_volume[lookback//2:])

            if avg_volume_first_half > 0:
                volume_trend = avg_volume_second_half / avg_volume_first_half
            else:
                volume_trend = 1.0

            # Calculate VFI
            vfi_result = self.calculate_vfi(df, period=lookback)

            if vfi_result is None:
                return {
                    'is_accumulating': False,
                    'accumulation_strength': 0.0,
                    'reason': 'VFI calculation failed'
                }

            vfi_trend = vfi_result['vfi_trend']
            current_vfi = vfi_result['current_vfi']

            # Determine accumulation
            # Accumulation: price stable/down + volume up + VFI positive
            is_accumulating = (
                price_change < 1.0 and  # Price not rising much
                volume_trend > 1.1 and  # Volume increasing
                current_vfi > 0  # Positive VFI
            )

            # Calculate accumulation strength
            if is_accumulating:
                strength = min(1.0, (volume_trend - 1.0) * 0.5 + abs(current_vfi) * 0.1)
            else:
                strength = 0.0

            return {
                'is_accumulating': is_accumulating,
                'accumulation_strength': float(strength),
                'price_change_pct': float(price_change),
                'volume_trend': float(volume_trend),
                'vfi_trend': vfi_trend,
                'current_vfi': float(current_vfi),
                'reason': self._get_accumulation_reason(is_accumulating, price_change, volume_trend, current_vfi)
            }

        except Exception as e:
            self.logger.error(f"[VOLFLOW] Accumulation detection error: {e}")
            return {
                'is_accumulating': False,
                'accumulation_strength': 0.0,
                'reason': f'Error: {str(e)}'
            }

    # =========================================================================
    # DISTRIBUTION DETECTION
    # =========================================================================

    def detect_distribution(
        self, df: pd.DataFrame, lookback: int = None
    ) -> Dict:
        """
        Detect distribution patterns.
        
        Distribution occurs when:
          - Price is stable or rising
          - Volume is increasing
          - VFI is negative
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Lookback period
            
        Returns:
            Dict with distribution analysis
        """
        if lookback is None:
            lookback = self.ad_lookback

        if df is None or df.empty or len(df) < lookback:
            return {
                'is_distributing': False,
                'distribution_strength': 0.0,
                'reason': 'Insufficient data'
            }

        try:
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return {
                    'is_distributing': False,
                    'distribution_strength': 0.0,
                    'reason': 'No volume data'
                }

            # Use lookback window
            recent_close = close[-lookback:]
            recent_volume = volume[-lookback:]

            # Calculate price trend
            price_change = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100

            # Calculate volume trend
            avg_volume_first_half = np.mean(recent_volume[:lookback//2])
            avg_volume_second_half = np.mean(recent_volume[lookback//2:])

            if avg_volume_first_half > 0:
                volume_trend = avg_volume_second_half / avg_volume_first_half
            else:
                volume_trend = 1.0

            # Calculate VFI
            vfi_result = self.calculate_vfi(df, period=lookback)

            if vfi_result is None:
                return {
                    'is_distributing': False,
                    'distribution_strength': 0.0,
                    'reason': 'VFI calculation failed'
                }

            current_vfi = vfi_result['current_vfi']

            # Determine distribution
            # Distribution: price stable/up + volume up + VFI negative
            is_distributing = (
                price_change > -1.0 and  # Price not falling much
                volume_trend > 1.1 and  # Volume increasing
                current_vfi < 0  # Negative VFI
            )

            # Calculate distribution strength
            if is_distributing:
                strength = min(1.0, (volume_trend - 1.0) * 0.5 + abs(current_vfi) * 0.1)
            else:
                strength = 0.0

            return {
                'is_distributing': is_distributing,
                'distribution_strength': float(strength),
                'price_change_pct': float(price_change),
                'volume_trend': float(volume_trend),
                'current_vfi': float(current_vfi),
                'reason': self._get_distribution_reason(is_distributing, price_change, volume_trend, current_vfi)
            }

        except Exception as e:
            self.logger.error(f"[VOLFLOW] Distribution detection error: {e}")
            return {
                'is_distributing': False,
                'distribution_strength': 0.0,
                'reason': f'Error: {str(e)}'
            }

    # =========================================================================
    # VOLUME TREND
    # =========================================================================

    def calculate_volume_trend(
        self, df: pd.DataFrame, period: int = 20
    ) -> Dict:
        """
        Calculate volume trend.
        
        Args:
            df: DataFrame with OHLCV data
            period: Volume trend period
            
        Returns:
            Dict with volume trend analysis
        """
        if df is None or df.empty or len(df) < period + 5:
            return {
                'volume_trend': 'UNKNOWN',
                'current_volume': 0.0,
                'average_volume': 0.0,
                'volume_ratio': 1.0
            }

        try:
            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return {
                    'volume_trend': 'UNKNOWN',
                    'current_volume': 0.0,
                    'average_volume': 0.0,
                    'volume_ratio': 1.0
                }

            current_volume = volume[-1]
            average_volume = np.mean(volume[-period:])

            if average_volume > 0:
                volume_ratio = current_volume / average_volume
            else:
                volume_ratio = 1.0

            # Determine trend
            if volume_ratio > 1.5:
                volume_trend = 'HIGH'
            elif volume_ratio > 1.0:
                volume_trend = 'ABOVE_AVERAGE'
            elif volume_ratio > 0.5:
                volume_trend = 'BELOW_AVERAGE'
            else:
                volume_trend = 'LOW'

            return {
                'volume_trend': volume_trend,
                'current_volume': float(current_volume),
                'average_volume': float(average_volume),
                'volume_ratio': float(volume_ratio)
            }

        except Exception as e:
            self.logger.error(f"[VOLFLOW] Volume trend error: {e}")
            return {
                'volume_trend': 'UNKNOWN',
                'current_volume': 0.0,
                'average_volume': 0.0,
                'volume_ratio': 1.0
            }

    # =========================================================================
    # MONEY FLOW INDEX (MFI)
    # =========================================================================

    def calculate_mfi(
        self, df: pd.DataFrame, period: int = 14
    ) -> Optional[np.ndarray]:
        """
        Calculate Money Flow Index (MFI).
        
        MFI is a volume-weighted RSI.
        
        Args:
            df: DataFrame with OHLCV data
            period: MFI period
            
        Returns:
            MFI array, or None on failure
        """
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

            # Calculate typical price
            typical_price = (high + low + close) / 3

            # Calculate money flow
            money_flow = typical_price * volume

            # Calculate positive and negative money flow
            n = len(typical_price)
            positive_flow = np.zeros(n)
            negative_flow = np.zeros(n)

            for i in range(1, n):
                if typical_price[i] > typical_price[i-1]:
                    positive_flow[i] = money_flow[i]
                elif typical_price[i] < typical_price[i-1]:
                    negative_flow[i] = money_flow[i]

            # Calculate money ratio
            positive_sum = pd.Series(positive_flow).rolling(period, min_periods=1).sum().values
            negative_sum = pd.Series(negative_flow).rolling(period, min_periods=1).sum().values

            positive_sum = np.nan_to_num(positive_sum, nan=0.0)
            negative_sum = np.nan_to_num(negative_sum, nan=0.0)

            # Calculate MFI
            mfi = np.zeros(n)
            for i in range(period, n):
                if negative_sum[i] > 0:
                    money_ratio = positive_sum[i] / negative_sum[i]
                    mfi[i] = 100 - (100 / (1 + money_ratio))
                else:
                    mfi[i] = 100.0

            return mfi

        except Exception as e:
            self.logger.error(f"[VOLFLOW] MFI calculation error: {e}")
            return None

    # =========================================================================
    # ON BALANCE VOLUME (OBV)
    # =========================================================================

    def calculate_obv(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate On Balance Volume (OBV).
        
        OBV adds volume on up days, subtracts on down days.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            OBV array, or None on failure
        """
        if df is None or df.empty or len(df) < 10:
            return None

        try:
            close = df['close'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Get volume
            volume = self._get_volume(df)
            if volume is None:
                return None

            # Calculate OBV
            direction = np.where(close > np.roll(close, 1), 1, -1)
            direction[0] = 0

            obv = np.cumsum(direction * volume)

            return obv

        except Exception as e:
            self.logger.error(f"[VOLFLOW] OBV calculation error: {e}")
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

    def _get_accumulation_reason(
        self, is_accumulating: bool, price_change: float,
        volume_trend: float, vfi: float
    ) -> str:
        """Get reason for accumulation detection."""
        if is_accumulating:
            return f"Accumulation: price {price_change:.1f}%, volume trend {volume_trend:.2f}x, VFI {vfi:.2f}"
        else:
            return "No accumulation detected"

    def _get_distribution_reason(
        self, is_distributing: bool, price_change: float,
        volume_trend: float, vfi: float
    ) -> str:
        """Get reason for distribution detection."""
        if is_distributing:
            return f"Distribution: price {price_change:.1f}%, volume trend {volume_trend:.2f}x, VFI {vfi:.2f}"
        else:
            return "No distribution detected"

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_volume_flow_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive volume flow analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with complete volume flow analysis
        """
        result = {
            'vfi': None,
            'accumulation': None,
            'distribution': None,
            'volume_trend': None,
            'mfi': None,
            'obv': None
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            # Calculate VFI
            result['vfi'] = self.calculate_vfi(df)

            # Detect accumulation
            result['accumulation'] = self.detect_accumulation(df)

            # Detect distribution
            result['distribution'] = self.detect_distribution(df)

            # Calculate volume trend
            result['volume_trend'] = self.calculate_volume_trend(df)

            # Calculate MFI
            result['mfi'] = self.calculate_mfi(df)

            # Calculate OBV
            result['obv'] = self.calculate_obv(df)

            return result

        except Exception as e:
            self.logger.error(f"[VOLFLOW] Analysis error: {e}")
            return result

    def format_volume_flow_log(self, analysis_result: Dict) -> str:
        """
        Format volume flow analysis result as concise log string.
        
        Args:
            analysis_result: Result from get_volume_flow_analysis
            
        Returns:
            Formatted log string
        """
        if analysis_result is None:
            return "[VOLFLOW] Analysis failed"

        vfi = analysis_result.get('vfi', {})
        accumulation = analysis_result.get('accumulation', {})
        distribution = analysis_result.get('distribution', {})

        vfi_str = f"{vfi.get('current_vfi', 0):.2f}" if vfi else "N/A"
        vfi_trend = vfi.get('vfi_trend', 'UNKNOWN') if vfi else 'UNKNOWN'
        acc_str = "YES" if accumulation.get('is_accumulating', False) else "NO"
        dist_str = "YES" if distribution.get('is_distributing', False) else "NO"

        return (
            f"[VOLFLOW] VFI: {vfi_str} ({vfi_trend}) | "
            f"Accumulation: {acc_str} | "
            f"Distribution: {dist_str}"
        )

    def get_smart_money_signal(self, analysis_result: Dict) -> Dict:
        """
        Get smart money signal based on volume flow analysis.
        
        Args:
            analysis_result: Result from get_volume_flow_analysis
            
        Returns:
            Dict with smart money signal
        """
        if analysis_result is None:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        accumulation = analysis_result.get('accumulation', {})
        distribution = analysis_result.get('distribution', {})
        vfi = analysis_result.get('vfi', {})

        is_accumulating = accumulation.get('is_accumulating', False)
        is_distributing = distribution.get('is_distributing', False)
        vfi_trend = vfi.get('vfi_trend', 'NEUTRAL') if vfi else 'NEUTRAL'

        if is_accumulating and vfi_trend == 'BULLISH':
            return {
                'signal': 'SMART_MONEY_BUY',
                'reason': 'Accumulation with bullish VFI',
                'strength': accumulation.get('accumulation_strength', 0.5)
            }
        elif is_distributing and vfi_trend == 'BEARISH':
            return {
                'signal': 'SMART_MONEY_SELL',
                'reason': 'Distribution with bearish VFI',
                'strength': distribution.get('distribution_strength', 0.5)
            }
        elif is_accumulating:
            return {
                'signal': 'WEAK_BUY',
                'reason': 'Accumulation detected',
                'strength': accumulation.get('accumulation_strength', 0.3) * 0.5
            }
        elif is_distributing:
            return {
                'signal': 'WEAK_SELL',
                'reason': 'Distribution detected',
                'strength': distribution.get('distribution_strength', 0.3) * 0.5
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'reason': 'No smart money activity',
                'strength': 0.0
            }