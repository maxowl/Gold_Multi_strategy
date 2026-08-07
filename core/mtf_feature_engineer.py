"""
Multi-Timeframe Feature Engineerer.

Extracts technical features from multiple timeframes (M1, M5, M15, H1)
for LightGBM regime prediction and machine learning models.

Features Extracted:
  - Trend indicators (EMA, RSI, MACD, ADX)
  - Volatility indicators (ATR, Bollinger Bands, StdDev)
  - Momentum indicators (ROC, Stochastic)
  - Volume features (Volume Z-score, OBV)
  - Statistical features (Skewness, Kurtosis, Hurst)
  - Price action features (Candle patterns, Range)

Used by:
  - HybridMTFPredictor (LightGBM regime prediction)
  - Machine learning model training
  - Feature-based regime classification
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class MTFFeatureEngineer:
    """
    Multi-Timeframe Feature Engineering engine.
    
    Features:
      - Multi-timeframe feature extraction
      - Technical indicator calculation
      - Statistical feature computation
      - Feature normalization
      - Timeframe merging with alignment
    """

    def __init__(self):
        """Initialize MTFFeatureEngineer."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Feature configuration
        self.ema_periods = [10, 20, 50]
        self.rsi_period = 14
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.atr_period = 14
        self.bb_params = {'period': 20, 'std': 2.0}
        self.adx_period = 14

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def extract_all_features(
        self, df_m15: pd.DataFrame, df_m5: pd.DataFrame = None,
        df_h1: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Extract and merge features from all timeframes.
        
        Args:
            df_m15: M15 DataFrame (primary)
            df_m5: M5 DataFrame (optional)
            df_h1: H1 DataFrame (optional)
            
        Returns:
            DataFrame with merged features
        """
        if df_m15 is None or df_m15.empty:
            self.logger.warning("[MTF] Primary M15 data not available")
            return pd.DataFrame()

        try:
            # Extract features from each timeframe
            m15_features = self._extract_tf_features(df_m15, prefix='m15_')
            
            result = m15_features

            # Merge M5 features if available
            if df_m5 is not None and not df_m5.empty:
                m5_features = self._extract_tf_features(df_m5, prefix='m5_')
                if not m5_features.empty:
                    result = self._merge_timeframes(result, m5_features)

            # Merge H1 features if available
            if df_h1 is not None and not df_h1.empty:
                h1_features = self._extract_tf_features(df_h1, prefix='h1_')
                if not h1_features.empty:
                    result = self._merge_timeframes(result, h1_features)

            return result

        except Exception as e:
            self.logger.error(f"[MTF] Feature extraction error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # TIMEFRAME FEATURE EXTRACTION
    # =========================================================================

    def _extract_tf_features(self, df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """
        Extract features from a single timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            prefix: Prefix for feature names
            
        Returns:
            DataFrame with extracted features
        """
        if df is None or df.empty or len(df) < 50:
            return pd.DataFrame()

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            open_ = df['open'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))
            open_ = np.nan_to_num(open_, nan=np.nanmean(open_))

            # Get volume if available
            volume = None
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
                volume = np.nan_to_num(volume, nan=1.0)
            elif 'volume' in df.columns:
                volume = df['volume'].values.astype(float)
                volume = np.nan_to_num(volume, nan=1.0)

            features = {}

            # =========================================================================
            # TREND FEATURES
            # =========================================================================
            trend_features = self._calculate_trend_features(close, high, low, prefix)
            features.update(trend_features)

            # =========================================================================
            # VOLATILITY FEATURES
            # =========================================================================
            vol_features = self._calculate_volatility_features(close, high, low, prefix)
            features.update(vol_features)

            # =========================================================================
            # MOMENTUM FEATURES
            # =========================================================================
            momentum_features = self._calculate_momentum_features(close, high, low, prefix)
            features.update(momentum_features)

            # =========================================================================
            # VOLUME FEATURES
            # =========================================================================
            if volume is not None:
                volume_features = self._calculate_volume_features(close, volume, prefix)
                features.update(volume_features)

            # =========================================================================
            # STATISTICAL FEATURES
            # =========================================================================
            stat_features = self._calculate_statistical_features(close, prefix)
            features.update(stat_features)

            # =========================================================================
            # PRICE ACTION FEATURES
            # =========================================================================
            pa_features = self._calculate_price_action_features(open_, high, low, close, prefix)
            features.update(pa_features)

            # Create DataFrame
            result = pd.DataFrame(features)

            # Add time column if available
            if 'time' in df.columns:
                result['time'] = df['time'].values

            return result

        except Exception as e:
            self.logger.error(f"[MTF] TF feature extraction error: {e}")
            return pd.DataFrame()

    # =========================================================================
    # TREND FEATURES
    # =========================================================================

    def _calculate_trend_features(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate trend-related features."""
        features = {}

        try:
            # EMA features
            for period in self.ema_periods:
                ema = self._calculate_ema(close, period)
                features[f'{prefix}ema_{period}'] = ema
                features[f'{prefix}ema_{period}_ratio'] = close / (ema + 1e-10) - 1

            # RSI
            rsi = self._calculate_rsi(close, self.rsi_period)
            features[f'{prefix}rsi'] = rsi

            # MACD
            macd, signal, histogram = self._calculate_macd(close)
            features[f'{prefix}macd'] = macd
            features[f'{prefix}macd_signal'] = signal
            features[f'{prefix}macd_histogram'] = histogram

            # ADX
            adx = self._calculate_adx(high, low, close, self.adx_period)
            features[f'{prefix}adx'] = adx

            # Trend direction
            ema_20 = self._calculate_ema(close, 20)
            ema_50 = self._calculate_ema(close, 50)
            features[f'{prefix}trend'] = np.where(ema_20 > ema_50, 1, -1)

        except Exception as e:
            self.logger.debug(f"[MTF] Trend features error: {e}")

        return features

    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================

    def _calculate_volatility_features(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate volatility-related features."""
        features = {}

        try:
            # ATR
            atr = self._calculate_atr(high, low, close, self.atr_period)
            features[f'{prefix}atr'] = atr
            features[f'{prefix}atr_pct'] = atr / (close + 1e-10) * 100

            # Bollinger Bands
            bb_upper, bb_lower, bb_width = self._calculate_bollinger_bands(
                close, self.bb_params['period'], self.bb_params['std']
            )
            features[f'{prefix}bb_upper'] = bb_upper
            features[f'{prefix}bb_lower'] = bb_lower
            features[f'{prefix}bb_width'] = bb_width
            features[f'{prefix}bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

            # Standard deviation
            std_20 = pd.Series(close).rolling(20).std().values
            std_20 = np.nan_to_num(std_20, nan=0.0)
            features[f'{prefix}std_20'] = std_20
            features[f'{prefix}std_20_pct'] = std_20 / (close + 1e-10) * 100

            # Range
            range_pct = (high - low) / (close + 1e-10) * 100
            features[f'{prefix}range_pct'] = range_pct

        except Exception as e:
            self.logger.debug(f"[MTF] Volatility features error: {e}")

        return features

    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================

    def _calculate_momentum_features(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate momentum-related features."""
        features = {}

        try:
            # ROC (Rate of Change)
            roc_10 = self._calculate_roc(close, 10)
            roc_20 = self._calculate_roc(close, 20)
            features[f'{prefix}roc_10'] = roc_10
            features[f'{prefix}roc_20'] = roc_20

            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            features[f'{prefix}stoch_k'] = stoch_k
            features[f'{prefix}stoch_d'] = stoch_d

            # Momentum
            momentum = close - np.roll(close, 10)
            momentum[0:10] = 0
            features[f'{prefix}momentum'] = momentum

        except Exception as e:
            self.logger.debug(f"[MTF] Momentum features error: {e}")

        return features

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================

    def _calculate_volume_features(
        self, close: np.ndarray, volume: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate volume-related features."""
        features = {}

        try:
            # Volume Z-score
            vol_mean = pd.Series(volume).rolling(20).mean().values
            vol_std = pd.Series(volume).rolling(20).std().values
            vol_mean = np.nan_to_num(vol_mean, nan=np.mean(volume))
            vol_std = np.nan_to_num(vol_std, nan=1.0)

            vol_zscore = (volume - vol_mean) / (vol_std + 1e-10)
            features[f'{prefix}volume_zscore'] = vol_zscore

            # OBV (On Balance Volume)
            obv = self._calculate_obv(close, volume)
            features[f'{prefix}obv'] = obv

            # Volume ratio
            avg_volume = pd.Series(volume).rolling(20).mean().values
            avg_volume = np.nan_to_num(avg_volume, nan=1.0)
            features[f'{prefix}volume_ratio'] = volume / (avg_volume + 1e-10)

        except Exception as e:
            self.logger.debug(f"[MTF] Volume features error: {e}")

        return features

    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================

    def _calculate_statistical_features(
        self, close: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate statistical features."""
        features = {}

        try:
            # Returns
            returns = np.diff(close) / (close[:-1] + 1e-10)
            returns = np.insert(returns, 0, 0)

            # Skewness (20-bar)
            skew_20 = pd.Series(returns).rolling(20).skew().values
            skew_20 = np.nan_to_num(skew_20, nan=0.0)
            features[f'{prefix}skew_20'] = skew_20

            # Kurtosis (20-bar)
            kurt_20 = pd.Series(returns).rolling(20).apply(
                lambda x: pd.Series(x).kurtosis(), raw=False
            ).values
            kurt_20 = np.nan_to_num(kurt_20, nan=0.0)
            features[f'{prefix}kurt_20'] = kurt_20

            # Volatility of volatility (vol of vol)
            vol_20 = pd.Series(returns).rolling(20).std().values
            vol_of_vol = pd.Series(vol_20).rolling(20).std().values
            vol_of_vol = np.nan_to_num(vol_of_vol, nan=0.0)
            features[f'{prefix}vol_of_vol'] = vol_of_vol

        except Exception as e:
            self.logger.debug(f"[MTF] Statistical features error: {e}")

        return features

    # =========================================================================
    # PRICE ACTION FEATURES
    # =========================================================================

    def _calculate_price_action_features(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
        close: np.ndarray, prefix: str
    ) -> Dict:
        """Calculate price action features."""
        features = {}

        try:
            # Candle body
            body = close - open_
            body_pct = body / (open_ + 1e-10) * 100
            features[f'{prefix}body_pct'] = body_pct

            # Upper and lower wicks
            upper_wick = high - np.maximum(open_, close)
            lower_wick = np.minimum(open_, close) - low
            range_ = high - low + 1e-10

            features[f'{prefix}upper_wick_pct'] = upper_wick / range_ * 100
            features[f'{prefix}lower_wick_pct'] = lower_wick / range_ * 100

            # Candle direction
            features[f'{prefix}candle_direction'] = np.where(close > open_, 1, -1)

            # Gap
            gap = open_ - np.roll(close, 1)
            gap[0] = 0
            features[f'{prefix}gap'] = gap

        except Exception as e:
            self.logger.debug(f"[MTF] Price action features error: {e}")

        return features

    # =========================================================================
    # TIMEFRAME MERGING
    # =========================================================================

    def _merge_timeframes(self, primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        """
        Merge features from different timeframes.
        
        Uses forward fill to align different timeframe lengths.
        
        Args:
            primary: Primary timeframe features
            secondary: Secondary timeframe features
            
        Returns:
            Merged DataFrame
        """
        try:
            if primary.empty:
                return secondary
            if secondary.empty:
                return primary

            # Merge on index
            result = primary.copy()

            # Add secondary columns with forward fill
            for col in secondary.columns:
                if col not in result.columns and col != 'time':
                    result[col] = secondary[col].reindex(result.index, method='ffill')

            return result

        except Exception as e:
            self.logger.error(f"[MTF] Timeframe merge error: {e}")
            return primary

    # =========================================================================
    # HELPER METHODS - TECHNICAL INDICATORS
    # =========================================================================

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        try:
            ema = pd.Series(data).ewm(span=period, adjust=False).mean().values
            return np.nan_to_num(ema, nan=np.nanmean(data))
        except Exception:
            return np.zeros_like(data)

    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        try:
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Prepend first value
            rsi = np.insert(rsi, 0, 50)

            return np.nan_to_num(rsi, nan=50.0)
        except Exception:
            return np.full_like(data, 50.0)

    def _calculate_macd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, signal, and histogram."""
        try:
            fast = self._calculate_ema(data, self.macd_params['fast'])
            slow = self._calculate_ema(data, self.macd_params['slow'])

            macd = fast - slow
            signal = self._calculate_ema(macd, self.macd_params['signal'])
            histogram = macd - signal

            return macd, signal, histogram
        except Exception:
            zeros = np.zeros_like(data)
            return zeros, zeros, zeros

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average Directional Index."""
        try:
            n = len(high)

            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            tr = np.insert(tr, 0, high[0] - low[0])

            # Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            plus_dm = np.insert(plus_dm, 0, 0)
            minus_dm = np.insert(minus_dm, 0, 0)

            # Smoothed values
            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
            plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)

            # DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values

            return np.nan_to_num(adx, nan=25.0)
        except Exception:
            return np.full_like(close, 25.0)

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range."""
        try:
            n = len(high)

            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            tr = np.insert(tr, 0, high[0] - low[0])

            atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

            return np.nan_to_num(atr, nan=np.nanmean(tr))
        except Exception:
            return np.zeros_like(close)

    def _calculate_bollinger_bands(self, data: np.ndarray, period: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        try:
            sma = pd.Series(data).rolling(period).mean().values
            std = pd.Series(data).rolling(period).std().values

            sma = np.nan_to_num(sma, nan=np.nanmean(data))
            std = np.nan_to_num(std, nan=0.0)

            upper = sma + std_mult * std
            lower = sma - std_mult * std
            width = (upper - lower) / (sma + 1e-10)

            return upper, lower, width
        except Exception:
            mid = np.mean(data)
            return np.full_like(data, mid), np.full_like(data, mid), np.zeros_like(data)

    def _calculate_roc(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Rate of Change."""
        try:
            roc = (data - np.roll(data, period)) / (np.roll(data, period) + 1e-10) * 100
            roc[:period] = 0
            return roc
        except Exception:
            return np.zeros_like(data)

    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator."""
        try:
            period = 14

            lowest_low = pd.Series(low).rolling(period).min().values
            highest_high = pd.Series(high).rolling(period).max().values

            lowest_low = np.nan_to_num(lowest_low, nan=np.min(low))
            highest_high = np.nan_to_num(highest_high, nan=np.max(high))

            stoch_k = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
            stoch_d = pd.Series(stoch_k).rolling(3).mean().values

            stoch_k = np.nan_to_num(stoch_k, nan=50.0)
            stoch_d = np.nan_to_num(stoch_d, nan=50.0)

            return stoch_k, stoch_d
        except Exception:
            return np.full_like(close, 50.0), np.full_like(close, 50.0)

    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On Balance Volume."""
        try:
            direction = np.where(close > np.roll(close, 1), 1, -1)
            direction[0] = 0

            obv = np.cumsum(direction * volume)

            return obv
        except Exception:
            return np.zeros_like(close)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_feature_names(self, prefix: str = '') -> List[str]:
        """
        Get list of all feature names.
        
        Args:
            prefix: Prefix for feature names
            
        Returns:
            List of feature names
        """
        features = []

        # Trend features
        for period in self.ema_periods:
            features.append(f'{prefix}ema_{period}')
            features.append(f'{prefix}ema_{period}_ratio')
        features.append(f'{prefix}rsi')
        features.append(f'{prefix}macd')
        features.append(f'{prefix}macd_signal')
        features.append(f'{prefix}macd_histogram')
        features.append(f'{prefix}adx')
        features.append(f'{prefix}trend')

        # Volatility features
        features.append(f'{prefix}atr')
        features.append(f'{prefix}atr_pct')
        features.append(f'{prefix}bb_upper')
        features.append(f'{prefix}bb_lower')
        features.append(f'{prefix}bb_width')
        features.append(f'{prefix}bb_position')
        features.append(f'{prefix}std_20')
        features.append(f'{prefix}std_20_pct')
        features.append(f'{prefix}range_pct')

        # Momentum features
        features.append(f'{prefix}roc_10')
        features.append(f'{prefix}roc_20')
        features.append(f'{prefix}stoch_k')
        features.append(f'{prefix}stoch_d')
        features.append(f'{prefix}momentum')

        # Volume features
        features.append(f'{prefix}volume_zscore')
        features.append(f'{prefix}obv')
        features.append(f'{prefix}volume_ratio')

        # Statistical features
        features.append(f'{prefix}skew_20')
        features.append(f'{prefix}kurt_20')
        features.append(f'{prefix}vol_of_vol')

        # Price action features
        features.append(f'{prefix}body_pct')
        features.append(f'{prefix}upper_wick_pct')
        features.append(f'{prefix}lower_wick_pct')
        features.append(f'{prefix}candle_direction')
        features.append(f'{prefix}gap')

        return features

    def format_mtf_log(self, features: pd.DataFrame) -> str:
        """
        Format MTF features as concise log string.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Formatted log string
        """
        if features.empty:
            return "[MTF] No features extracted"

        num_features = len(features.columns)
        num_rows = len(features)

        return f"[MTF] Extracted {num_features} features from {num_rows} bars"