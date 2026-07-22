"""
Multi-Timeframe Feature Engineerer.
Extracts technical features from M5, M15, H1 for LightGBM training.
"""
import pandas as pd
import numpy as np
import logging


class MTFFeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_all_features(self, df_m5: pd.DataFrame, df_m15: pd.DataFrame = None,
                              df_h1: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract and merge features from all timeframes.
        Returns a single DataFrame with time-aligned features.
        """
        # [FIX] Early validation to prevent downstream errors
        if df_m5 is None or df_m5.empty or 'time' not in df_m5.columns:
            return pd.DataFrame()
        
        m5_features = self._extract_tf_features(df_m5, 'm5_')
        if m5_features.empty:
            return pd.DataFrame()
        
        result = m5_features
        
        if df_m15 is not None and not df_m15.empty:
            m15_features = self._extract_tf_features(df_m15, 'm15_')
            if not m15_features.empty:
                result = self._merge_timeframes(result, m15_features)
        
        if df_h1 is not None and not df_h1.empty:
            h1_features = self._extract_tf_features(df_h1, 'h1_')
            if not h1_features.empty:
                result = self._merge_timeframes(result, h1_features)
        
        return result
    
    def _extract_tf_features(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Extract technical features from a single timeframe.
        [FIX] Added early validation and minimum length check.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            required_cols = ['time', 'close']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(f"[WARN] {prefix} missing column: {col}")
                    return pd.DataFrame()
            
            # [FIX] Check minimum length to prevent indicator warmup issues
            if len(df) < 50:
                return pd.DataFrame()
            
            df_work = df[required_cols].copy()
            
            # EMA features
            df_work[f'{prefix}ema_20'] = df_work['close'].ewm(span=20, adjust=False).mean()
            df_work[f'{prefix}ema_50'] = df_work['close'].ewm(span=50, adjust=False).mean()
            df_work[f'{prefix}trend'] = (df_work[f'{prefix}ema_20'] > df_work[f'{prefix}ema_50']).astype(int) * 2 - 1
            
            # ROC (Rate of Change)
            df_work[f'{prefix}roc_10'] = df_work['close'].pct_change(10) * 100
            
            # RSI
            df_work[f'{prefix}rsi_14'] = self._calculate_rsi(df_work['close'], 14)
            
            # MACD
            ema12 = df_work['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_work['close'].ewm(span=26, adjust=False).mean()
            df_work[f'{prefix}macd'] = ema12 - ema26
            df_work[f'{prefix}macd_signal'] = df_work[f'{prefix}macd'].ewm(span=9, adjust=False).mean()
            df_work[f'{prefix}macd_hist'] = df_work[f'{prefix}macd'] - df_work[f'{prefix}macd_signal']
            
            # Bollinger Bands
            sma20 = df_work['close'].rolling(20).mean()
            std20 = df_work['close'].rolling(20).std()
            df_work[f'{prefix}bb_upper'] = sma20 + 2 * std20
            df_work[f'{prefix}bb_lower'] = sma20 - 2 * std20
            df_work[f'{prefix}bb_width'] = (df_work[f'{prefix}bb_upper'] - df_work[f'{prefix}bb_lower']) / (sma20 + 1e-10)
            df_work[f'{prefix}bb_position'] = (df_work['close'] - df_work[f'{prefix}bb_lower']) / (df_work[f'{prefix}bb_upper'] - df_work[f'{prefix}bb_lower'] + 1e-10)
            
            # Drop original close column to reduce size
            df_work = df_work.drop(columns=['close'])
            
            if df_work.empty or 'time' not in df_work.columns:
                return pd.DataFrame()
            
            return df_work
            
        except Exception as e:
            self.logger.error(f"[FAIL] {prefix} feature extraction error: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _merge_timeframes(self, base_df: pd.DataFrame, higher_tf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge higher timeframe features into base dataframe using merge_asof.
        Aligns each base bar with the most recent completed higher timeframe bar.
        """
        if base_df.empty or higher_tf_df.empty:
            return base_df
        
        try:
            # Ensure time columns are datetime and sorted
            base_df = base_df.sort_values('time').copy()
            higher_tf_df = higher_tf_df.sort_values('time').copy()
            
            # Merge using merge_asof for backward-looking alignment
            merged = pd.merge_asof(
                base_df,
                higher_tf_df,
                on='time',
                direction='backward',
                suffixes=('', '_htf')
            )
            
            return merged
            
        except Exception as e:
            self.logger.error(f"[FAIL] Timeframe merge error: {e}")
            return base_df