"""
Principal Component Analysis Engine.
Extracts cyclical components and dominant features from price data.
Used by S12_PCA_Cycle strategy.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict


class PCAEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_principal_components(self, df: pd.DataFrame, n_components: int = 3, lookback: int = 100) -> Optional[Dict]:
        """
        Extract principal components from OHLCV data using SVD.
        
        Returns dict with:
        - 'components': list of pd.Series (each is a principal component)
        - 'explained_variance': list of floats (variance ratio per component)
        - 'singular_values': list of floats
        """
        if df is None or len(df) < lookback:
            return None
        
        try:
            features = pd.DataFrame()
            
            # Feature 1: Returns
            features['returns'] = df['close'].pct_change()
            
            # Feature 2: Log returns
            features['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            
            # Feature 3: High-Low range normalized
            features['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
            
            # Feature 4: Close position within bar
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # Feature 5-6: Volume features (if available)
            if 'tick_volume' in df.columns:
                features['volume_change'] = df['tick_volume'].pct_change()
                features['volume_ma_ratio'] = df['tick_volume'] / (df['tick_volume'].rolling(20).mean() + 1e-10)
            
            # Feature 7: Rolling mean ratio [FIX] Added 1e-10 to prevent division by zero
            features['rolling_mean_20'] = df['close'].rolling(20).mean() / (df['close'] + 1e-10)
            
            # Feature 8: Rolling std ratio
            features['rolling_std_20'] = df['close'].rolling(20).std() / (df['close'] + 1e-10)
            
            # Take only the lookback window and drop NaNs
            features = features.tail(lookback).fillna(0)
            if features.empty or len(features) < 50:
                return None
            
            X = features.to_numpy().astype(float)
            
            # Standardize features
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            std = np.std(X, axis=0) + 1e-10
            X_standardized = X_centered / std
            
            # SVD decomposition with fallback
            try:
                U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)
            except np.linalg.LinAlgError:
                self.logger.warning("[WARN] SVD failed, falling back to eigen decomposition")
                cov = np.cov(X_standardized.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                S = np.sqrt(np.maximum(eigenvalues[::-1], 0))
                Vt = eigenvectors[:, ::-1].T
                U = (X_standardized @ eigenvectors[:, ::-1]) / (S + 1e-10)
            
            # Limit to requested number of components
            n_components = min(n_components, len(S))
            components = []
            explained_variance = []
            
            total_variance = np.sum(S ** 2)
            
            for i in range(n_components):
                component = U[:, i]
                components.append(pd.Series(component, index=features.index))
                explained_variance.append(float(S[i] ** 2 / total_variance) if total_variance > 0 else 0.0)
            
            return {
                'components': components,
                'explained_variance': explained_variance,
                'singular_values': S[:n_components].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"[FAIL] PCA calculation error: {e}")
            return None