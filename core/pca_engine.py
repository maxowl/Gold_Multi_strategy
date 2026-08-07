"""
PCA (Principal Component Analysis) Engine.

Provides dimensionality reduction and cycle detection using PCA.
Extracts dominant patterns from multi-dimensional market data.

PCA Applications:
  - Market cycle detection
  - Noise reduction
  - Feature extraction
  - Pattern recognition

Used by:
  - S12_PCA_Cycle (PCA-based cycle detection)
  - Market regime analysis
  - Multi-asset correlation analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple


class PCAEngine:
    """
    Principal Component Analysis engine.
    
    Features:
      - PCA transformation
      - Market cycle detection
      - Principal component analysis
      - Variance explained calculation
      - Signal reconstruction
    """

    def __init__(self):
        """Initialize PCAEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # PCA parameters
        self.n_components = 5  # Number of components to extract
        self.cycle_lookback = 100  # Lookback for cycle detection

    # =========================================================================
    # PCA TRANSFORMATION
    # =========================================================================

    def apply_pca(
        self, data: np.ndarray, n_components: int = None, normalize: bool = True
    ) -> Optional[Dict]:
        """
        Apply PCA transformation to data matrix.
        
        Args:
            data: 2D array (samples x features)
            n_components: Number of components to extract
            normalize: Whether to normalize data
            
        Returns:
            Dict with PCA results, or None on failure
        """
        if n_components is None:
            n_components = self.n_components

        if data is None or len(data) < n_components + 10:
            return None

        try:
            # Handle NaN
            data = np.nan_to_num(data, nan=0.0)

            # Ensure 2D
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            n_samples, n_features = data.shape

            # Limit components
            n_components = min(n_components, n_features, n_samples - 1)

            if n_components < 1:
                return None

            # Normalize if requested
            if normalize:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std == 0] = 1  # Avoid division by zero
                data_normalized = (data - mean) / std
            else:
                mean = np.zeros(n_features)
                std = np.ones(n_features)
                data_normalized = data.copy()

            # Center data
            data_centered = data_normalized - np.mean(data_normalized, axis=0)

            # Calculate covariance matrix
            covariance = np.cov(data_centered.T)

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)

            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top n_components
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]

            # Handle negative eigenvalues (numerical errors)
            eigenvalues = np.maximum(eigenvalues, 0)

            # Transform data
            transformed = data_centered @ eigenvectors

            # Calculate variance explained
            total_variance = np.sum(eigenvalues)
            if total_variance > 0:
                variance_explained = eigenvalues / total_variance
            else:
                variance_explained = np.zeros(n_components)

            return {
                'transformed': transformed,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'variance_explained': variance_explained,
                'cumulative_variance': np.cumsum(variance_explained),
                'mean': mean,
                'std': std,
                'n_components': n_components
            }

        except Exception as e:
            self.logger.error(f"[PCA] Transformation error: {e}")
            return None

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def detect_cycle(
        self, series: np.ndarray, lookback: int = None
    ) -> Dict:
        """
        Detect market cycles using PCA.
        
        Extracts dominant cyclical patterns from price series.
        
        Args:
            series: Price series
            lookback: Number of bars to analyze
            
        Returns:
            Dict with cycle analysis
        """
        if lookback is None:
            lookback = self.cycle_lookback

        if series is None or len(series) < lookback:
            return {
                'cycle_detected': False,
                'cycle_period': None,
                'cycle_phase': None,
                'cycle_strength': 0.0
            }

        try:
            # Handle NaN
            series = np.nan_to_num(series, nan=np.nanmean(series))

            # Detrend series
            detrended = self._detrend(series[-lookback:])

            # Create lagged matrix for PCA
            lag_matrix = self._create_lag_matrix(detrended, lags=5)

            if lag_matrix is None or len(lag_matrix) < 10:
                return {
                    'cycle_detected': False,
                    'cycle_period': None,
                    'cycle_phase': None,
                    'cycle_strength': 0.0
                }

            # Apply PCA
            pca_result = self.apply_pca(lag_matrix, n_components=3, normalize=True)

            if pca_result is None:
                return {
                    'cycle_detected': False,
                    'cycle_period': None,
                    'cycle_phase': None,
                    'cycle_strength': 0.0
                }

            # Analyze first principal component (dominant pattern)
            pc1 = pca_result['transformed'][:, 0]
            variance_pc1 = pca_result['variance_explained'][0]

            # Detect cycle using autocorrelation
            cycle_period = self._detect_cycle_period(pc1)
            cycle_phase = self._calculate_phase(pc1)

            # Calculate cycle strength
            cycle_strength = min(1.0, variance_pc1 * 2)  # Scale variance

            # Determine if cycle is significant
            cycle_detected = variance_pc1 > 0.3 and cycle_period is not None

            return {
                'cycle_detected': cycle_detected,
                'cycle_period': cycle_period,
                'cycle_phase': cycle_phase,
                'cycle_strength': float(cycle_strength),
                'variance_pc1': float(variance_pc1),
                'pc1': pc1
            }

        except Exception as e:
            self.logger.error(f"[PCA] Cycle detection error: {e}")
            return {
                'cycle_detected': False,
                'cycle_period': None,
                'cycle_phase': None,
                'cycle_strength': 0.0
            }

    # =========================================================================
    # COMPONENT ANALYSIS
    # =========================================================================

    def analyze_components(
        self, data: np.ndarray, n_components: int = None
    ) -> Dict:
        """
        Analyze principal components of data.
        
        Args:
            data: 2D array (samples x features)
            n_components: Number of components to analyze
            
        Returns:
            Dict with component analysis
        """
        if n_components is None:
            n_components = self.n_components

        pca_result = self.apply_pca(data, n_components)

        if pca_result is None:
            return {
                'components': [],
                'variance_explained': [],
                'dominant_component': None
            }

        try:
            components = []
            for i in range(pca_result['n_components']):
                components.append({
                    'component_id': i + 1,
                    'eigenvalue': float(pca_result['eigenvalues'][i]),
                    'variance_explained': float(pca_result['variance_explained'][i]),
                    'cumulative_variance': float(pca_result['cumulative_variance'][i]),
                    'loadings': pca_result['eigenvectors'][:, i].tolist()
                })

            # Find dominant component
            dominant_idx = np.argmax(pca_result['variance_explained'])

            return {
                'components': components,
                'variance_explained': pca_result['variance_explained'].tolist(),
                'cumulative_variance': pca_result['cumulative_variance'].tolist(),
                'dominant_component': dominant_idx + 1,
                'dominant_variance': float(pca_result['variance_explained'][dominant_idx])
            }

        except Exception as e:
            self.logger.error(f"[PCA] Component analysis error: {e}")
            return {
                'components': [],
                'variance_explained': [],
                'dominant_component': None
            }

    # =========================================================================
    # VARIANCE EXPLAINED
    # =========================================================================

    def calculate_variance_explained(
        self, data: np.ndarray, n_components: int = None
    ) -> Optional[np.ndarray]:
        """
        Calculate variance explained by each principal component.
        
        Args:
            data: 2D array (samples x features)
            n_components: Number of components
            
        Returns:
            Array of variance explained, or None on failure
        """
        if n_components is None:
            n_components = self.n_components

        pca_result = self.apply_pca(data, n_components)

        if pca_result is None:
            return None

        return pca_result['variance_explained']

    # =========================================================================
    # SIGNAL RECONSTRUCTION
    # =========================================================================

    def reconstruct_signal(
        self, data: np.ndarray, n_components: int = None, normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Reconstruct signal from principal components (denoising).
        
        Args:
            data: 2D array (samples x features)
            n_components: Number of components to use for reconstruction
            normalize: Whether data was normalized
            
        Returns:
            Reconstructed signal, or None on failure
        """
        if n_components is None:
            n_components = self.n_components

        pca_result = self.apply_pca(data, n_components, normalize)

        if pca_result is None:
            return None

        try:
            transformed = pca_result['transformed']
            eigenvectors = pca_result['eigenvectors']
            mean = pca_result['mean']
            std = pca_result['std']

            # Reconstruct
            reconstructed = transformed @ eigenvectors.T

            # Denormalize if needed
            if normalize:
                reconstructed = reconstructed * std + mean

            return reconstructed

        except Exception as e:
            self.logger.error(f"[PCA] Reconstruction error: {e}")
            return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _detrend(self, series: np.ndarray) -> np.ndarray:
        """Remove linear trend from series."""
        try:
            n = len(series)
            x = np.arange(n)
            slope, intercept = np.polyfit(x, series, 1)
            trend = slope * x + intercept
            return series - trend
        except Exception:
            return series

    def _create_lag_matrix(self, series: np.ndarray, lags: int = 5) -> Optional[np.ndarray]:
        """Create lagged matrix for time series analysis."""
        try:
            n = len(series)
            if n < lags + 10:
                return None

            # Create matrix with lagged values
            lag_matrix = np.zeros((n - lags, lags + 1))
            for i in range(lags + 1):
                lag_matrix[:, i] = series[lags - i:n - i]

            return lag_matrix

        except Exception:
            return None

    def _detect_cycle_period(self, series: np.ndarray) -> Optional[int]:
        """Detect dominant cycle period using autocorrelation."""
        try:
            n = len(series)
            if n < 20:
                return None

            # Calculate autocorrelation
            autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
            autocorr = autocorr[n-1:]  # Positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find first peak after minimum lag
            min_lag = 5
            max_lag = min(n // 2, 50)

            best_lag = None
            best_corr = 0

            for lag in range(min_lag, max_lag):
                if autocorr[lag] > best_corr and autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                    best_corr = autocorr[lag]
                    best_lag = lag

            return best_lag

        except Exception:
            return None

    def _calculate_phase(self, series: np.ndarray) -> Optional[float]:
        """Calculate current phase in cycle (0-2*pi)."""
        try:
            n = len(series)
            if n < 10:
                return None

            # Use Hilbert transform approximation
            # Simplified: use position in recent cycle
            recent = series[-20:]
            mean_val = np.mean(recent)
            std_val = np.std(recent)

            if std_val == 0:
                return 0.0

            # Normalize current position
            normalized = (series[-1] - mean_val) / std_val

            # Convert to phase (approximation)
            phase = np.arctan2(normalized, 1)

            return float(phase)

        except Exception:
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_pca_summary(self, data: np.ndarray) -> Dict:
        """
        Get comprehensive PCA summary.
        
        Args:
            data: 2D array (samples x features)
            
        Returns:
            Dict with complete PCA analysis
        """
        result = {
            'pca_result': None,
            'components': None,
            'variance_explained': None,
            'cycle_analysis': None
        }

        if data is None or len(data) < 20:
            return result

        try:
            # Apply PCA
            result['pca_result'] = self.apply_pca(data)

            # Analyze components
            result['components'] = self.analyze_components(data)

            # Calculate variance explained
            result['variance_explained'] = self.calculate_variance_explained(data)

            # Detect cycles (if 1D)
            if data.ndim == 1 or data.shape[1] == 1:
                series = data.flatten()
                result['cycle_analysis'] = self.detect_cycle(series)

            return result

        except Exception as e:
            self.logger.error(f"[PCA] Summary error: {e}")
            return result

    def format_pca_log(self, pca_result: Dict) -> str:
        """
        Format PCA result as concise log string.
        
        Args:
            pca_result: Result from apply_pca
            
        Returns:
            Formatted log string
        """
        if pca_result is None:
            return "[PCA] Analysis failed"

        n_components = pca_result.get('n_components', 0)
        variance = pca_result.get('variance_explained', [])

        if len(variance) == 0:
            return "[PCA] No variance data"

        total_variance = np.sum(variance)
        dominant_variance = variance[0] if len(variance) > 0 else 0

        return (
            f"[PCA] Components: {n_components} | "
            f"Total Variance: {total_variance:.1%} | "
            f"PC1: {dominant_variance:.1%}"
        )