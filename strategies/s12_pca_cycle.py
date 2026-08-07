"""
S12_PCA_Cycle - PCA Cycle Detection Strategy.

Trend-following strategy that uses Principal Component Analysis (PCA)
to detect market cycles and trade based on cycle phase.

Strategy Logic:
  1. Apply PCA to decompose price data
  2. Extract dominant cycle from principal components
  3. Analyze cycle phase and frequency
  4. Generate entry signal based on cycle alignment

PCA (Principal Component Analysis):
  A dimensionality reduction technique that transforms correlated
  variables into uncorrelated principal components. The first
  principal component captures the most variance (dominant pattern).
  
  For cycle detection:
    - PC1: Dominant trend/cycle
    - PC2: Secondary cycle
    - Higher PCs: Noise

Cycle Detection:
  By analyzing the principal components, we can identify:
    - Dominant cycle period
    - Current phase in the cycle
    - Cycle strength (variance explained)

Used Engines:
  - PCAEngine: PCA transformation and cycle detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: TREND
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.pca_engine import PCAEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S12_PCA_Cycle(BaseStrategy):
    """
    PCA Cycle Detection Strategy.
    
    This strategy uses Principal Component Analysis to detect
    market cycles and trade based on cycle phase.
    
    PCA Definition:
      PCA transforms correlated price data into uncorrelated
      principal components. The first component (PC1) captures
      the dominant pattern, which is often the trend or cycle.
      
    Cycle Detection:
      By analyzing PC1, we can identify:
        - Cycle period (from autocorrelation)
        - Current phase (position in cycle)
        - Cycle strength (variance explained)
        
    Entry Criteria:
      - Dominant cycle detected
      - Cycle phase aligns with trade direction
      - Cycle strength above threshold
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S12_PCA_Cycle strategy."""
        super().__init__(
            strategy_name='S12_PCA_Cycle',
            strategy_category='TREND',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.pca_engine = PCAEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.pca_lookback = 100  # Lookback for PCA
        self.n_components = 3  # Number of principal components
        self.min_cycle_period = 10  # Minimum cycle period
        self.max_cycle_period = 50  # Maximum cycle period
        self.min_variance_explained = 0.3  # Minimum variance for valid cycle

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m15: pd.DataFrame,
        df_m5: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S12_PCA_Cycle.
        
        Args:
            df_m15: M15 DataFrame
            df_m5: M5 DataFrame (optional)
            regime_context: Current regime information
            
        Returns:
            Signal dict with entry/exit information
        """
        # Default neutral signal
        default_signal = self._create_neutral_signal()

        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < self.pca_lookback:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Apply PCA
            # =========================================================================
            pca_result = self._apply_pca(df_m15)

            if pca_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Cycle
            # =========================================================================
            cycle_info = self._detect_cycle(pca_result)

            if cycle_info is None or not cycle_info.get('cycle_detected', False):
                return default_signal

            # =========================================================================
            # STEP 3: Analyze Phase
            # =========================================================================
            phase_info = self._analyze_phase(pca_result, cycle_info)

            if phase_info is None:
                return default_signal

            # =========================================================================
            # STEP 4: Determine Direction
            # =========================================================================
            direction_info = self._determine_direction(phase_info, cycle_info, df_m15)

            if direction_info is None:
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, direction_info):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, direction_info, cycle_info, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S12_PCA] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # PCA APPLICATION
    # =========================================================================

    def _apply_pca(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Apply PCA to price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            PCA result dict or None
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Create feature matrix from price and derivatives
            n = len(close)
            features = np.zeros((n - 5, 5))

            for i in range(5, n):
                features[i - 5, 0] = close[i]  # Price
                features[i - 5, 1] = close[i] - close[i - 1]  # Change
                features[i - 5, 2] = close[i] - close[i - 5]  # 5-bar change
                features[i - 5, 3] = np.mean(close[i - 5:i + 1])  # 5-bar mean
                features[i - 5, 4] = np.std(close[i - 5:i + 1])  # 5-bar std

            # Apply PCA
            pca_result = self.pca_engine.apply_pca(
                features, n_components=self.n_components, normalize=True
            )

            if pca_result is None:
                return None

            return pca_result

        except Exception as e:
            self.logger.debug(f"[S12_PCA] PCA application error: {e}")
            return None

    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================

    def _detect_cycle(self, pca_result: Dict) -> Optional[Dict]:
        """
        Detect cycle from principal components.
        
        Args:
            pca_result: PCA result dict
            
        Returns:
            Cycle info dict or None
        """
        try:
            transformed = pca_result.get('transformed')
            variance_explained = pca_result.get('variance_explained')

            if transformed is None or variance_explained is None:
                return None

            # Check if PC1 explains enough variance
            if variance_explained[0] < self.min_variance_explained:
                return None

            # Use PC1 for cycle detection
            pc1 = transformed[:, 0]

            # Detect cycle period using autocorrelation
            cycle_period = self._detect_cycle_period(pc1)

            if cycle_period is None:
                return None

            # Validate cycle period
            if cycle_period < self.min_cycle_period or cycle_period > self.max_cycle_period:
                return None

            return {
                'cycle_detected': True,
                'cycle_period': cycle_period,
                'variance_explained': float(variance_explained[0]),
                'pc1': pc1,
                'cumulative_variance': float(np.sum(variance_explained[:2]))
            }

        except Exception as e:
            self.logger.debug(f"[S12_PCA] Cycle detection error: {e}")
            return None

    def _detect_cycle_period(self, pc1: np.ndarray) -> Optional[int]:
        """Detect cycle period using autocorrelation."""
        try:
            n = len(pc1)
            if n < 30:
                return None

            # Calculate autocorrelation
            pc1_centered = pc1 - np.mean(pc1)
            autocorr = np.correlate(pc1_centered, pc1_centered, mode='full')
            autocorr = autocorr[n - 1:]  # Positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find first peak after minimum lag
            min_lag = self.min_cycle_period
            max_lag = min(self.max_cycle_period, n // 2)

            best_lag = None
            best_corr = 0

            for lag in range(min_lag, max_lag):
                if autocorr[lag] > best_corr and autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                    best_corr = autocorr[lag]
                    best_lag = lag

            return best_lag

        except Exception:
            return None

    # =========================================================================
    # PHASE ANALYSIS
    # =========================================================================

    def _analyze_phase(self, pca_result: Dict, cycle_info: Dict) -> Optional[Dict]:
        """
        Analyze current cycle phase.
        
        Args:
            pca_result: PCA result dict
            cycle_info: Cycle info dict
            
        Returns:
            Phase info dict or None
        """
        try:
            pc1 = cycle_info.get('pc1')
            cycle_period = cycle_info.get('cycle_period')

            if pc1 is None or cycle_period is None:
                return None

            # Calculate current phase using Hilbert-like approach
            n = len(pc1)

            # Use last cycle_period bars
            recent_pc1 = pc1[-cycle_period:]

            # Calculate phase position
            mean_val = np.mean(recent_pc1)
            std_val = np.std(recent_pc1)

            if std_val == 0:
                return None

            current_val = pc1[-1]
            normalized_val = (current_val - mean_val) / std_val

            # Determine phase (0 to 2*pi)
            phase = np.arctan2(normalized_val, 1)

            # Determine phase category
            if 0 <= phase < np.pi / 2:
                phase_category = 'RISING'
            elif np.pi / 2 <= phase < np.pi:
                phase_category = 'PEAKING'
            elif np.pi <= phase < 3 * np.pi / 2:
                phase_category = 'FALLING'
            else:
                phase_category = 'BOTTOMING'

            return {
                'phase': float(phase),
                'phase_normalized': float(normalized_val),
                'phase_category': phase_category,
                'current_pc1': float(current_val),
                'mean_pc1': float(mean_val),
                'std_pc1': float(std_val)
            }

        except Exception as e:
            self.logger.debug(f"[S12_PCA] Phase analysis error: {e}")
            return None

    # =========================================================================
    # DIRECTION DETERMINATION
    # =========================================================================

    def _determine_direction(
        self, phase_info: Dict, cycle_info: Dict, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Determine trend direction based on cycle phase.
        
        Args:
            phase_info: Phase info dict
            cycle_info: Cycle info dict
            df: DataFrame with OHLCV data
            
        Returns:
            Direction info dict or None
        """
        try:
            phase_category = phase_info.get('phase_category', 'UNKNOWN')
            phase_normalized = phase_info.get('phase_normalized', 0)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            # Calculate price momentum
            recent_close = close[-20:]
            momentum = recent_close[-1] - recent_close[0]
            momentum_pct = momentum / recent_close[0] * 100

            # Determine direction based on phase and momentum
            if phase_category == 'RISING' and momentum > 0:
                direction = 'BUY'
                strength = min(1.0, abs(momentum_pct) / 2.0 + 0.3)
            elif phase_category == 'PEAKING' and momentum > 0:
                direction = 'BUY'
                strength = min(1.0, abs(momentum_pct) / 3.0 + 0.2)
            elif phase_category == 'FALLING' and momentum < 0:
                direction = 'SELL'
                strength = min(1.0, abs(momentum_pct) / 2.0 + 0.3)
            elif phase_category == 'BOTTOMING' and momentum < 0:
                direction = 'SELL'
                strength = min(1.0, abs(momentum_pct) / 3.0 + 0.2)
            else:
                return None  # Phase and momentum don't align

            return {
                'direction': direction,
                'strength': float(strength),
                'phase_category': phase_category,
                'momentum_pct': float(momentum_pct),
                'cycle_period': cycle_info.get('cycle_period', 0)
            }

        except Exception as e:
            self.logger.debug(f"[S12_PCA] Direction error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, direction_info: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            direction_info: Direction information
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = direction_info.get('direction', 'BUY')

            # Check M5 momentum aligns with direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                return momentum > 0  # Bullish momentum on M5
            else:
                return momentum < 0  # Bearish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, direction_info: Dict,
        cycle_info: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            direction_info: Direction information
            cycle_info: Cycle information
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = direction_info.get('direction', 'BUY')
            strength = direction_info.get('strength', 0.5)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            if direction == 'BUY':
                sl_price = entry_price - atr * 2.0
            else:  # SELL
                sl_price = entry_price + atr * 2.0

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, direction == 'BUY',
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Fixed R:R
                risk = abs(entry_price - sl_price)
                if direction == 'BUY':
                    tp_price = entry_price + risk * 2.0
                else:
                    tp_price = entry_price - risk * 2.0

            # Calculate confidence
            variance_bonus = cycle_info.get('variance_explained', 0.3) * 0.2
            confidence = min(1.0, 0.4 + strength * 0.4 + variance_bonus)

            # Build signal
            signal = {
                'signal': f'{direction}_MARKET',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'phase_category': direction_info.get('phase_category', 'UNKNOWN'),
                    'cycle_period': direction_info.get('cycle_period', 0),
                    'momentum_pct': direction_info.get('momentum_pct', 0),
                    'variance_explained': cycle_info.get('variance_explained', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S12_PCA] Signal generated: {direction} | "
                f"Phase: {direction_info.get('phase_category')} | "
                f"Cycle: {direction_info.get('cycle_period')} bars | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S12_PCA] Signal generation error: {e}")
            return self._create_neutral_signal()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_regime_compatible(self, regime_context: Dict) -> bool:
        """
        Check if current regime is compatible with this strategy.
        
        Args:
            regime_context: Current regime information
            
        Returns:
            True if compatible
        """
        regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # TREND strategies work best in trending regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'FALSE_SIDEWAY', 'PRE_BREAKOUT'
        ]

        return regime_name in compatible_regimes

    def _create_neutral_signal(self) -> Dict:
        """Create neutral signal."""
        return {
            'signal': 'NEUTRAL',
            'meta': {
                'strategy': self.strategy_name,
                'strategy_category': self.strategy_category,
                'confidence': 0.0
            }
        }