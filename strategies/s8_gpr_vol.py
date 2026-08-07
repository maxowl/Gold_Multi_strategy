"""
S8_GPR_Vol - Gaussian Process Regression + Volatility Strategy.

Mean-reversion strategy that uses Gaussian Process Regression (GPR)
combined with volatility analysis for probabilistic price prediction.

Strategy Logic:
  1. Apply GPR to predict price movement with uncertainty
  2. Analyze current volatility regime
  3. Generate entry when GPR predicts reversion with high confidence
  4. Use volatility for position sizing and stop loss

Gaussian Process Regression:
  A non-parametric Bayesian approach that provides:
    - Mean prediction (expected price)
    - Uncertainty (confidence interval)
    - Probabilistic forecasts

Used Engines:
  - KalmanSqueezeEngine: Volatility analysis
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: MEAN_REVERSION
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.kalman_squeeze_engine import KalmanSqueezeEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S8_GPR_Vol(BaseStrategy):
    """
    Gaussian Process Regression + Volatility Strategy.
    
    This strategy uses Gaussian Process Regression for probabilistic
    price prediction combined with volatility analysis.
    
    GPR Definition:
      A Bayesian non-parametric method that provides predictions
      with uncertainty estimates. Unlike point estimates, GPR
      gives a distribution of possible outcomes.
      
    Volatility Integration:
      Volatility is used to:
        - Adjust prediction uncertainty
        - Set adaptive stop loss
        - Determine position size
        - Filter low-probability signals
      
    Entry Criteria:
      - GPR prediction indicates reversion
      - Prediction confidence above threshold
      - Volatility within acceptable range
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S8_GPR_Vol strategy."""
        super().__init__(
            strategy_name='S8_GPR_Vol',
            strategy_category='MEAN_REVERSION',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.4,
            min_rr_ratio=1.5,
            max_spread_points=30,
            trailing_enabled=False,  # No trailing for mean reversion
            partial_close_enabled=False,  # No partial close
            requires_dynamic_exit=True
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.kalman_engine = KalmanSqueezeEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.gpr_lookback = 50  # Lookback for GPR training
        self.prediction_horizon = 10  # Predict next 10 bars
        self.confidence_threshold = 0.65  # Minimum confidence
        self.volatility_threshold = 1.5  # Maximum volatility multiplier

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
        Main analysis method for S8_GPR_Vol.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < self.gpr_lookback:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Apply GPR Prediction
            # =========================================================================
            gpr_result = self._apply_gpr(df_m15)

            if gpr_result is None:
                return default_signal

            # =========================================================================
            # STEP 2: Analyze Volatility
            # =========================================================================
            volatility_result = self._analyze_volatility(df_m15)

            if volatility_result is None:
                return default_signal

            # Check if volatility is acceptable
            if not self._is_volatility_acceptable(volatility_result):
                return default_signal

            # =========================================================================
            # STEP 3: Detect Reversion Signal
            # =========================================================================
            reversion = self._detect_reversion(gpr_result, volatility_result, df_m15)

            if reversion is None or not reversion.get('signal_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, reversion):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, reversion, volatility_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S8_GPR] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # GPR PREDICTION
    # =========================================================================

    def _apply_gpr(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Apply Gaussian Process Regression for price prediction.
        
        Uses simplified GPR with RBF kernel approximation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with GPR prediction, or None on failure
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            n = len(close)
            if n < self.gpr_lookback:
                return None

            # Use recent data for training
            train_data = close[-self.gpr_lookback:]

            # Normalize data
            mean_price = np.mean(train_data)
            std_price = np.std(train_data)
            if std_price == 0:
                return None

            normalized = (train_data - mean_price) / std_price

            # Create features (lagged values)
            X = np.zeros((self.gpr_lookback - 5, 5))
            y = np.zeros(self.gpr_lookback - 5)

            for i in range(5, self.gpr_lookback):
                X[i - 5] = normalized[i - 5:i]
                y[i - 5] = normalized[i]

            # Simplified GPR prediction using linear regression as approximation
            # In production, use sklearn.gaussian_process.GaussianProcessRegressor
            try:
                # Linear regression approximation
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)

                # Calculate weights (simplified)
                X_centered = X - X_mean
                y_centered = y - y_mean

                weights = np.linalg.lstsq(X_centered, y_centered, rcond=None)[0]

                # Predict next value
                current_features = normalized[-5:]
                predicted_normalized = np.dot(weights, current_features - X_mean) + y_mean

                # Calculate prediction uncertainty
                residuals = y - (np.dot(X - X_mean, weights) + y_mean)
                uncertainty = np.std(residuals)

                # Denormalize
                predicted_price = predicted_normalized * std_price + mean_price
                uncertainty_price = uncertainty * std_price

                # Determine prediction direction
                current_price = close[-1]
                if predicted_price > current_price * 1.001:  # > 0.1% up
                    direction = 'UP'
                elif predicted_price < current_price * 0.999:  # > 0.1% down
                    direction = 'DOWN'
                else:
                    direction = 'FLAT'

                return {
                    'predicted_price': float(predicted_price),
                    'current_price': float(current_price),
                    'direction': direction,
                    'uncertainty': float(uncertainty_price),
                    'uncertainty_pct': float(uncertainty_price / current_price * 100),
                    'confidence': float(1.0 - uncertainty),
                    'mean_price': float(mean_price),
                    'std_price': float(std_price)
                }

            except Exception as e:
                self.logger.debug(f"[S8_GPR] GPR calculation error: {e}")
                return None

        except Exception as e:
            self.logger.error(f"[S8_GPR] GPR application error: {e}")
            return None

    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================

    def _analyze_volatility(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze current volatility regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with volatility analysis, or None on failure
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Calculate ATR
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate volatility ratio
            recent_volatility = np.std(close[-20:]) / np.mean(close[-20:]) * 100
            baseline_volatility = np.std(close[-50:]) / np.mean(close[-50:]) * 100

            if baseline_volatility > 0:
                volatility_ratio = recent_volatility / baseline_volatility
            else:
                volatility_ratio = 1.0

            # Use Kalman engine for squeeze detection
            squeeze_result = self.kalman_engine.detect_squeeze(df)

            return {
                'atr': float(atr),
                'atr_pct': float(atr / close[-1] * 100),
                'recent_volatility': float(recent_volatility),
                'baseline_volatility': float(baseline_volatility),
                'volatility_ratio': float(volatility_ratio),
                'squeeze_on': squeeze_result.get('squeeze_on', False) if squeeze_result else False,
                'band_width': squeeze_result.get('band_width', 0) if squeeze_result else 0
            }

        except Exception as e:
            self.logger.error(f"[S8_GPR] Volatility analysis error: {e}")
            return None

    def _is_volatility_acceptable(self, volatility_result: Dict) -> bool:
        """Check if volatility is acceptable for trading."""
        try:
            volatility_ratio = volatility_result.get('volatility_ratio', 1.0)

            # Volatility should be within acceptable range
            if volatility_ratio > self.volatility_threshold:
                return False  # Too volatile

            if volatility_ratio < 0.3:
                return False  # Too low volatility (no opportunity)

            return True

        except Exception:
            return False

    # =========================================================================
    # REVERSION DETECTION
    # =========================================================================

    def _detect_reversion(
        self, gpr_result: Dict, volatility_result: Dict, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Detect mean reversion signal based on GPR prediction.
        
        Args:
            gpr_result: GPR prediction result
            volatility_result: Volatility analysis result
            df: DataFrame with OHLCV data
            
        Returns:
            Reversion dict or None
        """
        try:
            predicted_price = gpr_result.get('predicted_price', 0)
            current_price = gpr_result.get('current_price', 0)
            direction = gpr_result.get('direction', 'FLAT')
            confidence = gpr_result.get('confidence', 0)
            uncertainty = gpr_result.get('uncertainty', 0)

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                return None

            # Check prediction direction
            if direction == 'FLAT':
                return None

            # Calculate deviation from mean
            mean_price = gpr_result.get('mean_price', 0)
            std_price = gpr_result.get('std_price', 0)

            if std_price <= 0:
                return None

            # Z-score of current price
            zscore = (current_price - mean_price) / std_price

            # Determine reversion direction
            if direction == 'UP' and zscore < -1.0:
                # Price below mean, predicted up → BUY reversion
                reversion_direction = 'BUY'
                signal_strength = min(1.0, abs(zscore) / 2.0)
            elif direction == 'DOWN' and zscore > 1.0:
                # Price above mean, predicted down → SELL reversion
                reversion_direction = 'SELL'
                signal_strength = min(1.0, abs(zscore) / 2.0)
            else:
                return None  # No clear reversion signal

            return {
                'signal_detected': True,
                'direction': reversion_direction,
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'zscore': float(zscore),
                'confidence': float(confidence),
                'uncertainty': float(uncertainty),
                'signal_strength': float(signal_strength)
            }

        except Exception as e:
            self.logger.debug(f"[S8_GPR] Reversion detection error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, reversion: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            reversion: Reversion dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = reversion.get('direction', 'BUY')

            # Check M5 momentum aligns with reversion direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY reversion: M5 should show reversal signs
                return momentum > -0.2 * abs(recent_close[0])
            else:  # SELL
                # For SELL reversion: M5 should show reversal signs
                return momentum < 0.2 * abs(recent_close[0])

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, reversion: Dict, volatility_result: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            reversion: Reversion dict
            volatility_result: Volatility analysis result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = reversion.get('direction', 'BUY')
            entry_price = reversion.get('current_price', 0)
            predicted_price = reversion.get('predicted_price', 0)
            confidence = reversion.get('confidence', 0.5)
            uncertainty = reversion.get('uncertainty', 0)

            if entry_price <= 0:
                return self._create_neutral_signal()

            # Calculate ATR for stop loss
            atr = volatility_result.get('atr', 5.0)

            # Calculate Stop Loss based on ATR and uncertainty
            sl_multiplier = 1.5 + uncertainty / entry_price  # Add uncertainty buffer

            if direction == 'BUY':
                sl_price = entry_price - atr * sl_multiplier
                tp_price = predicted_price  # TP at predicted price
            else:  # SELL
                sl_price = entry_price + atr * sl_multiplier
                tp_price = predicted_price  # TP at predicted price

            # Validate SL and TP
            if sl_price <= 0 or sl_price == entry_price or tp_price <= 0:
                return self._create_neutral_signal()

            # Check R:R ratio
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)

            if risk <= 0 or reward / risk < self.min_rr_ratio:
                # Adjust TP to meet minimum R:R
                if direction == 'BUY':
                    tp_price = entry_price + risk * self.min_rr_ratio
                else:
                    tp_price = entry_price - risk * self.min_rr_ratio

            # Calculate confidence
            signal_strength = reversion.get('signal_strength', 0.5)
            final_confidence = min(1.0, 0.3 + confidence * 0.4 + signal_strength * 0.3)

            # Build signal
            signal = {
                'signal': f'{direction}_MARKET',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': final_confidence,
                    'predicted_price': round(predicted_price, 2),
                    'zscore': reversion.get('zscore', 0),
                    'gpr_confidence': confidence,
                    'signal_strength': signal_strength,
                    'volatility_ratio': volatility_result.get('volatility_ratio', 1.0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S8_GPR] Signal generated: {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Z-score: {reversion.get('zscore', 0):.2f} | "
                f"Confidence: {final_confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S8_GPR] Signal generation error: {e}")
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

        # MEAN_REVERSION strategies work best in ranging regimes
        compatible_regimes = [
            'CLASSIC_RANGE', 'TIGHT_RANGE',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'ANOMALY_BULL', 'ANOMALY_BEAR',
            'FALSE_SIDEWAY'
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