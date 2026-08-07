"""
S2_VI_Sweep - Volume Imbalance Sweep Strategy.

Scalping strategy that identifies volume imbalance zones and trades
liquidity sweeps at these levels.

Strategy Logic:
  1. Detect volume imbalance zones using OrderFlow engine
  2. Identify liquidity sweeps at swing highs/lows
  3. Confirm sweep completion with volume analysis
  4. Generate entry signal for reversal trade

Used Engines:
  - OrderFlowEngine: Volume imbalance detection
  - SMCStructuralEngine: Liquidity sweep detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SCALP
Timeframe: M5 (primary), M1 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.orderflow_engine import OrderFlowEngine
from core.smc_engine import SMCStructuralEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S2_VI_Sweep(BaseStrategy):
    """
    Volume Imbalance Sweep Strategy.
    
    This strategy identifies volume imbalance zones and trades
    liquidity sweeps at these high-probability levels.
    
    Volume Imbalance Definition:
      A zone where price moved quickly with significantly more
      volume on one side, indicating institutional activity.
      
    Liquidity Sweep Definition:
      Price breaks a swing high/low to trigger stop losses,
      then quickly reverses - a trap for retail traders.
      
    Entry Criteria:
      - Volume imbalance zone detected
      - Liquidity sweep at the zone
      - Sweep confirmed with reversal
      - Volume spike during sweep
    """

    def __init__(self):
        """Initialize S2_VI_Sweep strategy."""
        super().__init__(
            strategy_name='S2_VI_Sweep',
            strategy_category='SCALP',
            timeframes=['M5', 'M1'],
            risk_per_trade_pct=0.3,
            min_rr_ratio=1.5,
            max_spread_points=25,
            trailing_enabled=True,
            partial_close_enabled=False,
            requires_dynamic_exit=True,
            friction_sensitive=True
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.orderflow_engine = OrderFlowEngine()
        self.smc_engine = SMCStructuralEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.sweep_lookback = 30  # Lookback for sweep detection
        self.imbalance_threshold = 2.0  # Minimum imbalance ratio
        self.sweep_depth_threshold = 0.002  # Minimum sweep depth (0.2%)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m5: pd.DataFrame,
        df_m1: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S2_VI_Sweep.
        
        Args:
            df_m5: M5 DataFrame
            df_m1: M1 DataFrame (optional)
            regime_context: Current regime information
            
        Returns:
            Signal dict with entry/exit information
        """
        # Default neutral signal
        default_signal = self._create_neutral_signal()

        # Validate input
        if df_m5 is None or df_m5.empty or len(df_m5) < 50:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Detect Volume Imbalance
            # =========================================================================
            imbalances = self.orderflow_engine.detect_volume_imbalance(
                df_m5, min_imbalance_ratio=self.imbalance_threshold
            )

            if not imbalances:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Liquidity Sweep
            # =========================================================================
            sweeps = self.smc_engine.detect_liquidity_sweep(df_m5, lookback=self.sweep_lookback)

            if not sweeps:
                return default_signal

            # =========================================================================
            # STEP 3: Match Sweep with Imbalance
            # =========================================================================
            matched_sweep = self._match_sweep_with_imbalance(sweeps, imbalances, df_m5)

            if matched_sweep is None:
                return default_signal

            # =========================================================================
            # STEP 4: Confirm Sweep
            # =========================================================================
            if not self._confirm_sweep(df_m5, matched_sweep):
                return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m5, matched_sweep, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S2_VI] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SWEEP-IMBALANCE MATCHING
    # =========================================================================

    def _match_sweep_with_imbalance(
        self, sweeps: List[Dict], imbalances: List[Dict], df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Match liquidity sweep with volume imbalance zone.
        
        Args:
            sweeps: List of liquidity sweeps
            imbalances: List of volume imbalances
            df: DataFrame with OHLCV data
            
        Returns:
            Matched sweep dict or None
        """
        if not sweeps or not imbalances:
            return None

        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            # Find recent sweeps
            recent_sweeps = [s for s in sweeps if s.get('index', 0) >= len(df) - 10]

            if not recent_sweeps:
                return None

            # Check each sweep for matching imbalance
            for sweep in recent_sweeps:
                sweep_type = sweep.get('type', 'UNKNOWN')
                sweep_level = sweep.get('sweep_level', 0)

                # Find matching imbalance near sweep level
                for imbalance in imbalances:
                    imbalance_price = imbalance.get('price', 0)
                    imbalance_type = imbalance.get('type', 'UNKNOWN')

                    # Check if imbalance is near sweep level
                    distance = abs(imbalance_price - sweep_level) / sweep_level * 100

                    if distance < 0.5:  # Within 0.5%
                        # Match types
                        if sweep_type == 'UPSIDE_SWEEP' and imbalance_type == 'BULLISH':
                            return {
                                'sweep': sweep,
                                'imbalance': imbalance,
                                'direction': 'SELL',  # Upside sweep = sell signal
                                'level': sweep_level
                            }
                        elif sweep_type == 'DOWNSIDE_SWEEP' and imbalance_type == 'BEARISH':
                            return {
                                'sweep': sweep,
                                'imbalance': imbalance,
                                'direction': 'BUY',  # Downside sweep = buy signal
                                'level': sweep_level
                            }

            return None

        except Exception as e:
            self.logger.debug(f"[S2_VI] Sweep-imbalance matching error: {e}")
            return None

    # =========================================================================
    # SWEEP CONFIRMATION
    # =========================================================================

    def _confirm_sweep(self, df: pd.DataFrame, matched_sweep: Dict) -> bool:
        """
        Confirm sweep completion with volume analysis.
        
        Args:
            df: DataFrame with OHLCV data
            matched_sweep: Matched sweep dict
            
        Returns:
            True if sweep is confirmed
        """
        if df is None or df.empty or len(df) < 20:
            return False

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
                return False

            direction = matched_sweep.get('direction', 'BUY')
            sweep_level = matched_sweep.get('level', 0)

            current_price = close[-1]
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-20:])

            # Check volume spike during sweep
            volume_spike = current_volume > avg_volume * 1.5

            # Check price reversal
            if direction == 'BUY':
                # For BUY: Price should be above sweep level (reversed)
                price_reversed = current_price > sweep_level
            else:  # SELL
                # For SELL: Price should be below sweep level (reversed)
                price_reversed = current_price < sweep_level

            # Check sweep depth
            if direction == 'BUY':
                sweep_depth = (sweep_level - np.min(low[-5:])) / sweep_level
            else:
                sweep_depth = (np.max(high[-5:]) - sweep_level) / sweep_level

            depth_confirmed = sweep_depth >= self.sweep_depth_threshold

            # Confirmation: volume spike + price reversal + depth
            return volume_spike and price_reversed and depth_confirmed

        except Exception as e:
            self.logger.debug(f"[S2_VI] Sweep confirmation error: {e}")
            return False

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, matched_sweep: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal based on sweep.
        
        Args:
            df: DataFrame with OHLCV data
            matched_sweep: Matched sweep dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = matched_sweep.get('direction', 'BUY')
            sweep_level = matched_sweep.get('level', 0)
            imbalance = matched_sweep.get('imbalance', {})

            if sweep_level <= 0:
                return self._create_neutral_signal()

            # Calculate entry price
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below sweep level
                sl_buffer = abs(entry_price - sweep_level) * 0.3
                sl_price = sweep_level - sl_buffer
            else:  # SELL
                # SL above sweep level
                sl_buffer = abs(entry_price - sweep_level) * 0.3
                sl_price = sweep_level + sl_buffer

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
                    tp_price = entry_price + risk * 1.5
                else:
                    tp_price = entry_price - risk * 1.5

            # Calculate confidence based on imbalance strength
            imbalance_strength = imbalance.get('strength', 0.5)
            confidence = min(1.0, 0.5 + imbalance_strength * 0.3)

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
                    'sweep_type': matched_sweep.get('sweep', {}).get('type', 'UNKNOWN'),
                    'sweep_level': round(sweep_level, 2),
                    'imbalance_strength': imbalance_strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S2_VI] Signal generated: {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S2_VI] Signal generation error: {e}")
            return self._create_neutral_signal()

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

    def _is_regime_compatible(self, regime_context: Dict) -> bool:
        """
        Check if current regime is compatible with this strategy.
        
        Args:
            regime_context: Current regime information
            
        Returns:
            True if compatible
        """
        regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # SCALP strategies work best in choppy and high volatility regimes
        compatible_regimes = [
            'VOLATILE_CHOP', 'WHIPSAW_MARKET',
            'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
            'CLASSIC_RANGE', 'TIGHT_RANGE',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR'
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