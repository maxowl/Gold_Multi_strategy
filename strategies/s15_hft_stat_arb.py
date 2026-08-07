"""
S15_HFT_StatArb - High Frequency Statistical Arbitrage Strategy.

Mean-reversion strategy that uses statistical arbitrage techniques
for high-frequency trading opportunities.

Strategy Logic:
  1. Calculate z-score of price relative to moving average
  2. Detect statistical deviation from mean
  3. Generate entry when deviation exceeds threshold
  4. Exit when price reverts to mean

Statistical Arbitrage Concept:
  Statistical arbitrage exploits temporary price inefficiencies
  by identifying when price deviates significantly from its
  statistical mean and betting on reversion.
  
  Z-score = (Price - Mean) / Std
  
  Z > 2.0: Price significantly above mean → SELL (expect reversion down)
  Z < -2.0: Price significantly below mean → BUY (expect reversion up)

Used Engines:
  - StatArbEngine: Z-score and mean reversion detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: MEAN_REVERSION
Timeframe: M5 (primary), M1 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.stat_arb_engine import StatArbEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S15_HFT_StatArb(BaseStrategy):
    """
    High Frequency Statistical Arbitrage Strategy.
    
    This strategy uses statistical arbitrage techniques for
    high-frequency mean reversion trading.
    
    Statistical Arbitrage Definition:
      Statistical arbitrage identifies temporary price inefficiencies
      and exploits them by betting on price reversion to the mean.
      
    Z-Score Definition:
      Z-score measures how many standard deviations a price is
      from its mean. High z-score indicates statistical deviation.
      
    Entry Criteria:
      - Z-score exceeds threshold (±2.0)
      - Mean reversion detected
      - Statistical confirmation
      - Quick execution for HFT
    """

    def __init__(self):
        """Initialize S15_HFT_StatArb strategy."""
        super().__init__(
            strategy_name='S15_HFT_StatArb',
            strategy_category='MEAN_REVERSION',
            timeframes=['M5', 'M1'],
            risk_per_trade_pct=0.3,
            min_rr_ratio=1.0,  # Lower R:R for HFT
            max_spread_points=20,  # Tighter spread for HFT
            trailing_enabled=False,  # No trailing for mean reversion
            partial_close_enabled=False,  # No partial close
            requires_dynamic_exit=True,
            friction_sensitive=True  # HFT is friction sensitive
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.stat_arb_engine = StatArbEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.zscore_period = 20  # Z-score lookback
        self.entry_threshold = 2.0  # Z-score entry threshold
        self.exit_threshold = 0.5  # Z-score exit threshold
        self.min_half_life = 5  # Minimum half-life for valid mean reversion
        self.max_half_life = 50  # Maximum half-life for valid mean reversion

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
        Main analysis method for S15_HFT_StatArb.
        
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
            # STEP 1: Calculate Z-Score
            # =========================================================================
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            zscore = self.stat_arb_engine.calculate_zscore(close, period=self.zscore_period)

            if zscore is None or len(zscore) == 0:
                return default_signal

            current_zscore = zscore[-1]

            # =========================================================================
            # STEP 2: Check Z-Score Threshold
            # =========================================================================
            if abs(current_zscore) < self.entry_threshold:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Mean Reversion
            # =========================================================================
            mr_result = self.stat_arb_engine.detect_mean_reversion(close)

            if mr_result is None or not mr_result.get('is_mean_reverting', False):
                return default_signal

            # Check half-life
            half_life = mr_result.get('half_life')
            if half_life is None or half_life < self.min_half_life or half_life > self.max_half_life:
                return default_signal

            # =========================================================================
            # STEP 4: Determine Direction
            # =========================================================================
            direction = self._determine_direction(current_zscore)

            if direction is None:
                return default_signal

            # =========================================================================
            # STEP 5: M1 Confirmation (if available)
            # =========================================================================
            if df_m1 is not None and not df_m1.empty:
                if not self._confirm_m1(df_m1, direction):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m5, direction, current_zscore,
                                            mr_result, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S15_HFT] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # DIRECTION DETERMINATION
    # =========================================================================

    def _determine_direction(self, zscore: float) -> Optional[str]:
        """
        Determine trade direction based on z-score.
        
        Args:
            zscore: Current z-score
            
        Returns:
            Direction (BUY/SELL) or None
        """
        try:
            if zscore > self.entry_threshold:
                # Price above mean → expect reversion down → SELL
                return 'SELL'
            elif zscore < -self.entry_threshold:
                # Price below mean → expect reversion up → BUY
                return 'BUY'
            else:
                return None

        except Exception:
            return None

    # =========================================================================
    # M1 CONFIRMATION
    # =========================================================================

    def _confirm_m1(self, df_m1: pd.DataFrame, direction: str) -> bool:
        """
        Confirm signal on M1 timeframe.
        
        Args:
            df_m1: M1 DataFrame
            direction: Direction (BUY/SELL)
            
        Returns:
            True if confirmed
        """
        if df_m1 is None or df_m1.empty or len(df_m1) < 20:
            return True  # Skip confirmation if no M1 data

        try:
            close = df_m1['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Check M1 momentum for reversal signs
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY: M1 should show slowing decline or reversal
                return momentum > -0.1 * abs(recent_close[0])
            else:  # SELL
                # For SELL: M1 should show slowing rise or reversal
                return momentum < 0.1 * abs(recent_close[0])

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, direction: str, zscore: float,
        mr_result: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            direction: Direction (BUY/SELL)
            zscore: Current z-score
            mr_result: Mean reversion result
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
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

            # Calculate mean price for TP
            mean_price = np.mean(close[-self.zscore_period:])

            # Calculate Stop Loss
            if direction == 'BUY':
                sl_price = entry_price - atr * 1.0  # Tighter SL for HFT
                tp_price = mean_price  # TP at mean
            else:  # SELL
                sl_price = entry_price + atr * 1.0
                tp_price = mean_price  # TP at mean

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
            zscore_strength = min(1.0, abs(zscore) / 3.0)  # Cap at 3.0
            half_life_bonus = 0.1 if mr_result.get('half_life', 0) < 20 else 0.0

            confidence = min(1.0, 0.4 + zscore_strength * 0.4 + half_life_bonus)

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
                    'zscore': float(zscore),
                    'half_life': mr_result.get('half_life', 0),
                    'mean_price': round(mean_price, 2),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S15_HFT] Signal generated: {direction} | "
                f"Z-score: {zscore:.2f} | "
                f"Half-life: {mr_result.get('half_life', 0):.1f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S15_HFT] Signal generation error: {e}")
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