"""
S11_LiquidityDelta - Liquidity + Volume Delta Strategy.

Scalping strategy that combines liquidity pool detection with
volume delta analysis for high-probability reversal entries.

Strategy Logic:
  1. Detect liquidity pools (swing highs/lows)
  2. Calculate volume delta (buy/sell pressure)
  3. Detect liquidity sweeps with delta confirmation
  4. Generate entry signal after sweep confirmation

Liquidity Concept:
  Liquidity pools form at obvious levels where retail traders
  place stop losses:
    - Swing highs: Buy-side liquidity
    - Swing lows: Sell-side liquidity
  
  Market makers sweep these levels to trigger stops before
  the real move begins.

Volume Delta:
  Delta = Buy Volume - Sell Volume
  Positive delta = buying pressure
  Negative delta = selling pressure
  
  Delta divergence from price indicates smart money activity.

Used Engines:
  - OrderFlowEngine: Volume delta and CVD
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


class S11_LiquidityDelta(BaseStrategy):
    """
    Liquidity + Volume Delta Strategy.
    
    This strategy combines liquidity pool detection with volume
    delta analysis for high-probability reversal entries.
    
    Liquidity Pool Definition:
      Zones where stop losses cluster, typically at swing
      highs and lows. Market makers sweep these levels to
      trigger stops before reversing.
      
    Volume Delta Definition:
      The difference between buy volume and sell volume.
      Positive delta indicates buying pressure, negative
      indicates selling pressure.
      
    Entry Criteria:
      - Liquidity pool identified
      - Sweep of liquidity detected
      - Volume delta confirms reversal
      - Price action confirmation
    """

    def __init__(self):
        """Initialize S11_LiquidityDelta strategy."""
        super().__init__(
            strategy_name='S11_LiquidityDelta',
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
        self.liquidity_lookback = 50  # Lookback for liquidity detection
        self.delta_lookback = 20  # Lookback for delta calculation
        self.sweep_depth_threshold = 0.002  # Minimum sweep depth (0.2%)
        self.delta_threshold = 0.3  # Minimum delta for confirmation

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
        Main analysis method for S11_LiquidityDelta.
        
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
            # STEP 1: Detect Liquidity Pools
            # =========================================================================
            liquidity_pools = self.orderflow_engine.detect_liquidity_pools(
                df_m5, lookback=self.liquidity_lookback
            )

            if not liquidity_pools:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Liquidity Sweep
            # =========================================================================
            sweeps = self.smc_engine.detect_liquidity_sweep(
                df_m5, lookback=self.liquidity_lookback
            )

            if not sweeps:
                return default_signal

            # =========================================================================
            # STEP 3: Calculate Volume Delta
            # =========================================================================
            cvd_result = self.orderflow_engine.calculate_cvd(
                df_m5, lookback=self.delta_lookback
            )

            if cvd_result is None:
                return default_signal

            # =========================================================================
            # STEP 4: Match Sweep with Delta
            # =========================================================================
            matched_signal = self._match_sweep_with_delta(
                sweeps, liquidity_pools, cvd_result, df_m5
            )

            if matched_signal is None:
                return default_signal

            # =========================================================================
            # STEP 5: M1 Confirmation (if available)
            # =========================================================================
            if df_m1 is not None and not df_m1.empty:
                if not self._confirm_m1(df_m1, matched_signal):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m5, matched_signal, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S11_LD] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SWEEP-DELTA MATCHING
    # =========================================================================

    def _match_sweep_with_delta(
        self, sweeps: List[Dict], liquidity_pools: List[Dict],
        cvd_result: Dict, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Match liquidity sweep with volume delta confirmation.
        
        Args:
            sweeps: List of liquidity sweeps
            liquidity_pools: List of liquidity pools
            cvd_result: CVD calculation result
            df: DataFrame with OHLCV data
            
        Returns:
            Matched signal dict or None
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            # Get recent sweeps
            recent_sweeps = [s for s in sweeps if s.get('index', 0) >= len(df) - 10]

            if not recent_sweeps:
                return None

            # Get CVD data
            cvd = cvd_result.get('cvd', [])
            cvd_slope = cvd_result.get('cvd_slope', [])

            if len(cvd) == 0:
                return None

            current_cvd = cvd[-1]
            cvd_trend = cvd_result.get('cvd_trend', 'UNKNOWN')

            # Check each sweep
            for sweep in recent_sweeps:
                sweep_type = sweep.get('type', 'UNKNOWN')
                sweep_level = sweep.get('sweep_level', 0)

                # Check delta confirmation
                if sweep_type == 'UPSIDE_SWEEP':
                    # Upside sweep → expect SELL signal
                    # Delta should be negative (selling pressure after sweep)
                    if cvd_trend in ['FALLING'] or current_cvd < 0:
                        return {
                            'direction': 'SELL',
                            'sweep_type': sweep_type,
                            'sweep_level': sweep_level,
                            'delta_confirmation': True,
                            'cvd': current_cvd,
                            'cvd_trend': cvd_trend
                        }

                elif sweep_type == 'DOWNSIDE_SWEEP':
                    # Downside sweep → expect BUY signal
                    # Delta should be positive (buying pressure after sweep)
                    if cvd_trend in ['RISING'] or current_cvd > 0:
                        return {
                            'direction': 'BUY',
                            'sweep_type': sweep_type,
                            'sweep_level': sweep_level,
                            'delta_confirmation': True,
                            'cvd': current_cvd,
                            'cvd_trend': cvd_trend
                        }

            return None

        except Exception as e:
            self.logger.debug(f"[S11_LD] Sweep-delta matching error: {e}")
            return None

    # =========================================================================
    # M1 CONFIRMATION
    # =========================================================================

    def _confirm_m1(self, df_m1: pd.DataFrame, matched_signal: Dict) -> bool:
        """
        Confirm signal on M1 timeframe.
        
        Args:
            df_m1: M1 DataFrame
            matched_signal: Matched signal dict
            
        Returns:
            True if confirmed
        """
        if df_m1 is None or df_m1.empty or len(df_m1) < 20:
            return True  # Skip confirmation if no M1 data

        try:
            close = df_m1['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = matched_signal.get('direction', 'BUY')

            # Check M1 momentum aligns with signal direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                return momentum > 0  # Bullish momentum on M1
            else:
                return momentum < 0  # Bearish momentum on M1

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, matched_signal: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            matched_signal: Matched signal dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = matched_signal.get('direction', 'BUY')
            sweep_level = matched_signal.get('sweep_level', 0)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            if entry_price <= 0 or sweep_level <= 0:
                return self._create_neutral_signal()

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below sweep level
                sl_buffer = abs(entry_price - sweep_level) * 0.3
                sl_price = sweep_level - sl_buffer
            else:  # SELL
                # SL above sweep level
                sl_buffer = abs(sweep_level - entry_price) * 0.3
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

            # Calculate confidence
            delta_strength = abs(matched_signal.get('cvd', 0)) / 100  # Normalize
            confidence = min(1.0, 0.4 + delta_strength * 0.3 + 0.2)

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
                    'sweep_type': matched_signal.get('sweep_type', 'UNKNOWN'),
                    'cvd': matched_signal.get('cvd', 0),
                    'cvd_trend': matched_signal.get('cvd_trend', 'UNKNOWN'),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S11_LD] Signal generated: {direction} | "
                f"Sweep: {matched_signal.get('sweep_type')} | "
                f"CVD: {matched_signal.get('cvd', 0):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S11_LD] Signal generation error: {e}")
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

        # SCALP strategies work best in choppy and volatile regimes
        compatible_regimes = [
            'VOLATILE_CHOP', 'WHIPSAW_MARKET',
            'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
            'CLASSIC_RANGE', 'TIGHT_RANGE',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'PRE_BREAKOUT'
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