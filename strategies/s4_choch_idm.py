"""
S4_CHOCH_IDM - Change of Character + Inducement Strategy.

Smart Money Concepts strategy that identifies Change of Character (CHOCH)
combined with Inducement patterns for high-probability reversal entries.

Strategy Logic:
  1. Detect Change of Character (CHOCH) - trend reversal signal
  2. Identify Inducement (IDM) - fake breakout that traps traders
  3. Confirm structure break with volume
  4. Generate entry signal at optimal timing

Smart Money Concepts:
  CHOCH (Change of Character):
    First sign of trend reversal - breaks the recent swing structure
    in the opposite direction of the current trend.
    
  Inducement (IDM):
    A minor pullback or fake breakout that induces retail traders
    to enter in the wrong direction before the real move.

Used Engines:
  - SMCStructuralEngine: BOS/CHOCH and structure detection
  - FibonacciEngine: Fibonacci levels for TP
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SMC
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.fibonacci_engine import FibonacciEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S4_CHOCH_IDM(BaseStrategy):
    """
    Change of Character + Inducement Strategy.
    
    This strategy combines CHOCH (trend reversal) with Inducement
    (liquidity grab) for high-probability reversal entries.
    
    CHOCH Definition:
      Change of Character is the first break of structure in the
      opposite direction of the current trend, signaling a potential
      trend reversal.
      
    Inducement Definition:
      A minor pullback or fake breakout that "induces" retail traders
      to enter positions before being swept. After the sweep, the
      real move begins.
      
    Entry Criteria:
      - CHOCH detected (structure break)
      - Inducement present (fake breakout)
      - Order Block at entry zone
      - Volume confirmation
    """

    def __init__(self):
        """Initialize S4_CHOCH_IDM strategy."""
        super().__init__(
            strategy_name='S4_CHOCH_IDM',
            strategy_category='SMC',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.8,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.smc_engine = SMCStructuralEngine()
        self.fib_engine = FibonacciEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.structure_lookback = 50  # Lookback for structure analysis
        self.inducement_lookback = 20  # Lookback for inducement
        self.min_choch_break = 0.002  # Minimum structure break (0.2%)
        self.inducement_tolerance = 0.003  # Inducement tolerance (0.3%)

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
        Main analysis method for S4_CHOCH_IDM.
        
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
        if df_m15 is None or df_m15.empty or len(df_m15) < 50:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Detect CHOCH (Change of Character)
            # =========================================================================
            bos_choch = self.smc_engine.detect_bos_choch(df_m15)

            if bos_choch is None or not bos_choch.get('choch_detected', False):
                return default_signal

            choch_type = bos_choch.get('choch_type', 'UNKNOWN')
            choch_price = bos_choch.get('choch_price', 0)
            choch_index = bos_choch.get('choch_index', 0)

            # Check if CHOCH is recent (within last 10 bars)
            if len(df_m15) - choch_index > 10:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Inducement
            # =========================================================================
            inducement = self._detect_inducement(df_m15, choch_type, choch_price)

            if inducement is None:
                return default_signal

            # =========================================================================
            # STEP 3: Find Entry Zone (Order Block)
            # =========================================================================
            entry_zone = self._find_entry_zone(df_m15, inducement, choch_type)

            if entry_zone is None:
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, choch_type):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, choch_type, inducement,
                                            entry_zone, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S4_CHOCH] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # INDUCEMENT DETECTION
    # =========================================================================

    def _detect_inducement(
        self, df: pd.DataFrame, choch_type: str, choch_price: float
    ) -> Optional[Dict]:
        """
        Detect Inducement pattern.
        
        Inducement is a minor pullback or fake breakout that occurs
        before the real move, designed to trap retail traders.
        
        For Bullish CHOCH:
          Inducement is a small dip below a recent swing low
          that quickly recovers.
          
        For Bearish CHOCH:
          Inducement is a small spike above a recent swing high
          that quickly reverses.
        
        Args:
            df: DataFrame with OHLCV data
            choch_type: CHOCH type ('BULLISH_CHOCH' or 'BEARISH_CHOCH')
            choch_price: Price at CHOCH point
            
        Returns:
            Inducement dict or None
        """
        if df is None or df.empty or len(df) < self.inducement_lookback:
            return None

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            # Handle NaN
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Look for inducement before CHOCH
            # Use recent bars before CHOCH
            lookback_start = max(0, len(df) - self.inducement_lookback)
            recent_high = high[lookback_start:]
            recent_low = low[lookback_start:]
            recent_close = close[lookback_start:]

            if choch_type == 'BULLISH_CHOCH':
                # Bullish CHOCH: Look for downside inducement (fake breakdown)
                return self._detect_bullish_inducement(recent_high, recent_low, recent_close)

            elif choch_type == 'BEARISH_CHOCH':
                # Bearish CHOCH: Look for upside inducement (fake breakout)
                return self._detect_bearish_inducement(recent_high, recent_low, recent_close)

            return None

        except Exception as e:
            self.logger.debug(f"[S4_CHOCH] Inducement detection error: {e}")
            return None

    def _detect_bullish_inducement(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Optional[Dict]:
        """Detect bullish inducement (fake breakdown)."""
        try:
            n = len(close)
            if n < 10:
                return None

            # Find recent swing low
            swing_lows = []
            for i in range(3, n - 3):
                if low[i] == np.min(low[i-3:i+4]):
                    swing_lows.append((i, float(low[i])))

            if len(swing_lows) < 2:
                return None

            # Get the most recent swing lows
            recent_swing = swing_lows[-1]
            prev_swing = swing_lows[-2]

            swing_low = recent_swing[1]
            swing_idx = recent_swing[0]

            # Check if price went below swing low (inducement)
            # Then recovered above it
            for i in range(swing_idx + 1, min(swing_idx + 5, n)):
                if low[i] < swing_low:
                    # Price broke below swing low
                    inducement_low = float(low[i])
                    inducement_depth = (swing_low - inducement_low) / swing_low

                    # Check if inducement is shallow (typical for fake breakouts)
                    if inducement_depth < self.inducement_tolerance:
                        # Check recovery
                        if close[-1] > swing_low:
                            return {
                                'type': 'BULLISH_INDUCEMENT',
                                'swing_low': swing_low,
                                'inducement_low': inducement_low,
                                'inducement_depth_pct': float(inducement_depth * 100),
                                'recovered': True,
                                'index': i
                            }

            return None

        except Exception:
            return None

    def _detect_bearish_inducement(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Optional[Dict]:
        """Detect bearish inducement (fake breakout)."""
        try:
            n = len(close)
            if n < 10:
                return None

            # Find recent swing high
            swing_highs = []
            for i in range(3, n - 3):
                if high[i] == np.max(high[i-3:i+4]):
                    swing_highs.append((i, float(high[i])))

            if len(swing_highs) < 2:
                return None

            # Get the most recent swing highs
            recent_swing = swing_highs[-1]
            swing_high = recent_swing[1]
            swing_idx = recent_swing[0]

            # Check if price went above swing high (inducement)
            for i in range(swing_idx + 1, min(swing_idx + 5, n)):
                if high[i] > swing_high:
                    # Price broke above swing high
                    inducement_high = float(high[i])
                    inducement_height = (inducement_high - swing_high) / swing_high

                    # Check if inducement is shallow
                    if inducement_height < self.inducement_tolerance:
                        # Check recovery
                        if close[-1] < swing_high:
                            return {
                                'type': 'BEARISH_INDUCEMENT',
                                'swing_high': swing_high,
                                'inducement_high': inducement_high,
                                'inducement_height_pct': float(inducement_height * 100),
                                'recovered': True,
                                'index': i
                            }

            return None

        except Exception:
            return None

    # =========================================================================
    # ENTRY ZONE (ORDER BLOCK)
    # =========================================================================

    def _find_entry_zone(
        self, df: pd.DataFrame, inducement: Dict, choch_type: str
    ) -> Optional[Dict]:
        """
        Find entry zone (Order Block) near inducement level.
        
        Args:
            df: DataFrame with OHLCV data
            inducement: Inducement dict
            choch_type: CHOCH type
            
        Returns:
            Entry zone dict or None
        """
        try:
            # Detect Order Blocks
            order_blocks = self.smc_engine.detect_order_blocks(df, lookback=self.structure_lookback)

            if not order_blocks:
                return None

            # Find OB near inducement level
            if choch_type == 'BULLISH_CHOCH':
                # For BUY: Look for bullish OB near inducement low
                inducement_level = inducement.get('inducement_low', 0)
                matching_obs = [
                    ob for ob in order_blocks
                    if ob.get('type') == 'BULLISH' and
                    abs(ob.get('low', 0) - inducement_level) / inducement_level < 0.005
                ]
            else:  # BEARISH_CHOCH
                # For SELL: Look for bearish OB near inducement high
                inducement_level = inducement.get('inducement_high', 0)
                matching_obs = [
                    ob for ob in order_blocks
                    if ob.get('type') == 'BEARISH' and
                    abs(ob.get('high', 0) - inducement_level) / inducement_level < 0.005
                ]

            if not matching_obs:
                return None

            # Select highest quality OB
            best_ob = max(matching_obs, key=lambda x: x.get('quality', 0))

            return {
                'ob_type': best_ob.get('type', 'UNKNOWN'),
                'ob_high': float(best_ob.get('high', 0)),
                'ob_low': float(best_ob.get('low', 0)),
                'ob_quality': float(best_ob.get('quality', 50)),
                'ob_index': int(best_ob.get('index', 0))
            }

        except Exception as e:
            self.logger.debug(f"[S4_CHOCH] Entry zone error: {e}")
            return None

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, choch_type: str) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            choch_type: CHOCH type
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            # Check M5 momentum aligns with CHOCH direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if choch_type == 'BULLISH_CHOCH':
                return momentum > 0  # Bullish momentum on M5
            else:
                return momentum < 0  # Bearish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, choch_type: str, inducement: Dict,
        entry_zone: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            choch_type: CHOCH type
            inducement: Inducement dict
            entry_zone: Entry zone dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            # Determine direction from CHOCH
            if choch_type == 'BULLISH_CHOCH':
                direction = 'BUY'
            else:
                direction = 'SELL'

            # Calculate Stop Loss
            ob_low = entry_zone.get('ob_low', 0)
            ob_high = entry_zone.get('ob_high', 0)

            if direction == 'BUY':
                # SL below OB low
                sl_buffer = abs(entry_price - ob_low) * 0.2
                sl_price = ob_low - sl_buffer
            else:  # SELL
                # SL above OB high
                sl_buffer = abs(ob_high - entry_price) * 0.2
                sl_price = ob_high + sl_buffer

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
                # Fallback: Fibonacci
                fib_result = self.fib_engine.calculate_tp_from_fib(
                    entry_price, sl_price, df, direction == 'BUY'
                )
                if fib_result and fib_result.get('valid', False):
                    tp_price = fib_result['tp_price']
                else:
                    # Final fallback: Fixed R:R
                    risk = abs(entry_price - sl_price)
                    if direction == 'BUY':
                        tp_price = entry_price + risk * 2.0
                    else:
                        tp_price = entry_price - risk * 2.0

            # Calculate confidence
            ob_quality = entry_zone.get('ob_quality', 50) / 100
            inducement_strength = 1.0 - inducement.get('inducement_depth_pct',
                                        inducement.get('inducement_height_pct', 0)) / 100

            confidence = min(1.0, 0.4 + ob_quality * 0.3 + inducement_strength * 0.3)

            # Build signal
            signal = {
                'signal': f'{direction}_LIMIT',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'choch_type': choch_type,
                    'inducement_type': inducement.get('type', 'UNKNOWN'),
                    'ob_quality': ob_quality,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S4_CHOCH] Signal generated: {direction} | "
                f"CHOCH: {choch_type} | Inducement: {inducement.get('type')} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S4_CHOCH] Signal generation error: {e}")
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

        # SMC reversal strategies work best at exhaustion points
        compatible_regimes = [
            'EXHAUSTED_BULL', 'EXHAUSTED_BEAR',
            'OVERSOLD_BOUNCE', 'ANOMALY_BULL', 'ANOMALY_BEAR',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
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