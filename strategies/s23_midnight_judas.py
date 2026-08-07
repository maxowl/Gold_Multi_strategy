"""
S23_MidnightJudas - Midnight Judas Strategy.

Scalping strategy that detects "Judas Swings" (fake breakouts) during
session transitions, particularly around midnight and session opens.

Strategy Logic:
  1. Detect session transitions (midnight, London open, NY open)
  2. Identify Judas Swings (fake breakouts that trap traders)
  3. Confirm sweep and reversal
  4. Generate entry signal after Judas completion

Judas Swing Definition:
  A "Judas Swing" is a fake breakout that occurs during session
  transitions, designed to trap retail traders:
    - Price breaks above resistance (traps buyers)
    - Or breaks below support (traps sellers)
    - Then quickly reverses
    
  The name comes from the idea of a "betrayal" - the move looks
  real but reverses, trapping traders on the wrong side.

Session Transitions:
  - Midnight: 00:00 UTC (Asian session start)
  - London Open: 07:00-08:00 UTC
  - NY Open: 13:00-14:00 UTC
  
  These transitions often have liquidity sweeps and fake moves.

Used Engines:
  - SessionVolatilityEngine: Session detection
  - SMCStructuralEngine: Swing and sweep detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SCALP
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.base_strategy import BaseStrategy
from core.session_volatility import SessionVolatilityEngine
from core.smc_engine import SMCStructuralEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S23_MidnightJudas(BaseStrategy):
    """
    Midnight Judas Strategy.
    
    This strategy detects Judas Swings (fake breakouts) during
    session transitions and trades the reversal.
    
    Judas Swing Definition:
      A fake breakout that traps traders on the wrong side,
      then reverses. Common during session transitions when
      liquidity is being swept.
      
    Judas Characteristics:
      - Price breaks key level (swing high/low)
      - Breakout is shallow (typically < 0.3%)
      - Quick reversal back through the level
      - Volume spike during the fake breakout
      
    Session Transition Times:
      - Midnight: 00:00 UTC
      - London Open: 07:00-08:00 UTC
      - NY Open: 13:00-14:00 UTC
      
    Entry Criteria:
      - Session transition detected
      - Judas Swing identified
      - Reversal confirmed
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S23_MidnightJudas strategy."""
        super().__init__(
            strategy_name='S23_MidnightJudas',
            strategy_category='SCALP',
            timeframes=['M15', 'M5'],
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
        self.session_engine = SessionVolatilityEngine()
        self.smc_engine = SMCStructuralEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.judas_lookback = 30  # Lookback for Judas detection
        self.judas_depth_threshold = 0.003  # Maximum Judas depth (0.3%)
        self.transition_window = 30  # Minutes around session transition
        self.min_judas_strength = 0.5  # Minimum Judas strength

        # Session transition times (UTC hours)
        self.transition_times = [0, 7, 8, 13, 14]  # Midnight, London, NY

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
        Main analysis method for S23_MidnightJudas.
        
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
            # STEP 1: Detect Session Transition
            # =========================================================================
            transition = self._detect_transition(df_m15)

            if transition is None:
                return default_signal

            # =========================================================================
            # STEP 2: Detect Judas Swing
            # =========================================================================
            judas = self._detect_judas(df_m15)

            if judas is None:
                return default_signal

            # =========================================================================
            # STEP 3: Confirm Reversal
            # =========================================================================
            if not self._confirm_reversal(df_m15, judas):
                return default_signal

            # =========================================================================
            # STEP 4: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, judas):
                    return default_signal

            # =========================================================================
            # STEP 5: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, judas, transition, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S23_JUDAS] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SESSION TRANSITION DETECTION
    # =========================================================================

    def _detect_transition(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect session transition.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Transition dict or None
        """
        try:
            # Get current time from last bar
            if 'time' not in df.columns:
                current_time = datetime.utcnow()
            else:
                last_time = df['time'].iloc[-1]
                if isinstance(last_time, pd.Timestamp):
                    current_time = last_time
                else:
                    current_time = pd.to_datetime(last_time)

            current_hour = current_time.hour
            current_minute = current_time.minute

            # Check if near session transition
            for transition_hour in self.transition_times:
                # Check if within transition window
                time_diff = abs(current_hour * 60 + current_minute - transition_hour * 60)

                if time_diff <= self.transition_window:
                    return {
                        'transition_detected': True,
                        'transition_hour': transition_hour,
                        'current_hour': current_hour,
                        'current_minute': current_minute,
                        'time_diff_minutes': time_diff
                    }

            return None

        except Exception as e:
            self.logger.debug(f"[S23_JUDAS] Transition detection error: {e}")
            return None

    # =========================================================================
    # JUDAS SWING DETECTION
    # =========================================================================

    def _detect_judas(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Judas Swing (fake breakout).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Judas dict or None
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Detect swings
            swings_high, swings_low = self.smc_engine.detect_swings(df, order=3)

            if not swings_high or not swings_low:
                return None

            current_price = close[-1]

            # Check for upside Judas (fake breakout above swing high)
            upside_judas = self._detect_upside_judas(
                close, high, low, swings_high, current_price
            )

            if upside_judas:
                return upside_judas

            # Check for downside Judas (fake breakout below swing low)
            downside_judas = self._detect_downside_judas(
                close, high, low, swings_low, current_price
            )

            if downside_judas:
                return downside_judas

            return None

        except Exception as e:
            self.logger.debug(f"[S23_JUDAS] Judas detection error: {e}")
            return None

    def _detect_upside_judas(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        swings_high: List[int], current_price: float
    ) -> Optional[Dict]:
        """Detect upside Judas Swing (fake breakout above resistance)."""
        try:
            n = len(close)
            if n < 5:
                return None

            # Get recent swing highs
            recent_swing_highs = [idx for idx in swings_high if idx >= n - 20]

            for swing_idx in recent_swing_highs:
                swing_level = high[swing_idx]

                # Check if price broke above swing level recently
                for i in range(max(1, n - 5), n):
                    if high[i] > swing_level:
                        judas_high = high[i]
                        judas_depth = (judas_high - swing_level) / swing_level

                        # Check if Judas is shallow
                        if judas_depth < self.judas_depth_threshold:
                            # Check if price reversed back below swing level
                            if current_price < swing_level:
                                reversal_strength = (judas_high - current_price) / (judas_high - swing_level + 1e-10)

                                return {
                                    'judas_type': 'UPSIDE_JUDAS',
                                    'direction': 'SELL',  # Fade the fake breakout
                                    'swing_level': float(swing_level),
                                    'judas_high': float(judas_high),
                                    'judas_depth_pct': float(judas_depth * 100),
                                    'reversal_strength': float(min(1.0, reversal_strength)),
                                    'reversed': True
                                }

            return None

        except Exception:
            return None

    def _detect_downside_judas(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        swings_low: List[int], current_price: float
    ) -> Optional[Dict]:
        """Detect downside Judas Swing (fake breakout below support)."""
        try:
            n = len(close)
            if n < 5:
                return None

            # Get recent swing lows
            recent_swing_lows = [idx for idx in swings_low if idx >= n - 20]

            for swing_idx in recent_swing_lows:
                swing_level = low[swing_idx]

                # Check if price broke below swing level recently
                for i in range(max(1, n - 5), n):
                    if low[i] < swing_level:
                        judas_low = low[i]
                        judas_depth = (swing_level - judas_low) / swing_level

                        # Check if Judas is shallow
                        if judas_depth < self.judas_depth_threshold:
                            # Check if price reversed back above swing level
                            if current_price > swing_level:
                                reversal_strength = (current_price - judas_low) / (swing_level - judas_low + 1e-10)

                                return {
                                    'judas_type': 'DOWNSIDE_JUDAS',
                                    'direction': 'BUY',  # Fade the fake breakout
                                    'swing_level': float(swing_level),
                                    'judas_low': float(judas_low),
                                    'judas_depth_pct': float(judas_depth * 100),
                                    'reversal_strength': float(min(1.0, reversal_strength)),
                                    'reversed': True
                                }

            return None

        except Exception:
            return None

    # =========================================================================
    # REVERSAL CONFIRMATION
    # =========================================================================

    def _confirm_reversal(self, df: pd.DataFrame, judas: Dict) -> bool:
        """
        Confirm reversal after Judas Swing.
        
        Args:
            df: DataFrame with OHLCV data
            judas: Judas dict
            
        Returns:
            True if reversal is confirmed
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = judas.get('direction', 'BUY')
            reversal_strength = judas.get('reversal_strength', 0)

            # Check recent momentum
            recent_close = close[-5:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY: Price should be rising
                return momentum > 0 and reversal_strength > self.min_judas_strength
            else:  # SELL
                # For SELL: Price should be falling
                return momentum < 0 and reversal_strength > self.min_judas_strength

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, judas: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            judas: Judas dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = judas.get('direction', 'BUY')

            # Check M5 momentum aligns with Judas direction
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
        self, df: pd.DataFrame, judas: Dict, transition: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            judas: Judas dict
            transition: Transition dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = judas.get('direction', 'BUY')
            swing_level = judas.get('swing_level', 0)

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            entry_price = close[-1]

            if entry_price <= 0 or swing_level <= 0:
                return self._create_neutral_signal()

            # Calculate ATR for stop loss
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below Judas low
                judas_low = judas.get('judas_low', swing_level)
                sl_buffer = abs(entry_price - judas_low) * 0.3
                sl_price = judas_low - sl_buffer
            else:  # SELL
                # SL above Judas high
                judas_high = judas.get('judas_high', swing_level)
                sl_buffer = abs(judas_high - entry_price) * 0.3
                sl_price = judas_high + sl_buffer

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
            reversal_strength = judas.get('reversal_strength', 0.5)
            transition_bonus = 0.1 if transition.get('time_diff_minutes', 60) < 15 else 0.0
            confidence = min(1.0, 0.4 + reversal_strength * 0.4 + transition_bonus)

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
                    'judas_type': judas.get('judas_type', 'UNKNOWN'),
                    'judas_depth_pct': judas.get('judas_depth_pct', 0),
                    'swing_level': round(swing_level, 2),
                    'transition_hour': transition.get('transition_hour', 0),
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S23_JUDAS] Signal generated: {direction} | "
                f"Judas: {judas.get('judas_type')} | "
                f"Depth: {judas.get('judas_depth_pct', 0):.2f}% | "
                f"Transition: {transition.get('transition_hour')}:00 | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S23_JUDAS] Signal generation error: {e}")
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