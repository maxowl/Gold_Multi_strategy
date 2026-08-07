"""
S9_SessionSweep - Session Liquidity Sweep Strategy.

Scalping strategy that identifies liquidity sweeps at session
highs and lows, trading the reversal after fake breakouts.

Strategy Logic:
  1. Detect trading sessions (Asian, London, NY)
  2. Identify session high/low levels
  3. Detect liquidity sweeps at these levels
  4. Generate entry signal after sweep confirmation

Session Liquidity Concept:
  Each trading session creates liquidity pools at:
    - Session High: Buy-side liquidity (stop losses)
    - Session Low: Sell-side liquidity (stop losses)
  
  Market makers often "sweep" these levels to trigger
  stop losses before reversing - creating trading opportunities.

Used Engines:
  - SessionVolatilityEngine: Session detection
  - SMCStructuralEngine: Swing detection for levels
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


class S9_SessionSweep(BaseStrategy):
    """
    Session Liquidity Sweep Strategy.
    
    This strategy identifies liquidity sweeps at session highs
    and lows, trading the reversal after fake breakouts.
    
    Session Definition:
      - Asian Session: 00:00 - 08:00 UTC
      - London Session: 07:00 - 16:00 UTC
      - NY Session: 13:00 - 22:00 UTC
      
    Sweep Definition:
      Price breaks above/below session high/low to trigger
      stop losses, then quickly reverses - a liquidity grab.
      
    Entry Criteria:
      - Session identified
      - Session high/low detected
      - Price sweeps the level
      - Reversal confirmed
    """

    def __init__(self):
        """Initialize S9_SessionSweep strategy."""
        super().__init__(
            strategy_name='S9_SessionSweep',
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
        self.session_lookback_hours = 8  # Lookback for session levels
        self.sweep_tolerance_pct = 0.3  # Sweep tolerance (% of price)
        self.min_sweep_depth = 0.1  # Minimum sweep depth (USD)

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
        Main analysis method for S9_SessionSweep.
        
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
            # STEP 1: Detect Current Session
            # =========================================================================
            current_session = self._detect_current_session(df_m15)

            if current_session is None:
                return default_signal

            # =========================================================================
            # STEP 2: Get Session Levels
            # =========================================================================
            session_levels = self._get_session_levels(df_m15, current_session)

            if session_levels is None:
                return default_signal

            # =========================================================================
            # STEP 3: Detect Sweep
            # =========================================================================
            sweep = self._detect_sweep(df_m15, session_levels)

            if sweep is None or not sweep.get('sweep_detected', False):
                return default_signal

            # =========================================================================
            # STEP 4: Confirm Reversal
            # =========================================================================
            if not self._confirm_reversal(df_m15, sweep):
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, sweep):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, sweep, current_session, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S9_SESSION] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # SESSION DETECTION
    # =========================================================================

    def _detect_current_session(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect current trading session.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Session dict or None
        """
        try:
            # Get current time from last bar
            if 'time' not in df.columns:
                # Use system time as fallback
                current_time = datetime.utcnow()
            else:
                last_time = df['time'].iloc[-1]
                if isinstance(last_time, pd.Timestamp):
                    current_time = last_time
                else:
                    current_time = pd.to_datetime(last_time)

            current_hour = current_time.hour

            # Define sessions (UTC hours)
            sessions = {
                'ASIAN': {'start': 0, 'end': 8, 'priority': 1},
                'LONDON': {'start': 7, 'end': 16, 'priority': 2},
                'NY': {'start': 13, 'end': 22, 'priority': 3},
                'LATE': {'start': 22, 'end': 24, 'priority': 1}
            }

            # Find current session
            for session_name, session_info in sessions.items():
                if session_info['start'] <= current_hour < session_info['end']:
                    return {
                        'name': session_name,
                        'start_hour': session_info['start'],
                        'end_hour': session_info['end'],
                        'priority': session_info['priority'],
                        'current_hour': current_hour
                    }

            return None

        except Exception as e:
            self.logger.debug(f"[S9_SESSION] Session detection error: {e}")
            return None

    # =========================================================================
    # SESSION LEVELS
    # =========================================================================

    def _get_session_levels(self, df: pd.DataFrame, session: Dict) -> Optional[Dict]:
        """
        Get session high/low levels.
        
        Args:
            df: DataFrame with OHLCV data
            session: Current session dict
            
        Returns:
            Session levels dict or None
        """
        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            # Calculate lookback based on session duration
            session_hours = session['end_hour'] - session['start_hour']
            bars_lookback = min(len(df), session_hours * 4)  # M15 = 4 bars per hour

            # Get session data
            session_high = high[-bars_lookback:]
            session_low = low[-bars_lookback:]

            # Find session high and low
            high_level = float(np.max(session_high))
            low_level = float(np.min(session_low))

            # Find recent swing highs/lows for confirmation
            swings_high, swings_low = self.smc_engine.detect_swings(df, order=3)

            # Get nearest swing high/low
            current_price = (high[-1] + low[-1]) / 2

            nearest_high = None
            nearest_low = None

            for idx in swings_high[-3:]:  # Last 3 swing highs
                if idx < len(df):
                    swing_price = high[idx]
                    if swing_price > current_price and (nearest_high is None or swing_price < nearest_high):
                        nearest_high = swing_price

            for idx in swings_low[-3:]:  # Last 3 swing lows
                if idx < len(df):
                    swing_price = low[idx]
                    if swing_price < current_price and (nearest_low is None or swing_price > nearest_low):
                        nearest_low = swing_price

            return {
                'session_high': high_level,
                'session_low': low_level,
                'swing_high': nearest_high,
                'swing_low': nearest_low,
                'session_name': session['name']
            }

        except Exception as e:
            self.logger.debug(f"[S9_SESSION] Session levels error: {e}")
            return None

    # =========================================================================
    # SWEEP DETECTION
    # =========================================================================

    def _detect_sweep(self, df: pd.DataFrame, session_levels: Dict) -> Optional[Dict]:
        """
        Detect liquidity sweep at session levels.
        
        Args:
            df: DataFrame with OHLCV data
            session_levels: Session levels dict
            
        Returns:
            Sweep dict or None
        """
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))

            current_price = close[-1]
            session_high = session_levels.get('session_high', 0)
            session_low = session_levels.get('session_low', 0)

            # Check for upside sweep (break above session high, then reverse)
            upside_sweep = self._detect_upside_sweep(
                close, high, low, session_high, current_price
            )

            if upside_sweep:
                upside_sweep['direction'] = 'SELL'  # Fade the breakout
                upside_sweep['sweep_level'] = session_high
                return upside_sweep

            # Check for downside sweep (break below session low, then reverse)
            downside_sweep = self._detect_downside_sweep(
                close, high, low, session_low, current_price
            )

            if downside_sweep:
                downside_sweep['direction'] = 'BUY'  # Fade the breakout
                downside_sweep['sweep_level'] = session_low
                return downside_sweep

            return {'sweep_detected': False}

        except Exception as e:
            self.logger.debug(f"[S9_SESSION] Sweep detection error: {e}")
            return None

    def _detect_upside_sweep(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        session_high: float, current_price: float
    ) -> Optional[Dict]:
        """Detect upside liquidity sweep."""
        try:
            n = len(close)
            if n < 5:
                return None

            # Check recent bars for sweep
            for i in range(max(1, n - 5), n):
                # Price broke above session high
                if high[i] > session_high:
                    sweep_high = high[i]
                    sweep_depth = (sweep_high - session_high) / session_high * 100

                    # Check if sweep is shallow (typical for fake breakouts)
                    if sweep_depth < self.sweep_tolerance_pct * 10:
                        # Check if price reversed below session high
                        if current_price < session_high:
                            reversal_strength = (sweep_high - current_price) / (sweep_high - session_high + 1e-10)

                            return {
                                'sweep_detected': True,
                                'sweep_type': 'UPSIDE_SWEEP',
                                'sweep_high': float(sweep_high),
                                'sweep_depth_pct': float(sweep_depth),
                                'reversal_strength': float(min(1.0, reversal_strength)),
                                'reversed': True
                            }

            return None

        except Exception:
            return None

    def _detect_downside_sweep(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray,
        session_low: float, current_price: float
    ) -> Optional[Dict]:
        """Detect downside liquidity sweep."""
        try:
            n = len(close)
            if n < 5:
                return None

            # Check recent bars for sweep
            for i in range(max(1, n - 5), n):
                # Price broke below session low
                if low[i] < session_low:
                    sweep_low = low[i]
                    sweep_depth = (session_low - sweep_low) / session_low * 100

                    # Check if sweep is shallow
                    if sweep_depth < self.sweep_tolerance_pct * 10:
                        # Check if price reversed above session low
                        if current_price > session_low:
                            reversal_strength = (current_price - sweep_low) / (session_low - sweep_low + 1e-10)

                            return {
                                'sweep_detected': True,
                                'sweep_type': 'DOWNSIDE_SWEEP',
                                'sweep_low': float(sweep_low),
                                'sweep_depth_pct': float(sweep_depth),
                                'reversal_strength': float(min(1.0, reversal_strength)),
                                'reversed': True
                            }

            return None

        except Exception:
            return None

    # =========================================================================
    # REVERSAL CONFIRMATION
    # =========================================================================

    def _confirm_reversal(self, df: pd.DataFrame, sweep: Dict) -> bool:
        """
        Confirm reversal after sweep.
        
        Args:
            df: DataFrame with OHLCV data
            sweep: Sweep dict
            
        Returns:
            True if reversal is confirmed
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = sweep.get('direction', 'BUY')
            reversal_strength = sweep.get('reversal_strength', 0)

            # Check recent momentum
            recent_close = close[-5:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                # For BUY: Price should be rising
                return momentum > 0 and reversal_strength > 0.3
            else:  # SELL
                # For SELL: Price should be falling
                return momentum < 0 and reversal_strength > 0.3

        except Exception:
            return False

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, sweep: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            sweep: Sweep dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = sweep.get('direction', 'BUY')

            # Check M5 momentum aligns with sweep direction
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
        self, df: pd.DataFrame, sweep: Dict, session: Dict,
        regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            sweep: Sweep dict
            session: Session dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = sweep.get('direction', 'BUY')
            sweep_level = sweep.get('sweep_level', 0)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            if entry_price <= 0 or sweep_level <= 0:
                return self._create_neutral_signal()

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below sweep low with buffer
                sweep_low = sweep.get('sweep_low', sweep_level)
                sl_buffer = abs(entry_price - sweep_low) * 0.3
                sl_price = sweep_low - sl_buffer
            else:  # SELL
                # SL above sweep high with buffer
                sweep_high = sweep.get('sweep_high', sweep_level)
                sl_buffer = abs(sweep_high - entry_price) * 0.3
                sl_price = sweep_high + sl_buffer

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
            reversal_strength = sweep.get('reversal_strength', 0.5)
            session_priority = session.get('priority', 1)

            confidence = min(1.0, 0.4 + reversal_strength * 0.4 + session_priority * 0.1)

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
                    'sweep_type': sweep.get('sweep_type', 'UNKNOWN'),
                    'session_name': session.get('name', 'UNKNOWN'),
                    'sweep_depth_pct': sweep.get('sweep_depth_pct', 0),
                    'reversal_strength': reversal_strength,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit,
                    'friction_sensitive': self.friction_sensitive
                }
            }

            self.logger.info(
                f"[S9_SESSION] Signal generated: {direction} | "
                f"Session: {session.get('name')} | "
                f"Sweep: {sweep.get('sweep_type')} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S9_SESSION] Signal generation error: {e}")
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