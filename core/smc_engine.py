"""
Smart Money Concepts (SMC) Structural Engine.

Provides institutional-grade market structure analysis:
  - Swing High/Low Detection
  - Order Block Detection (Bullish/Bearish)
  - Breaker Block Detection (Failed OBs)
  - Fair Value Gap (FVG) Detection
  - Break of Structure (BOS) Detection
  - Change of Character (CHoCH) Detection
  - Liquidity Sweep Detection

Used by:
  - S1_IOB_Rejection
  - S4_CHOCH_IDM
  - S5_Breaker_Void
  - S7_MacroFVG
  - S21_BreakerFVGPOC
  - S22_WyckoffSpring
  - S30_VolumeProfileReversal
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque


class SMCStructuralEngine:
    """
    Smart Money Concepts structural analysis engine.
    
    Features:
      - Swing detection with configurable order
      - Order Block detection with quality scoring
      - Breaker Block detection
      - FVG detection with invalidation tracking
      - BOS/CHoCH detection
      - Liquidity sweep detection
      - OB freshness tracking
    """

    def __init__(self):
        """Initialize SMCStructuralEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default parameters
        self.swing_order = 5  # Number of bars on each side for swing
        self.ob_lookback = 100
        self.fvg_min_size_pct = 0.1  # Minimum FVG size as % of price

        # Tracking state
        self._recent_ob_history = deque(maxlen=50)
        self._recent_fvg_history = deque(maxlen=50)
        self._recent_swing_highs = deque(maxlen=20)
        self._recent_swing_lows = deque(maxlen=20)

    # =========================================================================
    # SWING DETECTION
    # =========================================================================

    def detect_swings(self, df: pd.DataFrame, order: int = None) -> Tuple[List[int], List[int]]:
        """
        Detect swing highs and swing lows.
        
        Swing High: Bar with 'order' lower highs on both sides
        Swing Low: Bar with 'order' higher lows on both sides
        
        Args:
            df: DataFrame with 'high', 'low' columns
            order: Number of bars on each side (default: 5)
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        if order is None:
            order = self.swing_order

        if df is None or df.empty or len(df) < order * 2 + 1:
            return [], []

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            swing_highs = []
            swing_lows = []

            # Find swing highs
            for i in range(order, len(high) - order):
                is_swing_high = True
                for j in range(1, order + 1):
                    if high[i] <= high[i - j] or high[i] <= high[i + j]:
                        is_swing_high = False
                        break
                if is_swing_high:
                    swing_highs.append(i)

            # Find swing lows
            for i in range(order, len(low) - order):
                is_swing_low = True
                for j in range(1, order + 1):
                    if low[i] >= low[i - j] or low[i] >= low[i + j]:
                        is_swing_low = False
                        break
                if is_swing_low:
                    swing_lows.append(i)

            return swing_highs, swing_lows

        except Exception as e:
            self.logger.error(f"[SMC] Swing detection error: {e}")
            return [], []

    # =========================================================================
    # ORDER BLOCK DETECTION
    # =========================================================================

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = None) -> List[Dict]:
        """
        Detect Order Blocks (OBs).
        
        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of bars to look back
            
        Returns:
            List of OB dicts with type, price levels, index, quality
        """
        if lookback is None:
            lookback = self.ob_lookback

        if df is None or df.empty or len(df) < 20:
            return []

        try:
            open_ = df['open'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            order_blocks = []
            start_idx = max(1, len(df) - lookback)

            for i in range(start_idx, len(df) - 2):
                # Determine if current candle is bullish or bearish
                is_current_bull = close[i] > open_[i]
                is_current_bear = close[i] < open_[i]

                # Look for strong move after current candle
                if i + 2 < len(df):
                    # Calculate strength of move (next 2 candles)
                    move_start = close[i]
                    move_end = close[i + 2]
                    move_pct = abs(move_end - move_start) / move_start * 100

                    # Strong move threshold (> 0.3%)
                    is_strong_move = move_pct > 0.3

                    if not is_strong_move:
                        continue

                    # Bullish OB: Bearish candle before strong bullish move
                    if is_current_bear and close[i + 2] > close[i]:
                        ob_quality = self._calculate_ob_quality(
                            i, df, 'BULLISH', move_pct
                        )

                        order_blocks.append({
                            'type': 'BULLISH',
                            'high': float(high[i]),
                            'low': float(low[i]),
                            'index': int(i),
                            'quality': ob_quality,
                            'move_pct': move_pct,
                            'fresh': True,  # Will be updated later
                            'mitigated': False
                        })

                        self._recent_ob_history.append({
                            'type': 'BULLISH',
                            'index': i,
                            'quality': ob_quality
                        })

                    # Bearish OB: Bullish candle before strong bearish move
                    elif is_current_bull and close[i + 2] < close[i]:
                        ob_quality = self._calculate_ob_quality(
                            i, df, 'BEARISH', move_pct
                        )

                        order_blocks.append({
                            'type': 'BEARISH',
                            'high': float(high[i]),
                            'low': float(low[i]),
                            'index': int(i),
                            'quality': ob_quality,
                            'move_pct': move_pct,
                            'fresh': True,
                            'mitigated': False
                        })

                        self._recent_ob_history.append({
                            'type': 'BEARISH',
                            'index': i,
                            'quality': ob_quality
                        })

            # Check OB freshness and mitigation
            self._check_ob_freshness(df, order_blocks)

            return order_blocks

        except Exception as e:
            self.logger.error(f"[SMC] Order block detection error: {e}")
            return []

    def _calculate_ob_quality(self, index: int, df: pd.DataFrame,
                               ob_type: str, move_pct: float) -> float:
        """
        Calculate Order Block quality score (0-100).
        
        Factors:
          - Move strength after OB
          - Volume confirmation (if available)
          - OB size (smaller is better)
          - Distance from recent swing
        """
        quality = 50.0  # Base quality

        try:
            open_ = df['open'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            # Factor 1: Move strength (max +25)
            if move_pct > 1.0:
                quality += 25
            elif move_pct > 0.5:
                quality += 15
            else:
                quality += 5

            # Factor 2: OB size (smaller is better, max +15)
            ob_range = high[index] - low[index]
            avg_range = np.mean(high[-20:] - low[-20:])

            if ob_range < avg_range * 0.7:
                quality += 15
            elif ob_range < avg_range:
                quality += 8

            # Factor 3: Volume confirmation (if available, max +10)
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values.astype(float)
                ob_volume = volume[index]
                avg_volume = np.mean(volume[-20:])

                if ob_volume > avg_volume * 1.5:
                    quality += 10
                elif ob_volume > avg_volume:
                    quality += 5

            return min(100.0, max(0.0, quality))

        except Exception:
            return 50.0

    def _check_ob_freshness(self, df: pd.DataFrame, order_blocks: List[Dict]):
        """Check if OBs are fresh or mitigated."""
        if not order_blocks or df is None or df.empty:
            return

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            last_idx = len(df) - 1

            for ob in order_blocks:
                ob_idx = ob['index']

                # Check if OB has been mitigated (price passed through)
                if ob['type'] == 'BULLISH':
                    # Bullish OB mitigated if price closed below OB low
                    for i in range(ob_idx + 1, last_idx + 1):
                        if close[i] < ob['low']:
                            ob['mitigated'] = True
                            break
                else:
                    # Bearish OB mitigated if price closed above OB high
                    for i in range(ob_idx + 1, last_idx + 1):
                        if close[i] > ob['high']:
                            ob['mitigated'] = True
                            break

                # Check freshness (OB age < 50 bars)
                age = last_idx - ob_idx
                ob['fresh'] = age < 50 and not ob['mitigated']

        except Exception:
            pass

    # =========================================================================
    # BREAKER BLOCK DETECTION
    # =========================================================================

    def detect_breaker_blocks(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """
        Detect Breaker Blocks (failed Order Blocks).
        
        Breaker: OB that was broken and now acts as S/R
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of bars to look back
            
        Returns:
            List of breaker block dicts
        """
        if df is None or df.empty or len(df) < 30:
            return []

        try:
            # First detect OBs
            order_blocks = self.detect_order_blocks(df, lookback)

            breakers = []
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            for ob in order_blocks:
                if ob['mitigated']:
                    # This OB was broken - now it's a breaker
                    ob_idx = ob['index']
                    last_idx = len(df) - 1

                    # Check if price is now reacting at this level
                    ob_mid = (ob['high'] + ob['low']) / 2
                    tolerance = (ob['high'] - ob['low']) * 0.3

                    # Check last 10 bars for reaction
                    recent_close = close[-10:]
                    recent_low = low[-10:]
                    recent_high = high[-10:]

                    if ob['type'] == 'BULLISH':
                        # Broken bullish OB now acts as resistance
                        if any(recent_high > ob_mid - tolerance and recent_high < ob_high + tolerance
                               for ob_high in [ob['high']]):
                            breakers.append({
                                'type': 'BEARISH_BREAKER',
                                'high': ob['high'],
                                'low': ob['low'],
                                'mid': ob_mid,
                                'index': ob_idx,
                                'quality': ob['quality'] * 0.8,  # Lower quality than OB
                                'original_type': 'BULLISH'
                            })
                    else:
                        # Broken bearish OB now acts as support
                        if any(recent_low < ob_mid + tolerance and recent_low > ob['low'] - tolerance
                               for ob_low in [ob['low']]):
                            breakers.append({
                                'type': 'BULLISH_BREAKER',
                                'high': ob['high'],
                                'low': ob['low'],
                                'mid': ob_mid,
                                'index': ob_idx,
                                'quality': ob['quality'] * 0.8,
                                'original_type': 'BEARISH'
                            })

            return breakers

        except Exception as e:
            self.logger.error(f"[SMC] Breaker block detection error: {e}")
            return []

    # =========================================================================
    # FVG (FAIR VALUE GAP) DETECTION
    # =========================================================================

    def detect_fvg(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVGs).
        
        Bullish FVG: Gap between candle 1 high and candle 3 low
        Bearish FVG: Gap between candle 1 low and candle 3 high
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of bars to look back
            
        Returns:
            List of FVG dicts
        """
        if df is None or df.empty or len(df) < 5:
            return []

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            fvgs = []
            start_idx = max(2, len(df) - lookback)

            for i in range(start_idx, len(df) - 2):
                # Bullish FVG: candle[i] high < candle[i+2] low
                if high[i] < low[i + 2]:
                    gap_size = low[i + 2] - high[i]
                    price_level = (high[i] + low[i + 2]) / 2
                    gap_pct = gap_size / price_level * 100

                    if gap_pct >= self.fvg_min_size_pct:
                        # Check if FVG is filled
                        filled = False
                        for j in range(i + 3, len(df)):
                            if low[j] < high[i]:
                                filled = True
                                break

                        fvgs.append({
                            'type': 'BULLISH',
                            'top': float(low[i + 2]),
                            'bottom': float(high[i]),
                            'mid': float(price_level),
                            'index': int(i + 1),  # Middle candle
                            'gap_size': float(gap_size),
                            'gap_pct': float(gap_pct),
                            'filled': filled,
                            'fresh': not filled
                        })

                        self._recent_fvg_history.append({
                            'type': 'BULLISH',
                            'index': i + 1,
                            'filled': filled
                        })

                # Bearish FVG: candle[i] low > candle[i+2] high
                elif low[i] > high[i + 2]:
                    gap_size = low[i] - high[i + 2]
                    price_level = (low[i] + high[i + 2]) / 2
                    gap_pct = gap_size / price_level * 100

                    if gap_pct >= self.fvg_min_size_pct:
                        # Check if FVG is filled
                        filled = False
                        for j in range(i + 3, len(df)):
                            if high[j] > low[i]:
                                filled = True
                                break

                        fvgs.append({
                            'type': 'BEARISH',
                            'top': float(low[i]),
                            'bottom': float(high[i + 2]),
                            'mid': float(price_level),
                            'index': int(i + 1),
                            'gap_size': float(gap_size),
                            'gap_pct': float(gap_pct),
                            'filled': filled,
                            'fresh': not filled
                        })

                        self._recent_fvg_history.append({
                            'type': 'BEARISH',
                            'index': i + 1,
                            'filled': filled
                        })

            return fvgs

        except Exception as e:
            self.logger.error(f"[SMC] FVG detection error: {e}")
            return []

    # =========================================================================
    # BOS / CHoCH DETECTION
    # =========================================================================

    def detect_bos_choch(self, df: pd.DataFrame) -> Dict:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH).
        
        BOS: Continuation of trend (higher high in uptrend, lower low in downtrend)
        CHoCH: Reversal signal (lower high in uptrend, higher low in downtrend)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with BOS and CHoCH information
        """
        result = {
            'current_trend': 'UNKNOWN',
            'bos_detected': False,
            'bos_type': None,
            'bos_price': 0.0,
            'bos_index': 0,
            'choch_detected': False,
            'choch_type': None,
            'choch_price': 0.0,
            'choch_index': 0,
            'structure_points': []
        }

        if df is None or df.empty or len(df) < 30:
            return result

        try:
            swing_highs, swing_lows = self.detect_swings(df, order=3)

            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return result

            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Get recent swings
            recent_highs = swing_highs[-3:]
            recent_lows = swing_lows[-3:]

            # Determine current trend
            higher_highs = high[recent_highs[-1]] > high[recent_highs[-2]]
            higher_lows = low[recent_lows[-1]] > low[recent_lows[-2]]
            lower_highs = high[recent_highs[-1]] < high[recent_highs[-2]]
            lower_lows = low[recent_lows[-1]] < low[recent_lows[-2]]

            if higher_highs and higher_lows:
                result['current_trend'] = 'UPTREND'
            elif lower_highs and lower_lows:
                result['current_trend'] = 'DOWNTREND'
            else:
                result['current_trend'] = 'SIDEWAY'

            # Check for BOS (continuation)
            if result['current_trend'] == 'UPTREND':
                # BOS in uptrend: new higher high
                if higher_highs:
                    result['bos_detected'] = True
                    result['bos_type'] = 'BULLISH_BOS'
                    result['bos_price'] = float(high[recent_highs[-1]])
                    result['bos_index'] = recent_highs[-1]

            elif result['current_trend'] == 'DOWNTREND':
                # BOS in downtrend: new lower low
                if lower_lows:
                    result['bos_detected'] = True
                    result['bos_type'] = 'BEARISH_BOS'
                    result['bos_price'] = float(low[recent_lows[-1]])
                    result['bos_index'] = recent_lows[-1]

            # Check for CHoCH (reversal)
            if result['current_trend'] == 'UPTREND' and lower_highs:
                result['choch_detected'] = True
                result['choch_type'] = 'BEARISH_CHOCH'
                result['choch_price'] = float(high[recent_highs[-1]])
                result['choch_index'] = recent_highs[-1]

            elif result['current_trend'] == 'DOWNTREND' and higher_lows:
                result['choch_detected'] = True
                result['choch_type'] = 'BULLISH_CHOCH'
                result['choch_price'] = float(low[recent_lows[-1]])
                result['choch_index'] = recent_lows[-1]

            # Build structure points
            for idx in recent_highs:
                result['structure_points'].append({
                    'type': 'HIGH',
                    'price': float(high[idx]),
                    'index': idx
                })
            for idx in recent_lows:
                result['structure_points'].append({
                    'type': 'LOW',
                    'price': float(low[idx]),
                    'index': idx
                })

            return result

        except Exception as e:
            self.logger.error(f"[SMC] BOS/CHoCH detection error: {e}")
            return result

    # =========================================================================
    # LIQUIDITY SWEEP DETECTION
    # =========================================================================

    def detect_liquidity_sweep(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Detect liquidity sweep patterns.
        
        Sweep: Price breaks swing high/low then quickly reverses
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of bars to look back
            
        Returns:
            List of sweep dicts
        """
        if df is None or df.empty or len(df) < 20:
            return []

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            swings_high, swings_low = self.detect_swings(df, order=3)

            sweeps = []
            start_idx = max(5, len(df) - lookback)

            # Check for upside sweep (break above swing high then reverse)
            for swing_idx in swings_high:
                if swing_idx < start_idx:
                    continue

                swing_level = high[swing_idx]

                # Check if price broke above then reversed
                for i in range(swing_idx + 1, min(swing_idx + 10, len(df))):
                    if high[i] > swing_level:
                        # Price broke above
                        # Check if it reversed (closed below swing level)
                        if close[i] < swing_level or (i + 1 < len(df) and close[i + 1] < swing_level):
                            sweeps.append({
                                'type': 'UPSIDE_SWEEP',
                                'sweep_level': float(swing_level),
                                'sweep_high': float(high[i]),
                                'index': int(i),
                                'depth_pct': float((high[i] - swing_level) / swing_level * 100)
                            })
                            break

            # Check for downside sweep (break below swing low then reverse)
            for swing_idx in swings_low:
                if swing_idx < start_idx:
                    continue

                swing_level = low[swing_idx]

                for i in range(swing_idx + 1, min(swing_idx + 10, len(df))):
                    if low[i] < swing_level:
                        # Price broke below
                        if close[i] > swing_level or (i + 1 < len(df) and close[i + 1] > swing_level):
                            sweeps.append({
                                'type': 'DOWNSIDE_SWEEP',
                                'sweep_level': float(swing_level),
                                'sweep_low': float(low[i]),
                                'index': int(i),
                                'depth_pct': float((swing_level - low[i]) / swing_level * 100)
                            })
                            break

            return sweeps

        except Exception as e:
            self.logger.error(f"[SMC] Liquidity sweep detection error: {e}")
            return []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_unmitigated_obs(self, df: pd.DataFrame, ob_type: str = None) -> List[Dict]:
        """
        Get unmitigated Order Blocks.
        
        Args:
            df: DataFrame with OHLC data
            ob_type: Filter by 'BULLISH' or 'BEARISH' (optional)
            
        Returns:
            List of unmitigated OBs
        """
        all_obs = self.detect_order_blocks(df)
        unmitigated = [ob for ob in all_obs if not ob.get('mitigated', False)]

        if ob_type:
            unmitigated = [ob for ob in unmitigated if ob['type'] == ob_type]

        return unmitigated

    def get_unfilled_fvgs(self, df: pd.DataFrame, fvg_type: str = None) -> List[Dict]:
        """
        Get unfilled Fair Value Gaps.
        
        Args:
            df: DataFrame with OHLC data
            fvg_type: Filter by 'BULLISH' or 'BEARISH' (optional)
            
        Returns:
            List of unfilled FVGs
        """
        all_fvgs = self.detect_fvg(df)
        unfilled = [fvg for fvg in all_fvgs if not fvg.get('filled', False)]

        if fvg_type:
            unfilled = [fvg for fvg in unfilled if fvg['type'] == fvg_type]

        return unfilled

    def get_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive market structure analysis.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict with complete structure analysis
        """
        swings_high, swings_low = self.detect_swings(df, order=3)
        order_blocks = self.detect_order_blocks(df)
        fvgs = self.detect_fvg(df)
        bos_choch = self.detect_bos_choch(df)
        sweeps = self.detect_liquidity_sweep(df)

        return {
            'swing_highs': swings_high,
            'swing_lows': swings_low,
            'order_blocks': order_blocks,
            'fvgs': fvgs,
            'bos_choch': bos_choch,
            'liquidity_sweeps': sweeps,
            'structure_summary': {
                'trend': bos_choch.get('current_trend', 'UNKNOWN'),
                'bos_detected': bos_choch.get('bos_detected', False),
                'choch_detected': bos_choch.get('choch_detected', False),
                'unmitigated_obs': len([ob for ob in order_blocks if not ob.get('mitigated', False)]),
                'unfilled_fvgs': len([fvg for fvg in fvgs if not fvg.get('filled', False)])
            }
        }