"""
Market Structure Analyzer - Dow Theory Implementation
Detects trend direction based on Swing Highs/Lows structure (Non-Lagging).
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List


class MarketStructureAnalyzer:
    """
    Analyzes market structure using Dow Theory principles.
    Provides non-lagging trend detection based on swing points.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_swings(self, df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Detect Swing Highs and Swing Lows using pivot point logic.
        
        Returns:
            - swing_high_indices: List of bar indices
            - swing_low_indices: List of bar indices  
            - swing_high_prices: List of prices
            - swing_low_prices: List of prices
        """
        if df is None or len(df) < (left_bars + right_bars + 1):
            return [], [], [], []
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        
        swing_highs_idx = []
        swing_lows_idx = []
        swing_highs_price = []
        swing_lows_price = []
        
        for i in range(left_bars, len(df) - right_bars):
            # Swing High detection
            is_swing_high = True
            for j in range(1, left_bars + 1):
                if high[i] <= high[i - j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                for j in range(1, right_bars + 1):
                    if high[i] <= high[i + j]:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                swing_highs_idx.append(i)
                swing_highs_price.append(float(high[i]))
            
            # Swing Low detection
            is_swing_low = True
            for j in range(1, left_bars + 1):
                if low[i] >= low[i - j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                for j in range(1, right_bars + 1):
                    if low[i] >= low[i + j]:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                swing_lows_idx.append(i)
                swing_lows_price.append(float(low[i]))
        
        return swing_highs_idx, swing_lows_idx, swing_highs_price, swing_lows_price
    
    def analyze_market_structure(self, df: pd.DataFrame, lookback_swings: int = 5) -> Dict:
        """
        Analyze market structure based on Dow Theory.
        
        Returns:
            {
                'structure': 'BULL' | 'BEAR' | 'TRANSITION' | 'RANGING',
                'last_break': 'BOS_UP' | 'BOS_DOWN' | 'CHOCH_UP' | 'CHOCH_DOWN' | None,
                'confidence': 0-100,
                'swing_count': int,
                'details': str
            }
        """
        if df is None or len(df) < 30:
            return {
                'structure': 'UNKNOWN',
                'last_break': None,
                'confidence': 0,
                'swing_count': 0,
                'details': 'Insufficient data'
            }
        
        try:
            # Detect swings with 3-bar pivot
            high_idx, low_idx, high_prices, low_prices = self.detect_swings(df, 3, 3)
            
            if len(high_prices) < 2 or len(low_prices) < 2:
                return {
                    'structure': 'RANGING',
                    'last_break': None,
                    'confidence': 30,
                    'swing_count': len(high_prices) + len(low_prices),
                    'details': 'Not enough swings detected'
                }
            
            # Analyze recent swings (last 5)
            recent_highs = high_prices[-lookback_swings:]
            recent_lows = low_prices[-lookback_swings:]
            
            # Count Higher Highs (HH) and Lower Highs (LH)
            hh_count = 0
            lh_count = 0
            for i in range(1, len(recent_highs)):
                if recent_highs[i] > recent_highs[i-1]:
                    hh_count += 1
                elif recent_highs[i] < recent_highs[i-1]:
                    lh_count += 1
            
            # Count Higher Lows (HL) and Lower Lows (LL)
            hl_count = 0
            ll_count = 0
            for i in range(1, len(recent_lows)):
                if recent_lows[i] > recent_lows[i-1]:
                    hl_count += 1
                elif recent_lows[i] < recent_lows[i-1]:
                    ll_count += 1
            
            # Determine structure
            total_highs = hh_count + lh_count
            total_lows = hl_count + ll_count
            
            if total_highs == 0 or total_lows == 0:
                return {
                    'structure': 'RANGING',
                    'last_break': None,
                    'confidence': 40,
                    'swing_count': len(high_prices) + len(low_prices),
                    'details': 'Incomplete swing data'
                }
            
            hh_ratio = hh_count / total_highs
            hl_ratio = hl_count / total_lows
            lh_ratio = lh_count / total_highs
            ll_ratio = ll_count / total_lows
            
            # BULL: HH + HL pattern (both > 60%)
            if hh_ratio >= 0.6 and hl_ratio >= 0.6:
                structure = 'BULL'
                confidence = int((hh_ratio + hl_ratio) / 2 * 100)
                details = f"HH:{hh_count}/{total_highs}, HL:{hl_count}/{total_lows}"
            
            # BEAR: LH + LL pattern (both > 60%)
            elif lh_ratio >= 0.6 and ll_ratio >= 0.6:
                structure = 'BEAR'
                confidence = int((lh_ratio + ll_ratio) / 2 * 100)
                details = f"LH:{lh_count}/{total_highs}, LL:{ll_count}/{total_lows}"
            
            # TRANSITION: Mixed patterns
            else:
                structure = 'TRANSITION'
                confidence = 50
                details = f"Mixed: HH:{hh_count}, LH:{lh_count}, HL:{hl_count}, LL:{ll_count}"
            
            # Detect Break of Structure (BOS) or Change of Character (CHOCH)
            last_break = self._detect_structure_break(df, high_idx, low_idx, high_prices, low_prices, structure)
            
            return {
                'structure': structure,
                'last_break': last_break,
                'confidence': confidence,
                'swing_count': len(high_prices) + len(low_prices),
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"[MARKET_STRUCTURE] Error: {e}")
            return {
                'structure': 'UNKNOWN',
                'last_break': None,
                'confidence': 0,
                'swing_count': 0,
                'details': f'Error: {e}'
            }
    
    def _detect_structure_break(self, df: pd.DataFrame, high_idx: List[int], low_idx: List[int],
                                high_prices: List[float], low_prices: List[float],
                                current_structure: str) -> str:
        """
        Detect Break of Structure (BOS) or Change of Character (CHOCH).
        
        BOS: Continuation break (BULL breaking new high, BEAR breaking new low)
        CHOCH: Reversal break (BULL breaking last low, BEAR breaking last high)
        """
        if len(high_prices) < 2 or len(low_prices) < 2:
            return None
        
        current_price = float(df['close'].iloc[-1])
        
        # Last significant high and low
        last_swing_high = high_prices[-1]
        last_swing_low = low_prices[-1]
        
        # Previous swing high and low
        prev_swing_high = high_prices[-2]
        prev_swing_low = low_prices[-2]
        
        if current_structure == 'BULL':
            # CHOCH: Price breaks below last swing low (trend reversal signal)
            if current_price < last_swing_low:
                return 'CHOCH_DOWN'
            # BOS: Price breaks above last swing high (continuation)
            elif current_price > last_swing_high:
                return 'BOS_UP'
        
        elif current_structure == 'BEAR':
            # CHOCH: Price breaks above last swing high (trend reversal signal)
            if current_price > last_swing_high:
                return 'CHOCH_UP'
            # BOS: Price breaks below last swing low (continuation)
            elif current_price < last_swing_low:
                return 'BOS_DOWN'
        
        elif current_structure == 'TRANSITION':
            # In transition, any break is significant
            if current_price > last_swing_high:
                return 'BOS_UP'
            elif current_price < last_swing_low:
                return 'BOS_DOWN'
        
        return None
    
    def get_trend_score(self, df: pd.DataFrame) -> Tuple[int, str]:
        """
        Get trend score for integration with RegimeRouter.
        
        Returns:
            - score: -2 to +2 (BEAR to BULL)
            - reason: explanation
        """
        result = self.analyze_market_structure(df)
        structure = result['structure']
        last_break = result['last_break']
        confidence = result['confidence']
        
        score = 0
        reason = f"Structure: {structure}, Conf: {confidence}%"
        
        if structure == 'BULL':
            score = 2
            if last_break == 'BOS_UP':
                reason += " + BOS_UP (strong continuation)"
            elif last_break == 'CHOCH_DOWN':
                score = 0  # Reversal signal cancels bull structure
                reason += " but CHOCH_DOWN detected (reversal warning)"
        
        elif structure == 'BEAR':
            score = -2
            if last_break == 'BOS_DOWN':
                reason += " + BOS_DOWN (strong continuation)"
            elif last_break == 'CHOCH_UP':
                score = 0  # Reversal signal cancels bear structure
                reason += " but CHOCH_UP detected (reversal warning)"
        
        elif structure == 'TRANSITION':
            if last_break == 'BOS_UP':
                score = 1
                reason += " + BOS_UP (potential bullish shift)"
            elif last_break == 'BOS_DOWN':
                score = -1
                reason += " + BOS_DOWN (potential bearish shift)"
        
        return score, reason