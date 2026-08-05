"""
Dangerous Pattern Detection Engine - CORRECTED VERSION.
Fixed 5 Critical Issues and 7 Moderate Issues.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from core.atr_cache import ATRCache
from core.smc_engine import SMCStructuralEngine


class PatternDetector:
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_LOW = 'LOW'
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smc_engine = SMCStructuralEngine()
        self._last_detection_bar = {}  # Cooldown tracking
    
    def detect_dangerous_patterns(
        self, 
        df: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        regime_name: str = 'UNKNOWN',
        timeframe: str = 'M15'  # [NEW] Add TF parameter
    ) -> Dict:
        """Main entry point with all corrections applied."""
        if df is None or len(df) < 50:
            return {
                'pattern_detected': False,
                'pattern_type': 'NONE',
                'confidence': 0.0,
                'action': 'NONE',
                'reason': 'Insufficient data'
            }
        
        is_buy = 'BUY' in signal_type

        # [NEW] Adjust parameters based on timeframe
        tf_multiplier = self._get_tf_multiplier(timeframe)
        
        # Check cooldown (don't detect same pattern in 10 bars)
        current_bar = len(df)
        cooldown_bars = int(10 * tf_multiplier)
        
        patterns_detected = []
        
        # Category 1: Liquidity Trap
        liquidity_trap = self._detect_liquidity_trap(df, is_buy, entry_price)
        if liquidity_trap['detected']:
            last_bar = liquidity_trap.get('bar_index', current_bar)
            if current_bar - self._last_detection_bar.get('LIQUIDITY', 0) > cooldown_bars:
                patterns_detected.append(liquidity_trap)
                self._last_detection_bar['LIQUIDITY'] = current_bar
        
        # Category 2: False Breakout
        false_breakout = self._detect_false_breakout(df, is_buy, entry_price)
        if false_breakout['detected']:
            if current_bar - self._last_detection_bar.get('FALSE_BREAKOUT', 0) > cooldown_bars:
                # Adjust severity based on regime
                if 'TREND' in regime_name:
                    false_breakout['severity'] = self.SEVERITY_LOW  # Less severe in trending
                patterns_detected.append(false_breakout)
                self._last_detection_bar['FALSE_BREAKOUT'] = current_bar
        
        # Category 3: Continuation Trap
        continuation_trap = self._detect_continuation_trap(df, is_buy)
        if continuation_trap['detected']:
            if current_bar - self._last_detection_bar.get('CONTINUATION', 0) > cooldown_bars:
                patterns_detected.append(continuation_trap)
                self._last_detection_bar['CONTINUATION'] = current_bar
        
        # Category 4: News/Event Pattern
        news_pattern = self._detect_news_pattern(df)
        if news_pattern['detected']:
            if current_bar - self._last_detection_bar.get('NEWS', 0) > cooldown_bars:
                patterns_detected.append(news_pattern)
                self._last_detection_bar['NEWS'] = current_bar
        
        if not patterns_detected:
            return {
                'pattern_detected': False,
                'pattern_type': 'NONE',
                'confidence': 0.0,
                'action': 'NONE',
                'reason': 'No dangerous patterns detected',
                'patterns': []
            }
        
        # Filter by minimum confidence (0.6)
        patterns_detected = [p for p in patterns_detected if p['confidence'] >= 0.6]
        
        if not patterns_detected:
            return {
                'pattern_detected': False,
                'pattern_type': 'NONE',
                'confidence': 0.0,
                'action': 'NONE',
                'reason': 'Patterns detected but below confidence threshold',
                'patterns': []
            }
        
        highest_severity_pattern = max(
            patterns_detected, 
            key=lambda p: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(p['severity'], 0)
        )
        
        action = self._determine_action(highest_severity_pattern, patterns_detected)
        
        pattern_names = [p['type'] for p in patterns_detected]
        reason = f"Detected {len(patterns_detected)} pattern(s): {', '.join(pattern_names)}"
        
        return {
            'pattern_detected': True,
            'pattern_type': highest_severity_pattern['type'],
            'confidence': highest_severity_pattern['confidence'],
            'severity': highest_severity_pattern['severity'],
            'action': action,
            'reason': reason,
            'patterns': patterns_detected,
            'recommendation': highest_severity_pattern.get('recommendation', ''),
            'timeframe': timeframe  # [NEW]
        }
    def _get_tf_multiplier(self, timeframe: str) -> float:
        """Get multiplier for timeframe-specific adjustments."""
        tf_map = {
            'M1': 0.5,
            'M5': 0.7,
            'M15': 1.0,  # Baseline
            'M30': 1.3,
            'H1': 1.5,
            'H4': 2.0,
            'D1': 3.0
        }
        return tf_map.get(timeframe, 1.0)
    
    def _detect_equal_levels(self, prices: np.ndarray, df: pd.DataFrame, tolerance_atr: float = 0.5) -> List[float]:
        """[FIXED] Detect equal levels using ATR-based tolerance."""
        atr = ATRCache.get_atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return []
        
        tolerance = atr * tolerance_atr
        
        equal_levels = []
        used_indices = set()
        
        for i in range(len(prices)):
            if i in used_indices:
                continue
            
            level = prices[i]
            matches = [level]
            
            for j in range(i + 1, len(prices)):
                if j in used_indices:
                    continue
                
                if abs(prices[j] - level) <= tolerance:
                    matches.append(prices[j])
                    used_indices.add(j)
            
            if len(matches) >= 2:
                equal_levels.extend(matches)
                used_indices.add(i)
        
        return equal_levels
    
    def _detect_liquidity_trap(self, df: pd.DataFrame, is_buy: bool, entry_price: float) -> Dict:
        """[FIXED] Detect Liquidity Trap with time recency check."""
        if len(df) < 30:
            return {'detected': False}
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        atr = ATRCache.get_atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return {'detected': False}
        
        if is_buy:
            equal_highs = self._detect_equal_levels(high[-30:], df, tolerance_atr=0.5)
            
            if len(equal_highs) >= 3:
                max_equal_high = max(equal_highs)
                
                recent_bars = 3
                recent_high = high[-recent_bars:].max()
                current_close = close[-1]
                
                sweep_distance = atr * 0.3
                
                if (recent_high > max_equal_high and 
                    current_close < max_equal_high and
                    (recent_high - max_equal_high) < sweep_distance):
                    
                    return {
                        'detected': True,
                        'type': 'EQUAL_HIGHS_SWEEP',
                        'severity': self.SEVERITY_HIGH,
                        'confidence': 0.85,
                        'bar_index': len(df),
                        'description': (
                            f"Equal highs at {max_equal_high:.2f} (×{len(equal_highs)}). "
                            f"Swept to {recent_high:.2f} in last {recent_bars} bars then reversed."
                        ),
                        'recommendation': 'BLOCK_TRADE'
                    }
        
        else:
            equal_lows = self._detect_equal_levels(low[-30:], df, tolerance_atr=0.5)
            
            if len(equal_lows) >= 3:
                min_equal_low = min(equal_lows)
                
                recent_bars = 3
                recent_low = low[-recent_bars:].min()
                current_close = close[-1]
                
                sweep_distance = atr * 0.3
                
                if (recent_low < min_equal_low and 
                    current_close > min_equal_low and
                    (min_equal_low - recent_low) < sweep_distance):
                    
                    return {
                        'detected': True,
                        'type': 'EQUAL_LOWS_SWEEP',
                        'severity': self.SEVERITY_HIGH,
                        'confidence': 0.85,
                        'bar_index': len(df),
                        'description': (
                            f"Equal lows at {min_equal_low:.2f} (×{len(equal_lows)}). "
                            f"Swept to {recent_low:.2f} in last {recent_bars} bars then reversed."
                        ),
                        'recommendation': 'BLOCK_TRADE'
                    }
        
        return {'detected': False}
    
    def _detect_false_breakout(self, df: pd.DataFrame, is_buy: bool, entry_price: float) -> Dict:
        """[FIXED] Detect False Breakout with ATR-based thresholds."""
        if len(df) < 20:
            return {'detected': False}
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        if 'tick_volume' in df.columns:
            volume = df['tick_volume'].to_numpy()
        else:
            volume = np.ones(len(df))
        
        atr = ATRCache.get_atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return {'detected': False}
        
        if is_buy:
            recent_high_idx = np.argmax(high[-20:])
            recent_high = high[-20 + recent_high_idx]
            
            if close[-1] > recent_high:
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                
                if current_volume < avg_volume * 1.2:
                    weakness_threshold = atr * 0.3
                    if (high[-1] - close[-1]) > weakness_threshold:
                        return {
                            'detected': True,
                            'type': 'BULL_TRAP',
                            'severity': self.SEVERITY_MEDIUM,
                            'confidence': 0.75,
                            'bar_index': len(df),
                            'description': (
                                f"Broke above {recent_high:.2f} with low volume. "
                                f"Closed {high[-1] - close[-1]:.2f} below high."
                            ),
                            'recommendation': 'REDUCE_POSITION by 50%'
                        }
        
        else:
            recent_low_idx = np.argmin(low[-20:])
            recent_low = low[-20 + recent_low_idx]
            
            if close[-1] < recent_low:
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                
                if current_volume < avg_volume * 1.2:
                    weakness_threshold = atr * 0.3
                    if (close[-1] - low[-1]) > weakness_threshold:
                        return {
                            'detected': True,
                            'type': 'BEAR_TRAP',
                            'severity': self.SEVERITY_MEDIUM,
                            'confidence': 0.75,
                            'bar_index': len(df),
                            'description': (
                                f"Broke below {recent_low:.2f} with low volume. "
                                f"Closed {close[-1] - low[-1]:.2f} above low."
                            ),
                            'recommendation': 'REDUCE_POSITION by 50%'
                        }
        
        return {'detected': False}
    
    def _detect_continuation_trap(self, df: pd.DataFrame, is_buy: bool) -> Dict:
        """[FIXED] Detect Wedge Failure with correct slope logic."""
        if len(df) < 30:
            return {'detected': False}
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        
        if is_buy:
            recent_highs = high[-15:]
            recent_lows = low[-15:]
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # [FIXED] Rising wedge: high_slope > low_slope (not <)
            if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
                convergence = (high_slope - low_slope) / high_slope
                if convergence > 0.3:
                    if close[-1] < recent_lows[-1]:
                        return {
                            'detected': True,
                            'type': 'RISING_WEDGE_FAILURE',
                            'severity': self.SEVERITY_MEDIUM,
                            'confidence': 0.70,
                            'bar_index': len(df),
                            'description': f"Rising wedge (convergence: {convergence*100:.1f}%). Broke below.",
                            'recommendation': 'WIDEN_SL by 30%'
                        }
        
        else:
            recent_highs = high[-15:]
            recent_lows = low[-15:]
            
            high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            # [FIXED] Falling wedge: low_slope > high_slope (not <)
            if high_slope < 0 and low_slope < 0 and low_slope > high_slope:
                convergence = (low_slope - high_slope) / abs(high_slope)
                if convergence > 0.3:
                    if close[-1] > recent_highs[-1]:
                        return {
                            'detected': True,
                            'type': 'FALLING_WEDGE_FAILURE',
                            'severity': self.SEVERITY_MEDIUM,
                            'confidence': 0.70,
                            'bar_index': len(df),
                            'description': f"Falling wedge (convergence: {convergence*100:.1f}%). Broke above.",
                            'recommendation': 'WIDEN_SL by 30%'
                        }
        
        return {'detected': False}
    
    def _detect_news_pattern(self, df: pd.DataFrame) -> Dict:
        """[FIXED] Detect News patterns with sudden move check."""
        if len(df) < 10:
            return {'detected': False}
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        open_ = df['open'].to_numpy()
        
        atr = ATRCache.get_atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return {'detected': False}
        
        # [FIXED] Spike & Fade with sudden move detection
        spike_bars = 3
        recent_high = high[-spike_bars:].max()
        recent_low = low[-spike_bars:].min()
        recent_range = recent_high - recent_low
        
        if recent_range > atr * 3:
            max_single_bar_range = max(high[-spike_bars:] - low[-spike_bars:])
            
            if max_single_bar_range > atr * 2:
                spike_start_price = open_[-spike_bars]
                spike_end_price = close[-spike_bars]
                spike_direction = spike_end_price - spike_start_price
                
                current_retracement = close[-1] - spike_end_price
                retracement_pct = abs(current_retracement / spike_direction) if spike_direction != 0 else 0
                
                if retracement_pct > 0.5:
                    return {
                        'detected': True,
                        'type': 'SPIKE_AND_FADE',
                        'severity': self.SEVERITY_HIGH,
                        'confidence': 0.90,
                        'bar_index': len(df),
                        'description': f"Sudden spike ({max_single_bar_range:.2f} in 1 bar). Retraced {retracement_pct*100:.1f}%.",
                        'recommendation': 'BLOCK_TRADE'
                    }
        
        # Whipsaw with range check
        direction_changes = 0
        for i in range(-5, -1):
            if (close[i] > open_[i]) != (close[i+1] > open_[i+1]):
                direction_changes += 1
        
        if direction_changes >= 3:
            avg_range = np.mean(high[-5:] - low[-5:])
            if avg_range > atr * 0.5:
                return {
                    'detected': True,
                    'type': 'WHIPSAW',
                    'severity': self.SEVERITY_HIGH,
                    'confidence': 0.85,
                    'bar_index': len(df),
                    'description': f"Whipsaw ({direction_changes} changes). Avg range: {avg_range:.2f}.",
                    'recommendation': 'BLOCK_TRADE'
                }
        
        return {'detected': False}
    
    def _determine_action(self, highest_pattern: Dict, all_patterns: List[Dict]) -> str:
        """Determine action based on patterns."""
        severity = highest_pattern['severity']
        
        if severity == self.SEVERITY_HIGH:
            return 'BLOCK_TRADE'
        elif severity == self.SEVERITY_MEDIUM:
            medium_count = sum(1 for p in all_patterns if p['severity'] == self.SEVERITY_MEDIUM)
            if medium_count >= 2:
                return 'BLOCK_TRADE'
            else:
                return 'REDUCE_POSITION'
        else:
            return 'WIDEN_SL'