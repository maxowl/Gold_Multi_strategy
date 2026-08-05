"""
Range Position Filter for Sideways Market.
Prevents entry at the middle of range where R:R is unfavorable.
Uses Volume Profile + Swing Structure for robust range detection.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from core.breaker_vp_engine import BreakerVPEngine
from core.smc_engine import SMCStructuralEngine
from core.atr_cache import ATRCache


class RangePositionFilter:
    """
    Evaluates entry position within the current trading range.
    
    Position Quality Zones:
      0-20%:   STRONG_BUY_ZONE (Bottom of range)
      20-40%:  WEAK_BUY_ZONE
      40-60%:  DEATH_ZONE (Reject all entries)
      60-80%:  WEAK_SELL_ZONE
      80-100%: STRONG_SELL_ZONE (Top of range)
    """
    
    # Position thresholds (configurable)
    STRONG_ZONE_THRESHOLD = 0.20  # 20% from edge
    WEAK_ZONE_THRESHOLD = 0.40    # 40% from edge
    DEATH_ZONE_LOW = 0.40         # 40-60% is death zone
    DEATH_ZONE_HIGH = 0.60
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vp_engine = BreakerVPEngine()
        self.smc_engine = SMCStructuralEngine()
    
    def evaluate_entry_position(
        self, 
        df: pd.DataFrame, 
        entry_price: float,
        is_buy: bool,
        regime_name: str = 'UNKNOWN'
    ) -> Dict:
        """
        Evaluate if entry price is at a favorable position within the range.
        
        Returns:
            Dict with keys:
            - percentile: 0-100 position in range
            - zone: STRONG_BUY_ZONE, DEATH_ZONE, etc.
            - should_trade: bool
            - reason: explanation
            - range_high, range_low: detected range boundaries
            - position_score: 0-100 quality score
        """
        # =========================================================================
        # Layer 1: Range Detection
        # =========================================================================
        range_info = self._detect_current_range(df)
        
        if not range_info['valid']:
            return {
                'percentile': 50.0,
                'zone': 'UNKNOWN',
                'should_trade': True,  # Can't filter without range
                'reason': 'Range detection failed - pass through',
                'range_high': 0.0,
                'range_low': 0.0,
                'position_score': 50
            }
        
        range_high = range_info['high']
        range_low = range_info['low']
        range_width = range_high - range_low
        
        if range_width <= 0:
            return {
                'percentile': 50.0,
                'zone': 'UNKNOWN',
                'should_trade': True,
                'reason': 'Invalid range width',
                'range_high': range_high,
                'range_low': range_low,
                'position_score': 50
            }
        
        # =========================================================================
        # Layer 2: Calculate Position Percentile
        # =========================================================================
        # Percentile: 0 = at range low, 100 = at range high
        percentile = ((entry_price - range_low) / range_width) * 100.0
        percentile = max(0.0, min(100.0, percentile))
        
        # =========================================================================
        # Layer 3: Zone Classification
        # =========================================================================
        zone, position_score, should_trade, reason = self._classify_position(
            percentile, is_buy, regime_name
        )
        
        return {
            'percentile': round(percentile, 2),
            'zone': zone,
            'should_trade': should_trade,
            'reason': reason,
            'range_high': round(range_high, 2),
            'range_low': round(range_low, 2),
            'range_width': round(range_width, 2),
            'position_score': position_score,
            'distance_to_high': round(range_high - entry_price, 2),
            'distance_to_low': round(entry_price - range_low, 2)
        }
    
    def _detect_current_range(self, df: pd.DataFrame) -> Dict:
        """
        Detect current trading range using multiple methods:
        1. Volume Profile (VAH/VAL) - Primary
        2. Swing High/Low - Secondary
        3. Recent High/Low - Fallback
        """
        if df is None or len(df) < 50:
            return {'valid': False, 'high': 0.0, 'low': 0.0, 'method': 'insufficient_data'}
        
        # Method 1: Volume Profile (Most reliable for institutional ranges)
        try:
            ticks_df = self.vp_engine.create_synthetic_ticks(df)
            if not ticks_df.empty:
                vp_levels = self.vp_engine.calculate_volume_profile_levels(
                    ticks_df, bins=120, value_area_pct=0.70
                )
                vah = vp_levels.get('vah', 0.0)
                val = vp_levels.get('val', 0.0)
                
                if vah > 0 and val > 0 and vah > val:
                    # Extend range slightly beyond VA for safety
                    atr = ATRCache.get_atr(df, 14).iloc[-1] if 'atr' in df.columns else 0
                    buffer = float(atr) * 0.5 if not pd.isna(atr) else (vah - val) * 0.1
                    return {
                        'valid': True,
                        'high': vah + buffer,
                        'low': val - buffer,
                        'method': 'volume_profile'
                    }
        except Exception as e:
            self.logger.debug(f"[RANGE] VP detection failed: {e}")
        
        # Method 2: Swing High/Low Structure
        try:
            swings_high, swings_low = self.smc_engine.detect_swings(df, order=3)
            if len(swings_high) >= 2 and len(swings_low) >= 2:
                # Use recent swings (last 2 each)
                recent_highs = [df['high'].iloc[i] for i in swings_high[-2:]]
                recent_lows = [df['low'].iloc[i] for i in swings_low[-2:]]
                
                range_high = max(recent_highs)
                range_low = min(recent_lows)
                
                if range_high > range_low:
                    return {
                        'valid': True,
                        'high': float(range_high),
                        'low': float(range_low),
                        'method': 'swing_structure'
                    }
        except Exception as e:
            self.logger.debug(f"[RANGE] Swing detection failed: {e}")
        
        # Method 3: Fallback - Rolling High/Low (last 100 bars)
        try:
            lookback = min(100, len(df))
            recent_df = df.tail(lookback)
            range_high = float(recent_df['high'].max())
            range_low = float(recent_df['low'].min())
            
            if range_high > range_low:
                return {
                    'valid': True,
                    'high': range_high,
                    'low': range_low,
                    'method': 'rolling_hl'
                }
        except Exception as e:
            self.logger.debug(f"[RANGE] Rolling HL failed: {e}")
        
        return {'valid': False, 'high': 0.0, 'low': 0.0, 'method': 'all_failed'}
    
    def _classify_position(
        self, 
        percentile: float, 
        is_buy: bool,
        regime_name: str
    ) -> tuple:
        """
        Classify position into zones and determine if trade should be taken.
        
        Returns: (zone, position_score, should_trade, reason)
        """
        # =========================================================================
        # Sideways Regime - Strict Rules
        # =========================================================================
        is_sideways_regime = any(x in regime_name for x in [
            'SIDEWAY', 'RANGE', 'CONSOLIDATING', 'TIGHT', 'CHOP', 'WHIPSAW'
        ])
        
        if is_sideways_regime:
            # Death Zone: 40-60 percentile
            if self.DEATH_ZONE_LOW * 100 <= percentile <= self.DEATH_ZONE_HIGH * 100:
                return (
                    'DEATH_ZONE',
                    0,
                    False,
                    f"Entry at {percentile:.1f}% of range - Middle of range has poor R:R"
                )
            
            # Strong Buy Zone: 0-20 percentile (only for BUY)
            if percentile <= self.STRONG_ZONE_THRESHOLD * 100:
                if is_buy:
                    return (
                        'STRONG_BUY_ZONE',
                        100,
                        True,
                        f"Entry at {percentile:.1f}% - Bottom of range, strong mean-reversion edge"
                    )
                else:
                    return (
                        'STRONG_BUY_ZONE_SELL_ENTRY',
                        20,
                        False,
                        f"SELL at {percentile:.1f}% - Bottom of range, expect bounce up"
                    )
            
            # Strong Sell Zone: 80-100 percentile (only for SELL)
            if percentile >= (1 - self.STRONG_ZONE_THRESHOLD) * 100:
                if not is_buy:
                    return (
                        'STRONG_SELL_ZONE',
                        100,
                        True,
                        f"Entry at {percentile:.1f}% - Top of range, strong mean-reversion edge"
                    )
                else:
                    return (
                        'STRONG_SELL_ZONE_BUY_ENTRY',
                        20,
                        False,
                        f"BUY at {percentile:.1f}% - Top of range, expect rejection down"
                    )
            
            # Weak Buy Zone: 20-40 percentile
            if percentile <= self.WEAK_ZONE_THRESHOLD * 100:
                if is_buy:
                    return (
                        'WEAK_BUY_ZONE',
                        60,
                        True,
                        f"Entry at {percentile:.1f}% - Lower half of range, acceptable"
                    )
                else:
                    return (
                        'WEAK_BUY_ZONE_SELL_ENTRY',
                        40,
                        False,
                        f"SELL at {percentile:.1f}% - Lower half, prefer BUY here"
                    )
            
            # Weak Sell Zone: 60-80 percentile
            if is_buy:
                return (
                    'WEAK_SELL_ZONE_BUY_ENTRY',
                    40,
                    False,
                    f"BUY at {percentile:.1f}% - Upper half, prefer SELL here"
                )
            else:
                return (
                    'WEAK_SELL_ZONE',
                    60,
                    True,
                    f"Entry at {percentile:.1f}% - Upper half of range, acceptable"
                )
        
        # =========================================================================
        # Non-Sideways Regime - Lenient Rules
        # =========================================================================
        else:
            # Still avoid middle in most cases
            if 45 <= percentile <= 55:
                return (
                    'NEUTRAL_ZONE',
                    50,
                    True,  # Allow but score neutral
                    f"Entry at {percentile:.1f}% - Middle of range, neutral in trending regime"
                )
            
            # For trending regimes, middle is less problematic
            if is_buy and percentile <= 50:
                return (
                    'LOWER_HALF_BUY',
                    70,
                    True,
                    f"Entry at {percentile:.1f}% - Lower half, favorable for BUY"
                )
            elif not is_buy and percentile >= 50:
                return (
                    'UPPER_HALF_SELL',
                    70,
                    True,
                    f"Entry at {percentile:.1f}% - Upper half, favorable for SELL"
                )
            else:
                return (
                    'AGAINST_POSITION',
                    40,
                    True,  # Still allow but penalize
                    f"Entry at {percentile:.1f}% - Against favorable position"
                )