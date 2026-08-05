"""
Adaptive Take Profit Engine - Institutional Grade (Fixed Version).
Calculates dynamic TP based on:
  1. Market Structure (Swing High/Low, Order Blocks, FVGs)
  2. Volume Profile (VAH, VAL, POC, HVNs)
  3. Entry Position in Range (Percentile-based)
  4. Regime-Specific Strategy
  5. ATR-Adaptive Distance

FIXED:
  - SELL TP calculations
  - Division by zero guards
  - DataFrame index validation
  - Consistent return format
  - Edge case handling
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from core.atr_cache import ATRCache

# Optional imports with graceful degradation
try:
    from core.smc_engine import SMCStructuralEngine
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False

try:
    from core.breaker_vp_engine import BreakerVPEngine
    VP_AVAILABLE = True
except ImportError:
    VP_AVAILABLE = False


class AdaptiveTPEngine:
    """
    4-Layer Adaptive TP System with robust error handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smc_engine = SMCStructuralEngine() if SMC_AVAILABLE else None
        self.vp_engine = BreakerVPEngine() if VP_AVAILABLE else None
    
    def calculate_adaptive_tp(
        self,
        df: pd.DataFrame,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        regime_name: str = 'UNKNOWN',
        range_position: Dict = None
    ) -> Dict:
        """Main entry point: Calculate adaptive TP using 4 layers."""
        # Validate inputs
        if df is None or df.empty or len(df) < 20:
            return self._fallback_tp(entry_price, sl_price, is_buy, 'INSUFFICIENT_DATA')
        
        if entry_price <= 0 or sl_price <= 0 or entry_price == sl_price:
            return self._fallback_tp(entry_price, sl_price, is_buy, 'INVALID_PRICES')
        
        risk = abs(entry_price - sl_price)
        if risk < 0.01:  # Too small
            return self._fallback_tp(entry_price, sl_price, is_buy, 'RISK_TOO_SMALL')
        
        # Calculate ATR safely
        try:
            atr = ATRCache.get_atr(df, 14).iloc[-1]
            if pd.isna(atr) or atr <= 0:
                atr = risk * 0.5
        except Exception:
            atr = risk * 0.5
        
        # =========================================================================
        # LAYER 1: Structure-Based TP
        # =========================================================================
        structure_tp = self._calculate_structure_tp(df, entry_price, is_buy, atr)
        
        # =========================================================================
        # LAYER 2: Volume Profile TP
        # =========================================================================
        vp_tp = self._calculate_volume_profile_tp(df, entry_price, is_buy, atr)
        
        # =========================================================================
        # LAYER 3: Entry Position-Based TP
        # =========================================================================
        position_tp = self._calculate_position_based_tp(
            entry_price, sl_price, is_buy, range_position, regime_name
        )
        
        # =========================================================================
        # LAYER 4: Regime-Specific TP Strategy
        # =========================================================================
        regime_tp = self._calculate_regime_specific_tp(
            entry_price, sl_price, is_buy, regime_name, atr, structure_tp, vp_tp
        )
        
        # =========================================================================
        # META-ENSEMBLE: Combine 4 Layers
        # =========================================================================
        final_tp, method, confidence, reason = self._meta_ensemble_tp(
            entry_price, sl_price, is_buy,
            structure_tp, vp_tp, position_tp, regime_tp,
            regime_name, atr
        )
        
        # =========================================================================
        # VALIDATION: Ensure R:R >= 1.0 (relaxed from 1.5 for edge cases)
        # =========================================================================
        reward = abs(final_tp - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < 1.0:
            # Adjust TP to minimum 1.0R
            if is_buy:
                final_tp = entry_price + (risk * 1.0)
            else:
                final_tp = entry_price - (risk * 1.0)
            reason += f" [Adjusted to 1.0R]"
            rr = 1.0
        
        # =========================================================================
        # PARTIAL CLOSE TARGETS
        # =========================================================================
        partial_targets = self._calculate_partial_targets(
            entry_price, final_tp, sl_price, is_buy, regime_name
        )
        
        return {
            'tp_price': round(final_tp, 2),
            'tp_method': method,
            'confidence': confidence,
            'reason': reason,
            'partial_targets': partial_targets,
            'risk_reward': round(rr, 2),
            'layers': {
                'structure': structure_tp,
                'volume_profile': vp_tp,
                'position': position_tp,
                'regime': regime_tp
            }
        }
    
    def _calculate_structure_tp(
        self, df: pd.DataFrame, entry_price: float, is_buy: bool, atr: float
    ) -> Dict:
        """Layer 1: Use market structure (Swing High/Low, OB, FVG)."""
        default_result = {
            'tp_price': 0.0, 'method': 'NONE', 'confidence': 0.0,
            'target_type': 'NONE', 'target_price': 0.0, 'reason': 'No structure found'
        }
        
        if not SMC_AVAILABLE or self.smc_engine is None:
            return default_result
        
        try:
            swings_high, swings_low = self.smc_engine.detect_swings(df, order=3)
            order_blocks = self.smc_engine.detect_order_blocks(df, lookback=50)
            
            # Validate indices
            max_idx = len(df) - 1
            
            if is_buy:
                resistance_levels = []
                
                # Recent Swing Highs (last 3, validated)
                for idx in swings_high[-3:]:
                    if idx <= max_idx and idx >= 0:
                        swing_high = float(df['high'].iloc[idx])
                        if swing_high > entry_price + atr * 0.5:  # Must be meaningful distance
                            resistance_levels.append({
                                'price': swing_high,
                                'type': 'SWING_HIGH',
                                'strength': 0.8
                            })
                
                # Bearish Order Blocks above entry
                for ob in order_blocks:
                    if ob.get('type') == 'BEARISH' and ob.get('low', 0) > entry_price + atr * 0.5:
                        resistance_levels.append({
                            'price': float(ob['low']),
                            'type': 'ORDER_BLOCK',
                            'strength': 0.9
                        })
                
                if resistance_levels:
                    closest = min(resistance_levels, key=lambda x: abs(x['price'] - entry_price))
                    tp_price = closest['price'] - (atr * 0.3)  # Buffer below resistance
                    
                    return {
                        'tp_price': tp_price,
                        'method': 'STRUCTURE',
                        'confidence': closest['strength'],
                        'target_type': closest['type'],
                        'target_price': closest['price'],
                        'reason': f"{closest['type']} at {closest['price']:.2f}"
                    }
            
            else:  # SELL
                support_levels = []
                
                for idx in swings_low[-3:]:
                    if idx <= max_idx and idx >= 0:
                        swing_low = float(df['low'].iloc[idx])
                        if swing_low < entry_price - atr * 0.5:
                            support_levels.append({
                                'price': swing_low,
                                'type': 'SWING_LOW',
                                'strength': 0.8
                            })
                
                for ob in order_blocks:
                    if ob.get('type') == 'BULLISH' and ob.get('high', 0) < entry_price - atr * 0.5:
                        support_levels.append({
                            'price': float(ob['high']),
                            'type': 'ORDER_BLOCK',
                            'strength': 0.9
                        })
                
                if support_levels:
                    closest = min(support_levels, key=lambda x: abs(x['price'] - entry_price))
                    tp_price = closest['price'] + (atr * 0.3)  # Buffer above support
                    
                    return {
                        'tp_price': tp_price,
                        'method': 'STRUCTURE',
                        'confidence': closest['strength'],
                        'target_type': closest['type'],
                        'target_price': closest['price'],
                        'reason': f"{closest['type']} at {closest['price']:.2f}"
                    }
        
        except Exception as e:
            self.logger.debug(f"[ADAPTIVE_TP] Structure calculation error: {e}")
        
        return default_result
    
    def _calculate_volume_profile_tp(
        self, df: pd.DataFrame, entry_price: float, is_buy: bool, atr: float
    ) -> Dict:
        """Layer 2: Use Volume Profile levels (VAH, VAL, POC, HVN)."""
        default_result = {
            'tp_price': 0.0, 'method': 'NONE', 'confidence': 0.0,
            'target_type': 'NONE', 'target_price': 0.0, 'reason': 'No VP levels found'
        }
        
        if not VP_AVAILABLE or self.vp_engine is None:
            return default_result
        
        try:
            ticks_df = self.vp_engine.create_synthetic_ticks(df)
            if ticks_df.empty:
                return default_result
            
            vp_levels = self.vp_engine.calculate_volume_profile_levels(
                ticks_df, bins=120, value_area_pct=0.70
            )
            
            vah = vp_levels.get('vah', 0)
            val = vp_levels.get('val', 0)
            poc = vp_levels.get('poc', 0)
            hvns = vp_levels.get('hvns', [])
            
            if is_buy:
                # BUY: Target VAH or next HVN above entry
                if vah > entry_price + atr * 0.3:
                    return {
                        'tp_price': vah,
                        'method': 'VOLUME_PROFILE',
                        'confidence': 0.85,
                        'target_type': 'VAH',
                        'target_price': vah,
                        'reason': f"VAH at {vah:.2f}"
                    }
                elif hvns and len(hvns) > 0:
                    next_hvn = next((h for h in hvns if h > entry_price + atr * 0.3), None)
                    if next_hvn:
                        return {
                            'tp_price': next_hvn,
                            'method': 'VOLUME_PROFILE',
                            'confidence': 0.80,
                            'target_type': 'HVN',
                            'target_price': next_hvn,
                            'reason': f"HVN at {next_hvn:.2f}"
                        }
            else:  # SELL
                # SELL: Target VAL or next HVN below entry
                if val < entry_price - atr * 0.3 and val > 0:
                    return {
                        'tp_price': val,
                        'method': 'VOLUME_PROFILE',
                        'confidence': 0.85,
                        'target_type': 'VAL',
                        'target_price': val,
                        'reason': f"VAL at {val:.2f}"
                    }
                elif hvns and len(hvns) > 0:
                    next_hvn = next((h for h in hvns if h < entry_price - atr * 0.3 and h > 0), None)
                    if next_hvn:
                        return {
                            'tp_price': next_hvn,
                            'method': 'VOLUME_PROFILE',
                            'confidence': 0.80,
                            'target_type': 'HVN',
                            'target_price': next_hvn,
                            'reason': f"HVN at {next_hvn:.2f}"
                        }
        
        except Exception as e:
            self.logger.debug(f"[ADAPTIVE_TP] VP calculation error: {e}")
        
        return default_result
    
    def _calculate_position_based_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        range_position: Dict,
        regime_name: str
    ) -> Dict:
        """Layer 3: Adjust TP based on entry position in range."""
        default_result = {
            'tp_price': 0.0, 'method': 'NONE', 'confidence': 0.0,
            'target_type': 'NONE', 'target_price': 0.0, 'reason': 'No range data'
        }
        
        if not range_position or not isinstance(range_position, dict):
            return default_result
        
        percentile = range_position.get('percentile', 50)
        range_high = range_position.get('range_high', 0)
        range_low = range_position.get('range_low', 0)
        
        # Validate range
        if (range_high <= 0 or range_low <= 0 or 
            range_high <= range_low or 
            range_high - range_low < 1.0):  # Range too small
            return default_result
        
        range_width = range_high - range_low
        
        if is_buy:
            # BUY: TP at range high (if entry at bottom)
            if percentile < 30:
                # Strong zone: Target range high
                tp_price = range_high - (range_width * 0.05)  # 5% buffer
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.90,
                    'target_type': 'RANGE_HIGH',
                    'target_price': range_high,
                    'reason': f"Bottom zone ({percentile:.0f}%), target range high"
                }
            elif percentile < 50:
                # Weak zone: Closer TP (70% of remaining range)
                remaining_range = range_high - entry_price
                tp_price = entry_price + (remaining_range * 0.7)
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.70,
                    'target_type': 'PARTIAL_RANGE',
                    'target_price': tp_price,
                    'reason': f"Middle-low zone ({percentile:.0f}%), 70% of range"
                }
            else:
                # Top of range: Very close TP
                remaining_range = range_high - entry_price
                tp_price = entry_price + (remaining_range * 0.5)
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.50,
                    'target_type': 'QUICK_EXIT',
                    'target_price': tp_price,
                    'reason': f"Top zone ({percentile:.0f}%), quick exit"
                }
        else:  # SELL
            if percentile > 70:
                # Strong zone: Target range low
                tp_price = range_low + (range_width * 0.05)
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.90,
                    'target_type': 'RANGE_LOW',
                    'target_price': range_low,
                    'reason': f"Top zone ({percentile:.0f}%), target range low"
                }
            elif percentile > 50:
                # Weak zone: Closer TP
                remaining_range = entry_price - range_low
                tp_price = entry_price - (remaining_range * 0.7)
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.70,
                    'target_type': 'PARTIAL_RANGE',
                    'target_price': tp_price,
                    'reason': f"Middle-high zone ({percentile:.0f}%), 70% of range"
                }
            else:
                # Bottom of range: Very close TP
                remaining_range = entry_price - range_low
                tp_price = entry_price - (remaining_range * 0.5)
                return {
                    'tp_price': tp_price,
                    'method': 'POSITION_BASED',
                    'confidence': 0.50,
                    'target_type': 'QUICK_EXIT',
                    'target_price': tp_price,
                    'reason': f"Bottom zone ({percentile:.0f}%), quick exit"
                }
    
    def _calculate_regime_specific_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        regime_name: str,
        atr: float,
        structure_tp: Dict,
        vp_tp: Dict
    ) -> Dict:
        """Layer 4: Regime-specific TP strategy."""
        risk = abs(entry_price - sl_price)
        
        # TREND Regimes: Use Structure or 2.5R
        if any(x in regime_name for x in ['UPTREND', 'DOWNTREND', 'RALLY', 'BLEED']):
            if structure_tp['tp_price'] > 0 and structure_tp['confidence'] > 0.7:
                return structure_tp
            else:
                if is_buy:
                    tp_price = entry_price + (risk * 2.5)
                else:
                    tp_price = entry_price - (risk * 2.5)
                return {
                    'tp_price': tp_price,
                    'method': 'REGIME_TREND',
                    'confidence': 0.75,
                    'target_type': 'FIXED_RR',
                    'target_price': tp_price,
                    'reason': f"Trend regime, 2.5R target"
                }
        
        # SIDEWAY Regimes: Use VP or 1.5R
        elif any(x in regime_name for x in ['RANGE', 'SIDEWAY', 'CONSOLIDATING']):
            if vp_tp['tp_price'] > 0 and vp_tp['confidence'] > 0.7:
                return vp_tp
            else:
                if is_buy:
                    tp_price = entry_price + (risk * 1.5)
                else:
                    tp_price = entry_price - (risk * 1.5)
                return {
                    'tp_price': tp_price,
                    'method': 'REGIME_SIDEWAY',
                    'confidence': 0.70,
                    'target_type': 'QUICK_PROFIT',
                    'target_price': tp_price,
                    'reason': f"Sideways regime, 1.5R quick profit"
                }
        
        # HIGH_VOL Regimes: 1.2R (Quick Exit)
        elif any(x in regime_name for x in ['VOLATILE', 'WHIPSAW', 'PARABOLIC', 'PANIC']):
            if is_buy:
                tp_price = entry_price + (risk * 1.2)
            else:
                tp_price = entry_price - (risk * 1.2)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_HIGH_VOL',
                'confidence': 0.65,
                'target_type': 'QUICK_EXIT',
                'target_price': tp_price,
                'reason': f"High vol regime, 1.2R quick exit"
            }
        
        # REVERSAL Regimes: 2.0R
        elif any(x in regime_name for x in ['BOUNCE', 'EXHAUSTED', 'ANOMALY']):
            if vp_tp['tp_price'] > 0 and vp_tp['confidence'] > 0.6:
                return vp_tp
            else:
                if is_buy:
                    tp_price = entry_price + (risk * 2.0)
                else:
                    tp_price = entry_price - (risk * 2.0)
                return {
                    'tp_price': tp_price,
                    'method': 'REGIME_REVERSAL',
                    'confidence': 0.70,
                    'target_type': 'MEAN_REVERSION',
                    'target_price': tp_price,
                    'reason': f"Reversal regime, 2.0R target"
                }
        
        # Default: 2.0R
        else:
            if is_buy:
                tp_price = entry_price + (risk * 2.0)
            else:
                tp_price = entry_price - (risk * 2.0)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_DEFAULT',
                'confidence': 0.60,
                'target_type': 'FIXED_RR',
                'target_price': tp_price,
                'reason': f"Default regime, 2.0R target"
            }
    
    def _meta_ensemble_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        structure_tp: Dict,
        vp_tp: Dict,
        position_tp: Dict,
        regime_tp: Dict,
        regime_name: str,
        atr: float
    ) -> Tuple[float, str, float, str]:
        """Meta-ensemble: Combine 4 layers with weighted voting."""
        candidates = []
        
        # Weight by regime
        if any(x in regime_name for x in ['UPTREND', 'DOWNTREND']):
            weights = {'structure': 0.4, 'volume_profile': 0.3, 'position': 0.2, 'regime': 0.1}
        elif any(x in regime_name for x in ['RANGE', 'SIDEWAY']):
            weights = {'structure': 0.2, 'volume_profile': 0.35, 'position': 0.35, 'regime': 0.1}
        else:
            weights = {'structure': 0.3, 'volume_profile': 0.3, 'position': 0.2, 'regime': 0.2}
        
        # Add valid candidates
        if structure_tp['tp_price'] > 0:
            candidates.append({
                'price': structure_tp['tp_price'],
                'confidence': structure_tp['confidence'] * weights['structure'],
                'method': structure_tp['method'],
                'reason': structure_tp.get('reason', 'Structure-based')
            })
        
        if vp_tp['tp_price'] > 0:
            candidates.append({
                'price': vp_tp['tp_price'],
                'confidence': vp_tp['confidence'] * weights['volume_profile'],
                'method': vp_tp['method'],
                'reason': vp_tp.get('reason', 'Volume Profile-based')
            })
        
        if position_tp['tp_price'] > 0:
            candidates.append({
                'price': position_tp['tp_price'],
                'confidence': position_tp['confidence'] * weights['position'],
                'method': position_tp['method'],
                'reason': position_tp.get('reason', 'Position-based')
            })
        
        if regime_tp['tp_price'] > 0:
            candidates.append({
                'price': regime_tp['tp_price'],
                'confidence': regime_tp['confidence'] * weights['regime'],
                'method': regime_tp['method'],
                'reason': regime_tp.get('reason', 'Regime-based')
            })
        
        if not candidates:
            # Fallback to regime TP
            return (
                regime_tp['tp_price'],
                regime_tp['method'],
                regime_tp['confidence'],
                regime_tp.get('reason', f"Regime-specific TP ({regime_name})")
            )
        
        # Choose highest confidence candidate
        best = max(candidates, key=lambda x: x['confidence'])
        
        return (
            best['price'],
            best['method'],
            best['confidence'],
            f"Meta-ensemble: {best['method']} ({best['reason']})"
        )
    
    def _calculate_partial_targets(
        self,
        entry_price: float,
        final_tp: float,
        sl_price: float,
        is_buy: bool,
        regime_name: str
    ) -> List[Dict]:
        """Calculate partial close targets (TP1, TP2, TP3)."""
        risk = abs(entry_price - sl_price)
        total_reward = abs(final_tp - entry_price)
        
        # Regime-specific partial targets
        if any(x in regime_name for x in ['VOLATILE', 'WHIPSAW', 'PARABOLIC']):
            # HIGH_VOL: Quick partial (earlier)
            tp1_pct, tp2_pct, tp3_pct = 0.4, 0.4, 0.2
        elif any(x in regime_name for x in ['RANGE', 'SIDEWAY']):
            # SIDEWAY: Balanced partial
            tp1_pct, tp2_pct, tp3_pct = 0.5, 0.3, 0.2
        else:
            # TREND: Let profits run
            tp1_pct, tp2_pct, tp3_pct = 0.33, 0.33, 0.34
        
        # Calculate TP prices (FIXED for SELL)
        if is_buy:
            tp1_price = entry_price + (risk * 1.5)
            if total_reward > risk * 2.5:
                tp2_price = entry_price + (risk * 2.5)
            else:
                tp2_price = entry_price + (total_reward * 0.7)
            tp3_price = final_tp
        else:  # SELL
            tp1_price = entry_price - (risk * 1.5)
            if total_reward > risk * 2.5:
                tp2_price = entry_price - (risk * 2.5)
            else:
                tp2_price = entry_price - (total_reward * 0.7)
            tp3_price = final_tp
        
        # Ensure TP prices are in correct order
        if is_buy:
            tp1_price = min(tp1_price, tp3_price - 1.0)
            tp2_price = min(tp2_price, tp3_price - 0.5)
            tp2_price = max(tp2_price, tp1_price + 1.0)
        else:
            tp1_price = max(tp1_price, tp3_price + 1.0)
            tp2_price = max(tp2_price, tp3_price + 0.5)
            tp2_price = min(tp2_price, tp1_price - 1.0)
        
        return [
            {'price': round(tp1_price, 2), 'percent': tp1_pct, 'label': 'TP1'},
            {'price': round(tp2_price, 2), 'percent': tp2_pct, 'label': 'TP2'},
            {'price': round(tp3_price, 2), 'percent': tp3_pct, 'label': 'TP3 (Final)'}
        ]
    
    def _fallback_tp(
        self, entry_price: float, sl_price: float, is_buy: bool, reason: str
    ) -> Dict:
        """Fallback TP when calculation fails."""
        risk = abs(entry_price - sl_price)
        if risk < 0.01:
            risk = 8.0  # Default risk
        
        # Calculate TP (FIXED)
        if is_buy:
            tp_price = entry_price + (risk * 2.0)
            tp1_price = entry_price + (risk * 1.5)
        else:
            tp_price = entry_price - (risk * 2.0)
            tp1_price = entry_price - (risk * 1.5)
        
        return {
            'tp_price': round(tp_price, 2),
            'tp_method': 'FALLBACK',
            'confidence': 0.5,
            'reason': f"Fallback 2.0R ({reason})",
            'partial_targets': [
                {'price': round(tp1_price, 2), 'percent': 0.33, 'label': 'TP1'},
                {'price': round(tp_price, 2), 'percent': 0.67, 'label': 'TP2 (Final)'}
            ],
            'risk_reward': 2.0,
            'layers': {}
        }