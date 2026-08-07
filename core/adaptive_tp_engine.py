"""
Adaptive Take Profit Engine - 4-Layer Meta-Ensemble.

Calculates dynamic Take Profit based on 4 independent layers,
then combines them using weighted voting to produce the optimal TP.

4 Layers:
  1. Structure-Based TP: Swing High/Low, Order Blocks, FVG
  2. Volume Profile TP: VAH, VAL, POC, HVN/LVN
  3. Position-Based TP: Entry percentile in range
  4. Regime-Specific TP: Regime-specific R:R targets

Meta-Ensemble:
  - Each layer produces a TP candidate with confidence score
  - Weights are adjusted based on current regime
  - Highest weighted score wins
  - Minimum R:R validation
  - Partial targets calculation

Micro-Account Optimized:
  - Closer TP targets (1.2-1.8R instead of 2.0-2.5R)
  - Minimum profit threshold ($4 USD)
  - Quick profit strategy for range-bound markets
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from config import config
from core.atr_cache import ATRCache


class AdaptiveTPEngine:
    """
    Calculates adaptive Take Profit using 4-layer meta-ensemble.
    
    Features:
      - Structure-based TP detection
      - Volume profile level targeting
      - Position-based TP adjustment
      - Regime-specific R:R targets
      - Meta-ensemble weighted voting
      - Partial targets calculation
      - Fallback mechanism
    """

    def __init__(self):
        """Initialize AdaptiveTPEngine with sub-engines."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Try to load sub-engines with graceful degradation
        self.smc_engine = None
        self.vp_engine = None

        try:
            from core.smc_engine import SMCStructuralEngine
            self.smc_engine = SMCStructuralEngine()
            self.logger.info("[ADAPTIVE_TP] SMC Engine loaded")
        except ImportError as e:
            self.logger.warning(f"[ADAPTIVE_TP] SMC Engine not available: {e}")

        try:
            from core.breaker_vp_engine import BreakerVPEngine
            self.vp_engine = BreakerVPEngine()
            self.logger.info("[ADAPTIVE_TP] Volume Profile Engine loaded")
        except ImportError as e:
            self.logger.warning(f"[ADAPTIVE_TP] VP Engine not available: {e}")

        # Minimum adjustment threshold
        self.min_adjustment_usd = 1.0

        # R:R targets from config (Micro-Account optimized)
        self.tp_trend_rr = config.tp_trend_rr      # 1.8
        self.tp_sideway_rr = config.tp_sideway_rr  # 1.3
        self.tp_highvol_rr = config.tp_highvol_rr  # 1.2
        self.tp_reversal_rr = config.tp_reversal_rr  # 1.5
        self.min_tp_distance_usd = config.min_tp_distance_usd  # 1.0

        self.logger.info(
            f"[ADAPTIVE_TP] Initialized | "
            f"Trend: {self.tp_trend_rr}R | Sideway: {self.tp_sideway_rr}R | "
            f"HighVol: {self.tp_highvol_rr}R | Reversal: {self.tp_reversal_rr}R"
        )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def calculate_adaptive_tp(
        self,
        df: pd.DataFrame,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        regime_name: str = 'UNKNOWN',
        range_position: Dict = None
    ) -> Dict:
        """
        Main entry point: Calculate adaptive TP using 4 layers.
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price
            sl_price: Stop loss price
            is_buy: True for BUY, False for SELL
            regime_name: Current regime name
            range_position: Range position analysis (optional)
            
        Returns:
            Dict with:
              - tp_price: Final TP price
              - tp_method: Which method was used
              - confidence: Confidence score (0-1)
              - reason: Human-readable explanation
              - partial_targets: TP1, TP2, TP3 for partial close
              - risk_reward: Calculated R:R
              - layers: Individual layer results
        """
        # Validate input
        if df is None or df.empty or len(df) < 20:
            return self._fallback_tp(entry_price, sl_price, is_buy, 'INSUFFICIENT_DATA')

        if entry_price <= 0 or sl_price <= 0 or entry_price == sl_price:
            return self._fallback_tp(entry_price, sl_price, is_buy, 'INVALID_PRICES')

        risk = abs(entry_price - sl_price)
        if risk < 0.01:  # Too small
            return self._fallback_tp(entry_price, sl_price, is_buy, 'RISK_TOO_SMALL')

        # Calculate ATR for volatility context
        try:
            atr_series = ATRCache.get_atr(df, 14)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                atr = risk * 0.5
            else:
                atr = float(atr_series.iloc[-1])
        except Exception:
            atr = risk * 0.5

        # =========================================================================
        # LAYER 1: STRUCTURE-BASED TP
        # =========================================================================
        structure_tp = self._calculate_structure_tp(df, entry_price, is_buy, atr)

        # =========================================================================
        # LAYER 2: VOLUME PROFILE TP
        # =========================================================================
        vp_tp = self._calculate_volume_profile_tp(df, entry_price, is_buy, atr)

        # =========================================================================
        # LAYER 3: POSITION-BASED TP
        # =========================================================================
        position_tp = self._calculate_position_based_tp(
            entry_price, sl_price, is_buy, range_position, regime_name
        )

        # =========================================================================
        # LAYER 4: REGIME-SPECIFIC TP
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
        # VALIDATION: Ensure R:R >= 1.0 and minimum profit
        # =========================================================================
        reward = abs(final_tp - entry_price)
        rr = reward / risk if risk > 0 else 0

        # Check minimum profit for micro-account
        min_profit = config.min_profit_usd
        if reward < min_profit:
            # Adjust TP to minimum profit
            if is_buy:
                final_tp = entry_price + min_profit
            else:
                final_tp = entry_price - min_profit
            reason += f" [Adjusted to min profit ${min_profit}]"
            rr = min_profit / risk

        # Check minimum R:R
        if rr < 1.0:
            # Adjust TP to minimum 1.0R
            if is_buy:
                final_tp = entry_price + risk
            else:
                final_tp = entry_price - risk
            reason += " [Adjusted to 1.0R]"
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

    # =========================================================================
    # LAYER 1: STRUCTURE-BASED TP
    # =========================================================================

    def _calculate_structure_tp(
        self, df: pd.DataFrame, entry_price: float, is_buy: bool, atr: float
    ) -> Dict:
        """
        Layer 1: Calculate TP based on market structure.
        
        Uses swing high/low and order blocks as targets.
        
        Returns:
            Dict with tp_price, confidence, method, reason
        """
        default_result = {
            'tp_price': 0.0, 'method': 'NONE', 'confidence': 0.0,
            'target_type': 'NONE', 'target_price': 0.0, 'reason': 'No structure found'
        }

        if self.smc_engine is None:
            return default_result

        try:
            # Detect swings
            swings_high, swings_low = self.smc_engine.detect_swings(df, order=3)

            # Detect order blocks
            order_blocks = self.smc_engine.detect_order_blocks(df, lookback=50)

            # Validate indices
            max_idx = len(df) - 1

            if is_buy:
                # BUY: Find resistance above entry
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
                    # Choose closest resistance
                    closest = min(resistance_levels, key=lambda x: abs(x['price'] - entry_price))

                    # Add ATR buffer below resistance
                    tp_price = closest['price'] - (atr * 0.3)

                    return {
                        'tp_price': tp_price,
                        'method': 'STRUCTURE',
                        'confidence': closest['strength'],
                        'target_type': closest['type'],
                        'target_price': closest['price'],
                        'reason': f"{closest['type']} at {closest['price']:.2f}"
                    }

            else:  # SELL
                # SELL: Find support below entry
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

    # =========================================================================
    # LAYER 2: VOLUME PROFILE TP
    # =========================================================================

    def _calculate_volume_profile_tp(
        self, df: pd.DataFrame, entry_price: float, is_buy: bool, atr: float
    ) -> Dict:
        """
        Layer 2: Calculate TP based on Volume Profile levels.
        
        Uses VAH, VAL, POC, HVN as targets.
        
        Returns:
            Dict with tp_price, confidence, method, reason
        """
        default_result = {
            'tp_price': 0.0, 'method': 'NONE', 'confidence': 0.0,
            'target_type': 'NONE', 'target_price': 0.0, 'reason': 'No VP levels found'
        }

        if self.vp_engine is None:
            return default_result

        try:
            # Calculate volume profile
            vp_levels = self.vp_engine.calculate_volume_profile_levels(df, bins=120, value_area_pct=0.70)

            if not vp_levels:
                return default_result

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

    # =========================================================================
    # LAYER 3: POSITION-BASED TP
    # =========================================================================

    def _calculate_position_based_tp(
        self,
        entry_price: float,
        sl_price: float,
        is_buy: bool,
        range_position: Dict,
        regime_name: str
    ) -> Dict:
        """
        Layer 3: Calculate TP based on entry position in range.
        
        Entry at bottom → Target top of range
        Entry at middle → Closer target
        Entry at top → Very close target
        
        Returns:
            Dict with tp_price, confidence, method, reason
        """
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

    # =========================================================================
    # LAYER 4: REGIME-SPECIFIC TP
    # =========================================================================

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
        """
        Layer 4: Regime-specific TP strategy.
        
        Different regimes have different optimal R:R targets.
        
        Returns:
            Dict with tp_price, confidence, method, reason
        """
        risk = abs(entry_price - sl_price)

        # =========================================================================
        # TREND Regimes: Use Structure or Trend R:R
        # =========================================================================
        if any(x in regime_name for x in ['UPTREND', 'DOWNTREND', 'RALLY', 'BLEED']):
            # Use structure TP if available and confident
            if structure_tp['tp_price'] > 0 and structure_tp['confidence'] > 0.7:
                return structure_tp

            # Fallback to trend R:R
            if is_buy:
                tp_price = entry_price + (risk * self.tp_trend_rr)
            else:
                tp_price = entry_price - (risk * self.tp_trend_rr)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_TREND',
                'confidence': 0.75,
                'target_type': 'FIXED_RR',
                'target_price': tp_price,
                'reason': f"Trend regime, {self.tp_trend_rr}R target"
            }

        # =========================================================================
        # SIDEWAY Regimes: Use VP or Sideway R:R
        # =========================================================================
        elif any(x in regime_name for x in ['RANGE', 'SIDEWAY', 'CONSOLIDATING']):
            # Use VP TP if available and confident
            if vp_tp['tp_price'] > 0 and vp_tp['confidence'] > 0.7:
                return vp_tp

            # Fallback to sideway R:R (quick profit)
            if is_buy:
                tp_price = entry_price + (risk * self.tp_sideway_rr)
            else:
                tp_price = entry_price - (risk * self.tp_sideway_rr)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_SIDEWAY',
                'confidence': 0.70,
                'target_type': 'QUICK_PROFIT',
                'target_price': tp_price,
                'reason': f"Sideways regime, {self.tp_sideway_rr}R quick profit"
            }

        # =========================================================================
        # HIGH_VOL Regimes: Very close TP
        # =========================================================================
        elif any(x in regime_name for x in ['VOLATILE', 'WHIPSAW', 'PARABOLIC', 'PANIC']):
            if is_buy:
                tp_price = entry_price + (risk * self.tp_highvol_rr)
            else:
                tp_price = entry_price - (risk * self.tp_highvol_rr)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_HIGH_VOL',
                'confidence': 0.65,
                'target_type': 'QUICK_EXIT',
                'target_price': tp_price,
                'reason': f"High vol regime, {self.tp_highvol_rr}R quick exit"
            }

        # =========================================================================
        # REVERSAL Regimes: Use VP or Reversal R:R
        # =========================================================================
        elif any(x in regime_name for x in ['BOUNCE', 'EXHAUSTED', 'ANOMALY']):
            # Use VP TP if available
            if vp_tp['tp_price'] > 0 and vp_tp['confidence'] > 0.6:
                return vp_tp

            # Fallback to reversal R:R
            if is_buy:
                tp_price = entry_price + (risk * self.tp_reversal_rr)
            else:
                tp_price = entry_price - (risk * self.tp_reversal_rr)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_REVERSAL',
                'confidence': 0.70,
                'target_type': 'MEAN_REVERSION',
                'target_price': tp_price,
                'reason': f"Reversal regime, {self.tp_reversal_rr}R target"
            }

        # =========================================================================
        # Default: Moderate R:R
        # =========================================================================
        else:
            default_rr = 1.5
            if is_buy:
                tp_price = entry_price + (risk * default_rr)
            else:
                tp_price = entry_price - (risk * default_rr)
            return {
                'tp_price': tp_price,
                'method': 'REGIME_DEFAULT',
                'confidence': 0.60,
                'target_type': 'FIXED_RR',
                'target_price': tp_price,
                'reason': f"Default regime, {default_rr}R target"
            }

    # =========================================================================
    # META-ENSEMBLE
    # =========================================================================

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
        """
        Meta-ensemble: Combine 4 layers with weighted voting.
        
        Returns:
            Tuple of (tp_price, method, confidence, reason)
        """
        candidates = []

        # Determine weights based on regime
        if any(x in regime_name for x in ['UPTREND', 'DOWNTREND']):
            # TREND: Prefer structure
            weights = {'structure': 0.4, 'volume_profile': 0.3, 'position': 0.2, 'regime': 0.1}
        elif any(x in regime_name for x in ['RANGE', 'SIDEWAY']):
            # SIDEWAY: Prefer VP and position
            weights = {'structure': 0.2, 'volume_profile': 0.35, 'position': 0.35, 'regime': 0.1}
        else:
            # Default: Balanced
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

        # If no valid candidates, use regime TP as fallback
        if not candidates:
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

    # =========================================================================
    # PARTIAL TARGETS
    # =========================================================================

    def _calculate_partial_targets(
        self,
        entry_price: float,
        final_tp: float,
        sl_price: float,
        is_buy: bool,
        regime_name: str
    ) -> List[Dict]:
        """
        Calculate partial close targets (TP1, TP2, TP3).
        
        Different regimes have different partial close strategies.
        
        Returns:
            List of partial target dicts
        """
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

        # Calculate TP prices
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

    # =========================================================================
    # FALLBACK
    # =========================================================================

    def _fallback_tp(
        self, entry_price: float, sl_price: float, is_buy: bool, reason: str
    ) -> Dict:
        """
        Fallback TP when calculation fails.
        
        Returns:
            Dict with fallback TP
        """
        risk = abs(entry_price - sl_price)
        if risk < 0.01:
            risk = 8.0  # Default risk

        # Calculate TP (use 2.0R fallback)
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

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def format_tp_log(self, result: Dict) -> str:
        """
        Format TP result as concise log string.
        
        Args:
            result: Result from calculate_adaptive_tp
            
        Returns:
            Formatted log string
        """
        tp_price = result.get('tp_price', 0)
        method = result.get('tp_method', 'UNKNOWN')
        rr = result.get('risk_reward', 0)
        confidence = result.get('confidence', 0)

        return (
            f"[ADAPTIVE_TP] TP: {tp_price:.2f} | "
            f"Method: {method} | "
            f"R:R: {rr:.2f} | "
            f"Conf: {confidence:.2f}"
        )