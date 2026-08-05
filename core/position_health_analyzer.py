"""
Position Health Analyzer (Hardened Version)
Analyzes each active position and assigns a health score (0-100).
Includes robust handling for SQLite NULL values and missing data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import logging


class PositionHealthAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_position_health(self, position: Dict, current_price: float, 
                               df_m5: pd.DataFrame, regime_context: Dict) -> Dict:
        """
        Analyze position health and return comprehensive metrics.
        Hardened against NoneType errors from SQLite.
        """
        meta = position.get('meta_data', {}) or {}
        
        # =========================================================================
        # SAFE DATA EXTRACTION (Prevent NoneType Errors)
        # =========================================================================
        # Use `or 0.0` to handle cases where key exists but value is None
        entry_price = float(position.get('entry_price') or 0.0)
        sl_price = float(position.get('sl') or 0.0)
        tp_price = float(position.get('tp') or 0.0)
        position_type = str(position.get('position_type') or 'BUY')
        volume = float(position.get('volume') or 0.0)
        
        is_buy = (position_type == 'BUY')
        
        # Smart Fallback for Missing SL: Use configured Micro SL distance if SL is 0/None
        if sl_price == 0.0 and entry_price > 0:
            try:
                from config import config
                fallback_dist = getattr(config, 'micro_sl_distance_usd', 16.0)
                sl_price = entry_price - fallback_dist if is_buy else entry_price + fallback_dist
            except Exception:
                sl_price = entry_price * 0.98 if is_buy else entry_price * 1.02 # 2% hardcoded fallback

        # Safe Open Time Parsing
        open_time_str = position.get('open_time')
        try:
            open_time = pd.to_datetime(open_time_str) if open_time_str else datetime.now()
        except Exception:
            open_time = datetime.now()
        
        # Ensure current_price is valid
        if current_price is None or current_price <= 0:
            current_price = entry_price

        # =========================================================================
        # METRICS CALCULATION
        # =========================================================================
        initial_risk = abs(entry_price - sl_price)
        if initial_risk == 0:
            initial_risk = 1.0 # Prevent Division by Zero
            
        initial_reward = abs(tp_price - entry_price) if tp_price > 0 else initial_risk * 3.0
        
        current_pnl = (current_price - entry_price) if is_buy else (entry_price - current_price)
        current_pnl_r = current_pnl / initial_risk
        
        # Time analysis
        hours_open = (datetime.now() - open_time).total_seconds() / 3600.0
        if hours_open < 0: hours_open = 0

        # =========================================================================
        # HEALTH SCORE COMPONENTS
        # =========================================================================
        scores = {}
        reasons = []
        
        # 1. Risk/Reward Evolution (0-25 points)
        if current_pnl_r > 0:
            scores['rr_evolution'] = min(25, current_pnl_r * 10)
            if current_pnl_r > 2:
                reasons.append(f"Strong profit: +{current_pnl_r:.1f}R")
        else:
            scores['rr_evolution'] = max(0, 10 + current_pnl_r * 5)
            if current_pnl_r < -0.5:
                reasons.append(f"Underperforming: {current_pnl_r:.1f}R")
        
        # 2. Time Efficiency (0-20 points)
        strategy_category = meta.get('strategy_category', 'GENERAL')
        expected_hours = {
            'SCALP': 2, 'SMC': 8, 'MEAN_REVERSION': 12, 'TREND': 24
        }.get(strategy_category, 12)
        
        if current_pnl_r > 0:
            time_efficiency = min(1.0, (current_pnl_r / max(hours_open / expected_hours, 0.1)))
        else:
            time_efficiency = max(0, 1 - hours_open / expected_hours)
        
        scores['time_efficiency'] = time_efficiency * 20
        
        if hours_open > expected_hours * 2 and current_pnl_r < 0.5:
            reasons.append(f"Time stall: {hours_open:.1f}h open, only {current_pnl_r:.1f}R")
        
        # 3. Regime Alignment (0-25 points)
        regime_name = regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
        regime_trend = self._get_regime_direction(regime_name)
        
        if regime_trend == 'BULL' and is_buy:
            scores['regime_alignment'] = 25
        elif regime_trend == 'BEAR' and not is_buy:
            scores['regime_alignment'] = 25
        elif regime_trend == 'SIDEWAY':
            scores['regime_alignment'] = 15
        else:
            scores['regime_alignment'] = 0
            reasons.append(f"Regime conflict: {position_type} vs {regime_name}")
        
        # 4. Momentum Score (0-20 points)
        if df_m5 is not None and len(df_m5) >= 20:
            momentum = self._calculate_momentum(df_m5, is_buy)
            scores['momentum'] = momentum * 20
            if momentum < 0.3:
                reasons.append("Weak momentum")
        else:
            scores['momentum'] = 10  # Neutral
        
        # 5. Stop Loss Management (0-10 points)
        # SAFE EXTRACTION: Handle None from SQLite explicitly
        current_sl = position.get('trailing_stop_level')
        if current_sl is None or current_sl == 0.0:
            current_sl = sl_price
        
        sl_distance = abs(current_price - float(current_sl))
        
        if current_pnl_r > 1.0 and sl_distance < initial_risk * 0.5:
            scores['sl_management'] = 10
            reasons.append("Good SL management")
        elif current_pnl_r < 0 and sl_distance > initial_risk * 1.1:
            scores['sl_management'] = 0
            reasons.append("SL widened beyond risk")
        else:
            scores['sl_management'] = 5
        
        # =========================================================================
        # FINAL HEALTH SCORE
        # =========================================================================
        health_score = sum(scores.values())
        
        if health_score >= 70:
            recommendation = 'HOLD'
            confidence = health_score
        elif health_score >= 50:
            recommendation = 'HOLD'
            confidence = health_score
        elif health_score >= 30:
            recommendation = 'REDUCE'
            confidence = 100 - health_score
        else:
            recommendation = 'CLOSE'
            confidence = 100 - health_score
        
        return {
            'health_score': health_score,
            'risk_reward_current': current_pnl_r,
            'time_efficiency': time_efficiency,
            'regime_alignment': scores['regime_alignment'] / 25.0,
            'momentum_score': scores['momentum'] / 20.0,
            'recommendation': recommendation,
            'confidence': confidence,
            'reasons': reasons,
            'breakdown': scores,
            'hours_open': hours_open,
            'volume': volume
        }
    
    def _get_regime_direction(self, regime_name: str) -> str:
        bull_regimes = ['QUIET_RALLY', 'HEALTHY_UPTREND', 'PARABOLIC_RALLY', 'ANOMALY_BULL']
        bear_regimes = ['SLOW_BLEED', 'HEALTHY_DOWNTREND', 'PANIC_CAPITULATION', 'ANOMALY_BEAR']
        if regime_name in bull_regimes: return 'BULL'
        elif regime_name in bear_regimes: return 'BEAR'
        return 'SIDEWAY'
    
    def _calculate_momentum(self, df: pd.DataFrame, is_buy: bool) -> float:
        try:
            close = df['close'].values
            if len(close) < 20: return 0.5
            price_mom = (close[-1] - close[-10]) / close[-10]
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values
                vol_ma = np.mean(volume[-20:])
                vol_recent = np.mean(volume[-5:])
                vol_mom = (vol_recent - vol_ma) / vol_ma if vol_ma > 0 else 0
            else:
                vol_mom = 0
            
            if is_buy:
                momentum = 0.5 + (price_mom * 5) + (vol_mom * 0.2)
            else:
                momentum = 0.5 - (price_mom * 5) - (vol_mom * 0.2)
            return max(0, min(1, momentum))
        except Exception:
            return 0.5