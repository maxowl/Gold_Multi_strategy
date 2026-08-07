"""
Strategy 30: Volume Profile Reversal (SMC).

Uses multiple volume engines to identify high-volume nodes and trade reversals.
Combines OrderFlow, VolumeIndicators, and WyckoffVSA engines.

Category: SMC
Optimal Regimes: HEALTHY_UPTREND, HEALTHY_DOWNTREND, PRE_BREAKOUT
Timeframe: M15 (Primary), H1 (Confirmation)
"""
import pandas as pd
import numpy as np
import logging
from core.base_strategy import BaseStrategy
from core.orderflow_engine import OrderFlowEngine
from core.volume_indicators import VolumeIndicatorsEngine
from core.wyckoff_vsa_engine import WyckoffVSAEngine
from core.atr_cache import ATRCache


class S30_VolumeProfileReversal(BaseStrategy):
    """
    Volume Profile Reversal Strategy.
    
    Logic:
    - Detect volume imbalances (OrderFlow)
    - Calculate money flow indicators (TMF, CMF, EOM)
    - Identify Wyckoff patterns (Spring, Upthrust)
    - Enter on confluence of multiple volume signals
    
    Advantages:
    - Institutional-grade volume analysis
    - Multiple confirmation layers
    - Works in trending and reversal markets
    """
    
    def __init__(self):
        super().__init__(
            name='S30_VolumeProfileReversal',
            category='SMC',
            description='Multi-Engine Volume Profile Analysis'
        )
        self.orderflow_engine = OrderFlowEngine()
        self.volume_indicators = VolumeIndicatorsEngine()
        self.wyckoff_engine = WyckoffVSAEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, df_primary: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
        """
        Evaluate volume profile reversal signal.
        
        Args:
            df_primary: M15 data
            df_htf: H1 data
            
        Returns:
            Signal dict with BUY/SELL or NEUTRAL
        """
        if df_primary is None or df_primary.empty or len(df_primary) < 50:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Insufficient data'}}
        
        # Layer 1: Detect volume imbalances
        imbalance_zones = self.orderflow_engine.detect_volume_imbalance_zones(
            df_primary, min_imbalance_ratio=2.0, lookback=50
        )
        
        # Layer 2: Calculate money flow indicators
        tmf = self.volume_indicators.calculate_twiggs_money_flow(df_primary, period=21)
        cmf = self.volume_indicators.calculate_chaikin_money_flow(df_primary, period=20)
        eom = self.volume_indicators.calculate_ease_of_movement(df_primary, period=14)
        
        # Layer 3: Detect Wyckoff patterns
        spring = self.wyckoff_engine.detect_spring(df_primary, lookback=50)
        upthrust = self.wyckoff_engine.detect_upthrust(df_primary, lookback=50)
        
        # Layer 4: Check no supply/no demand conditions
        no_supply = self.wyckoff_engine.detect_no_supply(df_primary, lookback=20)
        no_demand = self.wyckoff_engine.detect_no_demand(df_primary, lookback=20)
        
        # Determine signal based on confluence
        signal_type = 'NEUTRAL'
        entry_price = df_primary['close'].iloc[-1]
        sl_price = 0.0
        tp_price = 0.0
        confidence = 0.0
        reason = ''
        
        # Get ATR for risk management
        atr_series = ATRCache.get_atr(df_primary, 14)
        if atr_series.empty or atr_series.isna().all():
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'ATR calculation failed'}}
        
        current_atr = atr_series.iloc[-1]
        
        # BUY signal conditions
        buy_signals = 0
        buy_reasons = []
        
        # Condition 1: Bullish volume imbalance
        if imbalance_zones:
            bullish_zones = [z for z in imbalance_zones if z['type'] == 'BULLISH']
            if bullish_zones:
                buy_signals += 1
                buy_reasons.append('Bullish Volume Imbalance')
        
        # Condition 2: Positive money flow
        if tmf is not None and not tmf.isna().all():
            tmf_value = tmf.iloc[-1]
            if tmf_value > 0.1:
                buy_signals += 1
                buy_reasons.append(f'Positive TMF ({tmf_value:.3f})')
        
        if cmf is not None and not cmf.isna().all():
            cmf_value = cmf.iloc[-1]
            if cmf_value > 0.1:
                buy_signals += 1
                buy_reasons.append(f'Positive CMF ({cmf_value:.3f})')
        
        # Condition 3: Wyckoff Spring (accumulation reversal)
        if spring is not None:
            buy_signals += 2  # Strong signal
            buy_reasons.append('Wyckoff Spring Detected')
        
        # Condition 4: No supply (low volume on down move)
        if no_supply:
            buy_signals += 1
            buy_reasons.append('No Supply Condition')
        
        # SELL signal conditions
        sell_signals = 0
        sell_reasons = []
        
        # Condition 1: Bearish volume imbalance
        if imbalance_zones:
            bearish_zones = [z for z in imbalance_zones if z['type'] == 'BEARISH']
            if bearish_zones:
                sell_signals += 1
                sell_reasons.append('Bearish Volume Imbalance')
        
        # Condition 2: Negative money flow
        if tmf is not None and not tmf.isna().all():
            tmf_value = tmf.iloc[-1]
            if tmf_value < -0.1:
                sell_signals += 1
                sell_reasons.append(f'Negative TMF ({tmf_value:.3f})')
        
        if cmf is not None and not cmf.isna().all():
            cmf_value = cmf.iloc[-1]
            if cmf_value < -0.1:
                sell_signals += 1
                sell_reasons.append(f'Negative CMF ({cmf_value:.3f})')
        
        # Condition 3: Wyckoff Upthrust (distribution reversal)
        if upthrust is not None:
            sell_signals += 2  # Strong signal
            sell_reasons.append('Wyckoff Upthrust Detected')
        
        # Condition 4: No demand (low volume on up move)
        if no_demand:
            sell_signals += 1
            sell_reasons.append('No Demand Condition')
        
        # Determine final signal (need at least 3 confluence signals)
        if buy_signals >= 3:
            signal_type = 'BUY_MARKET'
            sl_price = entry_price - (current_atr * 2.0)
            tp_price = entry_price + (current_atr * 4.0)  # 2R target
            confidence = min(0.90, 0.60 + (buy_signals - 3) * 0.1)
            reason = ' | '.join(buy_reasons)
        
        elif sell_signals >= 3:
            signal_type = 'SELL_MARKET'
            sl_price = entry_price + (current_atr * 2.0)
            tp_price = entry_price - (current_atr * 4.0)  # 2R target
            confidence = min(0.90, 0.60 + (sell_signals - 3) * 0.1)
            reason = ' | '.join(sell_reasons)
        
        if signal_type == 'NEUTRAL':
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'Insufficient volume confluence'}}
        
        # Validate R:R
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr = reward / risk if risk > 0 else 0
        
        if rr < 1.8:  # Minimum R:R for SMC strategy
            return {'signal': 'NEUTRAL', 'meta': {'reason': f'R:R too low ({rr:.2f})'}}
        
        # Build signal
        meta = {
            'strategy': self.name,
            'strategy_category': self.category,
            'entry_price': round(entry_price, 2),
            'sl_price': round(sl_price, 2),
            'tp_price': round(tp_price, 2),
            'risk_reward': round(rr, 2),
            'confidence': confidence,
            'timeframe': 'M15',
            'expiration_bars': 20,  # ~5 hours
            'requires_dynamic_exit': False,
            'position_multiplier': 1.0,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'trailing_method': 'atr_based',
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'tmf': round(tmf.iloc[-1], 4) if tmf is not None and not tmf.isna().all() else 0,
            'cmf': round(cmf.iloc[-1], 4) if cmf is not None and not cmf.isna().all() else 0,
            'eom': round(eom.iloc[-1], 4) if eom is not None and not eom.isna().all() else 0,
            'spring_detected': spring is not None,
            'upthrust_detected': upthrust is not None,
            'no_supply': no_supply,
            'no_demand': no_demand,
            'reason': reason
        }
        
        signal = {
            'signal': signal_type,
            'meta': meta
        }
        
        self.log_signal_summary(signal)
        return signal