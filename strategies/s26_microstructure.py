"""
S26_Microstructure Strategy
Order Flow Absorption + Statistical Expected Move

Category: SCALP
Optimal Regimes: TIGHT_RANGE, CLASSIC_RANGE, QUIET_RALLY, SLOW_BLEED, PRE_BREAKOUT
Timeframe: M5 (Primary), M15 (Confirmation)
"""
import pandas as pd
import logging
from typing import Dict, Optional
from core.base_strategy import BaseStrategy
from core.microstructure_predictor import MicrostructurePredictor


class S26_Microstructure(BaseStrategy):
    """
    Microstructure-based scalping strategy using Order Flow and Volatility Cone.
    """
    
    def __init__(self):
        super().__init__(
            name='S26_Microstructure',
            category='SCALP',
            description='Order Flow Absorption + Statistical Expected Move'
        )
        self.predictor = MicrostructurePredictor()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(self, df_primary: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None) -> Dict:
        """
        Evaluate microstructure signal.
        
        Args:
            df_primary: Primary timeframe data (M5)
            df_htf: Higher timeframe data (M15) for confirmation
            
        Returns:
            Dict with signal and metadata
        """
        if df_primary is None or df_primary.empty:
            return {'signal': 'NEUTRAL', 'meta': {'reason': 'No primary data'}}
        
        # Generate base signal
        signal_data = self.predictor.generate_signal(df_primary)
        
        if signal_data['signal'] == 'NEUTRAL':
            return signal_data
        
        # HTF Confirmation (Optional but recommended)
        if df_htf is not None and not df_htf.empty and len(df_htf) >= 20:
            htf_close = df_htf['close'].to_numpy()
            htf_trend_up = htf_close[-1] > htf_close[-10]
            htf_trend_down = htf_close[-1] < htf_close[-10]
            
            is_buy_signal = 'BUY' in signal_data['signal']
            
            # Reject counter-trend signals on HTF
            if is_buy_signal and htf_trend_down:
                signal_data['meta']['confidence'] *= 0.7
                signal_data['meta']['htf_warning'] = 'HTF trend is DOWN, BUY signal weakened'
                self.logger.debug(
                    f"[S26] HTF conflict: BUY signal but M15 trend is DOWN"
                )
            elif not is_buy_signal and htf_trend_up:
                signal_data['meta']['confidence'] *= 0.7
                signal_data['meta']['htf_warning'] = 'HTF trend is UP, SELL signal weakened'
                self.logger.debug(
                    f"[S26] HTF conflict: SELL signal but M15 trend is UP"
                )
            
            # Boost confidence if HTF aligned
            if (is_buy_signal and htf_trend_up) or (not is_buy_signal and htf_trend_down):
                signal_data['meta']['confidence'] = min(0.95, signal_data['meta']['confidence'] * 1.15)
                signal_data['meta']['htf_confirmation'] = 'HTF trend aligned'
        
        # Apply base strategy signal building
        meta = signal_data['meta']
        return self.build_signal(
            signal_type=signal_data['signal'],
            entry_price=meta['entry_price'],
            sl_price=meta['sl_price'],
            tp_price=meta['tp_price'],
            timeframe=meta['timeframe'],
            confidence=meta['confidence'],
            expiration_bars=12,
            requires_dynamic_exit=meta.get('requires_dynamic_exit', True),
            dynamic_exit_threshold='cvd_reversal',
            position_multiplier=1.0,
            extra_meta=meta
        )