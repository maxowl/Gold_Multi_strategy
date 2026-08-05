"""
Entry Optimizer Engine
Converts Market Orders to Limit Orders based on Regime and Volatility
to reduce Execution Friction and improve Asymmetric Payoff.
"""
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from typing import Dict, Tuple, Optional


class EntryOptimizer:
    def __init__(self, symbol: str = "XAUUSDm"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_entry(self, signal: Dict, df_m5: pd.DataFrame) -> Dict:
        """
        Analyze signal and determine optimal execution method.
        Returns updated signal dict with execution metadata.
        """
        meta = signal.get('meta', {})
        regime_name = meta.get('regime_name', 'UNKNOWN')
        signal_type = signal.get('signal', '')
        
        # Regimes where Limit Orders are highly effective (Pullback expected)
        limit_friendly_regimes = [
            'CLASSIC_RANGE', 'TIGHT_RANGE', 'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
            'OVERSOLD_BOUNCE', 'EXHAUSTED_BULL', 'EXHAUSTED_BEAR', 'FALSE_SIDEWAY'
        ]
        
        # Regimes where Market Orders are mandatory (Momentum/Breakout)
        market_mandatory_regimes = [
            'PARABOLIC_RALLY', 'PANIC_CAPITULATION', 'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'PRE_BREAKOUT', 'VOLATILE_CHOP'
        ]
        
        execution_method = 'MARKET'
        limit_price = 0.0
        expiration_minutes = 0
        
        if regime_name in limit_friendly_regimes and 'MARKET' in signal_type:
            # Calculate ATR for limit offset
            offset_multiplier = 0.5
            atr = self._get_current_atr(df_m5)
            
            if atr > 0:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    is_buy = 'BUY' in signal_type
                    offset = atr * offset_multiplier
                    
                    if is_buy:
                        # Place Buy Limit below current Ask to catch pullback
                        limit_price = tick.ask - offset
                        execution_method = 'BUY_LIMIT'
                    else:
                        # Place Sell Limit above current Bid to catch pullback
                        limit_price = tick.bid + offset
                        execution_method = 'SELL_LIMIT'
                        
                    expiration_minutes = 45  # Cancel if not filled in 45 mins
                    self.logger.info(
                        f"[ENTRY OPT] Converting {signal_type} to {execution_method} "
                        f"at {limit_price:.2f} (Offset: {offset:.2f}) due to Regime: {regime_name}"
                    )
                    
        elif regime_name in market_mandatory_regimes:
            self.logger.debug(f"[ENTRY OPT] Forcing MARKET order due to momentum regime: {regime_name}")
            
        # Inject execution metadata
        meta['execution_method'] = execution_method
        meta['optimized_limit_price'] = limit_price
        meta['limit_expiration_minutes'] = expiration_minutes
        signal['meta'] = meta
        
        return signal

    def _get_current_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR safely."""
        if df is None or len(df) < period + 1:
            return 0.0
            
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.mean(tr[-period:])
        return float(atr)