"""
core/volume_price_trend.py
Volume Price Trend (VPT) - Cumulative volume-pressure indicator.
More comprehensive than VFI for trend strength assessment.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging


class VolumePriceTrend:
    """
    Volume Price Trend (VPT) Indicator.
    
    Formula:
    VPT = Previous VPT + (Volume × ((Close - Previous Close) / Previous Close))
    
    Interpretation:
    - VPT Rising + Price Rising: Strong Uptrend (Confirmation)
    - VPT Rising + Price Falling: Bullish Divergence (Accumulation)
    - VPT Falling + Price Rising: Bearish Divergence (Distribution)
    - VPT Falling + Price Falling: Strong Downtrend (Confirmation)
    """
    
    def __init__(self, signal_period: int = 7):
        self.signal_period = signal_period
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        if df is None or len(df) < 2:
            return pd.Series(dtype=float)
        
        close = df['close']
        volume = df['tick_volume'] if 'tick_volume' in df.columns else df['volume']
        
        # Calculate price change percentage
        price_change_pct = close.pct_change()
        
        # VPT = Cumulative sum of (Volume × Price Change %)
        vpt = (volume * price_change_pct).cumsum()
        
        return vpt
    
    def calculate_signal_line(self, vpt: pd.Series) -> pd.Series:
        """Calculate VPT signal line (MA of VPT)."""
        return vpt.rolling(window=self.signal_period, min_periods=self.signal_period).mean()
    
    def analyze_vpt(self, df: pd.DataFrame) -> Dict:
        """
        Analyze VPT for trend strength and divergences.
        
        Returns:
            {
                'vpt_trend': 'UP' | 'DOWN' | 'NEUTRAL',
                'price_trend': 'UP' | 'DOWN' | 'NEUTRAL',
                'confirmation': 'CONFIRMED' | 'DIVERGENCE' | 'NEUTRAL',
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'strength': int (0-100),
                'details': str
            }
        """
        if df is None or len(df) < 20:
            return {
                'vpt_trend': 'NEUTRAL',
                'price_trend': 'NEUTRAL',
                'confirmation': 'NEUTRAL',
                'signal': 'NEUTRAL',
                'strength': 0,
                'details': 'Insufficient data'
            }
        
        try:
            vpt = self.calculate_vpt(df)
            signal = self.calculate_signal_line(vpt)
            close = df['close']
            
            # Determine VPT trend (last 10 bars)
            vpt_recent = vpt.tail(10)
            vpt_slope = (vpt_recent.iloc[-1] - vpt_recent.iloc[0]) / (abs(vpt_recent.iloc[0]) + 1e-10)
            
            if vpt_slope > 0.05:
                vpt_trend = 'UP'
            elif vpt_slope < -0.05:
                vpt_trend = 'DOWN'
            else:
                vpt_trend = 'NEUTRAL'
            
            # Determine Price trend (last 10 bars)
            price_recent = close.tail(10)
            price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0]
            
            if price_slope > 0.005:  # 0.5% move
                price_trend = 'UP'
            elif price_slope < -0.005:
                price_trend = 'DOWN'
            else:
                price_trend = 'NEUTRAL'
            
            # Determine confirmation/divergence
            confirmation = 'NEUTRAL'
            signal_type = 'NEUTRAL'
            strength = 0
            details = []
            
            if vpt_trend == 'UP' and price_trend == 'UP':
                confirmation = 'CONFIRMED'
                signal_type = 'BULLISH'
                strength = 80
                details.append("Strong Uptrend: VPT and Price both rising")
            
            elif vpt_trend == 'DOWN' and price_trend == 'DOWN':
                confirmation = 'CONFIRMED'
                signal_type = 'BEARISH'
                strength = 80
                details.append("Strong Downtrend: VPT and Price both falling")
            
            elif vpt_trend == 'UP' and price_trend == 'DOWN':
                confirmation = 'DIVERGENCE'
                signal_type = 'BULLISH'
                strength = 60
                details.append("Bullish Divergence: VPT rising while price falling (accumulation)")
            
            elif vpt_trend == 'DOWN' and price_trend == 'UP':
                confirmation = 'DIVERGENCE'
                signal_type = 'BEARISH'
                strength = 60
                details.append("Bearish Divergence: VPT falling while price rising (distribution)")
            
            # Check for VPT-Signal crossover
            if len(vpt) >= 2 and len(signal) >= 2:
                if not pd.isna(vpt.iloc[-1]) and not pd.isna(signal.iloc[-1]):
                    if vpt.iloc[-1] > signal.iloc[-1] and vpt.iloc[-2] <= signal.iloc[-2]:
                        details.append("VPT crossed above signal (bullish)")
                        strength += 10
                    elif vpt.iloc[-1] < signal.iloc[-1] and vpt.iloc[-2] >= signal.iloc[-2]:
                        details.append("VPT crossed below signal (bearish)")
                        strength -= 10
            
            strength = max(0, min(100, strength))
            
            return {
                'vpt_trend': vpt_trend,
                'price_trend': price_trend,
                'confirmation': confirmation,
                'signal': signal_type,
                'strength': strength,
                'details': ' | '.join(details)
            }
            
        except Exception as e:
            self.logger.error(f"[VPT] Error: {e}")
            return {
                'vpt_trend': 'NEUTRAL',
                'price_trend': 'NEUTRAL',
                'confirmation': 'NEUTRAL',
                'signal': 'NEUTRAL',
                'strength': 0,
                'details': f'Error: {e}'
            }
    
    def get_trend_confirmation(self, df: pd.DataFrame, is_buy: bool) -> Tuple[bool, str]:
        """
        Check if VPT confirms a trade direction.
        
        Returns:
            - confirmed: bool
            - reason: str
        """
        analysis = self.analyze_vpt(df)
        
        if is_buy:
            # For BUY: Want VPT UP or Bullish Divergence
            if analysis['signal'] == 'BULLISH':
                return True, f"VPT confirms BUY: {analysis['details']}"
            elif analysis['vpt_trend'] == 'DOWN':
                return False, f"VPT diverges from BUY: {analysis['details']}"
            else:
                return True, f"VPT neutral, no strong signal"
        
        else:
            # For SELL: Want VPT DOWN or Bearish Divergence
            if analysis['signal'] == 'BEARISH':
                return True, f"VPT confirms SELL: {analysis['details']}"
            elif analysis['vpt_trend'] == 'UP':
                return False, f"VPT diverges from SELL: {analysis['details']}"
            else:
                return True, f"VPT neutral, no strong signal"