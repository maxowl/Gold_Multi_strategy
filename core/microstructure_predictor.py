"""
Microstructure Predictor for XAUUSD (M1, M5, M15).
Calculates Directional Bias using Order Flow (CVD Absorption)
and Estimates Magnitude using Statistical Volatility and Liquidity Magnets.

Institutional-Grade Implementation:
  - Cumulative Volume Delta (CVD) Analysis
  - Absorption Detection (Price/Volume Divergence)
  - Statistical Expected Move (Volatility Cone)
  - Noise Filtering for CFD Tick Volume
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional


class MicrostructurePredictor:
    """
    Predicts short-term directional bias and expected magnitude.
    
    Parameters:
        symbol: Trading symbol
        cvd_lookback: Bars for CVD analysis (default 20)
        volatility_lookback: Bars for StdDev calculation (default 50)
        z_score_multiplier: Sigma boundary for expected move (default 1.8)
        min_tick_volume: Minimum average tick volume to filter noise (default 50)
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration Parameters
        self.cvd_lookback = 20
        self.volatility_lookback = 50
        self.z_score_multiplier = 1.8
        self.min_tick_volume = 50
        self.min_rr_ratio = 1.2
        self.atr_buffer_multiplier = 0.5

    def predict_direction(self, df: pd.DataFrame) -> Dict:
        """
        Analyze Order Flow to detect Absorption (Directional Bias).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with direction (UP, DOWN, NEUTRAL), confidence, and reason
        """
        if df is None or len(df) < self.cvd_lookback + 5:
            return {'direction': 'NEUTRAL', 'confidence': 0.0, 'reason': 'Insufficient data'}

        try:
            close = df['close'].to_numpy()
            open_ = df['open'].to_numpy()
            
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].to_numpy().astype(float)
            elif 'volume' in df.columns:
                volume = df['volume'].to_numpy().astype(float)
            else:
                return {'direction': 'NEUTRAL', 'confidence': 0.0, 'reason': 'No volume data'}

            # Calculate Volume Delta per bar
            direction_mult = np.where(close >= open_, 1, -1)
            delta = direction_mult * volume
            cvd = np.cumsum(delta)

            # Analyze recent structural swings vs CVD
            recent_close = close[-self.cvd_lookback:]
            recent_cvd = cvd[-self.cvd_lookback:]

            price_low_idx = np.argmin(recent_close)
            price_high_idx = np.argmax(recent_close)
            
            cvd_at_price_low = recent_cvd[price_low_idx]
            cvd_at_price_high = recent_cvd[price_high_idx]
            
            current_close = close[-1]
            current_cvd = cvd[-1]
            
            # Bullish Absorption: Price makes lower low (or tests low), CVD makes higher low
            price_near_low = current_close <= (recent_close.min() * 1.002)
            cvd_diverging_up = current_cvd > (cvd_at_price_low + np.std(recent_cvd))
            
            # Bearish Absorption: Price near high, CVD dropping
            price_near_high = current_close >= (recent_close.max() * 0.998)
            cvd_diverging_down = current_cvd < (cvd_at_price_high - np.std(recent_cvd))

            if price_near_low and cvd_diverging_up:
                return {
                    'direction': 'UP',
                    'confidence': 0.85,
                    'reason': 'Bullish Absorption detected (Price at low, CVD rising)'
                }
            elif price_near_high and cvd_diverging_down:
                return {
                    'direction': 'DOWN',
                    'confidence': 0.85,
                    'reason': 'Bearish Absorption detected (Price at high, CVD dropping)'
                }
            else:
                # Trend continuation check
                if current_cvd > cvd[-5] and current_close > close[-5]:
                    return {'direction': 'UP', 'confidence': 0.60, 'reason': 'Order flow continuation UP'}
                elif current_cvd < cvd[-5] and current_close < close[-5]:
                    return {'direction': 'DOWN', 'confidence': 0.60, 'reason': 'Order flow continuation DOWN'}
                    
                return {'direction': 'NEUTRAL', 'confidence': 0.30, 'reason': 'No clear absorption or trend'}

        except Exception as e:
            self.logger.error(f"[MICRO_PREDICT] Direction error: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}

    def estimate_magnitude(self, df: pd.DataFrame, is_buy: bool) -> Dict:
        """
        Estimate the expected magnitude (target distance) using Volatility Cone.
        
        Args:
            df: DataFrame with OHLCV data
            is_buy: True for BUY positions, False for SELL
            
        Returns:
            Dict with expected_move, target_price, atr, std_dev, and reason
        """
        if df is None or len(df) < self.volatility_lookback:
            return {'expected_move': 0.0, 'target_price': 0.0, 'reason': 'Insufficient data'}

        try:
            close = df['close'].to_numpy()
            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            current_price = close[-1]

            # Calculate True Range for ATR
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

            # Calculate Standard Deviation of returns
            returns = np.diff(np.log(close[-self.volatility_lookback:]))
            std_dev = np.std(returns)
            
            # Expected Move (Statistical boundary)
            expected_move_pct = std_dev * self.z_score_multiplier
            expected_move_price = current_price * expected_move_pct
            
            # Ensure expected move is at least 1.5x ATR to cover spread/friction
            min_move = atr * 1.5
            final_expected_move = max(expected_move_price, min_move)

            if is_buy:
                target_price = current_price + final_expected_move
            else:
                target_price = current_price - final_expected_move

            return {
                'expected_move': round(final_expected_move, 2),
                'target_price': round(target_price, 2),
                'atr': round(atr, 2),
                'std_dev': round(std_dev, 6),
                'reason': f'Statistical Expected Move ({self.z_score_multiplier} Sigma)'
            }

        except Exception as e:
            self.logger.error(f"[MICRO_PREDICT] Magnitude error: {e}")
            return {'expected_move': 0.0, 'target_price': 0.0, 'reason': f'Error: {str(e)}'}

    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """
        Master method to generate a complete microstructure signal.
        Combines Directional Bias and Magnitude Estimation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with signal type and metadata
        """
        if df is None or df.empty:
            return {'signal': 'NEUTRAL', 'meta': {}}

        # Apply Volume Noise Filter
        if 'tick_volume' in df.columns:
            avg_vol = df['tick_volume'].rolling(20).mean().iloc[-1]
            if pd.notna(avg_vol) and avg_vol < self.min_tick_volume:
                return {
                    'signal': 'NEUTRAL', 
                    'meta': {'reason': f'Volume too low ({avg_vol:.0f} < {self.min_tick_volume})'}
                }

        direction_data = self.predict_direction(df)
        
        if direction_data['direction'] == 'NEUTRAL' or direction_data['confidence'] < 0.55:
            return {
                'signal': 'NEUTRAL',
                'meta': {'reason': direction_data['reason'], 'confidence': direction_data['confidence']}
            }

        is_buy = direction_data['direction'] == 'UP'
        magnitude_data = self.estimate_magnitude(df, is_buy)

        current_price = float(df['close'].iloc[-1])
        
        # Calculate SL based on recent structural swing + ATR buffer
        recent_low = float(df['low'].iloc[-20:].min())
        recent_high = float(df['high'].iloc[-20:].max())
        atr = magnitude_data.get('atr', 5.0)

        if is_buy:
            sl_price = recent_low - (atr * self.atr_buffer_multiplier)
            entry_price = current_price
        else:
            sl_price = recent_high + (atr * self.atr_buffer_multiplier)
            entry_price = current_price

        risk = abs(entry_price - sl_price)
        reward = abs(magnitude_data['target_price'] - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        # Reject if R:R is mathematically unsound for M1/M5 friction
        if rr_ratio < self.min_rr_ratio:
            return {
                'signal': 'NEUTRAL',
                'meta': {'reason': f'R:R too low ({rr_ratio:.2f}) after magnitude estimation'}
            }

        signal_type = 'BUY_MARKET' if is_buy else 'SELL_MARKET'

        meta = {
            'strategy': 'S26_Microstructure',
            'strategy_category': 'SCALP',
            'entry_price': round(entry_price, 2),
            'sl_price': round(sl_price, 2),
            'tp_price': magnitude_data['target_price'],
            'risk_reward': round(rr_ratio, 2),
            'confidence': direction_data['confidence'],
            'timeframe': 'M5',
            'direction_reason': direction_data['reason'],
            'magnitude_reason': magnitude_data['reason'],
            'expected_move_usd': magnitude_data['expected_move'],
            'requires_dynamic_exit': True,
            'trailing_enabled': True,
            'partial_close_enabled': True,
            'friction_sensitive': True
        }

        return {
            'signal': signal_type,
            'meta': meta
        }