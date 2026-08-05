"""
Reversal Detection Engine - Multi-Timeframe Version.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from core.atr_cache import ATRCache


class ReversalDetector:
    """Multi-Timeframe Reversal Detection System."""
    
    # Recommended TF per signal type
    SIGNAL_TF_MAP = {
        # Layer 1: Momentum
        'RSI_BEARISH_DIVERGENCE': 'H1',
        'RSI_BULLISH_DIVERGENCE': 'H1',
        'MACD_HISTOGRAM_REVERSAL': 'M15',
        'STOCHASTIC_OVERBOUGHT_REVERSAL': 'M15',
        'STOCHASTIC_OVERSOLD_REVERSAL': 'M15',
        
        # Layer 2: Volume
        'VOLUME_CLIMAX_BEARISH': 'M5',
        'VOLUME_CLIMAX_BULLISH': 'M5',
        'VOLUME_DELTA_REVERSAL_BEARISH': 'M1',
        'VOLUME_DELTA_REVERSAL_BULLISH': 'M1',
        
        # Layer 3: Price Action
        'SHOOTING_STAR': 'H1',
        'HAMMER': 'H1',
        'BEARISH_ENGULFING': 'M15',
        'BULLISH_ENGULFING': 'M15',
    }
    
    # Confidence boost when signal confirmed on multiple TFs
    MULTI_TF_CONFIDENCE_BOOST = 0.15
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_reversal_signals(
        self,
        df_dict: Dict[str, pd.DataFrame],
        is_buy: bool,
        current_profit_usd: float = 0.0
    ) -> Dict:
        """
        Multi-TF Reversal Detection.
        
        Args:
            df_dict: Dict of {timeframe: DataFrame} e.g., {'M1': df_m1, 'M5': df_m5, ...}
            is_buy: True if position is BUY
            current_profit_usd: Current unrealized profit
        
        Returns:
            Dict with reversal analysis
        """
        if not df_dict:
            return {
                'reversal_score': 0,
                'signals': [],
                'action': 'NONE',
                'confidence': 0.0,
                'recommendation': 'No data available'
            }
        
        all_signals = []
        
        # =========================================================================
        # Check each timeframe for its assigned signals
        # =========================================================================
        for tf_name, df in df_dict.items():
            if df is None or len(df) < 50:
                continue
            
            # Detect all signals on this TF
            momentum_signals = self._detect_momentum_exhaustion(df, is_buy, tf_name)
            volume_signals = self._detect_volume_climax(df, is_buy, tf_name)
            price_action_signals = self._detect_price_action_reversal(df, is_buy, tf_name)
            
            # Filter: Keep only signals that match this TF's assignment
            for signal in momentum_signals + volume_signals + price_action_signals:
                expected_tf = self.SIGNAL_TF_MAP.get(signal['type'], 'M15')
                
                if tf_name == expected_tf:
                    # This is the correct TF for this signal
                    signal['primary_tf'] = True
                    all_signals.append(signal)
                else:
                    # Wrong TF but still detected - lower confidence
                    signal['primary_tf'] = False
                    signal['strength'] *= 0.6  # Reduce strength by 40%
                    all_signals.append(signal)
        
        # =========================================================================
        # Multi-TF Confirmation Boost
        # =========================================================================
        # If same signal type detected on multiple TFs, boost confidence
        signal_types = [s['type'] for s in all_signals]
        signal_type_counts = {}
        for sig_type in signal_types:
            signal_type_counts[sig_type] = signal_type_counts.get(sig_type, 0) + 1
        
        for signal in all_signals:
            if signal_type_counts[signal['type']] > 1:
                signal['strength'] = min(1.0, signal['strength'] + self.MULTI_TF_CONFIDENCE_BOOST)
                signal['multi_tf_confirmed'] = True
        
        # =========================================================================
        # Calculate reversal score (unique layers triggered)
        # =========================================================================
        unique_layers = set([s['layer'] for s in all_signals if s['strength'] >= 0.6])
        reversal_score = len(unique_layers)
        confidence = min(1.0, reversal_score / 3.0)
        
        # =========================================================================
        # Determine action
        # =========================================================================
        if reversal_score >= 2:
            action = 'PARTIAL_CLOSE'
            recommendation = (
                f"Strong reversal detected ({reversal_score}/3 layers). "
                f"Recommend closing 50% of position and tightening trailing SL."
            )
        elif reversal_score == 1:
            action = 'TIGHTEN_TRAIL'
            recommendation = (
                f"Weak reversal signal ({reversal_score}/3 layers). "
                f"Recommend tightening trailing SL by 50%."
            )
        else:
            action = 'NONE'
            recommendation = "No reversal signals detected. Maintain current trailing SL."
        
        # Profit threshold check
        if current_profit_usd < 10.0 and action == 'PARTIAL_CLOSE':
            action = 'TIGHTEN_TRAIL'
            recommendation += " (Profit < 10 USD, downgraded to tighten trail)"
        
        return {
            'reversal_score': reversal_score,
            'signals': all_signals,
            'action': action,
            'confidence': confidence,
            'recommendation': recommendation,
            'current_profit_usd': current_profit_usd,
            'timeframes_checked': list(df_dict.keys())
        }
    
    def _detect_momentum_exhaustion(
        self, df: pd.DataFrame, is_buy: bool, timeframe: str
    ) -> List[Dict]:
        """Detect momentum exhaustion signals."""
        signals = []
        close = df['close'].to_numpy()
        
        # RSI Calculation
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(14).mean().to_numpy()
        avg_loss = pd.Series(loss).rolling(14).mean().to_numpy()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # RSI Divergence (Best on H1)
        if len(rsi) >= 20:
            # Adjust lookback based on timeframe
            lookback_map = {'M1': 30, 'M5': 25, 'M15': 20, 'H1': 15, 'H4': 10}
            lookback = lookback_map.get(timeframe, 20)
            
            recent_price_high = close[-lookback//2:].max()
            prev_price_high = close[-lookback:-lookback//2].max()
            
            recent_rsi_high = rsi[-lookback//2:].max()
            prev_rsi_high = rsi[-lookback:-lookback//2].max()
            
            # Bearish Divergence
            if is_buy and recent_price_high > prev_price_high and recent_rsi_high < prev_rsi_high:
                # Strength varies by TF
                tf_strength = {'M1': 0.4, 'M5': 0.5, 'M15': 0.7, 'H1': 0.9, 'H4': 0.95}
                strength = tf_strength.get(timeframe, 0.7)
                
                signals.append({
                    'layer': 1,
                    'type': 'RSI_BEARISH_DIVERGENCE',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': (
                        f"RSI bearish divergence on {timeframe}: "
                        f"Price higher high but RSI lower high"
                    )
                })
            
            # Bullish Divergence
            elif not is_buy:
                recent_price_low = close[-lookback//2:].min()
                prev_price_low = close[-lookback:-lookback//2].min()
                
                recent_rsi_low = rsi[-lookback//2:].min()
                prev_rsi_low = rsi[-lookback:-lookback//2].min()
                
                if recent_price_low < prev_price_low and recent_rsi_low > prev_rsi_low:
                    tf_strength = {'M1': 0.4, 'M5': 0.5, 'M15': 0.7, 'H1': 0.9, 'H4': 0.95}
                    strength = tf_strength.get(timeframe, 0.7)
                    
                    signals.append({
                        'layer': 1,
                        'type': 'RSI_BULLISH_DIVERGENCE',
                        'strength': strength,
                        'timeframe': timeframe,
                        'description': (
                            f"RSI bullish divergence on {timeframe}: "
                            f"Price lower low but RSI higher low"
                        )
                    })
        
        # MACD Histogram Reversal (Best on M15)
        if len(close) >= 26:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy()
            macd = ema12 - ema26
            signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().to_numpy()
            histogram = macd - signal_line
            
            if len(histogram) >= 3:
                tf_strength = {'M1': 0.4, 'M5': 0.6, 'M15': 0.8, 'H1': 0.7, 'H4': 0.6}
                strength = tf_strength.get(timeframe, 0.7)
                
                if is_buy:
                    if histogram[-3] > 0 and histogram[-2] > 0 and histogram[-1] < histogram[-2] and histogram[-1] < 0:
                        signals.append({
                            'layer': 1,
                            'type': 'MACD_HISTOGRAM_REVERSAL',
                            'strength': strength,
                            'timeframe': timeframe,
                            'description': f"MACD histogram reversed on {timeframe}"
                        })
                else:
                    if histogram[-3] < 0 and histogram[-2] < 0 and histogram[-1] > histogram[-2] and histogram[-1] > 0:
                        signals.append({
                            'layer': 1,
                            'type': 'MACD_HISTOGRAM_REVERSAL',
                            'strength': strength,
                            'timeframe': timeframe,
                            'description': f"MACD histogram reversed on {timeframe}"
                        })
        
        # Stochastic (Best on M15)
        if len(close) >= 14:
            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            
            lowest_low = pd.Series(low).rolling(14).min().to_numpy()
            highest_high = pd.Series(high).rolling(14).max().to_numpy()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            
            tf_strength = {'M1': 0.4, 'M5': 0.6, 'M15': 0.7, 'H1': 0.6, 'H4': 0.5}
            strength = tf_strength.get(timeframe, 0.6)
            
            if is_buy and stoch_k[-2] > 80 and stoch_k[-1] < stoch_k[-2]:
                signals.append({
                    'layer': 1,
                    'type': 'STOCHASTIC_OVERBOUGHT_REVERSAL',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Stochastic overbought reversal on {timeframe}"
                })
            elif not is_buy and stoch_k[-2] < 20 and stoch_k[-1] > stoch_k[-2]:
                signals.append({
                    'layer': 1,
                    'type': 'STOCHASTIC_OVERSOLD_REVERSAL',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Stochastic oversold reversal on {timeframe}"
                })
        
        return signals
    
    def _detect_volume_climax(
        self, df: pd.DataFrame, is_buy: bool, timeframe: str
    ) -> List[Dict]:
        """Detect volume climax signals."""
        signals = []
        
        if 'tick_volume' in df.columns:
            volume = df['tick_volume'].to_numpy()
        elif 'volume' in df.columns:
            volume = df['volume'].to_numpy()
        else:
            return signals
        
        close = df['close'].to_numpy()
        open_ = df['open'].to_numpy()
        
        # Volume Spike (Best on M5)
        if len(volume) >= 20:
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            
            if current_volume > avg_volume * 3.0:
                tf_strength = {'M1': 0.7, 'M5': 0.9, 'M15': 0.8, 'H1': 0.7, 'H4': 0.6}
                strength = tf_strength.get(timeframe, 0.8)
                
                if is_buy and close[-1] < open_[-1]:
                    signals.append({
                        'layer': 2,
                        'type': 'VOLUME_CLIMAX_BEARISH',
                        'strength': strength,
                        'timeframe': timeframe,
                        'description': (
                            f"Volume climax on {timeframe}: "
                            f"{current_volume:.0f} vs avg {avg_volume:.0f}"
                        )
                    })
                elif not is_buy and close[-1] > open_[-1]:
                    signals.append({
                        'layer': 2,
                        'type': 'VOLUME_CLIMAX_BULLISH',
                        'strength': strength,
                        'timeframe': timeframe,
                        'description': (
                            f"Volume climax on {timeframe}: "
                            f"{current_volume:.0f} vs avg {avg_volume:.0f}"
                        )
                    })
        
        # Volume Delta Reversal (Best on M1)
        if len(volume) >= 10:
            direction = np.where(close >= open_, 1, -1)
            volume_delta = direction * volume
            
            recent_delta = np.sum(volume_delta[-5:])
            prev_delta = np.sum(volume_delta[-10:-5])
            
            tf_strength = {'M1': 0.8, 'M5': 0.6, 'M15': 0.4, 'H1': 0.3, 'H4': 0.2}
            strength = tf_strength.get(timeframe, 0.5)
            
            if is_buy and prev_delta > 0 and recent_delta < 0:
                signals.append({
                    'layer': 2,
                    'type': 'VOLUME_DELTA_REVERSAL_BEARISH',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Volume delta reversed on {timeframe}"
                })
            elif not is_buy and prev_delta < 0 and recent_delta > 0:
                signals.append({
                    'layer': 2,
                    'type': 'VOLUME_DELTA_REVERSAL_BULLISH',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Volume delta reversed on {timeframe}"
                })
        
        return signals
    
    def _detect_price_action_reversal(
        self, df: pd.DataFrame, is_buy: bool, timeframe: str
    ) -> List[Dict]:
        """Detect price action reversal patterns."""
        signals = []
        
        if len(df) < 3:
            return signals
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()
        open_ = df['open'].to_numpy()
        
        curr_high = high[-1]
        curr_low = low[-1]
        curr_close = close[-1]
        curr_open = open_[-1]
        curr_range = curr_high - curr_low
        
        if curr_range == 0:
            return signals
        
        # Shooting Star / Hammer (Best on H1)
        tf_strength = {'M1': 0.3, 'M5': 0.5, 'M15': 0.7, 'H1': 0.9, 'H4': 0.95}
        strength = tf_strength.get(timeframe, 0.7)
        
        if is_buy:
            upper_wick = curr_high - max(curr_close, curr_open)
            lower_wick = min(curr_close, curr_open) - curr_low
            body = abs(curr_close - curr_open)
            
            if (upper_wick > body * 2 and 
                upper_wick > lower_wick * 2 and 
                body < curr_range * 0.3):
                signals.append({
                    'layer': 3,
                    'type': 'SHOOTING_STAR',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Shooting star on {timeframe}"
                })
        
        elif not is_buy:
            upper_wick = curr_high - max(curr_close, curr_open)
            lower_wick = min(curr_close, curr_open) - curr_low
            body = abs(curr_close - curr_open)
            
            if (lower_wick > body * 2 and 
                lower_wick > upper_wick * 2 and 
                body < curr_range * 0.3):
                signals.append({
                    'layer': 3,
                    'type': 'HAMMER',
                    'strength': strength,
                    'timeframe': timeframe,
                    'description': f"Hammer on {timeframe}"
                })
        
        # Engulfing Pattern (Best on M15)
        if len(df) >= 2:
            prev_close = close[-2]
            prev_open = open_[-2]
            prev_range = abs(prev_close - prev_open)
            
            tf_strength = {'M1': 0.4, 'M5': 0.6, 'M15': 0.9, 'H1': 0.85, 'H4': 0.8}
            strength = tf_strength.get(timeframe, 0.8)
            
            if is_buy:
                if (curr_close < curr_open and 
                    prev_close > prev_open and 
                    curr_close < prev_open and 
                    curr_open > prev_close and 
                    abs(curr_close - curr_open) > prev_range):
                    signals.append({
                        'layer': 3,
                        'type': 'BEARISH_ENGULFING',
                        'strength': strength,
                        'timeframe': timeframe,
                        'description': f"Bearish engulfing on {timeframe}"
                    })
            
            elif not is_buy:
                if (curr_close > curr_open and 
                    prev_close < prev_open and 
                    curr_close > prev_open and 
                    curr_open < prev_close and 
                    abs(curr_close - curr_open) > prev_range):
                    signals.append({
                        'layer': 3,
                        'type': 'BULLISH_ENGULFING',
                        'strength': strength,
                        'timeframe': timeframe,
                        'description': f"Bullish engulfing on {timeframe}"
                    })
        
        return signals