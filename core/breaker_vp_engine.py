"""
Breaker Block & Volume Profile Engine.
Includes NaN handling, infinite loop guard, and minimum distance validation.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional
from core.smc_engine import SMCStructuralEngine
from core.atr_cache import ATRCache


class BreakerVPEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.smc = SMCStructuralEngine()

    def detect_breaker_blocks(self, df: pd.DataFrame, lookback: int = 100, order: int = 3) -> List[Dict]:
        """
        Detect Breaker Blocks (failed Order Blocks that flipped role).
        """
        if df is None or len(df) < lookback:
            return []
        
        swings_high, swings_low = self.smc.detect_swings(df, order=order)
        breakers = []
        
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        
        # Bearish Breakers: Swing Low that got broken (now acts as resistance)
        for idx in swings_low:
            if idx >= len(df) - 5:
                continue
            for j in range(idx + 1, min(idx + 20, len(df))):
                if close_arr[j] < low_arr[idx]:
                    breakers.append({
                        'type': 'BEARISH',
                        'upper': float(high_arr[idx]),
                        'lower': float(low_arr[idx]),
                        'bar_index': int(idx),
                        'break_bar': int(j)
                    })
                    break
        
        # Bullish Breakers: Swing High that got broken (now acts as support)
        for idx in swings_high:
            if idx >= len(df) - 5:
                continue
            for j in range(idx + 1, min(idx + 20, len(df))):
                if close_arr[j] > high_arr[idx]:
                    breakers.append({
                        'type': 'BULLISH',
                        'upper': float(high_arr[idx]),
                        'lower': float(low_arr[idx]),
                        'bar_index': int(idx),
                        'break_bar': int(j)
                    })
                    break
        
        return breakers

    def detect_fvg_boxes(self, df: pd.DataFrame, atr_multiplier: float = 0.3) -> List[Dict]:
        """
        Detect FVG boxes with ATR-based minimum gap filter.
        """
        if df is None or len(df) < 3:
            return []
        
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        
        # Auto-calculate ATR if not present
        if 'atr' in df.columns:
            atr = df['atr'].to_numpy()
        else:
            atr = ATRCache.get_atr(df, 14).to_numpy()
        
        atr = np.nan_to_num(atr, nan=1.0)
        
        boxes = []
        min_gap = atr * atr_multiplier
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle[i-2].high and candle[i].low
            gap_up = low[i] - high[i-2]
            if gap_up > min_gap[i] and gap_up > 0:
                boxes.append({
                    'type': 'BULLISH',
                    'upper': float(low[i]),
                    'lower': float(high[i-2]),
                    'bar_index': int(i)
                })
            
            # Bearish FVG: Gap between candle[i].high and candle[i-2].low
            gap_down = low[i-2] - high[i]
            if gap_down > min_gap[i] and gap_down > 0:
                boxes.append({
                    'type': 'BEARISH',
                    'upper': float(low[i-2]),
                    'lower': float(high[i]),
                    'bar_index': int(i)
                })
        
        return boxes

    def calculate_session_vp_poc(self, ticks_df: pd.DataFrame, bins: int = 240) -> float:
        """Calculate Point of Control (POC) from tick data."""
        if ticks_df is None or ticks_df.empty:
            return 0.0
        
        if 'last' not in ticks_df.columns:
            return 0.0
        
        prices = ticks_df['last'].to_numpy().astype(float)
        volumes = ticks_df['volume'].to_numpy().astype(float) if 'volume' in ticks_df.columns else np.ones(len(prices))
        
        valid_mask = ~(np.isnan(prices) | np.isnan(volumes))
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(prices) == 0 or np.max(prices) == np.min(prices):
            return float(prices[-1]) if len(prices) > 0 else 0.0
        
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        poc_idx = np.argmax(hist)
        return float((bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2.0)

    def validate_triple_confluence(self, breakers: List[Dict], fvgs: List[Dict], 
                                    poc: float, tolerance: float) -> Optional[Dict]:
        """Validate triple confluence of Breaker + FVG + POC."""
        if not breakers or not fvgs or poc == 0:
            return None
        
        for b in reversed(breakers):
            for f in reversed(fvgs):
                if b['type'] != f['type']:
                    continue
                
                overlap_lower = max(b['lower'], f['lower'])
                overlap_upper = min(b['upper'], f['upper'])
                
                if overlap_lower < overlap_upper and (overlap_lower - tolerance) <= poc <= (overlap_upper + tolerance):
                    return {
                        'type': b['type'],
                        'breaker_lower': b['lower'],
                        'breaker_upper': b['upper'],
                        'fvg_lower': f['lower'],
                        'fvg_upper': f['upper'],
                        'entry_level': poc
                    }
        
        return None

    def create_synthetic_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic tick data from OHLCV for Volume Profile calculation.
        ENHANCED: Microstructure-aware tick generation.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        ticks_list = []
        session_volatility_mult = self._get_session_volatility_multiplier(df)
        
        for idx, row in df.iterrows():
            bar_range = row.high - row.low
            if bar_range <= 0:
                continue
            
            bar_body = abs(row.close - row.open)
            upper_wick = row.high - max(row.open, row.close)
            lower_wick = min(row.open, row.close) - row.low
            
            upper_wick_ratio = upper_wick / bar_range if bar_range > 0 else 0
            lower_wick_ratio = lower_wick / bar_range if bar_range > 0 else 0
            
            base_ticks = max(20, int(row.tick_volume / 8))
            n_ticks = min(base_ticks, 200)
            
            base_std = bar_range * 0.15 * session_volatility_mult
            wick_factor = 1.0 + (upper_wick_ratio + lower_wick_ratio) * 0.5
            adjusted_std = base_std * wick_factor
            
            for i in range(n_ticks):
                tick_position = i / n_ticks
                
                if tick_position < 0.2:
                    anchor_price = row.open
                    std_mult = 0.8
                elif tick_position > 0.8:
                    anchor_price = row.close
                    std_mult = 0.8
                else:
                    progress = (tick_position - 0.2) / 0.6
                    anchor_price = row.open + (row.close - row.open) * progress
                    std_mult = 1.2
                
                tick_std = adjusted_std * std_mult
                price = anchor_price + np.random.normal(0, tick_std)
                price = self._apply_key_level_clustering(price, row.low, row.high)
                price = float(np.clip(price, row.low, row.high))
                
                tick_volume = row.tick_volume / n_ticks
                
                ticks_list.append({
                    'time': row.time,
                    'last': price,
                    'volume': tick_volume
                })
        
        if not ticks_list:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(ticks_list)
        result_df = self._remove_tick_outliers(result_df)
        result_df = result_df.sort_values(['time', 'last']).reset_index(drop=True)
        
        return result_df
    
    def _get_session_volatility_multiplier(self, df: pd.DataFrame) -> float:
        """Get volatility multiplier based on trading session."""
        if df is None or df.empty or 'time' not in df.columns:
            return 1.0
        
        try:
            last_time = df['time'].iloc[-1]
            if not isinstance(last_time, pd.Timestamp):
                last_time = pd.to_datetime(last_time)
            
            import pytz
            if last_time.tzinfo is None:
                last_time = last_time.tz_localize('UTC')
            
            ny_tz = pytz.timezone('America/New_York')
            ny_time = last_time.astimezone(ny_tz)
            hour = ny_time.hour
            
            if 2 <= hour < 5:
                return 1.0
            elif 8 <= hour < 11:
                return 1.3
            elif 5 <= hour < 8:
                return 1.0
            elif 11 <= hour < 12:
                return 1.5
            elif (19 <= hour <= 23) or (0 <= hour <= 1):
                return 0.8
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"[VP] Session detection failed: {e}")
            return 1.0
    
    def _apply_key_level_clustering(self, price: float, bar_low: float, bar_high: float) -> float:
        """Apply clustering around key psychological levels."""
        key_interval = 5.0
        
        lower_key = (price // key_interval) * key_interval
        upper_key = lower_key + key_interval
        
        dist_lower = abs(price - lower_key)
        dist_upper = abs(price - upper_key)
        
        attraction_threshold = 0.3
        
        if dist_lower < attraction_threshold:
            attraction_strength = (attraction_threshold - dist_lower) / attraction_threshold
            price = price - (price - lower_key) * attraction_strength * 0.5
        elif dist_upper < attraction_threshold:
            attraction_strength = (attraction_threshold - dist_upper) / attraction_threshold
            price = price + (upper_key - price) * attraction_strength * 0.5
        
        return float(np.clip(price, bar_low, bar_high))
    
    def _remove_tick_outliers(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Remove synthetic ticks that are statistical outliers."""
        if ticks_df is None or ticks_df.empty:
            return ticks_df
        
        try:
            prices = ticks_df['last'].to_numpy()
            
            q1 = np.percentile(prices, 25)
            q3 = np.percentile(prices, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_mask = (prices >= lower_bound) & (prices <= upper_bound)
            filtered_df = ticks_df[valid_mask].reset_index(drop=True)
            
            return filtered_df
            
        except Exception as e:
            self.logger.warning(f"[VP] Outlier removal failed: {e}")
            return ticks_df

    def calculate_volume_profile_levels(
        self, ticks_df: pd.DataFrame, bins: int = 240, value_area_pct: float = 0.70
    ) -> Dict[str, float]:
        """Calculate Volume Profile levels (POC, VAH, VAL, HVNs, LVNs)."""
        if ticks_df is None or ticks_df.empty or 'last' not in ticks_df.columns:
            return {'poc': 0.0, 'vah': 0.0, 'val': 0.0, 'hvns': [], 'lvns': []}
        
        prices = ticks_df['last'].to_numpy().astype(float)
        volumes = ticks_df['volume'].to_numpy().astype(float) if 'volume' in ticks_df.columns else np.ones(len(prices))
        
        valid_mask = ~(np.isnan(prices) | np.isnan(volumes))
        prices = prices[valid_mask]
        volumes = volumes[valid_mask]
        
        if len(prices) == 0 or np.max(prices) == np.min(prices):
            last_price = float(prices[-1]) if len(prices) > 0 else 0.0
            return {'poc': last_price, 'vah': last_price, 'val': last_price, 'hvns': [], 'lvns': []}
        
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        
        poc_idx = int(np.argmax(hist))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)
        
        total_volume = float(np.sum(hist))
        target_volume = total_volume * value_area_pct
        
        vah_idx = poc_idx
        val_idx = poc_idx
        accumulated_volume = float(hist[poc_idx])
        
        max_iterations = bins * 2
        iteration = 0
        
        while accumulated_volume < target_volume and iteration < max_iterations:
            iteration += 1
            
            upper_volume = float(hist[vah_idx + 1]) if vah_idx < len(hist) - 1 else 0.0
            lower_volume = float(hist[val_idx - 1]) if val_idx > 0 else 0.0
            
            if upper_volume == 0 and lower_volume == 0:
                break
            
            if upper_volume >= lower_volume and vah_idx < len(hist) - 1:
                vah_idx += 1
                accumulated_volume += float(hist[vah_idx])
            elif val_idx > 0:
                val_idx -= 1
                accumulated_volume += float(hist[val_idx])
            else:
                break
        
        vah_price = float(bin_edges[min(vah_idx + 1, len(bin_edges) - 1)])
        val_price = float(bin_edges[max(val_idx, 0)])
        
        hvns = []
        mean_hist = float(np.mean(hist))
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > (mean_hist * 1.5):
                hvn_price = float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                hvns.append(hvn_price)
        
        lvns = []
        for i in range(1, len(hist) - 1):
            if hist[i] < hist[i-1] and hist[i] < hist[i+1] and hist[i] < (mean_hist * 0.5):
                lvn_price = float((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                lvns.append(lvn_price)
        
        current_price = float(prices[-1])
        hvns.sort(key=lambda x: abs(x - current_price))
        lvns.sort(key=lambda x: abs(x - current_price))
        
        return {
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'hvns': hvns[:5],
            'lvns': lvns[:5]
        }

    def calculate_vp_based_sl_tp(
        self, entry_price: float, is_buy: bool, vp_levels: Dict, atr: float,
        strategy_type: str = 'TREND', min_distance_atr: float = 1.0
    ) -> Dict[str, float]:
        """Calculate SL and TP based on Volume Profile levels."""
        poc = vp_levels.get('poc', entry_price)
        vah = vp_levels.get('vah', entry_price)
        val = vp_levels.get('val', entry_price)
        hvns = vp_levels.get('hvns', [])
        lvns = vp_levels.get('lvns', [])
        
        sl = 0.0
        tp = 0.0
        sl_reason = ''
        tp_reason = ''
        
        min_distance = min_distance_atr * atr
        
        if strategy_type == 'TREND':
            if is_buy:
                sl = val - (0.5 * atr)
                sl_reason = f"Below VAL ({val:.2f}) - 0.5*ATR"
                next_hvns = [h for h in hvns if h > entry_price]
                if next_hvns and next_hvns[0] > entry_price + min_distance:
                    tp = next_hvns[0]
                    tp_reason = f"Next HVN at {next_hvns[0]:.2f}"
                else:
                    tp = vah
                    tp_reason = f"VAH at {vah:.2f}"
            else:
                sl = vah + (0.5 * atr)
                sl_reason = f"Above VAH ({vah:.2f}) + 0.5*ATR"
                next_hvns = [h for h in hvns if h < entry_price]
                if next_hvns and next_hvns[0] < entry_price - min_distance:
                    tp = next_hvns[0]
                    tp_reason = f"Next HVN at {next_hvns[0]:.2f}"
                else:
                    tp = val
                    tp_reason = f"VAL at {val:.2f}"
        
        elif strategy_type == 'MEAN_REVERSION':
            if is_buy:
                sl = poc - (1.0 * atr)
                sl_reason = f"Below POC ({poc:.2f}) - 1.0*ATR"
                tp = vah
                tp_reason = f"VAH at {vah:.2f}"
            else:
                sl = poc + (1.0 * atr)
                sl_reason = f"Above POC ({poc:.2f}) + 1.0*ATR"
                tp = val
                tp_reason = f"VAL at {val:.2f}"
        
        else:  # SCALP
            if is_buy:
                nearest_lvn = lvns[0] if lvns else val
                sl = nearest_lvn - (0.3 * atr)
                sl_reason = f"Below LVN ({nearest_lvn:.2f}) - 0.3*ATR"
                nearest_hvns = [h for h in hvns if h > entry_price]
                tp = nearest_hvns[0] if nearest_hvns else poc
                tp_reason = f"{'HVN' if nearest_hvns else 'POC'} at {tp:.2f}"
            else:
                nearest_lvn = lvns[0] if lvns else vah
                sl = nearest_lvn + (0.3 * atr)
                sl_reason = f"Above LVN ({nearest_lvn:.2f}) + 0.3*ATR"
                nearest_hvns = [h for h in hvns if h < entry_price]
                tp = nearest_hvns[0] if nearest_hvns else poc
                tp_reason = f"{'HVN' if nearest_hvns else 'POC'} at {tp:.2f}"
        
        # Validate minimum distance
        sl_distance = abs(entry_price - sl)
        tp_distance = abs(tp - entry_price)
        
        if sl_distance < min_distance:
            sl = entry_price - min_distance if is_buy else entry_price + min_distance
            sl_reason += f" [ADJUSTED to min {min_distance:.2f}]"
        
        if tp_distance < min_distance:
            tp = entry_price + min_distance if is_buy else entry_price - min_distance
            tp_reason += f" [ADJUSTED to min {min_distance:.2f}]"
        
        return {
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'sl_reason': sl_reason,
            'tp_reason': tp_reason,
            'poc': round(poc, 2),
            'vah': round(vah, 2),
            'val': round(val, 2)
        }