"""
Loss Attribution Engine
Analyzes closed trades and categorizes the root cause of the loss.
Crucial for quantitative edge decay analysis.
"""
import logging
import pandas as pd
from typing import Dict

class LossAttributionEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_loss(self, trade_data: Dict, df_at_close: pd.DataFrame, spread_at_close: float) -> str:
        """
        Determine the primary reason a trade resulted in a loss.
        
        Returns one of:
        - 'STATISTICAL_VARIANCE' (Normal loss, hit SL cleanly)
        - 'STOP_HUNT_WICK' (SL hit by a wick, price closed back in direction)
        - 'SPREAD_WIDENING' (SL hit due to abnormal spread)
        - 'REGIME_SHIFT' (Closed by Regime Conflict Liquidation)
        - 'TIME_STALL' (Closed by Time Stop due to choppy market)
        - 'FAT_TAIL_GAP' (Closed with massive slippage/gap)
        """
        entry_price = trade_data.get('entry_price', 0)
        sl_price = trade_data.get('sl', 0)
        exit_price = trade_data.get('exit_price', 0)
        is_buy = trade_data.get('position_type') == 'BUY'
        exit_reason = trade_data.get('exit_reason', 'UNKNOWN')
        
        # 1. Check Systemic Exits First
        if 'Regime Conflict' in exit_reason:
            return 'REGIME_SHIFT'
        if 'Time Stop' in exit_reason:
            return 'TIME_STALL'
        if 'Dynamic Exit' in exit_reason:
            return 'DYNAMIC_REVERSAL'
            
        # 2. Check for Fat-Tail / Gap (Slippage > 20% of initial risk)
        initial_risk = abs(entry_price - sl_price)
        if initial_risk > 0:
            slippage = abs(exit_price - sl_price)
            if slippage > (initial_risk * 0.20):
                return 'FAT_TAIL_GAP'
                
        # 3. Check Spread Widening
        # If the distance from exit to SL is less than the spread at that moment
        if spread_at_close > 0:
            distance_to_sl = abs(exit_price - sl_price)
            if distance_to_sl <= spread_at_close:
                return 'SPREAD_WIDENING'
                
        # 4. Check Stop Hunt (Wick Analysis)
        if df_at_close is not None and not df_at_close.empty:
            last_candle = df_at_close.iloc[-1]
            candle_close = last_candle['close']
            
            # If BUY: SL was hit (low went below SL), but candle closed ABOVE SL
            if is_buy and last_candle['low'] <= sl_price and candle_close > sl_price:
                return 'STOP_HUNT_WICK'
            # If SELL: SL was hit (high went above SL), but candle closed BELOW SL
            elif not is_buy and last_candle['high'] >= sl_price and candle_close < sl_price:
                return 'STOP_HUNT_WICK'
                
        # 5. Default: Clean Statistical Loss (Edge just didn't work out this time)
        return 'STATISTICAL_VARIANCE'

    def generate_decay_report(self, loss_categories: Dict[str, int]) -> str:
        """Generate a summary report of loss causes for portfolio review."""
        total_losses = sum(loss_categories.values())
        if total_losses == 0:
            return "No losses recorded."
            
        report_lines = ["[LOSS ATTRIBUTION REPORT]"]
        for category, count in sorted(loss_categories.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_losses) * 100
            report_lines.append(f"  - {category}: {count} trades ({pct:.1f}%)")
            
        # Actionable Advice based on data
        if loss_categories.get('STOP_HUNT_WICK', 0) / total_losses > 0.30:
            report_lines.append("  [WARNING] >30% losses are Stop Hunts. Increase ATR Buffer on SL.")
        if loss_categories.get('SPREAD_WIDENING', 0) / total_losses > 0.15:
            report_lines.append("  [WARNING] >15% losses from Spread. Tighten Time-Based Filters.")
        if loss_categories.get('TIME_STALL', 0) / total_losses > 0.25:
            report_lines.append("  [WARNING] High Time-Stall rate. Market is too choppy, reduce size.")
            
        return "\n".join(report_lines)