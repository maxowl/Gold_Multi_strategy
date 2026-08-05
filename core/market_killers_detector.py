"""
Market Killers Detection System
Identifies 5 dangerous market conditions that destroy trading systems:
1. Liquidity Void / Flash Crash
2. Holiday / Thin Market
3. Correlation Breakdown
4. Central Bank Intervention
5. Contract Roll-over / Triple Witching
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5


class MarketKillersDetector:
    def __init__(self, symbol: str = "XAUUSDm"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Holiday Calendar 2026 (Major Global Holidays)
        self.holidays_2026 = [
            '2026-01-01',  # New Year's Day
            '2026-01-19',  # Martin Luther King Jr. Day (US)
            '2026-02-16',  # Presidents' Day (US)
            '2026-04-03',  # Good Friday
            '2026-04-06',  # Easter Monday
            '2026-05-25',  # Memorial Day (US)
            '2026-07-03',  # Independence Day Observed (US)
            '2026-07-04',  # Independence Day (US)
            '2026-09-07',  # Labor Day (US)
            '2026-11-26',  # Thanksgiving (US)
            '2026-11-27',  # Black Friday (Early Close)
            '2026-12-24',  # Christmas Eve (Early Close)
            '2026-12-25',  # Christmas Day
            '2026-12-31',  # New Year's Eve (Early Close)
        ]
        
        # Triple Witching Dates 2026 (3rd Friday of Mar, Jun, Sep, Dec)
        self.triple_witching_2026 = [
            '2026-03-20',
            '2026-06-19',
            '2026-09-18',
            '2026-12-18',
        ]
        
        # Cache for correlation data
        self._correlation_cache = {}
        self._correlation_cache_time = 0

    def detect_all_killers(self, df_m5: pd.DataFrame, df_dxy: pd.DataFrame = None) -> Dict:
        """
        Run all 5 killer detectors and return comprehensive report.
        
        Returns:
            {
                'active_killers': List[str],
                'severity': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                'recommended_actions': List[str],
                'details': Dict
            }
        """
        active_killers = []
        details = {}
        recommended_actions = []
        
        # =========================================================================
        # Killer #1: Liquidity Void / Flash Crash
        # =========================================================================
        liquidity_result = self._detect_liquidity_void(df_m5)
        details['liquidity_void'] = liquidity_result
        if liquidity_result['detected']:
            active_killers.append('LIQUIDITY_VOID')
            recommended_actions.append('Cancel pending orders, tighten SL to 50% Risk')
        
        # =========================================================================
        # Killer #2: Holiday / Thin Market
        # =========================================================================
        thin_market_result = self._detect_thin_market(df_m5)
        details['thin_market'] = thin_market_result
        if thin_market_result['detected']:
            active_killers.append('THIN_MARKET')
            recommended_actions.append('Block new entries, reduce size 75%')
        
        # =========================================================================
        # Killer #3: Correlation Breakdown
        # =========================================================================
        if df_dxy is not None:
            correlation_result = self._detect_correlation_breakdown(df_m5, df_dxy)
            details['correlation_breakdown'] = correlation_result
            if correlation_result['detected']:
                active_killers.append('CORRELATION_BREAKDOWN')
                recommended_actions.append('Disable DXY filter, reduce size 50%')
        
        # =========================================================================
        # Killer #4: Central Bank Intervention
        # =========================================================================
        intervention_result = self._detect_intervention(df_m5)
        details['intervention'] = intervention_result
        if intervention_result['detected']:
            active_killers.append('CENTRAL_BANK_INTERVENTION')
            recommended_actions.append('Freeze trading 2 hours, close positions')
        
        # =========================================================================
        # Killer #5: Contract Roll-over / Triple Witching
        # =========================================================================
        rollover_result = self._detect_rollover_period()
        details['rollover'] = rollover_result
        if rollover_result['detected']:
            active_killers.append('ROLLOVER_PERIOD')
            recommended_actions.append('Block trend strategies, reduce size 50%')
        
        # Determine overall severity
        if len(active_killers) == 0:
            severity = 'LOW'
        elif len(active_killers) == 1:
            severity = 'MEDIUM'
        elif len(active_killers) == 2:
            severity = 'HIGH'
        else:
            severity = 'CRITICAL'
        
        return {
            'active_killers': active_killers,
            'severity': severity,
            'recommended_actions': recommended_actions,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

    def _detect_liquidity_void(self, df: pd.DataFrame) -> Dict:
        """Detect Liquidity Void / Flash Crash conditions."""
        if df is None or len(df) < 100:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        try:
            # Calculate spread (if available) or use high-low as proxy
            if 'spread' in df.columns:
                current_spread = df['spread'].iloc[-1]
                spread_ma = df['spread'].rolling(window=100, min_periods=50).mean().iloc[-1]
                spread_ratio = current_spread / (spread_ma + 1e-10)
            else:
                # Use high-low range as spread proxy
                hl_range = (df['high'] - df['low']).values
                current_range = hl_range[-1]
                range_ma = np.mean(hl_range[-100:])
                spread_ratio = current_range / (range_ma + 1e-10)
            
            # Volume dry-up detection
            if 'tick_volume' in df.columns:
                volume = df['tick_volume'].values
                current_volume = volume[-1]
                volume_percentile = (volume[-100:] < current_volume).sum() / 100.0
            else:
                volume_percentile = 0.5  # Neutral if no volume data
            
            # Price gap detection
            close = df['close'].values
            if len(close) >= 15:
                returns = np.diff(close[-15:]) / close[-15:-1]
                current_return = abs(returns[-1])
                return_std = np.std(returns)
                price_spike = current_return > (3.0 * return_std)
            else:
                price_spike = False
            
            # Detection logic
            spread_anomaly = spread_ratio > 5.0
            volume_dryup = volume_percentile < 0.05  # Bottom 5%
            
            detected = (spread_anomaly and volume_dryup) or price_spike
            
            return {
                'detected': detected,
                'spread_ratio': spread_ratio,
                'volume_percentile': volume_percentile,
                'price_spike': price_spike,
                'reason': f"Spread ratio: {spread_ratio:.2f}, Vol percentile: {volume_percentile:.2%}, Price spike: {price_spike}"
            }
            
        except Exception as e:
            self.logger.error(f"[LIQUIDITY] Error: {e}")
            return {'detected': False, 'reason': f'Error: {e}'}

    def _detect_thin_market(self, df: pd.DataFrame) -> Dict:
        """Detect Holiday / Thin Market conditions."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check holiday calendar
        is_holiday = today in self.holidays_2026
        
        # Check pre/post holiday (1 day before/after)
        try:
            today_dt = datetime.strptime(today, '%Y-%m-%d')
            pre_holiday = (today_dt + timedelta(days=1)).strftime('%Y-%m-%d') in self.holidays_2026
            post_holiday = (today_dt - timedelta(days=1)).strftime('%Y-%m-%d') in self.holidays_2026
        except Exception:
            pre_holiday = False
            post_holiday = False
        
        # Volume threshold check
        if df is not None and len(df) >= 100 and 'tick_volume' in df.columns:
            volume = df['tick_volume'].values
            current_volume = volume[-1]
            volume_10d = np.mean(volume[-100:])
            volume_ratio = current_volume / (volume_10d + 1e-10)
            volume_anomaly = volume_ratio < 0.30  # Less than 30% of normal
        else:
            volume_ratio = 1.0
            volume_anomaly = False
        
        detected = is_holiday or pre_holiday or post_holiday or volume_anomaly
        
        reason_parts = []
        if is_holiday:
            reason_parts.append("Today is holiday")
        if pre_holiday:
            reason_parts.append("Pre-holiday")
        if post_holiday:
            reason_parts.append("Post-holiday")
        if volume_anomaly:
            reason_parts.append(f"Volume {volume_ratio:.0%} of normal")
        
        return {
            'detected': detected,
            'is_holiday': is_holiday,
            'pre_holiday': pre_holiday,
            'post_holiday': post_holiday,
            'volume_ratio': volume_ratio,
            'reason': ' | '.join(reason_parts) if reason_parts else 'Normal market'
        }

    def _detect_correlation_breakdown(self, df_gold: pd.DataFrame, df_dxy: pd.DataFrame) -> Dict:
        """Detect Correlation Breakdown between Gold and DXY."""
        if df_gold is None or df_dxy is None:
            return {'detected': False, 'reason': 'Missing DXY data'}
        
        if len(df_gold) < 50 or len(df_dxy) < 50:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        try:
            # Align timestamps
            gold_close = df_gold['close'].values
            dxy_close = df_dxy['close'].values
            
            # Use shorter series
            min_len = min(len(gold_close), len(dxy_close))
            gold_close = gold_close[-min_len:]
            dxy_close = dxy_close[-min_len:]
            
            # Calculate returns
            gold_returns = np.diff(gold_close) / gold_close[:-1]
            dxy_returns = np.diff(dxy_close) / dxy_close[:-1]
            
            # Rolling correlation (last 20 bars)
            if len(gold_returns) >= 20:
                recent_gold = gold_returns[-20:]
                recent_dxy = dxy_returns[-20:]
                
                rolling_corr = np.corrcoef(recent_gold, recent_dxy)[0, 1]
                
                # Historical baseline (Gold vs DXY should be ~-0.75)
                historical_corr = -0.75
                corr_deviation = abs(rolling_corr - historical_corr)
                
                # Breakdown if deviation > 0.40
                detected = corr_deviation > 0.40
                
                return {
                    'detected': detected,
                    'rolling_correlation': rolling_corr,
                    'historical_baseline': historical_corr,
                    'deviation': corr_deviation,
                    'reason': f"Rolling corr: {rolling_corr:.2f}, Expected: {historical_corr:.2f}, Deviation: {corr_deviation:.2f}"
                }
            else:
                return {'detected': False, 'reason': 'Insufficient data for correlation'}
                
        except Exception as e:
            self.logger.error(f"[CORRELATION] Error: {e}")
            return {'detected': False, 'reason': f'Error: {e}'}

    def _detect_intervention(self, df: pd.DataFrame) -> Dict:
        """Detect possible Central Bank Intervention."""
        if df is None or len(df) < 100:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        try:
            close = df['close'].values
            
            # =========================================================================
            # FIXED: Correct array slicing to prevent broadcasting error
            # =========================================================================
            # Calculate returns (100 bars)
            if len(close) >= 100:
                # np.diff of 100 elements = 99 differences
                # close[-100:-1] = 99 elements (previous closes)
                returns = np.diff(close[-100:]) / close[-100:-1]
                
                # Current return (last bar)
                current_return = abs(returns[-1])
                
                # Historical volatility (all returns except last)
                return_std = np.std(returns[:-1])
                
                # Spike if > 5 standard deviations
                spike_threshold = 5.0
                price_spike = current_return > (spike_threshold * return_std)
                
                # Volume surge check
                if 'tick_volume' in df.columns:
                    volume = df['tick_volume'].values
                    current_volume = volume[-1]
                    volume_percentile = (volume[-100:] < current_volume).sum() / 100.0
                    volume_surge = volume_percentile > 0.99  # Top 1%
                else:
                    volume_surge = False
                
                detected = price_spike and volume_surge
                
                return {
                    'detected': detected,
                    'current_return': current_return,
                    'return_std': return_std,
                    'spike_multiple': current_return / (return_std + 1e-10),
                    'volume_surge': volume_surge,
                    'reason': f"Return: {current_return:.4f} ({current_return/(return_std+1e-10):.1f}x std), Volume surge: {volume_surge}"
                }
            else:
                return {'detected': False, 'reason': 'Insufficient data for intervention detection'}
                
        except Exception as e:
            self.logger.error(f"[INTERVENTION] Error: {e}")
            return {'detected': False, 'reason': f'Error: {e}'}

    def _detect_rollover_period(self) -> Dict:
        """Detect Contract Roll-over / Triple Witching period."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if today is triple witching
        is_triple_witching = today in self.triple_witching_2026
        
        # Check if within 5 days before triple witching (roll-over period)
        within_rollover = False
        days_to_witching = None
        
        try:
            today_dt = datetime.strptime(today, '%Y-%m-%d')
            for witch_date in self.triple_witching_2026:
                witch_dt = datetime.strptime(witch_date, '%Y-%m-%d')
                days_diff = (witch_dt - today_dt).days
                
                if 0 < days_diff <= 5:
                    within_rollover = True
                    days_to_witching = days_diff
                    break
        except Exception:
            pass
        
        detected = is_triple_witching or within_rollover
        
        reason_parts = []
        if is_triple_witching:
            reason_parts.append("Triple Witching Day")
        if within_rollover:
            reason_parts.append(f"Roll-over period ({days_to_witching} days to expiration)")
        
        return {
            'detected': detected,
            'is_triple_witching': is_triple_witching,
            'within_rollover': within_rollover,
            'days_to_witching': days_to_witching,
            'reason': ' | '.join(reason_parts) if reason_parts else 'Normal period'
        }

    def get_position_size_multiplier(self, killers_report: Dict) -> float:
        """
        Get position size multiplier based on active killers.
        
        Returns:
            multiplier: 0.0 to 1.0
        """
        severity = killers_report.get('severity', 'LOW')
        
        multipliers = {
            'LOW': 1.0,        # Full size
            'MEDIUM': 0.75,    # 75% size
            'HIGH': 0.50,      # 50% size
            'CRITICAL': 0.0    # Stop trading
        }
        
        return multipliers.get(severity, 1.0)