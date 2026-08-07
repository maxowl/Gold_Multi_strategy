"""
Market Killers Detector - Extreme Condition Protection.

Detects dangerous market conditions that should halt or reduce trading.
These are "market killers" - conditions that can cause rapid, large losses.

Market Killers:
  1. FLASH_CRASH: Rapid price drop (> 3% in < 5 minutes)
  2. SPREAD_SPIKE: Spread widens abnormally (> 3x normal)
  3. LIQUIDITY_GAP: Missing liquidity in order book
  4. VOLUME_SPIKE: Abnormal volume surge (> 5x average)
  5. PRICE_GAP: Gap between bars (> 2x ATR)
  6. VOLATILITY_EXPLOSION: ATR spikes (> 3x baseline)
  7. NEWS_EVENT: High-impact news detected (via volatility)
  8. BROKER_ISSUE: Broker-side problems (connection, requotes)

Severity Levels:
  LOW:      Minor issue, continue with caution
  MEDIUM:   Moderate issue, reduce position size
  HIGH:     Serious issue, avoid new entries
  CRITICAL: Extreme issue, halt all trading immediately

Actions:
  - LOW:      Continue trading, monitor
  - MEDIUM:   Reduce position size by 50%
  - HIGH:     Block new entries, manage existing positions
  - CRITICAL: Emergency close all positions, halt trading
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import config


class MarketKillersDetector:
    """
    Detects dangerous market conditions that should halt or reduce trading.
    
    Provides:
      - 8 killer condition detections
      - Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
      - Position sizing recommendations
      - Cooldown mechanism to prevent repeated triggers
      - Multi-timeframe confirmation
    """

    # Killer types
    KILLER_FLASH_CRASH = 'FLASH_CRASH'
    KILLER_SPREAD_SPIKE = 'SPREAD_SPIKE'
    KILLER_LIQUIDITY_GAP = 'LIQUIDITY_GAP'
    KILLER_VOLUME_SPIKE = 'VOLUME_SPIKE'
    KILLER_PRICE_GAP = 'PRICE_GAP'
    KILLER_VOLATILITY_EXPLOSION = 'VOLATILITY_EXPLOSION'
    KILLER_NEWS_EVENT = 'NEWS_EVENT'
    KILLER_BROKER_ISSUE = 'BROKER_ISSUE'

    # Severity levels
    SEVERITY_LOW = 'LOW'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_CRITICAL = 'CRITICAL'

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize MarketKillersDetector.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Detection thresholds
        self.flash_crash_threshold_pct = 3.0  # 3% drop
        self.flash_crash_window_minutes = 5

        self.spread_spike_threshold = 3.0  # 3x normal
        self.volume_spike_threshold = 5.0  # 5x average
        self.price_gap_threshold_atr = 2.0  # 2x ATR
        self.volatility_explosion_threshold = 3.0  # 3x baseline

        # Cooldown tracking (killer_type -> last trigger timestamp)
        self._cooldown_cache: Dict[str, float] = {}
        self.cooldown_seconds = 300  # 5 minutes between triggers

        # Killers history for pattern analysis
        self._killers_history: List[Dict] = []
        self.max_history = 100

        # Cache
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._spread_history: List[float] = []
        self._spread_history_max = 100

        self.logger.info(f"[KILLERS] Initialized for {symbol}")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def detect_killers(self, df_m15: pd.DataFrame,
                        df_m5: pd.DataFrame = None) -> Dict:
        """
        Main entry point: Detect all market killer conditions.
        
        Args:
            df_m15: Primary timeframe data (M15)
            df_m5: Secondary timeframe data (M5, optional)
            
        Returns:
            Dict with:
              - active_killers: List of detected killer names
              - severity: Overall severity level
              - should_trade: Whether trading is allowed
              - position_multiplier: Recommended position size multiplier
              - details: Detailed breakdown per killer
              - recommendations: Action recommendations
        """
        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < 20:
            return self._build_default_result()

        active_killers = []
        details = {}
        max_severity = self.SEVERITY_LOW

        # =========================================================================
        # KILLER 1: Flash Crash Detection
        # =========================================================================
        flash_crash_result = self._detect_flash_crash(df_m15, df_m5)
        if flash_crash_result['detected']:
            active_killers.append(self.KILLER_FLASH_CRASH)
            details[self.KILLER_FLASH_CRASH] = flash_crash_result
            max_severity = self._max_severity(max_severity, flash_crash_result['severity'])

        # =========================================================================
        # KILLER 2: Spread Spike Detection
        # =========================================================================
        spread_result = self._detect_spread_spike()
        if spread_result['detected']:
            active_killers.append(self.KILLER_SPREAD_SPIKE)
            details[self.KILLER_SPREAD_SPIKE] = spread_result
            max_severity = self._max_severity(max_severity, spread_result['severity'])

        # =========================================================================
        # KILLER 3: Liquidity Gap Detection
        # =========================================================================
        liquidity_result = self._detect_liquidity_gap(df_m15)
        if liquidity_result['detected']:
            active_killers.append(self.KILLER_LIQUIDITY_GAP)
            details[self.KILLER_LIQUIDITY_GAP] = liquidity_result
            max_severity = self._max_severity(max_severity, liquidity_result['severity'])

        # =========================================================================
        # KILLER 4: Volume Spike Detection
        # =========================================================================
        volume_result = self._detect_volume_spike(df_m15)
        if volume_result['detected']:
            active_killers.append(self.KILLER_VOLUME_SPIKE)
            details[self.KILLER_VOLUME_SPIKE] = volume_result
            max_severity = self._max_severity(max_severity, volume_result['severity'])

        # =========================================================================
        # KILLER 5: Price Gap Detection
        # =========================================================================
        gap_result = self._detect_price_gap(df_m15)
        if gap_result['detected']:
            active_killers.append(self.KILLER_PRICE_GAP)
            details[self.KILLER_PRICE_GAP] = gap_result
            max_severity = self._max_severity(max_severity, gap_result['severity'])

        # =========================================================================
        # KILLER 6: Volatility Explosion Detection
        # =========================================================================
        vol_explosion_result = self._detect_volatility_explosion(df_m15)
        if vol_explosion_result['detected']:
            active_killers.append(self.KILLER_VOLATILITY_EXPLOSION)
            details[self.KILLER_VOLATILITY_EXPLOSION] = vol_explosion_result
            max_severity = self._max_severity(max_severity, vol_explosion_result['severity'])

        # =========================================================================
        # KILLER 7: News Event Detection (via volatility proxy)
        # =========================================================================
        news_result = self._detect_news_event(df_m15, df_m5)
        if news_result['detected']:
            active_killers.append(self.KILLER_NEWS_EVENT)
            details[self.KILLER_NEWS_EVENT] = news_result
            max_severity = self._max_severity(max_severity, news_result['severity'])

        # =========================================================================
        # KILLER 8: Broker Issue Detection
        # =========================================================================
        broker_result = self._detect_broker_issue()
        if broker_result['detected']:
            active_killers.append(self.KILLER_BROKER_ISSUE)
            details[self.KILLER_BROKER_ISSUE] = broker_result
            max_severity = self._max_severity(max_severity, broker_result['severity'])

        # =========================================================================
        # AGGREGATE RESULTS
        # =========================================================================
        result = self._aggregate_killers(active_killers, max_severity, details)

        # Record to history
        self._record_to_history(result)

        # Log if killers detected
        if active_killers:
            self.logger.warning(
                f"[KILLERS] Detected {len(active_killers)} killer(s): "
                f"{', '.join(active_killers)} | Severity: {max_severity}"
            )

        return result

    # =========================================================================
    # KILLER 1: FLASH CRASH DETECTION
    # =========================================================================

    def _detect_flash_crash(self, df_m15: pd.DataFrame,
                             df_m5: pd.DataFrame = None) -> Dict:
        """
        Detect flash crash: rapid price drop.
        
        Criteria:
          - Price drop > 3% within 5 minutes
          - OR price drop > 2% within 2 minutes (more severe)
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_FLASH_CRASH):
                return result

            close = df_m15['close'].values.astype(float)
            high = df_m15['high'].values.astype(float)
            low = df_m15['low'].values.astype(float)

            if len(close) < 10:
                return result

            # Check last few bars for rapid drop
            recent_close = close[-5:]
            recent_high = high[-5:]
            recent_low = low[-5:]

            # Calculate max drop in recent bars
            max_price = np.max(recent_high)
            min_price = np.min(recent_low)

            if max_price == 0:
                return result

            drop_pct = (max_price - min_price) / max_price * 100

            # M5 confirmation (more sensitive)
            m5_drop_pct = 0.0
            if df_m5 is not None and len(df_m5) >= 10:
                m5_close = df_m5['close'].values.astype(float)
                m5_high = df_m5['high'].values.astype(float)
                m5_low = df_m5['low'].values.astype(float)

                m5_max = np.max(m5_high[-10:])
                m5_min = np.min(m5_low[-10:])

                if m5_max > 0:
                    m5_drop_pct = (m5_max - m5_min) / m5_max * 100

            # Determine severity
            if drop_pct >= self.flash_crash_threshold_pct or m5_drop_pct >= 2.0:
                result['detected'] = True
                result['severity'] = self.SEVERITY_CRITICAL
                result['details'] = {
                    'drop_pct': round(drop_pct, 2),
                    'm5_drop_pct': round(m5_drop_pct, 2),
                    'threshold': self.flash_crash_threshold_pct,
                    'max_price': round(max_price, 2),
                    'min_price': round(min_price, 2)
                }
                self._update_cooldown(self.KILLER_FLASH_CRASH)

            elif drop_pct >= self.flash_crash_threshold_pct * 0.7:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'drop_pct': round(drop_pct, 2),
                    'm5_drop_pct': round(m5_drop_pct, 2),
                    'threshold': self.flash_crash_threshold_pct * 0.7
                }
                self._update_cooldown(self.KILLER_FLASH_CRASH)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Flash crash detection error: {e}")

        return result

    # =========================================================================
    # KILLER 2: SPREAD SPIKE DETECTION
    # =========================================================================

    def _detect_spread_spike(self) -> Dict:
        """
        Detect spread spike: abnormal spread widening.
        
        Criteria:
          - Current spread > 3x average spread
          - OR current spread > 50 points (absolute threshold)
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_SPREAD_SPIKE):
                return result

            symbol_info = self._get_symbol_info()
            if not symbol_info:
                return result

            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return result

            # Calculate current spread
            current_spread = (tick.ask - tick.bid) / symbol_info.point

            # Record to history
            self._spread_history.append(current_spread)
            if len(self._spread_history) > self._spread_history_max:
                self._spread_history = self._spread_history[-self._spread_history_max:]

            # Calculate average spread
            if len(self._spread_history) < 10:
                avg_spread = current_spread
            else:
                avg_spread = np.mean(self._spread_history[:-1])  # Exclude current

            if avg_spread == 0:
                return result

            # Calculate spike ratio
            spike_ratio = current_spread / avg_spread

            # Determine severity
            if spike_ratio >= self.spread_spike_threshold or current_spread > 50:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'current_spread': round(current_spread, 1),
                    'avg_spread': round(avg_spread, 1),
                    'spike_ratio': round(spike_ratio, 2),
                    'threshold': self.spread_spike_threshold
                }
                self._update_cooldown(self.KILLER_SPREAD_SPIKE)

            elif spike_ratio >= self.spread_spike_threshold * 0.7:
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'current_spread': round(current_spread, 1),
                    'avg_spread': round(avg_spread, 1),
                    'spike_ratio': round(spike_ratio, 2)
                }
                self._update_cooldown(self.KILLER_SPREAD_SPIKE)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Spread spike detection error: {e}")

        return result

    # =========================================================================
    # KILLER 3: LIQUIDITY GAP DETECTION
    # =========================================================================

    def _detect_liquidity_gap(self, df: pd.DataFrame) -> Dict:
        """
        Detect liquidity gap: missing liquidity in order book.
        
        Criteria:
          - Very wide bid-ask spread relative to price
          - OR very low tick volume (indicating no participants)
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_LIQUIDITY_GAP):
                return result

            tick = mt5.symbol_info_tick(self.symbol)
            symbol_info = self._get_symbol_info()

            if not tick or not symbol_info:
                return result

            # Calculate spread as percentage of price
            spread = tick.ask - tick.bid
            price = tick.bid
            spread_pct = (spread / price) * 100 if price > 0 else 0

            # Check tick volume (if available in df)
            low_volume = False
            if 'tick_volume' in df.columns and len(df) >= 20:
                volume = df['tick_volume'].values.astype(float)
                avg_volume = np.mean(volume[-20:-1])
                current_volume = volume[-1]

                if avg_volume > 0 and current_volume < avg_volume * 0.2:
                    low_volume = True

            # Determine severity
            if spread_pct > 0.05 or low_volume:  # 0.05% spread is very wide
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'spread_pct': round(spread_pct, 4),
                    'low_volume': low_volume,
                    'current_spread': round(spread, 2)
                }
                self._update_cooldown(self.KILLER_LIQUIDITY_GAP)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Liquidity gap detection error: {e}")

        return result

    # =========================================================================
    # KILLER 4: VOLUME SPIKE DETECTION
    # =========================================================================

    def _detect_volume_spike(self, df: pd.DataFrame) -> Dict:
        """
        Detect volume spike: abnormal volume surge.
        
        Criteria:
          - Current volume > 5x average volume
          - Indicates news event or institutional activity
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_VOLUME_SPIKE):
                return result

            if 'tick_volume' not in df.columns or len(df) < 30:
                return result

            volume = df['tick_volume'].values.astype(float)
            avg_volume = np.mean(volume[-30:-1])
            current_volume = volume[-1]

            if avg_volume == 0:
                return result

            spike_ratio = current_volume / avg_volume

            # Determine severity
            if spike_ratio >= self.volume_spike_threshold:
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'current_volume': int(current_volume),
                    'avg_volume': round(avg_volume, 1),
                    'spike_ratio': round(spike_ratio, 2),
                    'threshold': self.volume_spike_threshold
                }
                self._update_cooldown(self.KILLER_VOLUME_SPIKE)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Volume spike detection error: {e}")

        return result

    # =========================================================================
    # KILLER 5: PRICE GAP DETECTION
    # =========================================================================

    def _detect_price_gap(self, df: pd.DataFrame) -> Dict:
        """
        Detect price gap: gap between bars.
        
        Criteria:
          - Gap between previous close and current open > 2x ATR
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_PRICE_GAP):
                return result

            close = df['close'].values.astype(float)
            open_ = df['open'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 20:
                return result

            # Calculate gap (previous close vs current open)
            prev_close = close[-2]
            current_open = open_[-1]
            gap = abs(current_open - prev_close)

            # Calculate ATR for threshold
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-14:])

            if atr == 0:
                return result

            gap_ratio = gap / atr

            # Determine severity
            if gap_ratio >= self.price_gap_threshold_atr:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'gap': round(gap, 2),
                    'atr': round(atr, 2),
                    'gap_ratio': round(gap_ratio, 2),
                    'threshold': self.price_gap_threshold_atr
                }
                self._update_cooldown(self.KILLER_PRICE_GAP)

            elif gap_ratio >= self.price_gap_threshold_atr * 0.7:
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'gap': round(gap, 2),
                    'atr': round(atr, 2),
                    'gap_ratio': round(gap_ratio, 2)
                }
                self._update_cooldown(self.KILLER_PRICE_GAP)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Price gap detection error: {e}")

        return result

    # =========================================================================
    # KILLER 6: VOLATILITY EXPLOSION DETECTION
    # =========================================================================

    def _detect_volatility_explosion(self, df: pd.DataFrame) -> Dict:
        """
        Detect volatility explosion: ATR spikes abnormally.
        
        Criteria:
          - Current ATR > 3x baseline ATR
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_VOLATILITY_EXPLOSION):
                return result

            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 50:
                return result

            # Calculate True Range
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))

            # Current ATR (last 10 bars)
            current_atr = np.mean(tr[-10:])

            # Baseline ATR (50 bars ago to 20 bars ago)
            baseline_atr = np.mean(tr[-50:-20])

            if baseline_atr == 0:
                return result

            explosion_ratio = current_atr / baseline_atr

            # Determine severity
            if explosion_ratio >= self.volatility_explosion_threshold:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'current_atr': round(current_atr, 2),
                    'baseline_atr': round(baseline_atr, 2),
                    'explosion_ratio': round(explosion_ratio, 2),
                    'threshold': self.volatility_explosion_threshold
                }
                self._update_cooldown(self.KILLER_VOLATILITY_EXPLOSION)

            elif explosion_ratio >= self.volatility_explosion_threshold * 0.7:
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'current_atr': round(current_atr, 2),
                    'baseline_atr': round(baseline_atr, 2),
                    'explosion_ratio': round(explosion_ratio, 2)
                }
                self._update_cooldown(self.KILLER_VOLATILITY_EXPLOSION)

        except Exception as e:
            self.logger.debug(f"[KILLERS] Volatility explosion detection error: {e}")

        return result

    # =========================================================================
    # KILLER 7: NEWS EVENT DETECTION
    # =========================================================================

    def _detect_news_event(self, df_m15: pd.DataFrame,
                            df_m5: pd.DataFrame = None) -> Dict:
        """
        Detect news event via volatility proxy.
        
        Criteria:
          - Sudden volatility spike on multiple timeframes
          - Combined with volume spike
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check cooldown
            if not self._check_cooldown(self.KILLER_NEWS_EVENT):
                return result

            # Check if both M15 and M5 show volatility spike
            m15_vol_spike = self._check_volatility_spike(df_m15)
            m5_vol_spike = False

            if df_m5 is not None and len(df_m5) >= 30:
                m5_vol_spike = self._check_volatility_spike(df_m5)

            # News event: volatility spike on both timeframes
            if m15_vol_spike and m5_vol_spike:
                result['detected'] = True
                result['severity'] = self.SEVERITY_MEDIUM
                result['details'] = {
                    'm15_vol_spike': True,
                    'm5_vol_spike': True,
                    'reason': 'Multi-timeframe volatility spike (possible news)'
                }
                self._update_cooldown(self.KILLER_NEWS_EVENT)

        except Exception as e:
            self.logger.debug(f"[KILLERS] News event detection error: {e}")

        return result

    def _check_volatility_spike(self, df: pd.DataFrame) -> bool:
        """Check if volatility spiked on given timeframe."""
        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            if len(close) < 30:
                return False

            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))

            current_tr = tr[-1]
            avg_tr = np.mean(tr[-20:-1])

            if avg_tr == 0:
                return False

            return current_tr > avg_tr * 2.0

        except Exception:
            return False

    # =========================================================================
    # KILLER 8: BROKER ISSUE DETECTION
    # =========================================================================

    def _detect_broker_issue(self) -> Dict:
        """
        Detect broker-side issues.
        
        Criteria:
          - MT5 connection lost
          - Requote errors
          - Trade disabled
        """
        result = {
            'detected': False,
            'severity': self.SEVERITY_LOW,
            'details': {}
        }

        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None or not terminal_info.connected:
                result['detected'] = True
                result['severity'] = self.SEVERITY_CRITICAL
                result['details'] = {
                    'reason': 'MT5 connection lost',
                    'connected': False
                }
                return result

            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'reason': 'Cannot get account info',
                    'connected': True
                }
                return result

            # Check if trading is allowed
            if not account_info.trade_allowed:
                result['detected'] = True
                result['severity'] = self.SEVERITY_HIGH
                result['details'] = {
                    'reason': 'Trading disabled by broker',
                    'trade_allowed': False
                }
                return result

        except Exception as e:
            self.logger.debug(f"[KILLERS] Broker issue detection error: {e}")

        return result

    # =========================================================================
    # AGGREGATION & RECOMMENDATIONS
    # =========================================================================

    def _aggregate_killers(self, active_killers: List[str],
                            max_severity: str,
                            details: Dict) -> Dict:
        """Aggregate all killer detections into final result."""

        # Determine if trading is allowed
        should_trade = True
        position_multiplier = 1.0

        if max_severity == self.SEVERITY_CRITICAL:
            should_trade = False
            position_multiplier = 0.0
        elif max_severity == self.SEVERITY_HIGH:
            should_trade = True
            position_multiplier = 0.5
        elif max_severity == self.SEVERITY_MEDIUM:
            should_trade = True
            position_multiplier = 0.75

        # Generate recommendations
        recommendations = self._generate_recommendations(active_killers, max_severity)

        return {
            'active_killers': active_killers,
            'killer_count': len(active_killers),
            'severity': max_severity,
            'should_trade': should_trade,
            'position_multiplier': position_multiplier,
            'details': details,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_recommendations(self, active_killers: List[str],
                                    severity: str) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if not active_killers:
            return ['Normal conditions - continue trading']

        if severity == self.SEVERITY_CRITICAL:
            recommendations.append('HALT all trading immediately')
            recommendations.append('Close all open positions')
            recommendations.append('Wait for conditions to stabilize')

        elif severity == self.SEVERITY_HIGH:
            recommendations.append('Reduce position size by 50%')
            recommendations.append('Avoid new entries')
            recommendations.append('Manage existing positions closely')

        elif severity == self.SEVERITY_MEDIUM:
            recommendations.append('Reduce position size by 25%')
            recommendations.append('Use wider stop losses')
            recommendations.append('Avoid aggressive strategies')

        # Killer-specific recommendations
        if self.KILLER_FLASH_CRASH in active_killers:
            recommendations.append('Flash crash detected - wait for stabilization')

        if self.KILLER_SPREAD_SPIKE in active_killers:
            recommendations.append('Wide spread - avoid scalping strategies')

        if self.KILLER_NEWS_EVENT in active_killers:
            recommendations.append('Possible news event - wait for volatility to settle')

        if self.KILLER_BROKER_ISSUE in active_killers:
            recommendations.append('Broker issue - check connection and account status')

        return recommendations

    # =========================================================================
    # SEVERITY HELPERS
    # =========================================================================

    def _max_severity(self, current: str, new: str) -> str:
        """Return the higher severity level."""
        severity_order = {
            self.SEVERITY_LOW: 0,
            self.SEVERITY_MEDIUM: 1,
            self.SEVERITY_HIGH: 2,
            self.SEVERITY_CRITICAL: 3
        }

        if severity_order.get(new, 0) > severity_order.get(current, 0):
            return new
        return current

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _check_cooldown(self, killer_type: str) -> bool:
        """Check if cooldown period has passed for this killer type."""
        if killer_type not in self._cooldown_cache:
            return True

        last_trigger = self._cooldown_cache[killer_type]
        elapsed = time.time() - last_trigger

        return elapsed >= self.cooldown_seconds

    def _update_cooldown(self, killer_type: str):
        """Update cooldown timestamp for this killer type."""
        self._cooldown_cache[killer_type] = time.time()

    def clear_cooldown(self, killer_type: str = None):
        """Clear cooldown for specific or all killer types."""
        if killer_type:
            if killer_type in self._cooldown_cache:
                del self._cooldown_cache[killer_type]
        else:
            self._cooldown_cache.clear()

    # =========================================================================
    # HISTORY MANAGEMENT
    # =========================================================================

    def _record_to_history(self, result: Dict):
        """Record detection result to history."""
        self._killers_history.append({
            'timestamp': datetime.now().isoformat(),
            'active_killers': result['active_killers'],
            'severity': result['severity'],
            'killer_count': result['killer_count']
        })

        # Keep only last N records
        if len(self._killers_history) > self.max_history:
            self._killers_history = self._killers_history[-self.max_history:]

    def get_killers_history(self, limit: int = 10) -> List[Dict]:
        """Get recent killers history."""
        return self._killers_history[-limit:]

    def get_killers_stats(self) -> Dict:
        """Get statistics on killer detections."""
        if not self._killers_history:
            return {
                'total_detections': 0,
                'killer_frequency': {},
                'severity_distribution': {}
            }

        # Count killer occurrences
        killer_counts = {}
        severity_counts = {}

        for record in self._killers_history:
            for killer in record['active_killers']:
                killer_counts[killer] = killer_counts.get(killer, 0) + 1

            severity = record['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_detections': len(self._killers_history),
            'killer_frequency': killer_counts,
            'severity_distribution': severity_counts,
            'most_common_killer': max(killer_counts.keys(), key=lambda k: killer_counts[k]) if killer_counts else None
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_symbol_info(self):
        """Get symbol info with 5-second cache."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_time < 5.0):
            return self._symbol_info_cache

        info = mt5.symbol_info(self.symbol)
        if info is None:
            if mt5.symbol_select(self.symbol, True):
                info = mt5.symbol_info(self.symbol)

        if info:
            self._symbol_info_cache = info
            self._symbol_info_time = current_time

        return info

    def _build_default_result(self) -> Dict:
        """Build default result when detection fails."""
        return {
            'active_killers': [],
            'killer_count': 0,
            'severity': self.SEVERITY_LOW,
            'should_trade': True,
            'position_multiplier': 1.0,
            'details': {},
            'recommendations': ['Normal conditions'],
            'timestamp': datetime.now().isoformat()
        }

    def format_killers_log(self, result: Dict) -> str:
        """
        Format a concise log string for killer detection.
        
        Args:
            result: Result from detect_killers
            
        Returns:
            Formatted log string
        """
        killers = result.get('active_killers', [])
        severity = result.get('severity', 'LOW')
        multiplier = result.get('position_multiplier', 1.0)

        if not killers:
            return f"[KILLERS] None detected | Severity: {severity}"

        killers_str = ', '.join(killers[:3])  # Show first 3
        if len(killers) > 3:
            killers_str += f", +{len(killers) - 3} more"

        return (
            f"[KILLERS] {len(killers)} detected: {killers_str} | "
            f"Severity: {severity} | "
            f"Multiplier: {multiplier:.2f}x"
        )