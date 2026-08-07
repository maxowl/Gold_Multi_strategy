"""
Emergency Defense Engine - Worst-Case Scenario Protection.

Multi-layer emergency defense system that protects the trading account
from catastrophic market events and system failures.

Emergency Triggers:
  1. FLASH_CRASH: Rapid price drop (2%+ in 5 bars)
  2. SPREAD_CRISIS: Spread widens 5x+ normal
  3. POSITION_ABNORMALITY: Position loss > 5% of equity
  4. DAILY_LOSS_EMERGENCY: Daily loss > 3%
  5. BROKER_ISSUE: Broker connection problems
  6. NEWS_EVENT: High-impact news within 30 minutes
  7. WEEKEND_RISK: Weekend approaching with open positions

Emergency Actions:
  - EMERGENCY_CLOSE_ALL: Close all positions immediately
  - KILL_SWITCH: Halt all trading operations
  - MONITOR: Continue but with reduced exposure
  - PAUSE_NEW_ENTRIES: Stop new entries but manage existing

Features:
  - Flash crash detection with severity levels
  - Spread crisis monitoring
  - Position abnormality detection
  - Kill switch activation
  - Weekend protection
  - News event awareness
  - Emergency history tracking
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import config


class EmergencyDefenseEngine:
    """
    Multi-layer emergency defense system.
    
    Features:
      - Flash crash detection
      - Spread crisis monitoring
      - Position abnormality detection
      - Emergency close all positions
      - Kill switch management
      - Weekend protection
      - News event awareness
      - Emergency history tracking
    """

    # Emergency action types
    ACTION_CLOSE_ALL = 'EMERGENCY_CLOSE_ALL'
    ACTION_KILL_SWITCH = 'KILL_SWITCH'
    ACTION_MONITOR = 'MONITOR'
    ACTION_PAUSE_NEW = 'PAUSE_NEW_ENTRIES'
    ACTION_NO_ACTION = 'NO_ACTION'

    # Severity levels
    SEVERITY_NONE = 'NONE'
    SEVERITY_LOW = 'LOW'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_EXTREME = 'EXTREME'

    # Emergency trigger types
    TRIGGER_FLASH_CRASH = 'FLASH_CRASH'
    TRIGGER_SPREAD_CRISIS = 'SPREAD_CRISIS'
    TRIGGER_POSITION_ABNORMALITY = 'POSITION_ABNORMALITY'
    TRIGGER_DAILY_LOSS = 'DAILY_LOSS_EMERGENCY'
    TRIGGER_BROKER_ISSUE = 'BROKER_ISSUE'
    TRIGGER_NEWS_EVENT = 'NEWS_EVENT'
    TRIGGER_WEEKEND_RISK = 'WEEKEND_RISK'
    TRIGGER_MANUAL = 'MANUAL'

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize EmergencyDefenseEngine.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Thresholds from config
        self.flash_crash_threshold_pct = config.emergency_flash_crash_threshold_pct  # 0.02 (2%)
        self.flash_crash_window_bars = config.emergency_flash_crash_window_bars  # 5
        self.spread_crisis_multiplier = config.emergency_spread_crisis_multiplier  # 5.0
        self.max_daily_loss_pct = config.emergency_max_daily_loss_pct  # 3.0
        self.position_abnormal_loss_pct = config.emergency_position_abnormal_loss_pct  # 5.0
        self.news_buffer_minutes = config.emergency_news_buffer_minutes  # 30

        # State tracking
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_activated_at = None

        # Emergency history (last 20 events)
        self._emergency_history = deque(maxlen=20)

        # Trigger statistics
        self._trigger_stats: Dict[str, int] = {}

        # Spread tracking
        self._spread_history = deque(maxlen=100)
        self._normal_spread_points = 25.0  # Baseline for XAUUSD

        # Cache
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._tick_cache = None
        self._tick_time = 0

        # High impact news hours (UTC) - simplified calendar
        self.high_impact_news_hours_utc = [
            (13, 30),  # 8:30 AM NY (NFP, CPI, GDP)
            (15, 0),   # 10:00 AM NY (ISM)
            (19, 0),   # 2:00 PM NY (FOMC)
        ]

        self.logger.info(
            f"[EMERGENCY] Initialized for {symbol} | "
            f"Flash: {self.flash_crash_threshold_pct:.0%} in {self.flash_crash_window_bars} bars | "
            f"Spread: {self.spread_crisis_multiplier}x | "
            f"Daily Loss: {self.max_daily_loss_pct}%"
        )

    # =========================================================================
    # MAIN EMERGENCY CHECK
    # =========================================================================

    def run_emergency_check(self, df: pd.DataFrame, positions: List[Dict]) -> Dict:
        """
        Run all emergency checks and return consolidated result.
        
        Args:
            df: DataFrame with price data
            positions: List of active positions
            
        Returns:
            Dict with emergency status and recommended actions
        """
        results = {
            'emergency_detected': False,
            'actions': [],
            'flash_crash': None,
            'spread_crisis': None,
            'news_event': False,
            'abnormalities': [],
            'weekend_close': False,
            'kill_switch': self.kill_switch_active,
            'broker_healthy': True,
            'severity': self.SEVERITY_NONE
        }

        # Check if kill switch is active
        if self.kill_switch_active:
            results['actions'].append({
                'type': self.ACTION_KILL_SWITCH,
                'reason': self.kill_switch_reason,
                'recommendation': 'NO_TRADING'
            })
            return results

        # =========================================================================
        # CHECK 1: Flash Crash
        # =========================================================================
        if df is not None and len(df) >= self.flash_crash_window_bars + 1:
            flash_crash = self.detect_flash_crash(df)
            results['flash_crash'] = flash_crash

            if flash_crash['is_flash_crash']:
                severity = flash_crash['severity']
                results['severity'] = severity

                if severity in [self.SEVERITY_HIGH, self.SEVERITY_EXTREME]:
                    results['emergency_detected'] = True
                    results['actions'].append({
                        'type': self.TRIGGER_FLASH_CRASH,
                        'severity': severity,
                        'recommendation': self.ACTION_CLOSE_ALL,
                        'reason': f"Flash crash {severity}: {flash_crash['price_change_pct']:.2%} in {self.flash_crash_window_bars} bars"
                    })
                    self._record_emergency(self.TRIGGER_FLASH_CRASH, flash_crash)

                elif severity == self.SEVERITY_MEDIUM:
                    results['actions'].append({
                        'type': self.TRIGGER_FLASH_CRASH,
                        'severity': severity,
                        'recommendation': self.ACTION_PAUSE_NEW,
                        'reason': f"Medium flash crash: {flash_crash['price_change_pct']:.2%}"
                    })

        # =========================================================================
        # CHECK 2: Spread Crisis
        # =========================================================================
        spread_crisis = self.detect_spread_crisis()
        results['spread_crisis'] = spread_crisis

        if spread_crisis['is_crisis']:
            results['emergency_detected'] = True
            results['actions'].append({
                'type': self.TRIGGER_SPREAD_CRISIS,
                'severity': self.SEVERITY_HIGH,
                'recommendation': self.ACTION_PAUSE_NEW,
                'reason': f"Spread crisis: {spread_crisis['current_spread_points']:.0f} pts ({spread_crisis['multiplier']:.1f}x normal)"
            })
            self._record_emergency(self.TRIGGER_SPREAD_CRISIS, spread_crisis)

        # =========================================================================
        # CHECK 3: News Event
        # =========================================================================
        if self.is_near_high_impact_news():
            results['news_event'] = True
            results['actions'].append({
                'type': self.TRIGGER_NEWS_EVENT,
                'severity': self.SEVERITY_MEDIUM,
                'recommendation': self.ACTION_PAUSE_NEW,
                'reason': f"High-impact news within {self.news_buffer_minutes} minutes"
            })

        # =========================================================================
        # CHECK 4: Position Abnormalities
        # =========================================================================
        if positions:
            abnormalities = self.detect_position_abnormalities(positions)
            results['abnormalities'] = abnormalities

            critical_abnormalities = [a for a in abnormalities if a['severity'] == 'CRITICAL']
            if critical_abnormalities:
                results['emergency_detected'] = True
                results['actions'].append({
                    'type': self.TRIGGER_POSITION_ABNORMALITY,
                    'severity': self.SEVERITY_HIGH,
                    'recommendation': self.ACTION_CLOSE_ALL,
                    'reason': f"{len(critical_abnormalities)} position(s) with abnormal loss"
                })
                self._record_emergency(self.TRIGGER_POSITION_ABNORMALITY, {'count': len(critical_abnormalities)})

        # =========================================================================
        # CHECK 5: Weekend Protection
        # =========================================================================
        if positions and self.should_close_for_weekend():
            results['weekend_close'] = True
            results['actions'].append({
                'type': self.TRIGGER_WEEKEND_RISK,
                'severity': self.SEVERITY_MEDIUM,
                'recommendation': self.ACTION_CLOSE_ALL,
                'reason': "Weekend approaching - close all positions"
            })

        # =========================================================================
        # CHECK 6: Broker Health
        # =========================================================================
        broker_health = self._check_broker_health()
        results['broker_healthy'] = broker_health['healthy']

        if not broker_health['healthy']:
            results['emergency_detected'] = True
            results['actions'].append({
                'type': self.TRIGGER_BROKER_ISSUE,
                'severity': self.SEVERITY_EXTREME,
                'recommendation': self.ACTION_KILL_SWITCH,
                'reason': broker_health['reason']
            })
            self._record_emergency(self.TRIGGER_BROKER_ISSUE, broker_health)

        return results

    # =========================================================================
    # FLASH CRASH DETECTION
    # =========================================================================

    def detect_flash_crash(self, df: pd.DataFrame) -> Dict:
        """
        Detect flash crash conditions.
        
        Criteria:
          - Price move > threshold% in window bars
          - OR single bar move > 60% of threshold
        
        Returns:
            Dict with flash crash status and severity
        """
        result = {
            'is_flash_crash': False,
            'severity': self.SEVERITY_NONE,
            'price_change_pct': 0.0,
            'direction': 'NONE',
            'recommendation': self.ACTION_NO_ACTION
        }

        if df is None or len(df) < self.flash_crash_window_bars + 1:
            return result

        try:
            # Get last N bars
            window = df.tail(self.flash_crash_window_bars)

            # Calculate price change
            start_price = float(window['open'].iloc[0])
            end_price = float(window['close'].iloc[-1])

            if start_price == 0:
                return result

            price_change = end_price - start_price
            price_change_pct = abs(price_change / start_price)
            direction = 'UP' if price_change > 0 else 'DOWN'

            # Calculate max single-bar move
            bar_changes = window['close'].diff().abs()
            max_bar_change = float(bar_changes.max()) if not bar_changes.empty else 0
            max_bar_change_pct = max_bar_change / start_price if start_price > 0 else 0

            # Detect flash crash
            is_flash_crash = (
                price_change_pct >= self.flash_crash_threshold_pct or
                max_bar_change_pct >= self.flash_crash_threshold_pct * 0.6
            )

            if not is_flash_crash:
                return result

            # Determine severity
            if price_change_pct >= 0.05:  # 5%
                severity = self.SEVERITY_EXTREME
            elif price_change_pct >= 0.03:  # 3%
                severity = self.SEVERITY_HIGH
            elif price_change_pct >= self.flash_crash_threshold_pct:  # 2%
                severity = self.SEVERITY_MEDIUM
            else:
                severity = self.SEVERITY_LOW

            # Determine recommendation
            if severity == self.SEVERITY_EXTREME:
                recommendation = self.ACTION_CLOSE_ALL
            elif severity == self.SEVERITY_HIGH:
                recommendation = self.ACTION_PAUSE_NEW
            elif severity == self.SEVERITY_MEDIUM:
                recommendation = self.ACTION_MONITOR
            else:
                recommendation = self.ACTION_NO_ACTION

            result = {
                'is_flash_crash': True,
                'severity': severity,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'max_bar_change_pct': max_bar_change_pct,
                'recommendation': recommendation
            }

            # Log if flash crash detected
            self.logger.warning(
                f"[EMERGENCY] FLASH CRASH {severity} | "
                f"Direction: {direction} | "
                f"Change: {price_change_pct:.2%} in {self.flash_crash_window_bars} bars"
            )

        except Exception as e:
            self.logger.error(f"[EMERGENCY] Flash crash detection error: {e}")

        return result

    # =========================================================================
    # SPREAD CRISIS DETECTION
    # =========================================================================

    def detect_spread_crisis(self) -> Dict:
        """
        Detect abnormal spread widening.
        
        Returns:
            Dict with spread crisis status
        """
        result = {
            'is_crisis': False,
            'current_spread_points': 0.0,
            'normal_spread_points': self._normal_spread_points,
            'multiplier': 1.0,
            'recommendation': self.ACTION_NO_ACTION
        }

        try:
            symbol_info = self._get_symbol_info()
            tick = self._get_current_tick()

            if not symbol_info or not tick:
                return result

            current_spread = tick.ask - tick.bid
            current_spread_points = current_spread / symbol_info.point

            # Record to history
            self._spread_history.append(current_spread_points)

            # Update normal spread estimate (moving average)
            if len(self._spread_history) >= 20:
                self._normal_spread_points = np.mean(list(self._spread_history)[-50:])

            # Calculate multiplier
            multiplier = current_spread_points / self._normal_spread_points if self._normal_spread_points > 0 else 1.0

            result['current_spread_points'] = current_spread_points
            result['multiplier'] = multiplier

            # Determine if crisis
            is_crisis = multiplier >= self.spread_crisis_multiplier

            if is_crisis:
                result['is_crisis'] = True
                result['recommendation'] = self.ACTION_PAUSE_NEW

                self.logger.warning(
                    f"[EMERGENCY] SPREAD CRISIS | "
                    f"Current: {current_spread_points:.0f} pts | "
                    f"Normal: {self._normal_spread_points:.0f} pts | "
                    f"Multiplier: {multiplier:.1f}x"
                )

        except Exception as e:
            self.logger.error(f"[EMERGENCY] Spread crisis detection error: {e}")

        return result

    # =========================================================================
    # POSITION ABNORMALITY DETECTION
    # =========================================================================

    def detect_position_abnormalities(self, positions: List[Dict]) -> List[Dict]:
        """
        Detect abnormal position states.
        
        Checks:
          - Position loss > threshold
          - Position age > threshold
          - Position without SL
        
        Returns:
            List of abnormality dicts
        """
        abnormalities = []

        for pos in positions:
            try:
                ticket = pos.get('ticket')
                if ticket is None:
                    continue

                mt5_pos_list = mt5.positions_get(ticket=ticket)
                if not mt5_pos_list:
                    continue

                mt5_pos = mt5_pos_list[0]
                entry_price = mt5_pos.price_open
                current_price = mt5_pos.price_current
                sl = mt5_pos.sl

                # Check 1: No SL set
                if sl == 0:
                    abnormalities.append({
                        'ticket': ticket,
                        'type': 'NO_SL',
                        'severity': 'HIGH',
                        'description': 'Position has no Stop Loss'
                    })

                # Check 2: Abnormal loss
                is_buy = (mt5_pos.type == mt5.ORDER_TYPE_BUY)
                if is_buy:
                    loss_pct = (entry_price - current_price) / entry_price * 100
                else:
                    loss_pct = (current_price - entry_price) / entry_price * 100

                if loss_pct > self.position_abnormal_loss_pct:
                    abnormalities.append({
                        'ticket': ticket,
                        'type': 'ABNORMAL_LOSS',
                        'severity': 'CRITICAL',
                        'loss_pct': round(loss_pct, 2),
                        'description': f'Position loss {loss_pct:.2f}% exceeds threshold {self.position_abnormal_loss_pct}%'
                    })

                # Check 3: Position age > 24 hours
                open_time = datetime.fromtimestamp(mt5_pos.time)
                age_hours = (datetime.now() - open_time).total_seconds() / 3600

                if age_hours > 24:
                    abnormalities.append({
                        'ticket': ticket,
                        'type': 'OLD_POSITION',
                        'severity': 'MEDIUM',
                        'age_hours': round(age_hours, 1),
                        'description': f'Position open for {age_hours:.1f} hours'
                    })

            except Exception as e:
                self.logger.error(f"[EMERGENCY] Position abnormality check error: {e}")

        return abnormalities

    # =========================================================================
    # EMERGENCY CLOSE ALL POSITIONS
    # =========================================================================

    def emergency_close_all_positions(self, reason: str) -> Dict:
        """
        Emergency close ALL positions immediately.
        
        Args:
            reason: Reason for emergency close
            
        Returns:
            Dict with close results
        """
        self.logger.critical(f"[EMERGENCY] Closing ALL positions | Reason: {reason}")

        results = {
            'closed': 0,
            'failed': 0,
            'total_pnl': 0.0,
            'details': [],
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

        positions = mt5.positions_get(symbol=self.symbol) or []

        for pos in positions:
            try:
                tick = self._get_current_tick()
                if not tick:
                    results['failed'] += 1
                    results['details'].append({
                        'ticket': pos.ticket,
                        'status': 'FAILED',
                        'error': 'Cannot get tick'
                    })
                    continue

                is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
                close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
                price = tick.bid if is_buy else tick.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 100,  # Wide deviation for emergency
                    "magic": 999999,  # Emergency magic number
                    "comment": f"EMERGENCY: {reason[:20]}"
                }

                result = mt5.order_send(request)

                if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL):
                    results['closed'] += 1
                    pnl = pos.profit if hasattr(pos, 'profit') else 0
                    results['total_pnl'] += pnl
                    results['details'].append({
                        'ticket': pos.ticket,
                        'status': 'CLOSED',
                        'price': price,
                        'pnl': pnl
                    })
                    self.logger.warning(
                        f"[EMERGENCY] Closed ticket {pos.ticket} at {price:.2f}"
                    )
                else:
                    results['failed'] += 1
                    error_code = result.retcode if result else 'None'
                    results['details'].append({
                        'ticket': pos.ticket,
                        'status': 'FAILED',
                        'error': str(error_code)
                    })
                    self.logger.error(
                        f"[EMERGENCY] Failed to close ticket {pos.ticket}: {error_code}"
                    )

            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'ticket': pos.ticket,
                    'status': 'ERROR',
                    'error': str(e)
                })
                self.logger.error(f"[EMERGENCY] Exception closing ticket {pos.ticket}: {e}")

        # Record to history
        self._record_emergency(self.TRIGGER_MANUAL, results)

        self.logger.critical(
            f"[EMERGENCY] COMPLETE | Closed: {results['closed']} | "
            f"Failed: {results['failed']} | Total PnL: ${results['total_pnl']:.2f}"
        )

        return results

    # =========================================================================
    # KILL SWITCH MANAGEMENT
    # =========================================================================

    def activate_kill_switch(self, reason: str):
        """
        Activate kill switch to halt all trading.
        
        Args:
            reason: Reason for kill switch activation
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_activated_at = datetime.now()

        self._record_emergency(self.TRIGGER_MANUAL, {'reason': reason, 'type': 'KILL_SWITCH'})

        self.logger.critical(f"[EMERGENCY] KILL SWITCH ACTIVATED | Reason: {reason}")

    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        if self.kill_switch_active:
            self.logger.info("[EMERGENCY] Kill switch deactivated")
            self.kill_switch_active = False
            self.kill_switch_reason = ""
            self.kill_switch_activated_at = None

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self.kill_switch_active

    def get_kill_switch_status(self) -> Dict:
        """
        Get kill switch status.
        
        Returns:
            Dict with status
        """
        return {
            'active': self.kill_switch_active,
            'reason': self.kill_switch_reason,
            'activated_at': self.kill_switch_activated_at.isoformat() if self.kill_switch_activated_at else None,
            'duration_minutes': (datetime.now() - self.kill_switch_activated_at).total_seconds() / 60 if self.kill_switch_activated_at else 0
        }

    # =========================================================================
    # WEEKEND PROTECTION
    # =========================================================================

    def should_close_for_weekend(self) -> bool:
        """
        Check if positions should be closed before weekend.
        
        Closes on Friday after 3 PM NY time (20:00 UTC).
        
        Returns:
            True if positions should be closed
        """
        now = datetime.now()

        # Friday after 3 PM NY time (20:00 UTC)
        if now.weekday() == 4:  # Friday
            if now.hour >= 20:  # 8 PM UTC = 3 PM NY
                self.logger.info("[EMERGENCY] Weekend protection triggered")
                return True

        return False

    # =========================================================================
    # NEWS EVENT DETECTION
    # =========================================================================

    def is_near_high_impact_news(self, buffer_minutes: int = None) -> bool:
        """
        Check if current time is near high-impact news.
        
        Args:
            buffer_minutes: Minutes before/after news to check
            
        Returns:
            True if near high-impact news
        """
        if buffer_minutes is None:
            buffer_minutes = self.news_buffer_minutes

        try:
            now_utc = datetime.utcnow()
            current_minutes = now_utc.hour * 60 + now_utc.minute

            for news_hour, news_minute in self.high_impact_news_hours_utc:
                news_minutes = news_hour * 60 + news_minute

                # Check before and after news
                time_diff_before = current_minutes - (news_minutes - buffer_minutes)
                time_diff_after = (news_minutes + buffer_minutes) - current_minutes

                if 0 <= time_diff_before <= buffer_minutes * 2:
                    self.logger.info(
                        f"[EMERGENCY] Near high-impact news at {news_hour:02d}:{news_minute:02d} UTC"
                    )
                    return True

        except Exception as e:
            self.logger.error(f"[EMERGENCY] News detection error: {e}")

        return False

    # =========================================================================
    # BROKER HEALTH CHECK
    # =========================================================================

    def _check_broker_health(self) -> Dict:
        """
        Check broker connection health.
        
        Returns:
            Dict with health status
        """
        result = {
            'healthy': True,
            'reason': ''
        }

        try:
            # Check terminal connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None or not terminal_info.connected:
                result['healthy'] = False
                result['reason'] = 'MT5 terminal disconnected'
                return result

            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                result['healthy'] = False
                result['reason'] = 'Cannot get account info'
                return result

            # Check symbol info
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                result['healthy'] = False
                result['reason'] = 'Cannot get symbol info'
                return result

        except Exception as e:
            result['healthy'] = False
            result['reason'] = f'Broker health check error: {str(e)}'

        return result

    # =========================================================================
    # EMERGENCY HISTORY
    # =========================================================================

    def _record_emergency(self, trigger_type: str, details: Dict):
        """Record emergency event to history."""
        self._emergency_history.append({
            'timestamp': datetime.now().isoformat(),
            'trigger': trigger_type,
            'details': details
        })

        # Update trigger stats
        self._trigger_stats[trigger_type] = self._trigger_stats.get(trigger_type, 0) + 1

    def get_emergency_history(self, limit: int = 10) -> List[Dict]:
        """
        Get emergency history.
        
        Args:
            limit: Maximum records to return
            
        Returns:
            List of emergency records
        """
        return list(self._emergency_history)[-limit:]

    def get_emergency_stats(self) -> Dict:
        """
        Get emergency statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            'total_emergencies': len(self._emergency_history),
            'trigger_stats': dict(self._trigger_stats),
            'kill_switch_active': self.kill_switch_active,
            'recent_emergencies': list(self._emergency_history)[-5:]
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

    def _get_current_tick(self):
        """Get current tick with 0.5-second cache."""
        current_time = time.time()
        if self._tick_cache and (current_time - self._tick_time < 0.5):
            return self._tick_cache

        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self._tick_cache = tick
            self._tick_time = current_time
        return tick

    def format_emergency_log(self, results: Dict) -> str:
        """
        Format emergency check results as concise log string.
        
        Args:
            results: Results from run_emergency_check
            
        Returns:
            Formatted log string
        """
        if not results.get('emergency_detected'):
            return "[EMERGENCY] No emergency detected"

        actions = results.get('actions', [])
        action_strs = [f"{a.get('type', 'UNKNOWN')}: {a.get('reason', '')[:50]}" for a in actions[:2]]

        return (
            f"[EMERGENCY] DETECTED | Severity: {results.get('severity')} | "
            f"Actions: {', '.join(action_strs)}"
        )