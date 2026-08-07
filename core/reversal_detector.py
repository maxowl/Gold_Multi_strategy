"""
Multi-Timeframe Reversal Detector - Institutional Grade.

Detects reversal signals across multiple timeframes (M1, M5, M15, H1)
to enable proactive position management.

Detection Layers:
  Layer 1: EMA Crossover (10/20 EMA cross)
  Layer 2: RSI Divergence (overbought/oversold extremes)
  Layer 3: MACD Divergence (histogram direction change)

Scoring System:
  - Each layer contributes 1 point if confirmed
  - Multi-TF confirmation adds weight
  - Score 0-3 determines action

Actions:
  - Score >= 2: PARTIAL_CLOSE (close 50%)
  - Score == 1: TIGHTEN_TRAIL (reduce trail by 50%)
  - Score == 0: NO_ACTION
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ReversalDetector:
    """
    Detects reversal signals across multiple timeframes.
    
    Used by Layer 2.5 of the 10-Layer Active Position Management system
    to proactively protect profits when reversal signs appear.
    """

    def __init__(self):
        """Initialize ReversalDetector with configuration."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # EMA periods
        self.ema_fast = 10
        self.ema_slow = 20

        # RSI configuration
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.rsi_extreme_ob = 80
        self.rsi_extreme_os = 20

        # MACD configuration
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # Multi-TF weights (higher TF = higher weight)
        self.tf_weights = {
            'M1': 0.5,
            'M5': 0.7,
            'M15': 1.0,
            'H1': 1.3,
            'H4': 1.5
        }

        # Cooldown tracking (ticket -> last trigger timestamp)
        self._cooldown_cache: Dict[int, float] = {}
        self.cooldown_minutes = 10  # 10 minutes between signals

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def detect_reversal_signals(self, df_dict: Dict[str, pd.DataFrame],
                                 is_buy: bool, current_profit_usd: float,
                                 ticket: int = 0) -> Dict:
        """
        Main entry point for reversal detection.
        
        Args:
            df_dict: Dict of timeframe -> DataFrame
            is_buy: True if position is BUY, False if SELL
            current_profit_usd: Current unrealized profit
            ticket: Position ticket for cooldown tracking
            
        Returns:
            Dict with:
              - reversal_score: 0-3
              - action: 'NO_ACTION', 'TIGHTEN_TRAIL', 'PARTIAL_CLOSE'
              - signals: List of detected signals
              - multi_tf_confirmed: bool
        """
        result = {
            'reversal_score': 0,
            'action': 'NO_ACTION',
            'signals': [],
            'multi_tf_confirmed': False,
            'profit_usd': current_profit_usd,
            'recommendation': 'Hold position'
        }

        # Check cooldown
        if ticket > 0 and not self._check_cooldown(ticket):
            return result

        # Minimum profit threshold for reversal detection
        if current_profit_usd < 3.0:  # $3 minimum profit to consider reversal
            return result

        # Collect signals from all timeframes
        all_signals = []
        tf_confirmations = {}

        for tf_name, df in df_dict.items():
            if df is None or df.empty or len(df) < 30:
                continue

            tf_signals = self._detect_signals_for_tf(df, tf_name, is_buy)
            all_signals.extend(tf_signals)

            # Track confirmations per layer
            for sig in tf_signals:
                layer = sig['layer']
                if layer not in tf_confirmations:
                    tf_confirmations[layer] = set()
                tf_confirmations[layer].add(tf_name)

        result['signals'] = all_signals

        # Calculate reversal score (0-3)
        score = self._calculate_reversal_score(tf_confirmations)
        result['reversal_score'] = score

        # Check multi-TF confirmation (2+ timeframes confirm same layer)
        multi_tf_count = sum(1 for tf_set in tf_confirmations.values() if len(tf_set) >= 2)
        result['multi_tf_confirmed'] = multi_tf_count > 0

        # Determine action
        action, recommendation = self._determine_action(
            score, current_profit_usd, result['multi_tf_confirmed']
        )
        result['action'] = action
        result['recommendation'] = recommendation

        # Update cooldown if action taken
        if action != 'NO_ACTION' and ticket > 0:
            self._update_cooldown(ticket)

        return result

    # =========================================================================
    # PER-TIMEFRAME DETECTION
    # =========================================================================

    def _detect_signals_for_tf(self, df: pd.DataFrame, tf_name: str,
                                is_buy: bool) -> List[Dict]:
        """
        Detect reversal signals for a specific timeframe.
        
        Returns:
            List of signal dicts
        """
        signals = []

        try:
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)

            # Handle NaN
            close = np.nan_to_num(close, nan=close[0])
            high = np.nan_to_num(high, nan=high[0])
            low = np.nan_to_num(low, nan=low[0])

            # Layer 1: EMA Crossover
            ema_signal = self._detect_ema_crossover(close, is_buy, tf_name)
            if ema_signal:
                signals.append(ema_signal)

            # Layer 2: RSI Divergence
            rsi_signal = self._detect_rsi_divergence(close, is_buy, tf_name)
            if rsi_signal:
                signals.append(rsi_signal)

            # Layer 3: MACD Divergence
            macd_signal = self._detect_macd_divergence(close, is_buy, tf_name)
            if macd_signal:
                signals.append(macd_signal)

        except Exception as e:
            self.logger.error(f"[REVERSAL] Error detecting signals for {tf_name}: {e}")

        return signals

    # =========================================================================
    # LAYER 1: EMA CROSSOVER
    # =========================================================================

    def _detect_ema_crossover(self, close: np.ndarray, is_buy: bool,
                               tf_name: str) -> Optional[Dict]:
        """
        Detect EMA crossover reversal signal.
        
        BUY position reversal: Fast EMA crosses below Slow EMA
        SELL position reversal: Fast EMA crosses above Slow EMA
        """
        try:
            close_series = pd.Series(close)

            # Calculate EMAs using exponential weighted moving average
            ema_fast = close_series.ewm(span=self.ema_fast, adjust=False).mean().values
            ema_slow = close_series.ewm(span=self.ema_slow, adjust=False).mean().values

            if len(ema_fast) < 2 or len(ema_slow) < 2:
                return None

            # Current and previous values
            curr_fast = ema_fast[-1]
            prev_fast = ema_fast[-2]
            curr_slow = ema_slow[-1]
            prev_slow = ema_slow[-2]

            # Check for crossover
            if is_buy:
                # BUY reversal: fast crosses below slow
                if prev_fast >= prev_slow and curr_fast < curr_slow:
                    return {
                        'layer': 1,
                        'type': 'EMA_CROSS_DOWN',
                        'timeframe': tf_name,
                        'strength': 0.7,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'description': f'EMA {self.ema_fast}/{self.ema_slow} cross down on {tf_name}'
                    }
            else:
                # SELL reversal: fast crosses above slow
                if prev_fast <= prev_slow and curr_fast > curr_slow:
                    return {
                        'layer': 1,
                        'type': 'EMA_CROSS_UP',
                        'timeframe': tf_name,
                        'strength': 0.7,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'description': f'EMA {self.ema_fast}/{self.ema_slow} cross up on {tf_name}'
                    }

            return None

        except Exception as e:
            self.logger.debug(f"[REVERSAL] EMA detection error on {tf_name}: {e}")
            return None

    # =========================================================================
    # LAYER 2: RSI DIVERGENCE
    # =========================================================================

    def _detect_rsi_divergence(self, close: np.ndarray, is_buy: bool,
                                tf_name: str) -> Optional[Dict]:
        """
        Detect RSI extreme levels as reversal signal.
        
        BUY position reversal: RSI > 80 (extreme overbought)
        SELL position reversal: RSI < 20 (extreme oversold)
        """
        try:
            rsi = self._calculate_rsi(close, self.rsi_period)

            if rsi is None or len(rsi) < 2:
                return None

            current_rsi = rsi[-1]
            prev_rsi = rsi[-2]

            if np.isnan(current_rsi):
                return None

            if is_buy:
                # BUY reversal: RSI in extreme overbought zone
                if current_rsi >= self.rsi_extreme_ob:
                    strength = min(1.0, (current_rsi - self.rsi_extreme_ob) / 20 + 0.6)
                    return {
                        'layer': 2,
                        'type': 'RSI_EXTREME_OB',
                        'timeframe': tf_name,
                        'strength': strength,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'value': round(current_rsi, 2),
                        'description': f'RSI extreme overbought ({current_rsi:.1f}) on {tf_name}'
                    }
            else:
                # SELL reversal: RSI in extreme oversold zone
                if current_rsi <= self.rsi_extreme_os:
                    strength = min(1.0, (self.rsi_extreme_os - current_rsi) / 20 + 0.6)
                    return {
                        'layer': 2,
                        'type': 'RSI_EXTREME_OS',
                        'timeframe': tf_name,
                        'strength': strength,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'value': round(current_rsi, 2),
                        'description': f'RSI extreme oversold ({current_rsi:.1f}) on {tf_name}'
                    }

            return None

        except Exception as e:
            self.logger.debug(f"[REVERSAL] RSI detection error on {tf_name}: {e}")
            return None

    # =========================================================================
    # LAYER 3: MACD DIVERGENCE
    # =========================================================================

    def _detect_macd_divergence(self, close: np.ndarray, is_buy: bool,
                                 tf_name: str) -> Optional[Dict]:
        """
        Detect MACD histogram direction change as reversal signal.
        
        BUY position reversal: MACD histogram changes from positive to negative
        SELL position reversal: MACD histogram changes from negative to positive
        """
        try:
            close_series = pd.Series(close)

            # Calculate MACD
            ema_fast = close_series.ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = close_series.ewm(span=self.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
            histogram = macd_line - signal_line

            if len(histogram) < 2:
                return None

            curr_hist = float(histogram.iloc[-1])
            prev_hist = float(histogram.iloc[-2])

            if np.isnan(curr_hist) or np.isnan(prev_hist):
                return None

            if is_buy:
                # BUY reversal: histogram turns negative
                if prev_hist > 0 and curr_hist <= 0:
                    strength = min(1.0, abs(curr_hist) / (abs(prev_hist) + 1e-10) * 0.8)
                    return {
                        'layer': 3,
                        'type': 'MACD_HIST_NEGATIVE',
                        'timeframe': tf_name,
                        'strength': strength,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'value': round(curr_hist, 4),
                        'description': f'MACD histogram turned negative on {tf_name}'
                    }
            else:
                # SELL reversal: histogram turns positive
                if prev_hist < 0 and curr_hist >= 0:
                    strength = min(1.0, abs(curr_hist) / (abs(prev_hist) + 1e-10) * 0.8)
                    return {
                        'layer': 3,
                        'type': 'MACD_HIST_POSITIVE',
                        'timeframe': tf_name,
                        'strength': strength,
                        'weight': self.tf_weights.get(tf_name, 1.0),
                        'value': round(curr_hist, 4),
                        'description': f'MACD histogram turned positive on {tf_name}'
                    }

            return None

        except Exception as e:
            self.logger.debug(f"[REVERSAL] MACD detection error on {tf_name}: {e}")
            return None

    # =========================================================================
    # RSI CALCULATION
    # =========================================================================

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
        """
        Calculate RSI using Wilder's smoothing method.
        
        Args:
            close: Array of close prices
            period: RSI period (default 14)
            
        Returns:
            Array of RSI values or None on error
        """
        try:
            if len(close) < period + 1:
                return None

            # Calculate price changes
            deltas = np.diff(close)

            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Initialize with SMA for first period
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            rsi_values = np.zeros(len(close))
            rsi_values[:period] = np.nan

            # Calculate RSI using Wilder's smoothing
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi_values[i + 1] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))

            # Bounds check
            rsi_values = np.clip(rsi_values, 0, 100)

            return rsi_values

        except Exception as e:
            self.logger.debug(f"[REVERSAL] RSI calculation error: {e}")
            return None

    # =========================================================================
    # SCORING & ACTION DETERMINATION
    # =========================================================================

    def _calculate_reversal_score(self, tf_confirmations: Dict[int, set]) -> int:
        """
        Calculate reversal score based on number of confirmed layers.
        
        Each layer (1=EMA, 2=RSI, 3=MACD) contributes 1 point if
        at least one timeframe confirms it.
        
        Multi-TF confirmation (2+ timeframes) adds 0.5 bonus per layer.
        
        Returns:
            Score from 0 to 3
        """
        score = 0

        for layer, tf_set in tf_confirmations.items():
            if len(tf_set) >= 1:
                score += 1
                # Bonus for multi-TF confirmation
                if len(tf_set) >= 2:
                    score += 0.5

        return min(3, int(score))

    def _determine_action(self, score: int, profit_usd: float,
                           multi_tf_confirmed: bool) -> Tuple[str, str]:
        """
        Determine action based on reversal score and profit.
        
        Args:
            score: Reversal score (0-3)
            profit_usd: Current profit in USD
            multi_tf_confirmed: Whether multi-TF confirmation exists
            
        Returns:
            Tuple of (action, recommendation)
        """
        # Score 0: No action
        if score == 0:
            return 'NO_ACTION', 'Hold position - no reversal signals'

        # Score 1: Tighten trailing stop
        if score == 1:
            if multi_tf_confirmed:
                return 'TIGHTEN_TRAIL', 'Single layer reversal - tighten trailing stop'
            else:
                return 'NO_ACTION', 'Weak signal - continue monitoring'

        # Score 2+: Partial close
        if score >= 2:
            # Higher profit threshold for higher scores
            if profit_usd >= 5.0:
                return 'PARTIAL_CLOSE', f'Strong reversal (score {score}) - partial close recommended'
            else:
                return 'TIGHTEN_TRAIL', f'Moderate reversal (score {score}) - tighten trailing stop'

        return 'NO_ACTION', 'Hold position'

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _check_cooldown(self, ticket: int) -> bool:
        """
        Check if cooldown period has passed for this ticket.
        
        Returns:
            True if cooldown has passed, False otherwise
        """
        if ticket not in self._cooldown_cache:
            return True

        last_trigger = self._cooldown_cache[ticket]
        elapsed_minutes = (datetime.now().timestamp() - last_trigger) / 60.0

        return elapsed_minutes >= self.cooldown_minutes

    def _update_cooldown(self, ticket: int):
        """Update cooldown timestamp for this ticket."""
        self._cooldown_cache[ticket] = datetime.now().timestamp()

    def clear_cooldown(self, ticket: int):
        """Clear cooldown for a specific ticket."""
        if ticket in self._cooldown_cache:
            del self._cooldown_cache[ticket]

    def clear_all_cooldowns(self):
        """Clear all cooldowns."""
        self._cooldown_cache.clear()

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def format_reversal_log(self, result: Dict, strategy_name: str,
                             ticket: int) -> str:
        """
        Format a concise log string for reversal detection result.
        
        Args:
            result: Result from detect_reversal_signals
            strategy_name: Strategy name
            ticket: Position ticket
            
        Returns:
            Formatted log string
        """
        score = result.get('reversal_score', 0)
        action = result.get('action', 'NO_ACTION')
        multi_tf = result.get('multi_tf_confirmed', False)
        signals = result.get('signals', [])

        signal_summary = ', '.join([
            f"L{s['layer']}:{s['timeframe']}" for s in signals[:3]
        ]) if signals else 'None'

        return (
            f"[REVERSAL] Ticket {ticket} ({strategy_name}) | "
            f"Score: {score}/3 | "
            f"Action: {action} | "
            f"Multi-TF: {'YES' if multi_tf else 'NO'} | "
            f"Signals: {signal_summary}"
        )