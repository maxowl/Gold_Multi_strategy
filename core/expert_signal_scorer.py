"""
Expert Signal Scorer & Performance Tracker - Micro-Account-Only Edition.

Evaluates trade signals based on 9 institutional factors and assigns a grade (A+ to F).
Adjusts position sizing multiplier based on signal quality and historical edge.

9 Scoring Factors:
  1. Base Confidence (20 pts)
  2. Regime Alignment (20 pts)
  3. Risk/Reward Quality (15 pts)
  4. Session Quality (10 pts)
  5. Historical Edge (15 pts)
  6. Multi-Timeframe Alignment (10 pts)
  7. Portfolio Health (10 pts)
  8. 18-Regime Category Bonus (0-10 pts extra)
  9. Range Position Bonus (0-10 pts extra)

Penalties:
  - Choppy Score > 65: -15 pts
  - Choppy Score > 50: -8 pts
  - Market Killers active: -10 pts per killer
"""
import logging
import sqlite3
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from config import config


class PerformanceTracker:
    """
    Tracks historical performance of strategies per regime from SQLite.
    Used by ExpertSignalScorer to calculate the 'Historical Edge' factor.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize PerformanceTracker.

        Args:
            db_path: Path to SQLite database (defaults to config.state_db_path)
        """
        if db_path is None:
            db_path = config.state_db_path
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = 300  # 5 minutes cache TTL

    def get_strategy_stats(self, strategy_name: str, unified_regime: str, days: int = 30) -> Dict:
        """
        Get historical stats for a specific strategy in a specific unified regime.

        Args:
            strategy_name: Strategy name to query
            unified_regime: Unified regime to filter
            days: Number of days to look back

        Returns:
            Dict with trades, winrate, avg_win, avg_loss, profit_factor
        """
        cache_key = f"{strategy_name}_{unified_regime}_{days}"
        current_time = datetime.now().timestamp()

        # Return cached result if valid
        if cache_key in self._cache and current_time < self._cache_expiry.get(cache_key, 0):
            return self._cache[cache_key]

        default_stats = {
            'trades': 0,
            'winrate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

        try:
            # Read-Only connection to prevent locking the live bot
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)

            query = """
                SELECT profit, meta_data
                FROM trade_history
                WHERE strategy = ? AND close_time >= datetime('now', ?)
            """
            df = pd.read_sql_query(query, conn, params=(strategy_name, f"-{days} days"))
            conn.close()

            if df.empty:
                self._cache[cache_key] = default_stats
                self._cache_expiry[cache_key] = current_time + self.cache_ttl
                return default_stats

            # Parse meta_data to filter by unified_regime
            regimes = []
            for meta_str in df['meta_data']:
                try:
                    meta = json.loads(meta_str) if meta_str else {}
                    regimes.append(meta.get('regime', 'UNKNOWN'))
                except (json.JSONDecodeError, TypeError):
                    regimes.append('UNKNOWN')

            df['regime'] = regimes

            # Filter by unified regime (if specified and not UNKNOWN/SIDEWAY)
            if unified_regime not in ['UNKNOWN', 'SIDEWAY']:
                df_filtered = df[df['regime'] == unified_regime]
            else:
                df_filtered = df

            # If filtered df is empty, fallback to overall strategy stats
            if df_filtered.empty:
                df_filtered = df

            trades = len(df_filtered)
            winners = df_filtered[df_filtered['profit'] > 0]
            losers = df_filtered[df_filtered['profit'] <= 0]

            winrate = len(winners) / trades if trades > 0 else 0.0
            avg_win = float(winners['profit'].mean()) if not winners.empty else 0.0
            avg_loss = float(abs(losers['profit'].mean())) if not losers.empty else 0.0

            gross_profit = float(winners['profit'].sum()) if not winners.empty else 0.0
            gross_loss = float(abs(losers['profit'].sum())) if not losers.empty else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            stats = {
                'trades': trades,
                'winrate': winrate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }

            self._cache[cache_key] = stats
            self._cache_expiry[cache_key] = current_time + self.cache_ttl
            return stats

        except sqlite3.OperationalError as e:
            self.logger.debug(f"[PERF_TRACKER] DB Operational Error: {e}")
            return default_stats
        except Exception as e:
            self.logger.error(f"[PERF_TRACKER] Error fetching stats: {e}")
            return default_stats


class ExpertSignalScorer:
    """
    Evaluates trade signals using a 9-factor institutional scoring model.
    Assigns a grade (A+ to F) and a position multiplier.
    """

    def __init__(self):
        """Initialize ExpertSignalScorer with PerformanceTracker."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.perf_tracker = PerformanceTracker()

        # Scoring Weights (Total = 100 points)
        self.WEIGHTS = {
            'base_confidence': 20,
            'regime_alignment': 20,
            'risk_reward': 15,
            'session_quality': 10,
            'historical_edge': 15,
            'mtf_alignment': 10,
            'portfolio_health': 10
        }

        # Grade Thresholds: (Min Score, Grade, Multiplier)
        self.GRADE_THRESHOLDS = [
            (90, 'A+', 2.0),
            (80, 'A', 1.5),
            (70, 'B+', 1.2),
            (60, 'B', 1.0),
            (50, 'C+', 0.8),
            (0, 'F', 0.0)
        ]

        # 18-Regime Category Strength Mapping (For Factor 8 Bonus)
        self.REGIME_CATEGORY_STRENGTH = {
            'TREND': ['HEALTHY_UPTREND', 'HEALTHY_DOWNTREND', 'QUIET_RALLY', 'SLOW_BLEED', 'FALSE_SIDEWAY'],
            'MEAN_REVERSION': ['TIGHT_RANGE', 'CLASSIC_RANGE', 'ANOMALY_BULL', 'ANOMALY_BEAR',
                               'OVERSOLD_BOUNCE', 'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR'],
            'SMC': ['HEALTHY_UPTREND', 'HEALTHY_DOWNTREND', 'PRE_BREAKOUT'],
            'SCALP': ['PARABOLIC_RALLY', 'PANIC_CAPITULATION', 'VOLATILE_CHOP', 'WHIPSAW_MARKET', 'EXHAUSTED_BULL']
        }

    def score_signal(self, signal: Dict, context: Dict) -> Dict:
        """
        Score a trade signal based on 9 factors + penalties.

        Args:
            signal: Signal dict with meta
            context: Market context dict

        Returns:
            Dict with score, grade, should_trade, position_multiplier, reasons, breakdown
        """
        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')
        strategy_category = meta.get('strategy_category', 'GENERAL')

        scores = {}
        reasons = []

        # =========================================================================
        # FACTOR 1: Base Confidence (Max 20 pts)
        # =========================================================================
        base_conf = meta.get('confidence', 0.5)
        scores['base_confidence'] = max(0.0, min(20.0, base_conf * 20.0))

        # =========================================================================
        # FACTOR 2: Regime Alignment (Max 20 pts)
        # =========================================================================
        regime_name = context.get('regime_name', 'UNKNOWN')
        unified_regime = context.get('regime', context.get('unified_regime', 'SIDEWAY'))

        regime_score = self._score_regime_alignment(strategy_category, unified_regime)
        scores['regime_alignment'] = regime_score
        if regime_score >= 15:
            reasons.append(f"Strong regime alignment ({strategy_category} in {unified_regime})")
        elif regime_score <= 5:
            reasons.append(f"Poor regime alignment ({strategy_category} in {unified_regime})")

        # =========================================================================
        # FACTOR 3: Risk/Reward Quality (Max 15 pts)
        # =========================================================================
        rr_score, rr_reason = self._score_risk_reward(meta)
        scores['risk_reward'] = rr_score
        if rr_reason:
            reasons.append(rr_reason)

        # =========================================================================
        # FACTOR 4: Session Quality (Max 10 pts)
        # =========================================================================
        session = context.get('session', 'OTHER')
        session_score = self._score_session_quality(session)
        scores['session_quality'] = session_score

        # =========================================================================
        # FACTOR 5: Historical Edge (Max 15 pts)
        # =========================================================================
        hist_score, hist_reason = self._score_historical_edge(strategy_name, unified_regime)
        scores['historical_edge'] = hist_score
        if hist_reason:
            reasons.append(hist_reason)

        # =========================================================================
        # FACTOR 6: Multi-Timeframe Alignment (Max 10 pts)
        # =========================================================================
        mtf_score = self._score_mtf_alignment(meta, context)
        scores['mtf_alignment'] = mtf_score

        # =========================================================================
        # FACTOR 7: Portfolio Health / Drawdown (Max 10 pts)
        # =========================================================================
        portfolio_score = self._score_portfolio_health(context)
        scores['portfolio_health'] = portfolio_score

        # =========================================================================
        # CALCULATE BASE SCORE
        # =========================================================================
        base_score = sum(scores.values())

        # =========================================================================
        # FACTOR 8: 18-Regime Category Bonus (Max +10 pts extra)
        # =========================================================================
        regime_bonus = self._score_18_regime_bonus(strategy_category, regime_name)
        if regime_bonus > 0:
            reasons.append(f"18-Regime Bonus: +{regime_bonus:.1f} pts ({strategy_category} in {regime_name})")

        # =========================================================================
        # FACTOR 9: Range Position Bonus (Max +10 pts extra)
        # =========================================================================
        range_bonus = self._score_range_position_bonus(meta)
        if range_bonus > 0:
            reasons.append(f"Range Position Bonus: +{range_bonus:.1f} pts")

        # =========================================================================
        # PENALTIES (Choppy & Market Killers)
        # =========================================================================
        penalty = 0
        choppy_score = context.get('choppy_score', 0)
        if choppy_score > 65:
            penalty += 15
            reasons.append(f"Penalty: High choppy score ({choppy_score:.0f})")
        elif choppy_score > 50:
            penalty += 8
            reasons.append(f"Penalty: Medium choppy score ({choppy_score:.0f})")

        active_killers = context.get('active_killers', [])
        if active_killers:
            penalty += 10 * len(active_killers)
            reasons.append(f"Penalty: {len(active_killers)} Market Killers active")

        # =========================================================================
        # FINAL SCORE CALCULATION
        # =========================================================================
        final_score = max(0.0, min(110.0, (base_score + regime_bonus + range_bonus) - penalty))
        # Normalize back to 100 scale for grading
        normalized_score = min(100.0, final_score)

        # =========================================================================
        # GRADING & MULTIPLIER
        # =========================================================================
        grade, multiplier = self._get_grade_and_multiplier(normalized_score)

        # Minimum Grade Threshold (C+ = 50)
        min_grade_score = 50
        should_trade = normalized_score >= min_grade_score and multiplier > 0

        if not should_trade:
            reasons.append(f"Blocked: Score {normalized_score:.0f} below minimum threshold ({min_grade_score})")

        return {
            'score': normalized_score,
            'raw_score': final_score,
            'grade': grade,
            'should_trade': should_trade,
            'position_multiplier': multiplier,
            'reasons': reasons,
            'breakdown': scores
        }

    # =========================================================================
    # FACTOR SCORING METHODS
    # =========================================================================

    def _score_regime_alignment(self, strategy_category: str, unified_regime: str) -> float:
        """Score how well the strategy category matches the unified regime."""
        max_pts = self.WEIGHTS['regime_alignment']

        # Ideal matches
        ideal_matches = {
            'TREND': ['TREND'],
            'SIDEWAY': ['MEAN_REV', 'SCALP'],
            'HIGH_VOL': ['SCALP', 'MEAN_REV'],
            'REVERSAL': ['SMC', 'MEAN_REV']
        }

        # Good matches
        good_matches = {
            'TREND': ['SMC'],
            'SIDEWAY': ['SMC'],
            'HIGH_VOL': ['SMC'],
            'REVERSAL': ['TREND']
        }

        if unified_regime in ideal_matches and strategy_category in ideal_matches[unified_regime]:
            return float(max_pts)
        elif unified_regime in good_matches and strategy_category in good_matches[unified_regime]:
            return float(max_pts * 0.7)
        else:
            # Mismatch (e.g., Trend strategy in Sideways market)
            return float(max_pts * 0.2)

    def _score_risk_reward(self, meta: Dict) -> Tuple[float, str]:
        """Score the Risk/Reward ratio."""
        max_pts = self.WEIGHTS['risk_reward']
        rr = meta.get('risk_reward', 0)

        if rr >= 3.0:
            return float(max_pts), f"Excellent R:R ({rr:.1f})"
        elif rr >= 2.0:
            return float(max_pts * 0.8), f"Good R:R ({rr:.1f})"
        elif rr >= 1.5:
            return float(max_pts * 0.6), f"Acceptable R:R ({rr:.1f})"
        elif rr >= 1.0:
            return float(max_pts * 0.3), f"Poor R:R ({rr:.1f})"
        else:
            return 0.0, f"Negative/Invalid R:R ({rr:.1f})"

    def _score_session_quality(self, session: str) -> float:
        """Score the current trading session."""
        max_pts = self.WEIGHTS['session_quality']

        prime_sessions = ['LONDON_OPEN', 'NY_OPEN']
        active_sessions = ['LONDON', 'NY_MIDDAY']
        dead_sessions = ['ASIAN', 'US_CLOSE', 'OTHER']

        if session in prime_sessions:
            return float(max_pts)
        elif session in active_sessions:
            return float(max_pts * 0.7)
        else:
            return float(max_pts * 0.3)

    def _score_historical_edge(self, strategy_name: str, unified_regime: str) -> Tuple[float, str]:
        """Score based on historical performance of this strategy in this regime."""
        max_pts = self.WEIGHTS['historical_edge']

        stats = self.perf_tracker.get_strategy_stats(strategy_name, unified_regime, days=30)
        trades = stats['trades']
        winrate = stats['winrate']
        pf = stats['profit_factor']

        # Cold start (not enough data)
        if trades < 10:
            return float(max_pts * 0.5), f"Cold start ({trades} trades)"

        # Strong edge
        if winrate >= 0.60 and pf >= 1.5:
            return float(max_pts), f"Strong historical edge (WR:{winrate:.0%}, PF:{pf:.1f})"
        # Moderate edge
        elif winrate >= 0.50 and pf >= 1.2:
            return float(max_pts * 0.7), f"Moderate edge (WR:{winrate:.0%}, PF:{pf:.1f})"
        # Weak edge
        elif winrate >= 0.40:
            return float(max_pts * 0.4), f"Weak edge (WR:{winrate:.0%}, PF:{pf:.1f})"
        # Negative edge
        else:
            return 0.0, f"Negative edge (WR:{winrate:.0%}, PF:{pf:.1f})"

    def _score_mtf_alignment(self, meta: Dict, context: Dict) -> float:
        """Score Multi-Timeframe alignment."""
        max_pts = self.WEIGHTS['mtf_alignment']

        # If strategy provides MTF confirmation in meta
        mtf_confirmed = meta.get('mtf_confirmed', False)
        if mtf_confirmed:
            return float(max_pts)

        # Fallback: Check if HTF trend matches signal direction
        htf_trend = context.get('htf_trend', 'NEUTRAL')

        if htf_trend == 'NEUTRAL':
            return float(max_pts * 0.5)

        return float(max_pts * 0.7)  # Default moderate score if MTF logic is absent

    def _score_portfolio_health(self, context: Dict) -> float:
        """Score based on current portfolio drawdown state."""
        max_pts = self.WEIGHTS['portfolio_health']

        daily_pnl_pct = context.get('daily_pnl_pct', context.get('daily_pnl_percent', 0.0))

        # If we are in profit today, full score
        if daily_pnl_pct >= 0:
            return float(max_pts)
        # Mild drawdown
        elif daily_pnl_pct >= -1.5:
            return float(max_pts * 0.7)
        # Moderate drawdown (reduce size)
        elif daily_pnl_pct >= -3.0:
            return float(max_pts * 0.4)
        # Severe drawdown (block or minimal size)
        else:
            return 0.0

    def _score_18_regime_bonus(self, strategy_category: str, regime_name: str) -> float:
        """
        Factor 8: Award bonus points if the strategy category is in its strongest 18-Regime.
        Max Bonus: 10 points.
        """
        max_bonus = 10.0
        for cat, strong_regimes in self.REGIME_CATEGORY_STRENGTH.items():
            if strategy_category == cat and regime_name in strong_regimes:
                return max_bonus
        return 0.0

    def _score_range_position_bonus(self, meta: Dict) -> float:
        """
        Factor 9: Award bonus points based on Range Position Filter quality.
        Max Bonus: 10 points.
        """
        range_analysis = meta.get('range_analysis', {})
        if range_analysis and isinstance(range_analysis, dict):
            position_score = range_analysis.get('position_score', 50)
            # Convert 0-100 score to 0-10 points bonus
            return float((position_score / 100.0) * 10.0)
        return 0.0

    # =========================================================================
    # GRADING
    # =========================================================================

    def _get_grade_and_multiplier(self, score: float) -> Tuple[str, float]:
        """Map normalized score to Grade and Position Multiplier."""
        for threshold, grade, multiplier in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade, multiplier
        return 'F', 0.0

    def format_score_log(self, signal: Dict, score_result: Dict) -> str:
        """Format a concise log string for the signal score."""
        meta = signal.get('meta', {})
        strategy = meta.get('strategy', 'Unknown')
        signal_type = signal.get('signal', 'NEUTRAL')

        score = score_result['score']
        grade = score_result['grade']
        mult = score_result['position_multiplier']

        top_reasons = ', '.join(score_result['reasons'][:2]) if score_result['reasons'] else 'No specific reasons'

        return (
            f"[SCORER] {strategy} {signal_type} | "
            f"Score: {score:.0f} (Grade {grade}) | "
            f"Mult: {mult:.1f}x | "
            f"Trade: {'YES' if score_result['should_trade'] else 'NO'} | "
            f"Reasons: {top_reasons}"
        )