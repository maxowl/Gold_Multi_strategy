"""
Expert Signal Scorer - Grades signals from A+ to F based on multi-factor analysis.
Integrates with StrategyPerformanceTracker for adaptive grading.
"""
import logging
from typing import Dict
from core.strategy_performance_tracker import StrategyPerformanceTracker


class ExpertSignalScorer:
    GRADE_THRESHOLDS = {
        'A+': 90, 'A': 80, 'B+': 70, 'B': 60, 'C+': 50, 'C': 40, 'D': 30, 'F': 0
    }

    def __init__(self, db_path: str = "bot_state.db"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.perf_tracker = StrategyPerformanceTracker(db_path)

    def score_signal(self, signal: dict, context: dict) -> Dict:
        """
        Score a trading signal based on multi-factor analysis.
        Returns: dict with score, grade, position_multiplier, should_trade
        """
        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')
        base_confidence = meta.get('confidence', 0.5)
        
        score = 50.0  # Base score
        
        # Factor 1: Base Confidence (0-25 points)
        score += base_confidence * 25.0
        
        # Factor 2: Regime Alignment (0-15 points)
        regime_conf = context.get('regime_confidence', 0.5)
        score += regime_conf * 15.0
        
        # Factor 3: Risk-Reward Ratio (0-15 points)
        risk_reward = meta.get('risk_reward', 1.0)
        if risk_reward >= 2.5:
            score += 15.0
        elif risk_reward >= 2.0:
            score += 12.0
        elif risk_reward >= 1.5:
            score += 8.0
        elif risk_reward >= 1.2:
            score += 4.0
        
        # Factor 4: Session Quality (0-15 points)
        session = context.get('session', 'OTHER')
        session_scores = {
            'LONDON_OPEN': 15.0, 'NY_OPEN': 14.0, 'LONDON': 12.0,
            'NY_MIDDAY': 10.0, 'US_CLOSE': 8.0, 'ASIAN': 6.0, 'OTHER': 4.0
        }
        score += session_scores.get(session, 4.0)
        
        # Factor 5: Historical Performance (0-15 points)
        regime = context.get('regime', 'UNKNOWN')
        stats = self.perf_tracker.get_strategy_stats(strategy_name, regime, days=30)
        
        # Use .get() with defaults to prevent KeyError if stats dict is incomplete
        trade_count = stats.get('trades', 0)
        if trade_count >= 10:
            winrate = stats.get('winrate', 0.5)
            profit_factor = stats.get('profit_factor', 1.0)
            winrate_bonus = (winrate - 0.5) * 20.0
            profit_factor_bonus = min(10.0, (profit_factor - 1.0) * 5.0)
            score += winrate_bonus + profit_factor_bonus
        
        # Factor 6: MTF Alignment (0-10 points)
        mtf_alignment = context.get('mtf_alignment', 0.5)
        score += mtf_alignment * 10.0
        
        # Factor 7: Daily PnL Status (0-5 points)
        daily_pnl = context.get('daily_pnl_percent', 0.0)
        if daily_pnl > 0:
            score += 5.0
        elif daily_pnl < -1.0:
            score -= 10.0
        
        # Clamp score to 0-100 range
        score = max(0.0, min(100.0, score))
        
        # Determine grade
        grade = 'F'
        for g, threshold in sorted(self.GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                grade = g
                break
        
        # Calculate position multiplier based on grade
        multiplier_map = {
            'A+': 2.0, 'A': 1.5, 'B+': 1.2, 'B': 1.0,
            'C+': 0.8, 'C': 0.6, 'D': 0.4, 'F': 0.0
        }
        position_multiplier = multiplier_map.get(grade, 1.0)
        
        # Determine if should trade (minimum grade C+)
        should_trade = grade in ['A+', 'A', 'B+', 'B', 'C+']
        
        # Log the scoring
        self.logger.info(
            f"[SCORER] {strategy_name} | Score: {score:.0f} ({grade}) | "
            f"Mult: {position_multiplier}x | Trade: {'YES' if should_trade else 'NO'}"
        )
        
        return {
            'score': round(score, 2),
            'grade': grade,
            'position_multiplier': position_multiplier,
            'should_trade': should_trade,
            'breakdown': {
                'base_confidence': base_confidence,
                'regime_conf': regime_conf,
                'risk_reward': risk_reward,
                'session': session,
                'historical_winrate': stats.get('winrate', 0.5),
                'historical_profit_factor': stats.get('profit_factor', 1.0)
            }
        }