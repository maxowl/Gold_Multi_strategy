"""
Position Intelligence Manager - Smart Position Monitoring.

Provides intelligent analysis of all open positions and generates
actionable recommendations for position management.

Used by Layer 8 of the 10-Layer Active Position Management system.

Analysis Dimensions:
  1. P&L Status: Current profit/loss and trajectory
  2. Momentum: Is the trade thesis still valid?
  3. Risk/Reward: Has R:R deteriorated?
  4. Time: How long has position been open?
  5. Structure: Has key structure been broken?
  6. Regime: Does position still align with regime?

Recommendation Actions:
  - HOLD: Continue holding position
  - TIGHTEN_STOP: Move SL closer to lock profit
  - PARTIAL_CLOSE: Close portion of position
  - CLOSE: Close entire position
  - ADD: Add to position (rare, only in strong trends)

Priority Levels (1-5):
  1 = CRITICAL: Immediate action required
  2 = HIGH: Action needed soon
  3 = MEDIUM: Monitor closely
  4 = LOW: Normal monitoring
  5 = INFO: Informational only
"""
import pandas as pd
import numpy as np
import logging
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import config


class PositionIntelligenceManager:
    """
    Provides intelligent analysis of open positions.
    
    Features:
      - Multi-dimensional position analysis
      - Priority-based recommendations
      - Regime context awareness
      - Structure validation
      - Momentum tracking
      - Time-based recommendations
    """

    # Recommendation actions
    ACTION_HOLD = 'HOLD'
    ACTION_TIGHTEN_STOP = 'TIGHTEN_STOP'
    ACTION_PARTIAL_CLOSE = 'PARTIAL_CLOSE'
    ACTION_CLOSE = 'CLOSE'
    ACTION_ADD = 'ADD'

    # Priority levels
    PRIORITY_CRITICAL = 1
    PRIORITY_HIGH = 2
    PRIORITY_MEDIUM = 3
    PRIORITY_LOW = 4
    PRIORITY_INFO = 5

    def __init__(self):
        """Initialize PositionIntelligenceManager."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Analysis thresholds
        self.profit_target_pct = 50  # 50% of target reached
        self.loss_threshold_pct = 70  # 70% of risk realized
        self.time_threshold_minutes = 60  # 1 hour

        # Momentum thresholds
        self.momentum_reversal_threshold = 0.3  # 30% momentum reversal

        # Cache for analysis
        self._analysis_cache: Dict[int, Dict] = {}
        self._last_analysis_time: Dict[int, float] = {}
        self.cache_ttl_seconds = 60  # 1 minute cache

        self.logger.info("[POS_INTEL] Initialized with multi-dimensional analysis")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def analyze_all_positions(self, positions: List[Dict],
                               current_prices: Dict[int, float],
                               df_m5: pd.DataFrame = None,
                               regime_context: Dict = None) -> Dict:
        """
        Main entry point: Analyze all open positions.
        
        Args:
            positions: List of position dicts from StateManager
            current_prices: Dict of ticket -> current price
            df_m5: M5 DataFrame for technical analysis
            regime_context: Current regime information
            
        Returns:
            Dict with:
              - position_analyses: Dict of ticket -> analysis
              - recommendations: List of prioritized recommendations
              - summary: Overall portfolio health
        """
        if not positions:
            return {
                'position_analyses': {},
                'recommendations': [],
                'summary': {'total_positions': 0, 'health': 'NO_POSITIONS'}
            }

        position_analyses = {}
        all_recommendations = []

        for pos in positions:
            ticket = pos.get('ticket')
            if ticket is None:
                continue

            current_price = current_prices.get(ticket)
            if current_price is None:
                continue

            # Analyze position
            analysis = self.analyze_position(
                pos, current_price, df_m5, regime_context
            )

            position_analyses[ticket] = analysis

            # Extract recommendation
            if analysis.get('recommendation'):
                rec = analysis['recommendation']
                rec['ticket'] = ticket
                rec['strategy'] = pos.get('meta_data', {}).get('strategy', 'Unknown')
                all_recommendations.append(rec)

        # Sort recommendations by priority
        all_recommendations.sort(key=lambda r: r.get('priority', 5))

        # Generate summary
        summary = self._generate_summary(position_analyses)

        return {
            'position_analyses': position_analyses,
            'recommendations': all_recommendations,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }

    # =========================================================================
    # SINGLE POSITION ANALYSIS
    # =========================================================================

    def analyze_position(self, pos: Dict, current_price: float,
                          df_m5: pd.DataFrame = None,
                          regime_context: Dict = None) -> Dict:
        """
        Analyze a single position comprehensively.
        
        Args:
            pos: Position dict from StateManager
            current_price: Current market price
            df_m5: M5 DataFrame for technical analysis
            regime_context: Current regime information
            
        Returns:
            Dict with comprehensive analysis
        """
        ticket = pos.get('ticket', 0)
        entry_price = pos.get('entry_price', 0)
        sl_price = pos.get('sl', 0)
        tp_price = pos.get('tp', 0)
        position_type = pos.get('position_type', 'BUY')
        meta = pos.get('meta_data', {})

        is_buy = position_type == 'BUY'

        # Calculate basic metrics
        pnl_analysis = self._analyze_pnl(
            entry_price, current_price, sl_price, tp_price, is_buy
        )

        # Analyze momentum
        momentum_analysis = self._analyze_momentum(
            pos, current_price, df_m5, is_buy
        )

        # Analyze risk
        risk_analysis = self._analyze_risk(
            entry_price, current_price, sl_price, tp_price, is_buy
        )

        # Analyze time
        time_analysis = self._analyze_time(pos)

        # Analyze structure
        structure_analysis = self._analyze_structure(
            pos, current_price, df_m5, is_buy
        )

        # Analyze regime alignment
        regime_analysis = self._analyze_regime_alignment(
            pos, regime_context, is_buy
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            pnl_analysis, momentum_analysis, risk_analysis,
            time_analysis, structure_analysis, regime_analysis
        )

        # Calculate overall health score
        health_score = self._calculate_health_score(
            pnl_analysis, momentum_analysis, risk_analysis,
            structure_analysis, regime_analysis
        )

        return {
            'ticket': ticket,
            'entry_price': entry_price,
            'current_price': current_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_type': position_type,
            'pnl_analysis': pnl_analysis,
            'momentum_analysis': momentum_analysis,
            'risk_analysis': risk_analysis,
            'time_analysis': time_analysis,
            'structure_analysis': structure_analysis,
            'regime_analysis': regime_analysis,
            'health_score': health_score,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }

    # =========================================================================
    # DIMENSION 1: P&L ANALYSIS
    # =========================================================================

    def _analyze_pnl(self, entry_price: float, current_price: float,
                      sl_price: float, tp_price: float,
                      is_buy: bool) -> Dict:
        """
        Analyze current P&L status.
        
        Returns:
            Dict with P&L metrics and status
        """
        # Calculate P&L in price units
        if is_buy:
            pnl = current_price - entry_price
            distance_to_sl = current_price - sl_price if sl_price > 0 else 0
            distance_to_tp = tp_price - current_price if tp_price > 0 else 0
            total_risk = entry_price - sl_price if sl_price > 0 else 0
            total_reward = tp_price - entry_price if tp_price > 0 else 0
        else:
            pnl = entry_price - current_price
            distance_to_sl = sl_price - current_price if sl_price > 0 else 0
            distance_to_tp = current_price - tp_price if tp_price > 0 else 0
            total_risk = sl_price - entry_price if sl_price > 0 else 0
            total_reward = entry_price - tp_price if tp_price > 0 else 0

        # Calculate percentages
        pnl_pct_of_risk = (pnl / total_risk * 100) if total_risk > 0 else 0
        progress_to_tp = (pnl / total_reward * 100) if total_reward > 0 else 0

        # Determine status
        if pnl < 0:
            if pnl_pct_of_risk <= -self.loss_threshold_pct:
                status = 'CRITICAL_LOSS'
            elif pnl_pct_of_risk <= -50:
                status = 'MODERATE_LOSS'
            else:
                status = 'MINOR_LOSS'
        elif pnl > 0:
            if progress_to_tp >= 100:
                status = 'TARGET_REACHED'
            elif progress_to_tp >= self.profit_target_pct:
                status = 'STRONG_PROFIT'
            elif progress_to_tp >= 25:
                status = 'MODERATE_PROFIT'
            else:
                status = 'MINOR_PROFIT'
        else:
            status = 'BREAKEVEN'

        return {
            'pnl': round(pnl, 2),
            'pnl_pct_of_risk': round(pnl_pct_of_risk, 1),
            'progress_to_tp': round(progress_to_tp, 1),
            'distance_to_sl': round(distance_to_sl, 2),
            'distance_to_tp': round(distance_to_tp, 2),
            'total_risk': round(total_risk, 2),
            'total_reward': round(total_reward, 2),
            'status': status
        }

    # =========================================================================
    # DIMENSION 2: MOMENTUM ANALYSIS
    # =========================================================================

    def _analyze_momentum(self, pos: Dict, current_price: float,
                            df_m5: pd.DataFrame, is_buy: bool) -> Dict:
        """
        Analyze if trade thesis is still valid based on momentum.
        
        Returns:
            Dict with momentum status
        """
        result = {
            'momentum_valid': True,
            'momentum_score': 50.0,
            'details': {}
        }

        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return result

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=close[0])

            # Calculate recent momentum (last 10 bars)
            if len(close) >= 10:
                recent_momentum = close[-1] - close[-10]
                prev_momentum = close[-10] - close[-20] if len(close) >= 20 else recent_momentum

                # Check if momentum reversed
                if is_buy:
                    if recent_momentum < 0 and prev_momentum > 0:
                        reversal_strength = abs(recent_momentum / (prev_momentum + 1e-10))
                        if reversal_strength > self.momentum_reversal_threshold:
                            result['momentum_valid'] = False
                            result['momentum_score'] = max(0, 50 - reversal_strength * 50)
                            result['details']['reversal_detected'] = True
                            result['details']['reversal_strength'] = round(reversal_strength, 2)
                    else:
                        # Momentum still positive
                        result['momentum_valid'] = True
                        result['momentum_score'] = min(100, 50 + (recent_momentum / (prev_momentum + 1e-10)) * 30)

                else:  # SELL position
                    if recent_momentum > 0 and prev_momentum < 0:
                        reversal_strength = abs(recent_momentum / (prev_momentum + 1e-10))
                        if reversal_strength > self.momentum_reversal_threshold:
                            result['momentum_valid'] = False
                            result['momentum_score'] = max(0, 50 - reversal_strength * 50)
                            result['details']['reversal_detected'] = True
                            result['details']['reversal_strength'] = round(reversal_strength, 2)
                    else:
                        result['momentum_valid'] = True
                        result['momentum_score'] = min(100, 50 + (-recent_momentum / (-prev_momentum + 1e-10)) * 30)

        except Exception as e:
            self.logger.debug(f"[POS_INTEL] Momentum analysis error: {e}")

        return result

    # =========================================================================
    # DIMENSION 3: RISK ANALYSIS
    # =========================================================================

    def _analyze_risk(self, entry_price: float, current_price: float,
                       sl_price: float, tp_price: float,
                       is_buy: bool) -> Dict:
        """
        Analyze current risk/reward status.
        
        Returns:
            Dict with risk metrics
        """
        # Calculate current R:R
        if is_buy:
            current_risk = current_price - sl_price if sl_price > 0 else 0
            current_reward = tp_price - current_price if tp_price > 0 else 0
        else:
            current_risk = sl_price - current_price if sl_price > 0 else 0
            current_reward = current_price - tp_price if tp_price > 0 else 0

        current_rr = current_reward / current_risk if current_risk > 0 else 0

        # Original R:R
        if is_buy:
            original_risk = entry_price - sl_price if sl_price > 0 else 0
            original_reward = tp_price - entry_price if tp_price > 0 else 0
        else:
            original_risk = sl_price - entry_price if sl_price > 0 else 0
            original_reward = entry_price - tp_price if tp_price > 0 else 0

        original_rr = original_reward / original_risk if original_risk > 0 else 0

        # Determine if R:R deteriorated
        rr_deteriorated = current_rr < original_rr * 0.7  # 30% deterioration

        # Determine status
        if current_rr >= 2.0:
            status = 'EXCELLENT'
        elif current_rr >= 1.5:
            status = 'GOOD'
        elif current_rr >= 1.0:
            status = 'ACCEPTABLE'
        elif current_rr >= 0.5:
            status = 'POOR'
        else:
            status = 'CRITICAL'

        return {
            'current_rr': round(current_rr, 2),
            'original_rr': round(original_rr, 2),
            'current_risk': round(current_risk, 2),
            'current_reward': round(current_reward, 2),
            'rr_deteriorated': rr_deteriorated,
            'status': status
        }

    # =========================================================================
    # DIMENSION 4: TIME ANALYSIS
    # =========================================================================

    def _analyze_time(self, pos: Dict) -> Dict:
        """
        Analyze time-based factors.
        
        Returns:
            Dict with time metrics
        """
        result = {
            'elapsed_minutes': 0,
            'time_status': 'NORMAL',
            'details': {}
        }

        try:
            setup_time_str = pos.get('setup_time') or pos.get('open_time')
            if not setup_time_str:
                return result

            setup_time = datetime.fromisoformat(setup_time_str)
            elapsed = (datetime.now() - setup_time).total_seconds() / 60.0

            result['elapsed_minutes'] = round(elapsed, 1)

            # Determine time status
            if elapsed < 15:
                result['time_status'] = 'TOO_EARLY'
            elif elapsed > self.time_threshold_minutes * 2:
                result['time_status'] = 'TOO_LONG'
            elif elapsed > self.time_threshold_minutes:
                result['time_status'] = 'APPROACHING_LIMIT'
            else:
                result['time_status'] = 'NORMAL'

        except Exception as e:
            self.logger.debug(f"[POS_INTEL] Time analysis error: {e}")

        return result

    # =========================================================================
    # DIMENSION 5: STRUCTURE ANALYSIS
    # =========================================================================

    def _analyze_structure(self, pos: Dict, current_price: float,
                            df_m5: pd.DataFrame, is_buy: bool) -> Dict:
        """
        Analyze if key structure is still intact.
        
        Returns:
            Dict with structure status
        """
        result = {
            'structure_intact': True,
            'structure_score': 70.0,
            'details': {}
        }

        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return result

        try:
            high = df_m5['high'].values.astype(float)
            low = df_m5['low'].values.astype(float)
            high = np.nan_to_num(high, nan=high[0])
            low = np.nan_to_num(low, nan=low[0])

            # Find recent swing points
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])

            if is_buy:
                # For BUY, check if structure support is intact
                # Structure broken if price is below recent swing low
                if current_price < recent_low * 0.998:  # 0.2% tolerance
                    result['structure_intact'] = False
                    result['structure_score'] = 30.0
                    result['details']['structure_broken'] = True
                    result['details']['broken_level'] = round(recent_low, 2)
                else:
                    result['structure_intact'] = True
                    result['structure_score'] = 80.0
            else:
                # For SELL, check if structure resistance is intact
                if current_price > recent_high * 1.002:
                    result['structure_intact'] = False
                    result['structure_score'] = 30.0
                    result['details']['structure_broken'] = True
                    result['details']['broken_level'] = round(recent_high, 2)
                else:
                    result['structure_intact'] = True
                    result['structure_score'] = 80.0

        except Exception as e:
            self.logger.debug(f"[POS_INTEL] Structure analysis error: {e}")

        return result

    # =========================================================================
    # DIMENSION 6: REGIME ALIGNMENT
    # =========================================================================

    def _analyze_regime_alignment(self, pos: Dict, regime_context: Dict,
                                    is_buy: bool) -> Dict:
        """
        Analyze if position still aligns with current regime.
        
        Returns:
            Dict with regime alignment status
        """
        result = {
            'regime_aligned': True,
            'alignment_score': 70.0,
            'details': {}
        }

        if regime_context is None:
            return result

        regime_name = regime_context.get('regime_name', 'UNKNOWN')
        regime_category = regime_context.get('regime_category', 'UNKNOWN')

        # Define regime alignment rules
        BULL_REGIMES = ['HEALTHY_UPTREND', 'QUIET_RALLY', 'CONSOLIDATING_BULL', 'OVERSOLD_BOUNCE']
        BEAR_REGIMES = ['HEALTHY_DOWNTREND', 'SLOW_BLEED', 'CONSOLIDATING_BEAR', 'EXHAUSTED_BULL']

        if is_buy:
            if regime_name in BEAR_REGIMES:
                result['regime_aligned'] = False
                result['alignment_score'] = 20.0
                result['details']['conflict'] = f'BUY position in {regime_name}'
            elif regime_name in BULL_REGIMES:
                result['regime_aligned'] = True
                result['alignment_score'] = 90.0
            else:
                result['regime_aligned'] = True
                result['alignment_score'] = 60.0
        else:  # SELL position
            if regime_name in BULL_REGIMES:
                result['regime_aligned'] = False
                result['alignment_score'] = 20.0
                result['details']['conflict'] = f'SELL position in {regime_name}'
            elif regime_name in BEAR_REGIMES:
                result['regime_aligned'] = True
                result['alignment_score'] = 90.0
            else:
                result['regime_aligned'] = True
                result['alignment_score'] = 60.0

        return result

    # =========================================================================
    # RECOMMENDATION GENERATION
    # =========================================================================

    def _generate_recommendation(self, pnl_analysis: Dict,
                                   momentum_analysis: Dict,
                                   risk_analysis: Dict,
                                   time_analysis: Dict,
                                   structure_analysis: Dict,
                                   regime_analysis: Dict) -> Dict:
        """
        Generate action recommendation based on all analyses.
        
        Returns:
            Dict with recommendation
        """
        action = self.ACTION_HOLD
        priority = self.PRIORITY_LOW
        reason = 'Position performing normally'
        details = {}

        # CRITICAL: Regime conflict
        if not regime_analysis.get('regime_aligned', True):
            action = self.ACTION_CLOSE
            priority = self.PRIORITY_CRITICAL
            reason = f"Regime conflict: {regime_analysis['details'].get('conflict', 'Unknown')}"
            return {
                'action': action,
                'priority': priority,
                'reason': reason,
                'details': details
            }

        # CRITICAL: Structure broken
        if not structure_analysis.get('structure_intact', True):
            action = self.ACTION_CLOSE
            priority = self.PRIORITY_CRITICAL
            reason = f"Structure broken at {structure_analysis['details'].get('broken_level', 0):.2f}"
            return {
                'action': action,
                'priority': priority,
                'reason': reason,
                'details': details
            }

        # HIGH: Critical loss
        if pnl_analysis.get('status') == 'CRITICAL_LOSS':
            action = self.ACTION_CLOSE
            priority = self.PRIORITY_HIGH
            reason = f"Critical loss: {pnl_analysis['pnl_pct_of_risk']:.1f}% of risk realized"
            return {
                'action': action,
                'priority': priority,
                'reason': reason,
                'details': details
            }

        # HIGH: Momentum reversed
        if not momentum_analysis.get('momentum_valid', True):
            action = self.ACTION_TIGHTEN_STOP
            priority = self.PRIORITY_HIGH
            reason = "Momentum reversal detected"
            details['momentum_score'] = momentum_analysis.get('momentum_score', 0)

        # MEDIUM: Target reached
        if pnl_analysis.get('status') == 'TARGET_REACHED':
            action = self.ACTION_CLOSE
            priority = self.PRIORITY_MEDIUM
            reason = "Profit target reached"
            details['progress_to_tp'] = pnl_analysis.get('progress_to_tp', 0)

        # MEDIUM: Strong profit
        elif pnl_analysis.get('status') == 'STRONG_PROFIT':
            action = self.ACTION_PARTIAL_CLOSE
            priority = self.PRIORITY_MEDIUM
            reason = f"Strong profit: {pnl_analysis['progress_to_tp']:.1f}% to target"
            details['progress_to_tp'] = pnl_analysis.get('progress_to_tp', 0)
            details['suggested_close_pct'] = 50

        # MEDIUM: R:R deteriorated
        if risk_analysis.get('rr_deteriorated', False):
            if action == self.ACTION_HOLD:
                action = self.ACTION_TIGHTEN_STOP
                priority = max(priority, self.PRIORITY_MEDIUM)
                reason = f"R:R deteriorated from {risk_analysis['original_rr']:.2f} to {risk_analysis['current_rr']:.2f}"

        # MEDIUM: Too long
        if time_analysis.get('time_status') == 'TOO_LONG':
            if action == self.ACTION_HOLD:
                action = self.ACTION_TIGHTEN_STOP
                priority = max(priority, self.PRIORITY_MEDIUM)
                reason = f"Position open too long: {time_analysis['elapsed_minutes']:.0f} minutes"

        return {
            'action': action,
            'priority': priority,
            'reason': reason,
            'details': details
        }

    # =========================================================================
    # HEALTH SCORE CALCULATION
    # =========================================================================

    def _calculate_health_score(self, pnl_analysis: Dict,
                                  momentum_analysis: Dict,
                                  risk_analysis: Dict,
                                  structure_analysis: Dict,
                                  regime_analysis: Dict) -> float:
        """
        Calculate overall position health score (0-100).
        
        Returns:
            Health score
        """
        # Weights
        pnl_weight = 0.25
        momentum_weight = 0.20
        risk_weight = 0.20
        structure_weight = 0.20
        regime_weight = 0.15

        # P&L score
        pnl_status = pnl_analysis.get('status', 'BREAKEVEN')
        pnl_scores = {
            'TARGET_REACHED': 100,
            'STRONG_PROFIT': 85,
            'MODERATE_PROFIT': 70,
            'MINOR_PROFIT': 60,
            'BREAKEVEN': 50,
            'MINOR_LOSS': 40,
            'MODERATE_LOSS': 25,
            'CRITICAL_LOSS': 10
        }
        pnl_score = pnl_scores.get(pnl_status, 50)

        # Momentum score
        momentum_score = momentum_analysis.get('momentum_score', 50)

        # Risk score
        risk_status = risk_analysis.get('status', 'ACCEPTABLE')
        risk_scores = {
            'EXCELLENT': 100,
            'GOOD': 80,
            'ACCEPTABLE': 60,
            'POOR': 40,
            'CRITICAL': 20
        }
        risk_score = risk_scores.get(risk_status, 60)

        # Structure score
        structure_score = structure_analysis.get('structure_score', 70)

        # Regime score
        regime_score = regime_analysis.get('alignment_score', 70)

        # Weighted average
        health_score = (
            pnl_score * pnl_weight +
            momentum_score * momentum_weight +
            risk_score * risk_weight +
            structure_score * structure_weight +
            regime_score * regime_weight
        )

        return round(max(0, min(100, health_score)), 1)

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def _generate_summary(self, position_analyses: Dict[int, Dict]) -> Dict:
        """
        Generate portfolio health summary.
        
        Returns:
            Dict with summary metrics
        """
        if not position_analyses:
            return {
                'total_positions': 0,
                'health': 'NO_POSITIONS',
                'avg_health_score': 0,
                'critical_count': 0,
                'healthy_count': 0
            }

        health_scores = [a['health_score'] for a in position_analyses.values()]
        avg_health = np.mean(health_scores)

        critical_count = sum(1 for a in position_analyses.values()
                            if a['health_score'] < 40)
        healthy_count = sum(1 for a in position_analyses.values()
                           if a['health_score'] >= 70)

        # Determine overall health
        if avg_health >= 75:
            health = 'EXCELLENT'
        elif avg_health >= 60:
            health = 'GOOD'
        elif avg_health >= 45:
            health = 'FAIR'
        elif avg_health >= 30:
            health = 'POOR'
        else:
            health = 'CRITICAL'

        return {
            'total_positions': len(position_analyses),
            'health': health,
            'avg_health_score': round(avg_health, 1),
            'min_health_score': round(min(health_scores), 1),
            'max_health_score': round(max(health_scores), 1),
            'critical_count': critical_count,
            'healthy_count': healthy_count
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def log_position_intelligence(self, result: Dict):
        """
        Log position intelligence results.
        
        Args:
            result: Result from analyze_all_positions
        """
        summary = result.get('summary', {})
        recommendations = result.get('recommendations', [])

        self.logger.info(
            f"[POS_INTEL] Portfolio Health: {summary.get('health', 'UNKNOWN')} | "
            f"Avg Score: {summary.get('avg_health_score', 0):.1f} | "
            f"Positions: {summary.get('total_positions', 0)} | "
            f"Critical: {summary.get('critical_count', 0)}"
        )

        # Log high-priority recommendations
        for rec in recommendations:
            if rec.get('priority', 5) <= self.PRIORITY_HIGH:
                self.logger.warning(
                    f"[POS_INTEL] Ticket {rec.get('ticket')} ({rec.get('strategy', 'Unknown')}) | "
                    f"Action: {rec.get('action')} | "
                    f"Priority: {rec.get('priority')} | "
                    f"Reason: {rec.get('reason')}"
                )

    def get_position_health(self, ticket: int) -> Optional[Dict]:
        """
        Get cached health analysis for a specific position.
        
        Args:
            ticket: Position ticket
            
        Returns:
            Health analysis dict or None
        """
        return self._analysis_cache.get(ticket)