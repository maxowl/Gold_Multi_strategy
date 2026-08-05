"""
Position Recommendation Engine
Generates actionable recommendations for position management.
Hardened: Safe access to all dictionary keys with fallback values.
"""
from typing import Dict, List
import logging


class PositionRecommendationEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_recommendations(self, position_analytics: List[Dict], 
                                 portfolio_analytics: Dict) -> List[Dict]:
        """
        Generate prioritized recommendations with safe dictionary access.
        """
        recommendations = []
        
        # =========================================================================
        # 1. CRITICAL: Close unhealthy positions
        # =========================================================================
        for pos_analytics in position_analytics:
            health_score = pos_analytics.get('health_score', 0)
            if health_score < 30:
                reasons = pos_analytics.get('reasons', [])
                recommendations.append({
                    'priority': 1,
                    'action': 'CLOSE',
                    'ticket': pos_analytics.get('ticket', 0),
                    'strategy': pos_analytics.get('strategy', 'Unknown'),
                    'position_type': pos_analytics.get('position_type', 'BUY'),
                    'reason': f"Low health score ({health_score:.0f}/100): " + 
                             ', '.join(reasons[:2]) if reasons else "Multiple issues detected",
                    'confidence': pos_analytics.get('confidence', 70),
                    'expected_impact': 'Reduce risk exposure'
                })
        
        # =========================================================================
        # 2. HIGH: Reduce concentrated positions
        # =========================================================================
        concentration_risk = portfolio_analytics.get('concentration_risk', 0)
        if concentration_risk > 0.7 and position_analytics:
            # Find largest position by volume
            largest_pos = max(position_analytics, key=lambda x: x.get('volume', 0))
            volume = largest_pos.get('volume', 0)
            volume_pct = largest_pos.get('volume_pct', 0)
            
            recommendations.append({
                'priority': 2,
                'action': 'REDUCE',
                'ticket': largest_pos.get('ticket', 0),
                'strategy': largest_pos.get('strategy', 'Unknown'),
                'position_type': largest_pos.get('position_type', 'BUY'),
                'reason': f"High concentration risk ({concentration_risk:.0%}). " +
                         f"Position is {volume:.2f} lots ({volume_pct:.0%} of portfolio)",
                'confidence': 80,
                'expected_impact': 'Improve diversification'
            })
        
        # =========================================================================
        # 3. MEDIUM: Time-stalled positions
        # =========================================================================
        for pos_analytics in position_analytics:
            time_efficiency = pos_analytics.get('time_efficiency', 1.0)
            health_score = pos_analytics.get('health_score', 0)
            hours_open = pos_analytics.get('hours_open', 0)
            current_rr = pos_analytics.get('risk_reward_current', 0)
            
            if time_efficiency < 0.3 and health_score < 60:
                recommendations.append({
                    'priority': 3,
                    'action': 'CLOSE',
                    'ticket': pos_analytics.get('ticket', 0),
                    'strategy': pos_analytics.get('strategy', 'Unknown'),
                    'position_type': pos_analytics.get('position_type', 'BUY'),
                    'reason': f"Time stall: Open for {hours_open:.1f}h " +
                             f"with only {current_rr:.1f}R profit",
                    'confidence': 70,
                    'expected_impact': 'Free up capital'
                })
        
        # =========================================================================
        # 4. LOW: Regime-conflicted positions
        # =========================================================================
        for pos_analytics in position_analytics:
            regime_alignment = pos_analytics.get('regime_alignment', 1.0)
            if regime_alignment < 0.3:
                recommendations.append({
                    'priority': 4,
                    'action': 'CLOSE',
                    'ticket': pos_analytics.get('ticket', 0),
                    'strategy': pos_analytics.get('strategy', 'Unknown'),
                    'position_type': pos_analytics.get('position_type', 'BUY'),
                    'reason': f"Regime conflict: Position against current market regime",
                    'confidence': 75,
                    'expected_impact': 'Align with market direction'
                })
        
        # =========================================================================
        # 5. OPTIONAL: Add to strong positions
        # =========================================================================
        for pos_analytics in position_analytics:
            health_score = pos_analytics.get('health_score', 0)
            current_rr = pos_analytics.get('risk_reward_current', 0)
            
            if (health_score > 80 and 
                current_rr > 1.5 and
                concentration_risk < 0.5):
                recommendations.append({
                    'priority': 5,
                    'action': 'ADD',
                    'ticket': pos_analytics.get('ticket', 0),
                    'strategy': pos_analytics.get('strategy', 'Unknown'),
                    'position_type': pos_analytics.get('position_type', 'BUY'),
                    'reason': f"Strong performer: +{current_rr:.1f}R " +
                             f"with health score {health_score:.0f}/100",
                    'confidence': pos_analytics.get('confidence', 70),
                    'expected_impact': 'Increase profit potential'
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations