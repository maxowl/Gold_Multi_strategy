"""
Position Intelligence Manager
Integrates all position analytics into OrderManager.
Fixed: Added volume_pct calculation for each position.
"""
from core.position_health_analyzer import PositionHealthAnalyzer
from core.portfolio_correlation_analyzer import PortfolioCorrelationAnalyzer
from core.position_recommendation_engine import PositionRecommendationEngine
from typing import Dict, List
import logging


class PositionIntelligenceManager:
    def __init__(self):
        self.health_analyzer = PositionHealthAnalyzer()
        self.correlation_analyzer = PortfolioCorrelationAnalyzer()
        self.recommendation_engine = PositionRecommendationEngine()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_all_positions(self, positions: List[Dict], current_prices: Dict[int, float],
                            df_m5, regime_context: Dict) -> Dict:
        """
        Comprehensive analysis of all active positions.
        """
        if not positions:
            return {
                'position_analytics': [],
                'portfolio_analytics': {},
                'recommendations': [],
                'summary': 'No active positions'
            }
        
        # =========================================================================
        # 0. CALCULATE TOTAL VOLUME FOR PERCENTAGE CALCULATION
        # =========================================================================
        total_volume = sum([float(p.get('volume') or 0) for p in positions])
        
        # =========================================================================
        # 1. ANALYZE EACH POSITION
        # =========================================================================
        position_analytics = []
        for pos in positions:
            ticket = pos.get('ticket')
            current_price = current_prices.get(ticket, pos.get('entry_price', 0))
            
            analytics = self.health_analyzer.analyze_position_health(
                pos, current_price, df_m5, regime_context
            )
            
            # Add position metadata
            analytics['ticket'] = ticket
            analytics['strategy'] = pos.get('strategy', 'Unknown')
            analytics['position_type'] = pos.get('position_type', 'BUY')
            
            volume = float(pos.get('volume') or 0)
            analytics['volume'] = volume
            
            # [FIX] Calculate volume percentage
            if total_volume > 0:
                analytics['volume_pct'] = volume / total_volume
            else:
                analytics['volume_pct'] = 0.0
            
            entry_price = float(pos.get('entry_price') or 0)
            analytics['entry_price'] = entry_price
            analytics['current_price'] = current_price
            
            # Calculate PnL in USD (volume × 100 oz per lot for XAUUSD)
            pnl_usd = (current_price - entry_price) * volume * 100 if pos.get('position_type') == 'BUY' \
                      else (entry_price - current_price) * volume * 100
            analytics['pnl_usd'] = pnl_usd
            
            position_analytics.append(analytics)
        
        # =========================================================================
        # 2. ANALYZE PORTFOLIO
        # =========================================================================
        portfolio_analytics = self.correlation_analyzer.analyze_portfolio(positions, current_prices)
        
        # Update with average health score
        if position_analytics:
            avg_health = sum([p['health_score'] for p in position_analytics]) / len(position_analytics)
            portfolio_analytics['risk_summary']['avg_health_score'] = avg_health
            portfolio_analytics['risk_summary']['total_pnl_usd'] = sum([p['pnl_usd'] for p in position_analytics])
        
        # =========================================================================
        # 3. GENERATE RECOMMENDATIONS
        # =========================================================================
        recommendations = self.recommendation_engine.generate_recommendations(
            position_analytics, portfolio_analytics
        )
        
        # =========================================================================
        # 4. GENERATE SUMMARY
        # =========================================================================
        summary = self._generate_summary(position_analytics, portfolio_analytics, recommendations)
        
        return {
            'position_analytics': position_analytics,
            'portfolio_analytics': portfolio_analytics,
            'recommendations': recommendations,
            'summary': summary
        }
    
    def _generate_summary(self, position_analytics: List[Dict], 
                         portfolio_analytics: Dict,
                         recommendations: List[Dict]) -> str:
        """Generate human-readable summary."""
        
        total_positions = len(position_analytics)
        avg_health = portfolio_analytics['risk_summary'].get('avg_health_score', 0)
        total_pnl = portfolio_analytics['risk_summary'].get('total_pnl_usd', 0)
        
        # Count by recommendation
        close_count = sum(1 for r in recommendations if r['action'] == 'CLOSE')
        reduce_count = sum(1 for r in recommendations if r['action'] == 'REDUCE')
        add_count = sum(1 for r in recommendations if r['action'] == 'ADD')
        
        summary_parts = [
            f"Portfolio: {total_positions} positions",
            f"Avg Health: {avg_health:.0f}/100",
            f"Total PnL: ${total_pnl:+.2f}",
            f"Bias: {portfolio_analytics.get('directional_bias', 'NEUTRAL')}",
            f"Concentration: {portfolio_analytics.get('concentration_risk', 0):.0%}"
        ]
        
        if recommendations:
            summary_parts.append(f"Actions: {close_count} CLOSE, {reduce_count} REDUCE, {add_count} ADD")
        
        return " | ".join(summary_parts)
    
    def log_position_intelligence(self, analysis: Dict):
        """Log position intelligence summary."""
        self.logger.info("=" * 80)
        self.logger.info("[POSITION INTELLIGENCE]")
        self.logger.info(f"Summary: {analysis['summary']}")
        
        # Log top 3 recommendations
        if analysis['recommendations']:
            self.logger.info("\nTop Recommendations:")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                self.logger.info(
                    f"  {i}. [{rec['action']}] Ticket {rec['ticket']} ({rec['strategy']}) - "
                    f"{rec['reason']}"
                )
        
        # Log position health summary
        if analysis['position_analytics']:
            self.logger.info("\nPosition Health:")
            for pos in sorted(analysis['position_analytics'], key=lambda x: x['health_score'], reverse=True):
                status = "OK" if pos['health_score'] >= 60 else "WARN" if pos['health_score'] >= 40 else "BAD"
                self.logger.info(
                    f"  [{status}] Ticket {pos['ticket']} ({pos['strategy']}): "
                    f"{pos['health_score']:.0f}/100 | {pos['risk_reward_current']:+.1f}R | "
                    f"Vol: {pos['volume']:.2f} ({pos['volume_pct']:.0%}) | "
                    f"PnL: ${pos['pnl_usd']:+.2f}"
                )
        
        self.logger.info("=" * 80)