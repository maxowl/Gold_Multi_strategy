"""
Portfolio Correlation Analyzer
Analyzes correlation and concentration risk across active positions
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging


class PortfolioCorrelationAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_portfolio(self, positions: List[Dict], current_prices: Dict[int, float]) -> Dict:
        """
        Analyze portfolio-level metrics
        
        Returns:
            {
                'total_exposure': float,
                'concentration_risk': float (0-1),
                'directional_bias': 'BULL' | 'BEAR' | 'NEUTRAL',
                'correlation_matrix': Dict,
                'diversification_score': float (0-100),
                'risk_summary': Dict
            }
        """
        if not positions:
            return {
                'total_exposure': 0,
                'concentration_risk': 0,
                'directional_bias': 'NEUTRAL',
                'diversification_score': 100,
                'risk_summary': {'message': 'No active positions'}
            }
        
        # =========================================================================
        # 1. TOTAL EXPOSURE
        # =========================================================================
        total_volume = sum([float(p.get('volume') or 0) for p in positions])
        
        total_risk = 0.0
        for p in positions:
            vol = float(p.get('volume') or 0)
            entry = float(p.get('entry_price') or 0)
            sl = float(p.get('sl') or 0)
            
            # Fallback if SL is missing
            if sl == 0.0 and entry > 0:
                sl = entry * 0.98 if p.get('position_type') == 'BUY' else entry * 1.02
                
            risk_dist = abs(entry - sl)
            total_risk += (vol * risk_dist * 100)
        
        # =========================================================================
        # 2. DIRECTIONAL BIAS
        # =========================================================================
        buy_volume = sum([p.get('volume', 0) for p in positions if p.get('position_type') == 'BUY'])
        sell_volume = sum([p.get('volume', 0) for p in positions if p.get('position_type') == 'SELL'])
        
        if buy_volume > sell_volume * 1.5:
            directional_bias = 'BULL'
        elif sell_volume > buy_volume * 1.5:
            directional_bias = 'BEAR'
        else:
            directional_bias = 'NEUTRAL'
        
        # =========================================================================
        # 3. CONCENTRATION RISK
        # =========================================================================
        # Herfindahl-Hirschman Index (HHI) for concentration
        if total_volume > 0:
            volume_shares = [p.get('volume', 0) / total_volume for p in positions]
            hhi = sum([share ** 2 for share in volume_shares])
            # Normalize: HHI = 1/n for equal distribution, 1 for single position
            concentration_risk = min(1.0, hhi * len(positions))
        else:
            concentration_risk = 0
        
        # =========================================================================
        # 4. STRATEGY DIVERSIFICATION
        # =========================================================================
        strategies = [p.get('strategy', 'Unknown') for p in positions]
        unique_strategies = len(set(strategies))
        diversification_score = (unique_strategies / len(positions)) * 100 if positions else 100
        
        # =========================================================================
        # 5. CORRELATION ANALYSIS
        # =========================================================================
        correlation_matrix = self._calculate_strategy_correlations(positions)
        
        # =========================================================================
        # 6. RISK SUMMARY
        # =========================================================================
        risk_summary = {
            'total_positions': len(positions),
            'total_volume': total_volume,
            'total_risk_usd': total_risk,
            'buy_positions': sum(1 for p in positions if p.get('position_type') == 'BUY'),
            'sell_positions': sum(1 for p in positions if p.get('position_type') == 'SELL'),
            'avg_health_score': 0  # Will be updated by PositionHealthAnalyzer
        }
        
        return {
            'total_exposure': total_volume,
            'concentration_risk': concentration_risk,
            'directional_bias': directional_bias,
            'correlation_matrix': correlation_matrix,
            'diversification_score': diversification_score,
            'risk_summary': risk_summary
        }
    
    def _calculate_strategy_correlations(self, positions: List[Dict]) -> Dict:
        """
        Calculate correlation between positions based on strategy categories
        """
        if len(positions) < 2:
            return {}
        
        categories = {
            'TREND': ['S3', 'S10', 'S12', 'S13', 'S14', 'S17', 'S20', 'S24', 'S25'],
            'MEAN_REVERSION': ['S6', 'S8', 'S15', 'S16', 'S18', 'S22'],
            'SMC': ['S1', 'S4', 'S5', 'S7', 'S21'],
            'SCALP': ['S2', 'S9', 'S11', 'S19', 'S23']
        }
        
        # Count positions per category
        category_counts = {}
        for pos in positions:
            strategy = pos.get('strategy', '')
            for cat, strategies in categories.items():
                if any(strategy.startswith(s) for s in strategies):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    break
        
        # High correlation = many positions in same category
        max_count = max(category_counts.values()) if category_counts else 0
        total = len(positions)
        
        correlation_level = 'LOW'
        if max_count >= total * 0.7:
            correlation_level = 'HIGH'
        elif max_count >= total * 0.5:
            correlation_level = 'MEDIUM'
        
        return {
            'category_distribution': category_counts,
            'correlation_level': correlation_level,
            'max_concentration': max_count
        }