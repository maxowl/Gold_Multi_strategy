"""
Enhanced Strategy Allocator - 18-Regime Weight Distribution System.

Maps 30 strategies to 18 market regimes with institutional-grade precision.
Integrates Kelly Criterion multipliers for optimal capital allocation.
Includes backward compatibility wrapper for legacy code.
"""
import logging
from typing import Dict, List


class EnhancedStrategyAllocator:
    """
    18-Regime Strategy Weight Allocator.
    
    Allocates weights to 30 strategies based on current regime combination:
      - Trend Direction (BULL/BEAR/SIDEWAY)
      - Volatility Level (LOW/NORMAL/HIGH)
      - Fractal Behavior (TRENDING/MEAN_REVERTING)
    """

    # Strategy Category Mapping (30 strategies)
    STRATEGY_CATEGORIES = {
        # TREND-Following Strategies (10)
        'S3_EMD_HHT': 'TREND',
        'S10_EhlersMESA': 'TREND',
        'S12_PCA_Cycle': 'TREND',
        'S13_TMF_EOM': 'TREND',
        'S14_Propulsion': 'TREND',
        'S17_ChaoSqueeze': 'TREND',
        'S20_VFIAccumulation': 'TREND',
        'S24_KalmanMomentum': 'TREND',
        'S25_HurstWavelet': 'TREND',
        'S28_MTF_Confluence': 'TREND',  # NEW

        # MEAN-REVERSION Strategies (8)
        'S6_QuantumPDF': 'MEAN_REVERSION',
        'S8_GPR_Vol': 'MEAN_REVERSION',
        'S15_HFT_StatArb': 'MEAN_REVERSION',
        'S16_RoofingEMD': 'MEAN_REVERSION',
        'S18_EhlersVector': 'MEAN_REVERSION',
        'S22_WyckoffSpring': 'MEAN_REVERSION',
        'S27_VWAP_MeanReversion': 'MEAN_REVERSION',  # NEW
        'S29_QuantumMomentum': 'MEAN_REVERSION',  # NEW

        # SMC (Smart Money Concepts) Strategies (6)
        'S1_IOB_Rejection': 'SMC',
        'S4_CHOCH_IDM': 'SMC',
        'S5_Breaker_Void': 'SMC',
        'S7_MacroFVG': 'SMC',
        'S21_BreakerFVGPOC': 'SMC',
        'S30_VolumeProfileReversal': 'SMC',  # NEW

        # SCALP Strategies (6)
        'S2_VI_Sweep': 'SCALP',
        'S9_SessionSweep': 'SCALP',
        'S11_LiquidityDelta': 'SCALP',
        'S19_VoidReversal': 'SCALP',
        'S23_MidnightJudas': 'SCALP',
        'S26_Microstructure': 'SCALP',  # NEW
    }

    # 18-Regime Weight Mapping (Institutional Standard)
    REGIME_WEIGHTS = {
        # BULL TREND REGIMES
        'QUIET_RALLY': {
            'TREND': 0.70, 'MEAN_REVERSION': 0.05, 'SMC': 0.15, 'SCALP': 0.10
        },
        'ANOMALY_BULL': {
            'TREND': 0.00, 'MEAN_REVERSION': 0.60, 'SMC': 0.10, 'SCALP': 0.30
        },
        'HEALTHY_UPTREND': {
            'TREND': 0.50, 'MEAN_REVERSION': 0.05, 'SMC': 0.35, 'SCALP': 0.10
        },
        'CONSOLIDATING_BULL': {
            'TREND': 0.10, 'MEAN_REVERSION': 0.45, 'SMC': 0.25, 'SCALP': 0.20
        },
        'PARABOLIC_RALLY': {
            'TREND': 0.20, 'MEAN_REVERSION': 0.10, 'SMC': 0.10, 'SCALP': 0.60
        },
        'EXHAUSTED_BULL': {
            'TREND': 0.00, 'MEAN_REVERSION': 0.35, 'SMC': 0.15, 'SCALP': 0.50
        },

        # SIDEWAY REGIMES
        'PRE_BREAKOUT': {
            'TREND': 0.15, 'MEAN_REVERSION': 0.15, 'SMC': 0.50, 'SCALP': 0.20
        },
        'TIGHT_RANGE': {
            'TREND': 0.05, 'MEAN_REVERSION': 0.55, 'SMC': 0.15, 'SCALP': 0.25
        },
        'FALSE_SIDEWAY': {
            'TREND': 0.40, 'MEAN_REVERSION': 0.10, 'SMC': 0.25, 'SCALP': 0.25
        },
        'CLASSIC_RANGE': {
            'TREND': 0.05, 'MEAN_REVERSION': 0.50, 'SMC': 0.20, 'SCALP': 0.25
        },
        'VOLATILE_CHOP': {
            'TREND': 0.00, 'MEAN_REVERSION': 0.20, 'SMC': 0.20, 'SCALP': 0.60
        },
        'WHIPSAW_MARKET': {
            'TREND': 0.00, 'MEAN_REVERSION': 0.25, 'SMC': 0.15, 'SCALP': 0.60
        },

        # BEAR TREND REGIMES
        'SLOW_BLEED': {
            'TREND': 0.70, 'MEAN_REVERSION': 0.05, 'SMC': 0.15, 'SCALP': 0.10
        },
        'ANOMALY_BEAR': {
            'TREND': 0.00, 'MEAN_REVERSION': 0.60, 'SMC': 0.10, 'SCALP': 0.30
        },
        'HEALTHY_DOWNTREND': {
            'TREND': 0.50, 'MEAN_REVERSION': 0.05, 'SMC': 0.35, 'SCALP': 0.10
        },
        'CONSOLIDATING_BEAR': {
            'TREND': 0.10, 'MEAN_REVERSION': 0.45, 'SMC': 0.25, 'SCALP': 0.20
        },
        'PANIC_CAPITULATION': {
            'TREND': 0.15, 'MEAN_REVERSION': 0.10, 'SMC': 0.10, 'SCALP': 0.65
        },
        'OVERSOLD_BOUNCE': {
            'TREND': 0.05, 'MEAN_REVERSION': 0.50, 'SMC': 0.25, 'SCALP': 0.20
        },
    }

    # Regime-specific strategy adjustments
    REGIME_STRATEGY_OVERRIDES = {
        'QUIET_RALLY': {
            'S10_EhlersMESA': 1.3, 'S24_KalmanMomentum': 1.3, 'S25_HurstWavelet': 1.3,
            'S28_MTF_Confluence': 1.3  # NEW
        },
        'ANOMALY_BULL': {
            'S6_QuantumPDF': 1.2, 'S8_GPR_Vol': 1.2,
            'S27_VWAP_MeanReversion': 1.2, 'S29_QuantumMomentum': 1.2  # NEW
        },
        'HEALTHY_UPTREND': {
            'S3_EMD_HHT': 1.2, 'S14_Propulsion': 1.2,
            'S1_IOB_Rejection': 1.3, 'S4_CHOCH_IDM': 1.3, 'S5_Breaker_Void': 1.3,
            'S30_VolumeProfileReversal': 1.3, 'S28_MTF_Confluence': 1.2  # NEW
        },
        'PARABOLIC_RALLY': {
            'S2_VI_Sweep': 1.3, 'S9_SessionSweep': 1.3, 'S11_LiquidityDelta': 1.3,
            'S26_Microstructure': 1.3  # NEW
        },
        'EXHAUSTED_BULL': {
            'S17_ChaoSqueeze': 1.2, 'S22_WyckoffSpring': 1.2,
            'S29_QuantumMomentum': 1.2  # NEW
        },
        'PRE_BREAKOUT': {
            'S17_ChaoSqueeze': 1.4, 'S5_Breaker_Void': 1.3, 'S21_BreakerFVGPOC': 1.3,
            'S30_VolumeProfileReversal': 1.3  # NEW
        },
        'TIGHT_RANGE': {
            'S6_QuantumPDF': 1.3, 'S8_GPR_Vol': 1.3, 'S15_HFT_StatArb': 1.3,
            'S16_RoofingEMD': 1.3, 'S18_EhlersVector': 1.3,
            'S27_VWAP_MeanReversion': 1.3, 'S29_QuantumMomentum': 1.3  # NEW
        },
        'FALSE_SIDEWAY': {
            'S14_Propulsion': 1.3, 'S3_EMD_HHT': 1.2, 'S12_PCA_Cycle': 1.2,
            'S28_MTF_Confluence': 1.2  # NEW
        },
        'CLASSIC_RANGE': {
            'S6_QuantumPDF': 1.2, 'S9_SessionSweep': 1.2, 'S19_VoidReversal': 1.2,
            'S27_VWAP_MeanReversion': 1.2  # NEW
        },
        'VOLATILE_CHOP': {
            'S4_CHOCH_IDM': 1.3, 'S19_VoidReversal': 1.3,
            'S26_Microstructure': 1.3  # NEW
        },
        'WHIPSAW_MARKET': {
            'S9_SessionSweep': 1.3, 'S22_WyckoffSpring': 1.3, 'S23_MidnightJudas': 1.3,
            'S26_Microstructure': 1.3  # NEW
        },
        'SLOW_BLEED': {
            'S10_EhlersMESA': 1.3, 'S24_KalmanMomentum': 1.3, 'S25_HurstWavelet': 1.3,
            'S28_MTF_Confluence': 1.3  # NEW
        },
        'ANOMALY_BEAR': {
            'S6_QuantumPDF': 1.2, 'S8_GPR_Vol': 1.2,
            'S27_VWAP_MeanReversion': 1.2, 'S29_QuantumMomentum': 1.2  # NEW
        },
        'HEALTHY_DOWNTREND': {
            'S3_EMD_HHT': 1.2, 'S14_Propulsion': 1.2,
            'S1_IOB_Rejection': 1.3, 'S4_CHOCH_IDM': 1.3, 'S5_Breaker_Void': 1.3,
            'S30_VolumeProfileReversal': 1.3, 'S28_MTF_Confluence': 1.2  # NEW
        },
        'PANIC_CAPITULATION': {
            'S2_VI_Sweep': 1.3, 'S11_LiquidityDelta': 1.3, 'S19_VoidReversal': 1.3,
            'S26_Microstructure': 1.3  # NEW
        },
        'OVERSOLD_BOUNCE': {
            'S22_WyckoffSpring': 1.4, 'S9_SessionSweep': 1.3, 'S17_ChaoSqueeze': 1.2,
            'S29_QuantumMomentum': 1.3  # NEW
        },
    }

    # Blocked strategies per regime (force weight to 0)
    BLOCKED_STRATEGIES = {
        'ANOMALY_BULL': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
        'EXHAUSTED_BULL': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
        'VOLATILE_CHOP': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
        'WHIPSAW_MARKET': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
        'ANOMALY_BEAR': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
        'PANIC_CAPITULATION': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S14_Propulsion', 'S24_KalmanMomentum', 'S25_HurstWavelet', 'S28_MTF_Confluence'],
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def allocate_weights_18_regime(self, regime_name: str, kelly_multiplier: float,
                                     strategy_signals: Dict[str, dict]) -> Dict[str, float]:
        """
        Allocate weights to strategies based on 18-regime classification.
        
        Args:
            regime_name: One of 18 regime names (e.g., 'HEALTHY_UPTREND')
            kelly_multiplier: Kelly Criterion multiplier from RegimeRouter
            strategy_signals: Dict of strategy_name -> signal_dict
            
        Returns:
            Dict mapping strategy_name to weight (0.0-1.0)
        """
        weights = {}

        # Get base category weights for this regime
        regime_weights = self.REGIME_WEIGHTS.get(regime_name, self.REGIME_WEIGHTS['CLASSIC_RANGE'])

        # Get strategy overrides for this regime
        overrides = self.REGIME_STRATEGY_OVERRIDES.get(regime_name, {})

        # Get blocked strategies for this regime
        blocked = self.BLOCKED_STRATEGIES.get(regime_name, [])

        # Group strategies by category
        categories = {}
        for strategy_name, category in self.STRATEGY_CATEGORIES.items():
            if category not in categories:
                categories[category] = []
            categories[category].append(strategy_name)

        # Distribute weights within each category
        for category, strategies in categories.items():
            category_weight = regime_weights.get(category, 0.1)

            if not strategies:
                continue

            # Base weight per strategy in this category
            base_weight = category_weight / len(strategies)

            for strategy_name in strategies:
                # Check if blocked
                if strategy_name in blocked:
                    weights[strategy_name] = 0.0
                    continue

                # Apply override multiplier
                override_mult = overrides.get(strategy_name, 1.0)
                adjusted_weight = base_weight * override_mult

                # Apply signal quality adjustment
                if strategy_name in strategy_signals:
                    signal_conf = strategy_signals[strategy_name].get('meta', {}).get('confidence', 0.5)
                    adjusted_weight *= (0.5 + signal_conf)
                else:
                    adjusted_weight *= 0.5  # Reduce weight for strategies without signals

                # Apply Kelly multiplier
                final_weight = adjusted_weight * kelly_multiplier

                # Clamp to [0.0, 1.0]
                final_weight = max(0.0, min(1.0, final_weight))

                weights[strategy_name] = round(final_weight, 4)

        return weights

    def get_recommended_strategies(self, weights: Dict[str, float],
                                     min_weight: float = 0.05) -> List[str]:
        """Get list of recommended strategies based on weights."""
        recommended = [name for name, weight in weights.items() if weight >= min_weight]
        recommended.sort(key=lambda x: weights.get(x, 0.0), reverse=True)
        return recommended

    def get_blocked_strategies(self, regime_name: str) -> List[str]:
        """Get list of blocked strategies for a regime."""
        return self.BLOCKED_STRATEGIES.get(regime_name, [])

    def get_regime_analysis(self, regime_name: str) -> Dict:
        """Get detailed analysis of a regime's strategy allocation."""
        regime_weights = self.REGIME_WEIGHTS.get(regime_name, {})
        overrides = self.REGIME_STRATEGY_OVERRIDES.get(regime_name, {})
        blocked = self.BLOCKED_STRATEGIES.get(regime_name, [])

        # Count strategies per category
        category_counts = {}
        for strategy_name, category in self.STRATEGY_CATEGORIES.items():
            category_counts[category] = category_counts.get(category, 0) + 1

        # Calculate expected active strategies
        active_count = 0
        for strategy_name in self.STRATEGY_CATEGORIES.keys():
            if strategy_name not in blocked:
                active_count += 1

        return {
            'regime': regime_name,
            'category_weights': regime_weights,
            'strategy_overrides': overrides,
            'blocked_strategies': blocked,
            'category_counts': category_counts,
            'expected_active_strategies': active_count,
            'total_strategies': len(self.STRATEGY_CATEGORIES)
        }

    def compare_regimes(self, regime1: str, regime2: str) -> Dict:
        """Compare strategy allocation between two regimes."""
        weights1 = self.allocate_weights_18_regime(regime1, 1.0, {})
        weights2 = self.allocate_weights_18_regime(regime2, 1.0, {})

        # Find strategies that differ significantly
        differences = {}
        for strategy_name in self.STRATEGY_CATEGORIES.keys():
            w1 = weights1.get(strategy_name, 0.0)
            w2 = weights2.get(strategy_name, 0.0)
            diff = w2 - w1

            if abs(diff) > 0.1:  # Significant difference
                differences[strategy_name] = {
                    'regime1_weight': w1,
                    'regime2_weight': w2,
                    'change': diff,
                    'direction': 'INCREASE' if diff > 0 else 'DECREASE'
                }

        return {
            'regime1': regime1,
            'regime2': regime2,
            'significant_differences': differences,
            'num_changed_strategies': len(differences)
        }


# =========================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =========================================================================

class StrategyAllocator(EnhancedStrategyAllocator):
    """
    Backward-compatible wrapper for legacy code.
    
    Inherits all functionality from EnhancedStrategyAllocator
    and adds legacy method for old 3-regime system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def allocate_weights(self, regime_id: int, confidence: float,
                          strategy_signals: Dict[str, dict]) -> Dict[str, float]:
        """
        Legacy method for backward compatibility.
        
        Maps old 3-regime system to new 18-regime system.
        
        Args:
            regime_id: Old regime ID (0=BULL, 1=BEAR, 2=SIDEWAY)
            confidence: Confidence score (0.0-1.0)
            strategy_signals: Dict of strategy signals
            
        Returns:
            Dict mapping strategy_name to weight
        """
        # Map old 3-regime system to new 18-regime
        legacy_mapping = {
            0: 'HEALTHY_UPTREND',  # BULL_TREND
            1: 'HEALTHY_DOWNTREND',  # BEAR_TREND
            2: 'CLASSIC_RANGE'  # SIDEWAY
        }

        regime_name = legacy_mapping.get(regime_id, 'CLASSIC_RANGE')
        kelly_mult = confidence * 2.0  # Scale confidence to Kelly multiplier

        return self.allocate_weights_18_regime(regime_name, kelly_mult, strategy_signals)

    def get_weights_for_regime(self, regime_name: str,
                                 strategy_signals: Dict[str, dict] = None) -> Dict[str, float]:
        """
        Convenience method for getting weights with default Kelly multiplier.
        """
        if strategy_signals is None:
            strategy_signals = {}
        return self.allocate_weights_18_regime(regime_name, 1.0, strategy_signals)