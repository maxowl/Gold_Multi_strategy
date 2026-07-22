"""
Enhanced Strategy Allocator.
Assigns weights to strategies based on 18-regime system and Kelly multiplier.
"""
import logging
from typing import Dict


class EnhancedStrategyAllocator:
    """
    Allocates position weights to strategies based on current market regime.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # 18-Regime Weight Configuration
    REGIME_WEIGHTS = {
        # Healthy trends
        'HEALTHY_UPTREND': {'TREND': 0.6, 'SMC': 0.2, 'MEAN_REVERSION': 0.1, 'SCALP': 0.1},
        'HEALTHY_DOWNTREND': {'TREND': 0.6, 'SMC': 0.2, 'MEAN_REVERSION': 0.1, 'SCALP': 0.1},
        
        # Quiet trends
        'QUIET_RALLY': {'TREND': 0.5, 'SMC': 0.2, 'MEAN_REVERSION': 0.2, 'SCALP': 0.1},
        'SLOW_BLEED': {'TREND': 0.5, 'SMC': 0.2, 'MEAN_REVERSION': 0.2, 'SCALP': 0.1},
        
        # Ranges
        'TIGHT_RANGE': {'MEAN_REVERSION': 0.5, 'SCALP': 0.3, 'SMC': 0.15, 'TREND': 0.05},
        'CLASSIC_RANGE': {'MEAN_REVERSION': 0.4, 'SCALP': 0.3, 'SMC': 0.2, 'TREND': 0.1},
        'WIDE_RANGE': {'MEAN_REVERSION': 0.3, 'SCALP': 0.4, 'SMC': 0.2, 'TREND': 0.1},
        
        # Reversals
        'BOUNCE_REVERSAL': {'SMC': 0.5, 'MEAN_REVERSION': 0.3, 'TREND': 0.1, 'SCALP': 0.1},
        'EXHAUSTED_REVERSAL': {'MEAN_REVERSION': 0.5, 'SMC': 0.3, 'TREND': 0.1, 'SCALP': 0.1},
        
        # Volatile
        'CHOPPY_HIGH_VOL': {'SCALP': 0.5, 'MEAN_REVERSION': 0.3, 'SMC': 0.1, 'TREND': 0.1},
        'WHIPSAW': {'SCALP': 0.4, 'MEAN_REVERSION': 0.4, 'SMC': 0.1, 'TREND': 0.1},
        
        # Extreme
        'PARABOLIC_UP': {'MEAN_REVERSION': 0.6, 'SCALP': 0.3, 'SMC': 0.1, 'TREND': 0.0},
        'PARABOLIC_DOWN': {'MEAN_REVERSION': 0.6, 'SCALP': 0.3, 'SMC': 0.1, 'TREND': 0.0},
        'PANIC_SELL': {'MEAN_REVERSION': 0.5, 'SCALP': 0.4, 'SMC': 0.1, 'TREND': 0.0},
        'FOMO_BUY': {'MEAN_REVERSION': 0.5, 'SCALP': 0.4, 'SMC': 0.1, 'TREND': 0.0},
        
        # Anomalies
        'LIQUIDITY_ANOMALY': {'SCALP': 0.6, 'SMC': 0.3, 'MEAN_REVERSION': 0.1, 'TREND': 0.0},
        'VOLUME_ANOMALY': {'SCALP': 0.5, 'SMC': 0.3, 'MEAN_REVERSION': 0.2, 'TREND': 0.0},
        
        # Unknown
        'UNKNOWN': {'TREND': 0.25, 'SMC': 0.25, 'MEAN_REVERSION': 0.25, 'SCALP': 0.25}
    }
    
    # Strategy to Category mapping
    STRATEGY_CATEGORIES = {
        'S1_IOB_Rejection': 'SMC', 'S2_VI_Sweep': 'SCALP', 'S3_EMD_HHT': 'TREND',
        'S4_CHOCH_IDM': 'SMC', 'S5_Breaker_Void': 'SMC', 'S6_QuantumPDF': 'MEAN_REVERSION',
        'S7_MacroFVG': 'SMC', 'S8_GPR_Vol': 'MEAN_REVERSION', 'S9_SessionSweep': 'SCALP',
        'S10_EhlersMESA': 'TREND', 'S11_LiquidityDelta': 'SCALP', 'S12_PCA_Cycle': 'TREND',
        'S13_TMF_EOM': 'TREND', 'S14_Propulsion': 'TREND', 'S15_HFT_StatArb': 'MEAN_REVERSION',
        'S16_RoofingEMD': 'MEAN_REVERSION', 'S17_ChaosSqueeze': 'TREND', 'S18_EhlersVector': 'MEAN_REVERSION',
        'S19_VoidReversal': 'SCALP', 'S20_VFIAccumulation': 'TREND', 'S21_BreakerFVGPOC': 'SMC',
        'S22_WyckoffSpring': 'MEAN_REVERSION', 'S23_MidnightJudas': 'SCALP', 'S24_KalmanMomentum': 'TREND',
        'S25_HurstWavelet': 'TREND'
    }
    
    def allocate_weights_18_regime(self, regime_name: str, kelly_multiplier: float,
                                   strategy_signals: Dict[str, dict]) -> Dict[str, float]:
        """
        Allocate weights to strategies based on 18-regime system.
        [FIX] Clamps weights before applying Kelly multiplier to prevent overflow.
        """
        weights = {}
        regime_weights = self.REGIME_WEIGHTS.get(regime_name, self.REGIME_WEIGHTS['UNKNOWN'])
        
        # Group strategies by category
        categories = {'TREND': [], 'SMC': [], 'MEAN_REVERSION': [], 'SCALP': []}
        for strat_name, cat in self.STRATEGY_CATEGORIES.items():
            if cat in categories:
                categories[cat].append(strat_name)
        
        # Block strategies in incompatible regimes
        blocked = set()
        if 'PARABOLIC' in regime_name or 'PANIC' in regime_name or 'FOMO' in regime_name:
            # Block trend following in extreme regimes
            blocked.update(categories['TREND'])
        if 'WHIPSAW' in regime_name or 'CHOPPY' in regime_name:
            # Block trend following in choppy markets
            blocked.update(categories['TREND'])
        
        # Override multipliers for specific strategies
        overrides = {
            'S2_VI_Sweep': 1.2,
            'S9_SessionSweep': 1.2,
            'S23_MidnightJudas': 1.1
        }
        
        # Calculate weights per category
        for category, strategies in categories.items():
            category_weight = regime_weights.get(category, 0.1)
            
            if not strategies:
                continue
            
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
                    signal_mult = 0.5 + signal_conf
                else:
                    signal_mult = 0.5
                
                adjusted_weight *= signal_mult
                
                # [FIX] Clamp before applying Kelly multiplier to prevent overflow
                adjusted_weight = max(0.0, min(1.0, adjusted_weight))
                
                # Apply Kelly multiplier
                final_weight = adjusted_weight * kelly_multiplier
                
                # Final clamp to [0.0, 1.0]
                final_weight = max(0.0, min(1.0, final_weight))
                
                weights[strategy_name] = round(final_weight, 4)
        
        return weights
    
    def get_blocked_strategies(self, regime_name: str) -> set:
        """Return set of strategy names blocked in the given regime."""
        blocked = set()
        if 'PARABOLIC' in regime_name or 'PANIC' in regime_name or 'FOMO' in regime_name:
            for strat, cat in self.STRATEGY_CATEGORIES.items():
                if cat == 'TREND':
                    blocked.add(strat)
        if 'WHIPSAW' in regime_name or 'CHOPPY' in regime_name:
            for strat, cat in self.STRATEGY_CATEGORIES.items():
                if cat == 'TREND':
                    blocked.add(strat)
        return blocked