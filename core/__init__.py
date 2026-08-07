"""
Core module - Trading engines and indicators.

This module contains all the core engines and indicators used by strategies:
  - SMC Engine (Order Blocks, FVG, BOS/CHOCH)
  - Volume Profile Engine
  - DSP Engines (EMD, Hilbert, Ehlers)
  - Statistical Engines (PCA, Hurst, Wavelet)
  - Order Flow Engine
  - And many more...
"""

from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.breaker_vp_engine import BreakerVPEngine
from core.dsp_engine import DSPEngine
from core.dsp_ehlers_engine import EhlersDSPEngine
from core.fibonacci_engine import FibonacciEngine
from core.hurst_wavelet_engine import HurstWaveletEngine
from core.kalman_squeeze_engine import KalmanSqueezeEngine
from core.mtf_feature_engineer import MTFFeatureEngineer
from core.orderflow_engine import OrderFlowEngine
from core.pca_engine import PCAEngine
from core.propulsion_engine import PropulsionEngine
from core.quant_math_engine import QuantMathEngine
from core.stat_arb_engine import StatArbEngine
from core.time_price_engine import TimePriceEngine
from core.void_structural_engine import VoidStructuralEngine
from core.volume_flow_engine import VolumeFlowEngine
from core.volume_indicators import VolumeIndicatorsEngine
from core.wyckoff_vsa_engine import WyckoffVSAEngine
from core.atr_cache import ATRCache
from core.adaptive_tp_engine import AdaptiveTPEngine
from core.regime_router import RegimeRouter
from core.expert_signal_scorer import ExpertSignalScorer
from core.kelly_criterion import KellyCriterionEngine
from core.time_stop_manager import TimeStopManager
from core.emergency_defense_engine import EmergencyDefenseEngine
from core.dynamic_stops_manager import DynamicStopsManager
from core.chandelier_engine import ChandelierEngine
from core.entry_optimizer import EntryOptimizer
from core.invalidation_engine import InvalidationEngine
from core.reversal_detector import ReversalDetector
from core.loss_attribution_engine import LossAttributionEngine
from core.session_volatility import SessionVolatilityManager
from core.choppy_detector import ChoppyDetector
from core.market_killers_detector import MarketKillersDetector
from core.pattern_detector import PatternDetector
from core.hmm_regime_detector import HMMRegimeDetector
from core.regime_classifier import RegimeClassifier
from core.hybrid_mtf_predictor import HybridMTFPredictor
from core.strategy_allocator import StrategyAllocator, EnhancedStrategyAllocator
from core.strategy_performance_tracker import StrategyPerformanceTracker

__all__ = [
    'BaseStrategy',
    'SMCStructuralEngine',
    'BreakerVPEngine',
    'DSPEngine',
    'EhlersDSPEngine',
    'FibonacciEngine',
    'HurstWaveletEngine',
    'KalmanSqueezeEngine',
    'MTFFeatureEngineer',
    'OrderFlowEngine',
    'PCAEngine',
    'PropulsionEngine',
    'QuantMathEngine',
    'StatArbEngine',
    'TimePriceEngine',
    'VoidStructuralEngine',
    'VolumeFlowEngine',
    'VolumeIndicatorsEngine',
    'WyckoffVSAEngine',
    'ATRCache',
    'AdaptiveTPEngine',
    'RegimeRouter',
    'ExpertSignalScorer',
    'KellyCriterionEngine',
    'TimeStopManager',
    'EmergencyDefenseEngine',
    'DynamicStopsManager',
    'ChandelierEngine',
    'EntryOptimizer',
    'InvalidationEngine',
    'ReversalDetector',
    'LossAttributionEngine',
    'SessionVolatilityManager',
    'ChoppyDetector',
    'MarketKillersDetector',
    'PatternDetector',
    'HMMRegimeDetector',
    'RegimeClassifier',
    'HybridMTFPredictor',
    'StrategyAllocator',
    'EnhancedStrategyAllocator',
    'StrategyPerformanceTracker',
]