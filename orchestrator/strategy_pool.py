"""
Strategy Pool.
Loads, manages, and evaluates all 25 trading strategies.
"""
import logging
from typing import Dict, List, Tuple
import pandas as pd
from config import config

# Import all 25 strategies safely
try:
    from strategies.s1_iob_rejection import Strategy1_IOB_Rejection
    from strategies.s2_vi_sweep import Strategy2_VI_Sweep
    from strategies.s3_emd_hht import Strategy3_EMD_HHT
    from strategies.s4_choch_idm import Strategy4_CHOCH_IDM
    from strategies.s5_breaker_void import Strategy5_Breaker_Void
    from strategies.s6_quantum_pdf import Strategy6_QuantumPDF
    from strategies.s7_macro_fvg import Strategy7_MacroFVG
    from strategies.s8_gpr_vol import Strategy8_GPR_Vol
    from strategies.s9_session_sweep import Strategy9_SessionSweep
    from strategies.s10_ehlers_mesa import Strategy10_EhlersMESA
    from strategies.s11_liquidity_delta import Strategy11_LiquidityDelta
    from strategies.s12_pca_cycle import Strategy12_PCA_Cycle
    from strategies.s13_tmf_eom import Strategy13_TMF_EOM
    from strategies.s14_propulsion import Strategy14_Propulsion
    from strategies.s15_hft_stat_arb import Strategy15_HFT_StatArb
    from strategies.s16_roofing_emd import Strategy16_RoofingEMD
    from strategies.s17_chaos_squeeze import Strategy17_ChaosSqueeze
    from strategies.s18_ehlers_vector import Strategy18_EhlersVector
    from strategies.s19_void_reversal import Strategy19_VoidReversal
    from strategies.s20_vfi_accumulation import Strategy20_VFIAccumulation
    from strategies.s21_breaker_fvg_poc import Strategy21_BreakerFVGPOC
    from strategies.s22_wyckoff_spring import Strategy22_WyckoffSpring
    from strategies.s23_midnight_judas import Strategy23_MidnightJudas
    from strategies.s24_kalman_momentum import Strategy24_KalmanMomentum
    from strategies.s25_hurst_wavelet import Strategy25_HurstWavelet
except ImportError as e:
    logging.getLogger(__name__).error(f"[FAIL] Strategy import error: {e}")


class StrategyPool:
    ROUTE_MAP = {
        'S1_IOB_Rejection': ('M15', 'H1'), 'S2_VI_Sweep': ('M1', 'M15'), 'S3_EMD_HHT': ('M15', 'H1'),
        'S4_CHOCH_IDM': ('M15', 'H1'), 'S5_Breaker_Void': ('M15', 'H1'), 'S6_QuantumPDF': ('M5', 'M15'),
        'S7_MacroFVG': ('H1', 'H4'), 'S8_GPR_Vol': ('M15', 'H1'), 'S9_SessionSweep': ('M5', 'M15'),
        'S10_EhlersMESA': ('M15', 'H1'), 'S11_LiquidityDelta': ('M1', 'M5'), 'S12_PCA_Cycle': ('M15', 'H1'),
        'S13_TMF_EOM': ('M15', 'H1'), 'S14_Propulsion': ('M15', 'H1'), 'S15_HFT_StatArb': ('M5', 'M15'),
        'S16_RoofingEMD': ('M15', 'H1'), 'S17_ChaosSqueeze': ('M15', 'H1'), 'S18_EhlersVector': ('M15', 'H1'),
        'S19_VoidReversal': ('M5', 'M15'), 'S20_VFIAccumulation': ('M15', 'H1'), 'S21_BreakerFVGPOC': ('M5', 'H1'),
        'S22_WyckoffSpring': ('M15', 'H1'), 'S23_MidnightJudas': ('M15', 'H1'), 'S24_KalmanMomentum': ('M15', 'H1'),
        'S25_HurstWavelet': ('M15', 'H1')
    }
    
    SCALPING_PRIORITIES = {
        'S2_VI_Sweep': 1.5, 'S9_SessionSweep': 1.4, 'S11_LiquidityDelta': 1.3,
        'S19_VoidReversal': 1.2, 'S23_MidnightJudas': 1.1
    }
    
    # [FIX] Safe fallback for config attributes
    SCALPING_ALLOWED_REGIMES = getattr(config, 'scalping_regimes_allowed', ('TIGHT_RANGE', 'CLASSIC_RANGE', 'QUIET_RALLY', 'SLOW_BLEED'))

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategies = []
        self._load_strategies()

    def _load_strategies(self):
        # [FIX] Wrap each instantiation in try/except so one broken strategy doesn't kill the bot
        strategy_classes = [
            Strategy1_IOB_Rejection, Strategy2_VI_Sweep, Strategy3_EMD_HHT, Strategy4_CHOCH_IDM,
            Strategy5_Breaker_Void, Strategy6_QuantumPDF, Strategy7_MacroFVG, Strategy8_GPR_Vol,
            Strategy9_SessionSweep, Strategy10_EhlersMESA, Strategy11_LiquidityDelta, Strategy12_PCA_Cycle,
            Strategy13_TMF_EOM, Strategy14_Propulsion, Strategy15_HFT_StatArb, Strategy16_RoofingEMD,
            Strategy17_ChaosSqueeze, Strategy18_EhlersVector, Strategy19_VoidReversal, Strategy20_VFIAccumulation,
            Strategy21_BreakerFVGPOC, Strategy22_WyckoffSpring, Strategy23_MidnightJudas, Strategy24_KalmanMomentum,
            Strategy25_HurstWavelet
        ]
        
        for cls in strategy_classes:
            try:
                self.strategies.append(cls())
            except Exception as e:
                self.logger.error(f"[FAIL] Could not instantiate {cls.__name__}: {e}")
                
        self.logger.info(f"[OK] Loaded {len(self.strategies)} strategies successfully.")

    def evaluate_all(self, data: Dict[str, pd.DataFrame], triggered_tfs: set, regime_context: dict = None) -> Dict[str, dict]:
        signals = {}
        
        # Scalping mode regime check
        if config.scalping_mode and regime_context:
            current_regime = regime_context.get('regime_name', 'UNKNOWN')
            if current_regime not in self.SCALPING_ALLOWED_REGIMES:
                self.logger.debug(f"[SCALP BLOCK] Regime {current_regime} not suitable for scalping")
                return signals
        
        for strategy in self.strategies:
            name = strategy.name
            tf_primary, tf_htf = self.ROUTE_MAP.get(name, ('M15', None))
            
            if config.scalping_mode:
                if name in self.SCALPING_PRIORITIES:
                    tf_primary = config.scalping_primary_tf
                else:
                    continue
            
            if tf_primary not in triggered_tfs: 
                continue
            
            df_primary = data.get(tf_primary)
            df_htf = data.get(tf_htf) if tf_htf else None
            if df_primary is None or df_primary.empty: 
                continue
            
            try:
                signal = strategy.evaluate(df_primary, df_htf)
                if signal and signal.get('signal') != 'NEUTRAL':
                    if config.scalping_mode and name in self.SCALPING_PRIORITIES:
                        meta = signal.get('meta', {})
                        meta['position_multiplier'] = meta.get('position_multiplier', 1.0) * self.SCALPING_PRIORITIES[name]
                        signal['meta'] = meta
                    signals[name] = signal
            except Exception as e:
                self.logger.error(f"[FAIL] Exception in {name}: {e}", exc_info=True)
        
        return signals

    def get_active_signals(self, signals: Dict[str, dict]) -> List[Tuple[str, dict]]:
        active = []
        for name, sig in signals.items():
            if sig.get('signal') != 'NEUTRAL':
                active.append((name, sig))
        return active