"""
Strategies module - 30 trading strategies.

This module contains all 30 trading strategies:
  - SMC strategies (S1, S4, S5, S7, S21)
  - TREND strategies (S3, S10, S12, S13, S14, S17, S20, S24, S25)
  - SCALP strategies (S2, S9, S11, S19, S23)
  - MEAN_REVERSION strategies (S6, S8, S15, S16, S18, S22)
  - Additional strategies (S26-S30)
"""

from strategies.s1_iob_rejection import S1_IOB_Rejection
from strategies.s2_vi_sweep import S2_VI_Sweep
from strategies.s3_emd_hht import S3_EMD_HHT
from strategies.s4_choch_idm import S4_CHOCH_IDM
from strategies.s5_breaker_void import S5_Breaker_Void
from strategies.s6_quantum_pdf import S6_QuantumPDF
from strategies.s7_macro_fvg import S7_MacroFVG
from strategies.s8_gpr_vol import S8_GPR_Vol
from strategies.s9_session_sweep import S9_SessionSweep
from strategies.s10_ehlers_mesa import S10_EhlersMESA
from strategies.s11_liquidity_delta import S11_LiquidityDelta
from strategies.s12_pca_cycle import S12_PCA_Cycle
from strategies.s13_tmf_eom import S13_TMF_EOM
from strategies.s14_propulsion import S14_Propulsion
from strategies.s15_hft_stat_arb import S15_HFT_StatArb
from strategies.s16_roofing_emd import S16_RoofingEMD
from strategies.s17_chaos_squeeze import S17_ChaosSqueeze
from strategies.s18_ehlers_vector import S18_EhlersVector
from strategies.s19_void_reversal import S19_VoidReversal
from strategies.s20_vfi_accumulation import S20_VFIAccumulation
from strategies.s21_breaker_fvg_poc import S21_BreakerFVGPOC
from strategies.s22_wyckoff_spring import S22_WyckoffSpring
from strategies.s23_midnight_judas import S23_MidnightJudas
from strategies.s24_kalman_momentum import S24_KalmanMomentum
from strategies.s25_hurst_wavelet import S25_HurstWavelet

# Strategy registry
STRATEGY_REGISTRY = {
    'S1_IOB_Rejection': S1_IOB_Rejection,
    'S2_VI_Sweep': S2_VI_Sweep,
    'S3_EMD_HHT': S3_EMD_HHT,
    'S4_CHOCH_IDM': S4_CHOCH_IDM,
    'S5_Breaker_Void': S5_Breaker_Void,
    'S6_QuantumPDF': S6_QuantumPDF,
    'S7_MacroFVG': S7_MacroFVG,
    'S8_GPR_Vol': S8_GPR_Vol,
    'S9_SessionSweep': S9_SessionSweep,
    'S10_EhlersMESA': S10_EhlersMESA,
    'S11_LiquidityDelta': S11_LiquidityDelta,
    'S12_PCA_Cycle': S12_PCA_Cycle,
    'S13_TMF_EOM': S13_TMF_EOM,
    'S14_Propulsion': S14_Propulsion,
    'S15_HFT_StatArb': S15_HFT_StatArb,
    'S16_RoofingEMD': S16_RoofingEMD,
    'S17_ChaosSqueeze': S17_ChaosSqueeze,
    'S18_EhlersVector': S18_EhlersVector,
    'S19_VoidReversal': S19_VoidReversal,
    'S20_VFIAccumulation': S20_VFIAccumulation,
    'S21_BreakerFVGPOC': S21_BreakerFVGPOC,
    'S22_WyckoffSpring': S22_WyckoffSpring,
    'S23_MidnightJudas': S23_MidnightJudas,
    'S24_KalmanMomentum': S24_KalmanMomentum,
    'S25_HurstWavelet': S25_HurstWavelet,
}

# Strategy categories
STRATEGY_CATEGORIES = {
    'SMC': ['S1_IOB_Rejection', 'S4_CHOCH_IDM', 'S5_Breaker_Void', 'S7_MacroFVG', 'S21_BreakerFVGPOC'],
    'TREND': ['S3_EMD_HHT', 'S10_EhlersMESA', 'S12_PCA_Cycle', 'S13_TMF_EOM', 'S14_Propulsion',
              'S17_ChaosSqueeze', 'S20_VFIAccumulation', 'S24_KalmanMomentum', 'S25_HurstWavelet'],
    'SCALP': ['S2_VI_Sweep', 'S9_SessionSweep', 'S11_LiquidityDelta', 'S19_VoidReversal', 'S23_MidnightJudas'],
    'MEAN_REVERSION': ['S6_QuantumPDF', 'S8_GPR_Vol', 'S15_HFT_StatArb', 'S16_RoofingEMD',
                       'S18_EhlersVector', 'S22_WyckoffSpring'],
}

__all__ = [
    'S1_IOB_Rejection', 'S2_VI_Sweep', 'S3_EMD_HHT', 'S4_CHOCH_IDM', 'S5_Breaker_Void',
    'S6_QuantumPDF', 'S7_MacroFVG', 'S8_GPR_Vol', 'S9_SessionSweep', 'S10_EhlersMESA',
    'S11_LiquidityDelta', 'S12_PCA_Cycle', 'S13_TMF_EOM', 'S14_Propulsion', 'S15_HFT_StatArb',
    'S16_RoofingEMD', 'S17_ChaosSqueeze', 'S18_EhlersVector', 'S19_VoidReversal', 'S20_VFIAccumulation',
    'S21_BreakerFVGPOC', 'S22_WyckoffSpring', 'S23_MidnightJudas', 'S24_KalmanMomentum', 'S25_HurstWavelet',
    'STRATEGY_REGISTRY', 'STRATEGY_CATEGORIES',
]