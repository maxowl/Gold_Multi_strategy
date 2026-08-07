"""
Strategy Pool - 30 Strategies Edition (REVISED).
Loads, manages, and evaluates all 30 trading strategies.
Micro-Account-Only Mode (no scalping mode, no standard mode).

REVISION LOG:
  [REV-001] FIXED logging.getLogger(name) -> __name__ (NameError crash).
  [REV-002] FIXED Partial import failure. Each strategy class is now
            loaded individually with try/except so one broken strategy
            does not kill the entire bot.
  [REV-003] ADDED regime_context parameter pass-through to strategy.evaluate().
  [REV-004] ADDED Regime-kill pre-filtering before evaluation.
  [REV-005] ADDED Strategy cooldown mechanism after consecutive losses.
  [REV-006] ADDED Session-based strategy filtering.
  [REV-007] ADDED Signal deduplication per direction.
  [REV-008] ADDED Strategy performance tracking hook.

Responsibilities:
  - Import and instantiate all 30 strategies
  - Route strategies to correct timeframes
  - Evaluate all strategies on triggered timeframes
  - Category-based strategy grouping
  - Regime-based pre-filtering
  - Strategy cooldown management
"""
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from config import config

# =========================================================================
# IMPORT ALL 30 STRATEGIES
# [REV-001] FIXED: Use __name__ instead of undefined 'name'
# [REV-002] FIXED: Track import errors without crashing
# =========================================================================
_STRATEGY_IMPORT_ERRORS = []

try:
    from strategies.s1_iob_rejection import S1_IOB_Rejection
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S1: {e}")
    S1_IOB_Rejection = None

try:
    from strategies.s2_vi_sweep import S2_VI_Sweep
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S2: {e}")
    S2_VI_Sweep = None

try:
    from strategies.s3_emd_hht import S3_EMD_HHT
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S3: {e}")
    S3_EMD_HHT = None

try:
    from strategies.s4_choch_idm import S4_CHOCH_IDM
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S4: {e}")
    S4_CHOCH_IDM = None

try:
    from strategies.s5_breaker_void import S5_Breaker_Void
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S5: {e}")
    S5_Breaker_Void = None

try:
    from strategies.s6_quantum_pdf import S6_QuantumPDF
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S6: {e}")
    S6_QuantumPDF = None

try:
    from strategies.s7_macro_fvg import S7_MacroFVG
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S7: {e}")
    S7_MacroFVG = None

try:
    from strategies.s8_gpr_vol import S8_GPR_Vol
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S8: {e}")
    S8_GPR_Vol = None

try:
    from strategies.s9_session_sweep import S9_SessionSweep
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S9: {e}")
    S9_SessionSweep = None

try:
    from strategies.s10_ehlers_mesa import S10_EhlersMESA
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S10: {e}")
    S10_EhlersMESA = None

try:
    from strategies.s11_liquidity_delta import S11_LiquidityDelta
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S11: {e}")
    S11_LiquidityDelta = None

try:
    from strategies.s12_pca_cycle import S12_PCA_Cycle
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S12: {e}")
    S12_PCA_Cycle = None

try:
    from strategies.s13_tmf_eom import S13_TMF_EOM
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S13: {e}")
    S13_TMF_EOM = None

try:
    from strategies.s14_propulsion import S14_Propulsion
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S14: {e}")
    S14_Propulsion = None

try:
    from strategies.s15_hft_stat_arb import S15_HFT_StatArb
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S15: {e}")
    S15_HFT_StatArb = None

try:
    from strategies.s16_roofing_emd import S16_RoofingEMD
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S16: {e}")
    S16_RoofingEMD = None

try:
    from strategies.s17_chaos_squeeze import S17_ChaosSqueeze
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S17: {e}")
    S17_ChaosSqueeze = None

try:
    from strategies.s18_ehlers_vector import S18_EhlersVector
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S18: {e}")
    S18_EhlersVector = None

try:
    from strategies.s19_void_reversal import S19_VoidReversal
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S19: {e}")
    S19_VoidReversal = None

try:
    from strategies.s20_vfi_accumulation import S20_VFIAccumulation
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S20: {e}")
    S20_VFIAccumulation = None

try:
    from strategies.s21_breaker_fvg_poc import S21_BreakerFVGPOC
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S21: {e}")
    S21_BreakerFVGPOC = None

try:
    from strategies.s22_wyckoff_spring import S22_WyckoffSpring
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S22: {e}")
    S22_WyckoffSpring = None

try:
    from strategies.s23_midnight_judas import S23_MidnightJudas
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S23: {e}")
    S23_MidnightJudas = None

try:
    from strategies.s24_kalman_momentum import S24_KalmanMomentum
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S24: {e}")
    S24_KalmanMomentum = None

try:
    from strategies.s25_hurst_wavelet import S25_HurstWavelet
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S25: {e}")
    S25_HurstWavelet = None

try:
    from strategies.s26_microstructure import S26_Microstructure
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S26: {e}")
    S26_Microstructure = None

try:
    from strategies.s27_vwap_mean_reversion import S27_VWAP_MeanReversion
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S27: {e}")
    S27_VWAP_MeanReversion = None

try:
    from strategies.s28_mtf_confluence import S28_MTF_Confluence
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S28: {e}")
    S28_MTF_Confluence = None

try:
    from strategies.s29_quantum_momentum import S29_QuantumMomentum
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S29: {e}")
    S29_QuantumMomentum = None

try:
    from strategies.s30_volume_profile_reversal import S30_VolumeProfileReversal
except ImportError as e:
    _STRATEGY_IMPORT_ERRORS.append(f"S30: {e}")
    S30_VolumeProfileReversal = None


class StrategyPool:
    """
    Manages and evaluates all 30 trading strategies.

    ROUTE_MAP defines primary and higher timeframe for each strategy.
    Category groupings for regime-based filtering.

    Features:
      - Individual strategy error isolation [REV-002]
      - Regime-kill pre-filtering [REV-004]
      - Strategy cooldown management [REV-005]
      - Session-based filtering [REV-006]
      - Signal deduplication [REV-007]
    """

    # =========================================================================
    # ROUTE MAP: Strategy -> (Primary TF, Higher TF)
    # =========================================================================
    ROUTE_MAP = {
        'S1_IOB_Rejection': ('M15', 'H1'),
        'S2_VI_Sweep': ('M1', 'M15'),
        'S3_EMD_HHT': ('M15', 'H1'),
        'S4_CHOCH_IDM': ('M15', 'H1'),
        'S5_Breaker_Void': ('M15', 'H1'),
        'S6_QuantumPDF': ('M5', 'M15'),
        'S7_MacroFVG': ('H1', 'H4'),
        'S8_GPR_Vol': ('M15', 'H1'),
        'S9_SessionSweep': ('M5', 'M15'),
        'S10_EhlersMESA': ('M15', 'H1'),
        'S11_LiquidityDelta': ('M1', 'M5'),
        'S12_PCA_Cycle': ('M15', 'H1'),
        'S13_TMF_EOM': ('M15', 'H1'),
        'S14_Propulsion': ('M15', 'H1'),
        'S15_HFT_StatArb': ('M5', 'M15'),
        'S16_RoofingEMD': ('M15', 'H1'),
        'S17_ChaosSqueeze': ('M15', 'H1'),
        'S18_EhlersVector': ('M15', 'H1'),
        'S19_VoidReversal': ('M5', 'M15'),
        'S20_VFIAccumulation': ('M15', 'H1'),
        'S21_BreakerFVGPOC': ('M5', 'H1'),
        'S22_WyckoffSpring': ('M15', 'H1'),
        'S23_MidnightJudas': ('M15', 'H1'),
        'S24_KalmanMomentum': ('M15', 'H1'),
        'S25_HurstWavelet': ('M15', 'H1'),
        'S26_Microstructure': ('M5', 'M15'),
        'S27_VWAP_MeanReversion': ('M5', 'M15'),
        'S28_MTF_Confluence': ('M15', 'H1'),
        'S29_QuantumMomentum': ('M15', 'M5'),
        'S30_VolumeProfileReversal': ('M15', 'H1'),
    }

    # =========================================================================
    # CATEGORY GROUPINGS
    # =========================================================================
    SCALP_STRATEGIES = [
        'S2_VI_Sweep', 'S9_SessionSweep', 'S11_LiquidityDelta',
        'S19_VoidReversal', 'S23_MidnightJudas', 'S26_Microstructure',
        'S27_VWAP_MeanReversion'
    ]
    TREND_STRATEGIES = [
        'S3_EMD_HHT', 'S10_EhlersMESA', 'S12_PCA_Cycle',
        'S14_Propulsion', 'S16_RoofingEMD', 'S17_ChaosSqueeze',
        'S24_KalmanMomentum', 'S28_MTF_Confluence'
    ]
    SMC_STRATEGIES = [
        'S1_IOB_Rejection', 'S4_CHOCH_IDM', 'S5_Breaker_Void',
        'S7_MacroFVG', 'S21_BreakerFVGPOC', 'S22_WyckoffSpring',
        'S30_VolumeProfileReversal'
    ]
    MEAN_REVERSION_STRATEGIES = [
        'S6_QuantumPDF', 'S8_GPR_Vol', 'S13_TMF_EOM',
        'S15_HFT_StatArb', 'S18_EhlersVector', 'S20_VFIAccumulation',
        'S25_HurstWavelet', 'S29_QuantumMomentum'
    ]

    # =========================================================================
    # REGIME-KILL LIST [REV-004]
    # =========================================================================
    REGIME_KILL = [
        'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
        'VOLATILE_CHOP', 'WHIPSAW_MARKET'
    ]

    # =========================================================================
    # SESSION-BASED STRATEGY FILTERS [REV-006]
    # =========================================================================
    SESSION_DISABLED_STRATEGIES = {
        'ASIA': [],  # All strategies disabled during ASIA session (handled by event_loop)
        'LONDON': [],
        'NY': [],
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategies = []
        self._load_strategies()

        # [REV-005] Strategy cooldown tracking
        self._cooldowns = {}  # {strategy_name: cooldown_until_timestamp}
        self._cooldown_duration = 300  # 5 minutes

        # [REV-008] Consecutive loss tracking per strategy
        self._consecutive_losses = {}  # {strategy_name: count}
        self._max_consecutive_losses = 3  # Trigger cooldown after 3 losses

        # Log import errors if any
        if _STRATEGY_IMPORT_ERRORS:
            self.logger.warning(
                f"[STRATEGY_POOL] {len(_STRATEGY_IMPORT_ERRORS)} import error(s): "
                f"{'; '.join(_STRATEGY_IMPORT_ERRORS[:5])}"
            )

    # =========================================================================
    # STRATEGY LOADING [REV-002]
    # =========================================================================

    def _load_strategies(self):
        """
        Load all 30 strategies with individual error handling.

        [REV-002] Each strategy is wrapped in its own try/except so
        one broken strategy doesn't kill the entire bot.
        """
        # Build strategy map: name -> class (or None if import failed)
        strategy_map = {
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
            'S26_Microstructure': S26_Microstructure,
            'S27_VWAP_MeanReversion': S27_VWAP_MeanReversion,
            'S28_MTF_Confluence': S28_MTF_Confluence,
            'S29_QuantumMomentum': S29_QuantumMomentum,
            'S30_VolumeProfileReversal': S30_VolumeProfileReversal,
        }

        loaded_count = 0
        failed_count = 0

        for name, cls in strategy_map.items():
            # Skip if import failed
            if cls is None:
                self.logger.warning(
                    f"[STRATEGY_POOL] {name}: Import failed, skipping"
                )
                failed_count += 1
                continue

            # Try to instantiate
            try:
                instance = cls()
                self.strategies.append(instance)
                loaded_count += 1
            except Exception as e:
                self.logger.error(
                    f"[STRATEGY_POOL] Could not instantiate {name}: {e}"
                )
                failed_count += 1

        self.logger.info(
            f"[STRATEGY_POOL] Loaded {loaded_count}/30 strategies "
            f"({failed_count} failed)"
        )

        # Log category distribution
        scalp_count = len([s for s in self.strategies if s.name in self.SCALP_STRATEGIES])
        trend_count = len([s for s in self.strategies if s.name in self.TREND_STRATEGIES])
        smc_count = len([s for s in self.strategies if s.name in self.SMC_STRATEGIES])
        mr_count = len([s for s in self.strategies if s.name in self.MEAN_REVERSION_STRATEGIES])
        self.logger.info(
            f"[STRATEGY_POOL] Categories: SCALP={scalp_count}, TREND={trend_count}, "
            f"SMC={smc_count}, MEAN_REV={mr_count}"
        )

    # =========================================================================
    # STRATEGY COOLDOWN [REV-005]
    # =========================================================================

    def _is_in_cooldown(self, strategy_name: str) -> bool:
        """
        Check if strategy is in cooldown period.

        Args:
            strategy_name: Strategy name

        Returns:
            True if strategy is in cooldown
        """
        if strategy_name in self._cooldowns:
            if time.time() < self._cooldowns[strategy_name]:
                return True
            # Cooldown expired, remove it
            del self._cooldowns[strategy_name]
            self.logger.info(
                f"[STRATEGY_POOL] {strategy_name}: Cooldown expired"
            )
        return False

    def _set_cooldown(self, strategy_name: str):
        """
        Set cooldown for a strategy.

        Args:
            strategy_name: Strategy name
        """
        self._cooldowns[strategy_name] = time.time() + self._cooldown_duration
        self.logger.info(
            f"[STRATEGY_POOL] {strategy_name}: Cooldown set for "
            f"{self._cooldown_duration}s"
        )

    def record_loss(self, strategy_name: str):
        """
        Record a loss for a strategy and potentially trigger cooldown.

        Args:
            strategy_name: Strategy name
        """
        if strategy_name not in self._consecutive_losses:
            self._consecutive_losses[strategy_name] = 0

        self._consecutive_losses[strategy_name] += 1

        if self._consecutive_losses[strategy_name] >= self._max_consecutive_losses:
            self._set_cooldown(strategy_name)
            self._consecutive_losses[strategy_name] = 0

    def record_win(self, strategy_name: str):
        """
        Record a win for a strategy (resets consecutive loss counter).

        Args:
            strategy_name: Strategy name
        """
        self._consecutive_losses[strategy_name] = 0

    # =========================================================================
    # STRATEGY EVALUATION
    # =========================================================================

    def evaluate_all(
        self,
        data: Dict[str, pd.DataFrame],
        triggered_tfs: Set[str],
        regime_context: dict = None
    ) -> Dict[str, dict]:
        """
        Evaluate all strategies on triggered timeframes.

        [REV-004] Added regime-kill pre-filtering.
        [REV-003] Added regime_context pass-through.
        [REV-005] Added cooldown check.
        [REV-006] Added session-based filtering.

        Args:
            data: Dict of timeframe -> DataFrame
            triggered_tfs: Set of timeframes that triggered (new bar closed)
            regime_context: Current regime information (optional)

        Returns:
            Dict of strategy_name -> signal dict
        """
        signals = {}

        # =====================================================================
        # [REV-004] REGIME-KILL PRE-FILTER
        # =====================================================================
        if regime_context is not None:
            regime_name = regime_context.get(
                'regime_name', regime_context.get('regime', 'UNKNOWN')
            )
            if regime_name in self.REGIME_KILL:
                self.logger.info(
                    f"[STRATEGY_POOL] Regime-kill: {regime_name} | "
                    f"Skipping all strategy evaluation"
                )
                return signals

        # =====================================================================
        # [REV-006] SESSION-BASED FILTERING
        # =====================================================================
        session = None
        if regime_context is not None:
            session = regime_context.get('session', 'OTHER')

        for strategy in self.strategies:
            name = strategy.name
            tf_primary, tf_htf = self.ROUTE_MAP.get(name, ('M15', None))

            # Check if primary timeframe was triggered
            if tf_primary not in triggered_tfs:
                continue

            # [REV-005] Check cooldown
            if self._is_in_cooldown(name):
                continue

            # [REV-006] Check session-based filtering
            if session is not None and session in self.SESSION_DISABLED_STRATEGIES:
                if name in self.SESSION_DISABLED_STRATEGIES[session]:
                    continue

            df_primary = data.get(tf_primary)
            df_htf = data.get(tf_htf) if tf_htf else None

            if df_primary is None or df_primary.empty:
                self.logger.debug(f"[STRATEGY_POOL] {name}: No data for {tf_primary}")
                continue

            try:
                # [REV-003] Pass regime_context to strategy
                signal = strategy.evaluate(
                    df_primary, df_htf, regime_context=regime_context
                )

                if signal and signal.get('signal') != 'NEUTRAL':
                    # Add regime info to signal meta
                    if regime_context is not None:
                        meta = signal.get('meta', {})
                        meta['regime_name'] = regime_context.get(
                            'regime_name', regime_context.get('regime', 'UNKNOWN')
                        )
                        meta['session'] = regime_context.get('session', 'OTHER')
                        signal['meta'] = meta

                    signals[name] = signal
                    self.logger.debug(
                        f"[STRATEGY_POOL] {name}: Signal generated ({signal['signal']})"
                    )

            except Exception as e:
                self.logger.error(
                    f"[STRATEGY_POOL] Exception in {name}: {e}", exc_info=True
                )

        # =====================================================================
        # [REV-007] SIGNAL DEDUPLICATION PER DIRECTION
        # =====================================================================
        if signals:
            signals = self._deduplicate_signals(signals)

        if signals:
            self.logger.info(
                f"[STRATEGY_POOL] {len(signals)} signal(s) generated: "
                f"{list(signals.keys())}"
            )

        return signals

    def _deduplicate_signals(self, signals: Dict[str, dict]) -> Dict[str, dict]:
        """
        [REV-007] Deduplicate signals by direction.

        If multiple strategies generate signals in the same direction,
        keep the one with the highest expert_score or confidence.

        Args:
            signals: Dict of strategy_name -> signal dict

        Returns:
            Deduplicated dict of strategy_name -> signal dict
        """
        if len(signals) <= 1:
            return signals

        # Group by direction
        buy_signals = {}
        sell_signals = {}

        for name, sig in signals.items():
            signal_type = sig.get('signal', '')
            if 'BUY' in signal_type:
                buy_signals[name] = sig
            elif 'SELL' in signal_type:
                sell_signals[name] = sig

        # Keep the best signal per direction
        deduplicated = {}

        if buy_signals:
            best_buy = max(
                buy_signals.items(),
                key=lambda x: x[1].get('meta', {}).get('expert_score', 0)
            )
            deduplicated[best_buy[0]] = best_buy[1]
            if len(buy_signals) > 1:
                self.logger.info(
                    f"[STRATEGY_POOL] Deduplicated {len(buy_signals)} BUY signals "
                    f"to {best_buy[0]}"
                )

        if sell_signals:
            best_sell = max(
                sell_signals.items(),
                key=lambda x: x[1].get('meta', {}).get('expert_score', 0)
            )
            deduplicated[best_sell[0]] = best_sell[1]
            if len(sell_signals) > 1:
                self.logger.info(
                    f"[STRATEGY_POOL] Deduplicated {len(sell_signals)} SELL signals "
                    f"to {best_sell[0]}"
                )

        return deduplicated

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_strategy_by_name(self, name: str):
        """
        Get strategy instance by name.

        Args:
            name: Strategy name (e.g., 'S1_IOB_Rejection')

        Returns:
            Strategy instance or None
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None

    def get_strategies_by_category(self, category: str) -> List:
        """
        Get all strategies of a specific category.

        Args:
            category: Category name (SCALP, TREND, SMC, MEAN_REVERSION)

        Returns:
            List of strategy instances
        """
        category_map = {
            'SCALP': self.SCALP_STRATEGIES,
            'TREND': self.TREND_STRATEGIES,
            'SMC': self.SMC_STRATEGIES,
            'MEAN_REVERSION': self.MEAN_REVERSION_STRATEGIES,
        }
        target_names = category_map.get(category, [])
        return [s for s in self.strategies if s.name in target_names]

    def get_all_strategy_names(self) -> List[str]:
        """
        Get all loaded strategy names.

        Returns:
            List of strategy names
        """
        return [s.name for s in self.strategies]

    def get_strategy_count(self) -> int:
        """
        Get total number of loaded strategies.

        Returns:
            Integer count
        """
        return len(self.strategies)

    def get_route_map(self) -> Dict[str, tuple]:
        """
        Get the complete route map.

        Returns:
            Dict of strategy_name -> (primary_tf, htf)
        """
        return self.ROUTE_MAP.copy()

    def get_cooldown_status(self) -> Dict:
        """
        Get current cooldown status for all strategies.

        Returns:
            Dict of strategy_name -> cooldown_remaining_seconds
        """
        status = {}
        current_time = time.time()
        for name, until in self._cooldowns.items():
            remaining = until - current_time
            if remaining > 0:
                status[name] = round(remaining, 1)
        return status

    def get_consecutive_losses(self) -> Dict:
        """
        Get consecutive loss counts for all strategies.

        Returns:
            Dict of strategy_name -> consecutive_loss_count
        """
        return self._consecutive_losses.copy()