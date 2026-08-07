"""
Kelly Criterion Engine - Dynamic Risk Sizing for Micro-Account Trading.

Implements the Kelly Criterion formula for optimal position sizing,
with safety caps and regime-based adjustments for micro-account
trading ($500-$3000 portfolio).

Kelly Criterion Formula:
    f* = (p * b - q) / b

    Where:
        p = win rate (probability of winning)
        q = 1 - p (probability of losing)
        b = avg_win / avg_loss (profit/loss ratio)

    f* = optimal fraction of capital to risk

Half Kelly (Safety):
    f_half = f* / 2

    Half Kelly is used by default because:
    1. Full Kelly assumes perfect knowledge of win rate and P/L ratio
    2. In practice, estimates are noisy
    3. Half Kelly reduces variance by ~50% with only ~25% less growth
    4. Critical for micro-accounts where a single over-sized loss
       can be devastating

Safety Mechanisms:
    1. Minimum trade count (min_trades=50):
       Kelly requires sufficient data. Below threshold, use base risk.

    2. Maximum risk cap (max_risk_pct=3.0%):
       Even if Kelly suggests higher, never exceed this cap.

    3. Minimum winrate (min_winrate=0.4):
       If winrate is below this, Kelly fraction may be negative
       (meaning don't trade). We use base risk instead.

    4. Minimum profit factor (min_profit_factor=1.2):
       Ensures the strategy has a positive edge before applying Kelly.

    5. Regime-based adjustment:
       Different market conditions warrant different risk levels.

    6. Consecutive loss adjustment:
       After consecutive losses, reduce risk to prevent tilt.

    7. Drawdown-aware scaling:
       Reduce risk proportionally to current drawdown.

Usage:
    kelly = KellyCriterionEngine(
        min_trades=50,
        max_risk_pct=3.0,
        use_half_kelly=True,
        min_winrate=0.4,
        min_profit_factor=1.2
    )

    kelly_risk, kelly_fraction, reason = kelly.calculate_kelly_risk(
        winrate=0.60,
        avg_win=12.5,
        avg_loss=8.3,
        base_risk_pct=0.5,
        trade_count=75
    )

    # With regime adjustment
    adjusted_risk = kelly.get_regime_adjusted_risk(kelly_risk, 'TREND')
"""
import logging
import math
from typing import Dict, Optional, Tuple


class KellyCriterionEngine:
    """
    Kelly Criterion Engine for dynamic risk sizing.

    This engine calculates the optimal risk percentage per trade
    using the Kelly Criterion formula, with multiple safety
    mechanisms to protect micro-account capital.

    Key Features:
        - Full Kelly and Half Kelly calculation
        - Minimum trade count threshold
        - Maximum risk cap
        - Regime-based risk adjustment
        - Consecutive loss reduction
        - Drawdown-aware scaling
        - Comprehensive logging and diagnostics

    Thread Safety:
        This class is stateless (no mutable state after __init__),
        so it is inherently thread-safe.
    """

    # =========================================================================
    # REGIME-BASED RISK MULTIPLIERS
    # =========================================================================
    REGIME_RISK_MULTIPLIERS = {
        'TREND': 1.0,       # Full Kelly for trending markets
        'SIDEWAY': 0.75,    # Reduce 25% for sideways markets
        'REVERSAL': 0.75,   # Reduce 25% for reversal setups
        'HIGH_VOL': 0.50,   # Reduce 50% for high volatility
        'UNKNOWN': 0.50,    # Reduce 50% for unknown regime (safety)
    }

    # =========================================================================
    # CONSECUTIVE LOSS MULTIPLIERS
    # =========================================================================
    CONSECUTIVE_LOSS_MULTIPLIERS = {
        0: 1.00,   # No losses
        1: 1.00,   # 1 loss - no change
        2: 0.90,   # 2 losses - reduce 10%
        3: 0.75,   # 3 losses - reduce 25%
        4: 0.50,   # 4 losses - reduce 50%
        5: 0.25,   # 5+ losses - reduce 75%
    }

    # =========================================================================
    # DRAWDOWN RISK REDUCTION
    # =========================================================================
    DRAWDOWN_RISK_THRESHOLDS = [
        # (drawdown_pct, risk_multiplier)
        (0.0, 1.00),    # No drawdown
        (1.0, 0.90),    # 1% drawdown - reduce 10%
        (2.0, 0.75),    # 2% drawdown - reduce 25%
        (3.0, 0.50),    # 3% drawdown - reduce 50%
        (5.0, 0.25),    # 5% drawdown - reduce 75%
        (10.0, 0.00),   # 10%+ drawdown - stop trading
    ]

    def __init__(self, min_trades: int = 50, max_risk_pct: float = 3.0,
                 use_half_kelly: bool = True, min_winrate: float = 0.4,
                 min_profit_factor: float = 1.2):
        """
        Initialize KellyCriterionEngine.

        Args:
            min_trades: Minimum number of trades required before
                        applying Kelly. Below this, use base risk.
                        Default: 50 (per rule: min_trades=50)

            max_risk_pct: Maximum risk percentage cap. Kelly will
                          never suggest risk above this value.
                          Default: 3.0%

            use_half_kelly: Whether to use Half Kelly (recommended).
                            Half Kelly reduces variance significantly
                            with minimal growth reduction.
                            Default: True

            min_winrate: Minimum winrate required for Kelly to apply.
                         Below this, the strategy may not have edge.
                         Default: 0.4 (40%)

            min_profit_factor: Minimum profit factor required.
                               Ensures positive edge before Kelly.
                               Default: 1.2
        """
        self.min_trades = min_trades
        self.max_risk_pct = max_risk_pct
        self.use_half_kelly = use_half_kelly
        self.min_winrate = min_winrate
        self.min_profit_factor = min_profit_factor

        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            f"[KELLY] Initialized | min_trades: {self.min_trades} | "
            f"max_risk: {self.max_risk_pct}% | "
            f"half_kelly: {self.use_half_kelly} | "
            f"min_winrate: {self.min_winrate} | "
            f"min_pf: {self.min_profit_factor}"
        )

    # =========================================================================
    # CORE KELLY CALCULATION
    # =========================================================================

    def calculate_kelly_fraction(self, winrate: float, avg_win: float,
                                  avg_loss: float) -> Tuple[float, str]:
        """
        Calculate raw Kelly fraction from win rate and P/L ratio.

        Kelly Formula:
            f* = (p * b - q) / b

            Where:
                p = winrate
                q = 1 - winrate
                b = avg_win / avg_loss

        Args:
            winrate: Win rate as decimal (0.0 - 1.0)
            avg_win: Average winning trade amount (USD)
            avg_loss: Average losing trade amount (USD, positive)

        Returns:
            Tuple of (kelly_fraction, reason_string)
            kelly_fraction is the raw Kelly fraction (can be negative)
        """
        # Validate inputs
        if winrate <= 0 or winrate >= 1:
            return 0.0, f"Invalid winrate: {winrate}"

        if avg_win <= 0:
            return 0.0, f"Invalid avg_win: {avg_win}"

        if avg_loss <= 0:
            return 0.0, f"Invalid avg_loss: {avg_loss}"

        # Calculate Kelly components
        p = winrate           # Probability of winning
        q = 1.0 - winrate     # Probability of losing
        b = avg_win / avg_loss  # Profit/loss ratio

        # Kelly formula: f* = (p * b - q) / b
        kelly_fraction = (p * b - q) / b

        # Apply Half Kelly if enabled
        if self.use_half_kelly:
            kelly_fraction = kelly_fraction / 2.0

        reason = (
            f"p={p:.3f}, q={q:.3f}, b={b:.3f}, "
            f"f*={(p * b - q) / b:.4f}"
            f"{', half=' + f'{kelly_fraction:.4f}' if self.use_half_kelly else ''}"
        )

        return kelly_fraction, reason

    def calculate_kelly_risk(self, winrate: float, avg_win: float,
                              avg_loss: float, base_risk_pct: float,
                              trade_count: int) -> Tuple[float, float, str]:
        """
        Calculate Kelly-adjusted risk percentage with all safety checks.

        This is the main entry point for Kelly risk calculation.
        It applies all safety mechanisms:
            1. Minimum trade count check
            2. Minimum winrate check
            3. Minimum profit factor check
            4. Kelly fraction calculation
            5. Maximum risk cap

        Args:
            winrate: Win rate as decimal (0.0 - 1.0)
            avg_win: Average winning trade amount (USD)
            avg_loss: Average losing trade amount (USD, positive)
            base_risk_pct: Base risk percentage (e.g., 0.5 for 0.5%)
            trade_count: Number of trades in the sample

        Returns:
            Tuple of (kelly_risk_pct, kelly_fraction, reason_string)

            kelly_risk_pct: The final risk percentage to use
            kelly_fraction: The raw Kelly fraction (for diagnostics)
            reason_string: Explanation of the calculation
        """
        # =====================================================================
        # CHECK 1: Minimum Trade Count
        # =====================================================================
        if trade_count < self.min_trades:
            reason = (
                f"[BASE RISK] Insufficient data: {trade_count}/{self.min_trades} trades. "
                f"Using base risk {base_risk_pct:.2f}%"
            )
            self.logger.info(f"[KELLY] {reason}")
            return base_risk_pct, 0.0, reason

        # =====================================================================
        # CHECK 2: Minimum Winrate
        # =====================================================================
        if winrate < self.min_winrate:
            reason = (
                f"[BASE RISK] Winrate too low: {winrate:.2%} < {self.min_winrate:.2%}. "
                f"Using base risk {base_risk_pct:.2f}%"
            )
            self.logger.info(f"[KELLY] {reason}")
            return base_risk_pct, 0.0, reason

        # =====================================================================
        # CHECK 3: Minimum Profit Factor
        # =====================================================================
        if avg_loss > 0:
            profit_factor = (winrate * avg_win) / ((1 - winrate) * avg_loss)
        else:
            profit_factor = 0.0

        if profit_factor < self.min_profit_factor:
            reason = (
                f"[BASE RISK] Profit factor too low: {profit_factor:.2f} < "
                f"{self.min_profit_factor:.2f}. Using base risk {base_risk_pct:.2f}%"
            )
            self.logger.info(f"[KELLY] {reason}")
            return base_risk_pct, 0.0, reason

        # =====================================================================
        # CHECK 4: Calculate Kelly Fraction
        # =====================================================================
        kelly_fraction, kelly_reason = self.calculate_kelly_fraction(
            winrate, avg_win, avg_loss
        )

        # Negative Kelly means the strategy has negative expectation
        if kelly_fraction <= 0:
            reason = (
                f"[BASE RISK] Negative Kelly fraction: {kelly_fraction:.4f}. "
                f"Strategy may not have edge. Using base risk {base_risk_pct:.2f}%"
            )
            self.logger.warning(f"[KELLY] {reason}")
            return base_risk_pct, kelly_fraction, reason

        # =====================================================================
        # CHECK 5: Convert Kelly Fraction to Risk Percentage
        # =====================================================================
        # Kelly fraction is the fraction of capital to risk
        # Convert to percentage
        kelly_risk_pct = kelly_fraction * 100.0

        # =====================================================================
        # CHECK 6: Apply Maximum Risk Cap
        # =====================================================================
        if kelly_risk_pct > self.max_risk_pct:
            reason = (
                f"[KELLY CAPPED] Kelly suggests {kelly_risk_pct:.2f}% but "
                f"capped at {self.max_risk_pct:.2f}%. "
                f"({kelly_reason})"
            )
            self.logger.info(f"[KELLY] {reason}")
            return self.max_risk_pct, kelly_fraction, reason

        # =====================================================================
        # CHECK 7: Apply Minimum Risk Floor
        # =====================================================================
        # Don't go below 0.1% (micro-account minimum meaningful risk)
        min_risk_pct = 0.1
        if kelly_risk_pct < min_risk_pct:
            reason = (
                f"[KELLY FLOORED] Kelly suggests {kelly_risk_pct:.2f}% but "
                f"floored at {min_risk_pct:.2f}%. "
                f"({kelly_reason})"
            )
            self.logger.info(f"[KELLY] {reason}")
            return min_risk_pct, kelly_fraction, reason

        # =====================================================================
        # ALL CHECKS PASSED - Use Kelly Risk
        # =====================================================================
        reason = (
            f"[KELLY APPLIED] Risk: {kelly_risk_pct:.2f}% "
            f"(base: {base_risk_pct:.2f}%, multiplier: {kelly_risk_pct/base_risk_pct:.2f}x). "
            f"({kelly_reason})"
        )
        self.logger.info(f"[KELLY] {reason}")

        return kelly_risk_pct, kelly_fraction, reason

    # =========================================================================
    # REGIME-BASED RISK ADJUSTMENT
    # =========================================================================

    def get_regime_adjusted_risk(self, kelly_risk_pct: float,
                                  unified_regime: str) -> float:
        """
        Adjust Kelly risk based on current market regime.

        Different market conditions warrant different risk levels:
            - TREND: Full Kelly (1.0x) - trending markets are favorable
            - SIDEWAY: Reduce 25% (0.75x) - range-bound markets
            - REVERSAL: Reduce 25% (0.75x) - reversal setups
            - HIGH_VOL: Reduce 50% (0.50x) - high volatility
            - UNKNOWN: Reduce 50% (0.50x) - safety first

        Args:
            kelly_risk_pct: Kelly-calculated risk percentage
            unified_regime: Unified regime name
                            (TREND, SIDEWAY, REVERSAL, HIGH_VOL, UNKNOWN)

        Returns:
            Regime-adjusted risk percentage
        """
        multiplier = self.REGIME_RISK_MULTIPLIERS.get(unified_regime, 0.50)
        adjusted_risk = kelly_risk_pct * multiplier

        # Ensure we don't go below minimum meaningful risk
        min_risk_pct = 0.1
        if adjusted_risk < min_risk_pct:
            adjusted_risk = min_risk_pct

        self.logger.debug(
            f"[KELLY] Regime adjustment: {unified_regime} | "
            f"multiplier: {multiplier:.2f} | "
            f"risk: {kelly_risk_pct:.2f}% -> {adjusted_risk:.2f}%"
        )

        return adjusted_risk

    # =========================================================================
    # CONSECUTIVE LOSS ADJUSTMENT
    # =========================================================================

    def get_consecutive_loss_adjusted_risk(self, kelly_risk_pct: float,
                                            consecutive_losses: int) -> float:
        """
        Adjust Kelly risk based on consecutive losses.

        After consecutive losses, reduce risk to prevent tilt
        and protect capital during losing streaks.

        Multipliers:
            0-1 losses: 1.00 (no change)
            2 losses: 0.90 (reduce 10%)
            3 losses: 0.75 (reduce 25%)
            4 losses: 0.50 (reduce 50%)
            5+ losses: 0.25 (reduce 75%)

        Args:
            kelly_risk_pct: Kelly-calculated risk percentage
            consecutive_losses: Number of consecutive losses

        Returns:
            Consecutive-loss-adjusted risk percentage
        """
        # Cap at 5 for lookup
        capped_losses = min(consecutive_losses, 5)
        multiplier = self.CONSECUTIVE_LOSS_MULTIPLIERS.get(capped_losses, 0.25)
        adjusted_risk = kelly_risk_pct * multiplier

        # Ensure we don't go below minimum meaningful risk
        min_risk_pct = 0.1
        if adjusted_risk < min_risk_pct:
            adjusted_risk = min_risk_pct

        if consecutive_losses > 0:
            self.logger.info(
                f"[KELLY] Consecutive loss adjustment: {consecutive_losses} losses | "
                f"multiplier: {multiplier:.2f} | "
                f"risk: {kelly_risk_pct:.2f}% -> {adjusted_risk:.2f}%"
            )

        return adjusted_risk

    # =========================================================================
    # DRAWDOWN-AWARE RISK ADJUSTMENT
    # =========================================================================

    def get_drawdown_adjusted_risk(self, kelly_risk_pct: float,
                                    current_drawdown_pct: float) -> float:
        """
        Adjust Kelly risk based on current drawdown.

        As drawdown increases, reduce risk to protect remaining capital.

        Thresholds:
            0% drawdown: 1.00 (no change)
            1% drawdown: 0.90 (reduce 10%)
            2% drawdown: 0.75 (reduce 25%)
            3% drawdown: 0.50 (reduce 50%)
            5% drawdown: 0.25 (reduce 75%)
            10%+ drawdown: 0.00 (stop trading)

        Args:
            kelly_risk_pct: Kelly-calculated risk percentage
            current_drawdown_pct: Current drawdown percentage (positive)

        Returns:
            Drawdown-adjusted risk percentage
        """
        # Find the appropriate multiplier
        multiplier = 1.0
        for threshold, mult in self.DRAWDOWN_RISK_THRESHOLDS:
            if current_drawdown_pct >= threshold:
                multiplier = mult

        adjusted_risk = kelly_risk_pct * multiplier

        if current_drawdown_pct > 0:
            self.logger.info(
                f"[KELLY] Drawdown adjustment: {current_drawdown_pct:.2f}% DD | "
                f"multiplier: {multiplier:.2f} | "
                f"risk: {kelly_risk_pct:.2f}% -> {adjusted_risk:.2f}%"
            )

        return adjusted_risk

    # =========================================================================
    # COMPREHENSIVE RISK CALCULATION
    # =========================================================================

    def calculate_comprehensive_risk(
        self,
        winrate: float,
        avg_win: float,
        avg_loss: float,
        base_risk_pct: float,
        trade_count: int,
        unified_regime: str = 'UNKNOWN',
        consecutive_losses: int = 0,
        current_drawdown_pct: float = 0.0
    ) -> Dict:
        """
        Calculate comprehensive Kelly risk with all adjustments.

        This method applies all risk adjustments in sequence:
            1. Base Kelly calculation
            2. Regime adjustment
            3. Consecutive loss adjustment
            4. Drawdown adjustment

        Args:
            winrate: Win rate as decimal (0.0 - 1.0)
            avg_win: Average winning trade amount (USD)
            avg_loss: Average losing trade amount (USD, positive)
            base_risk_pct: Base risk percentage
            trade_count: Number of trades in the sample
            unified_regime: Unified regime name
            consecutive_losses: Number of consecutive losses
            current_drawdown_pct: Current drawdown percentage

        Returns:
            Dict with comprehensive risk calculation results:
            {
                'final_risk_pct': float,      # Final risk to use
                'kelly_risk_pct': float,      # Kelly-calculated risk
                'kelly_fraction': float,      # Raw Kelly fraction
                'reason': str,                # Explanation
                'adjustments': {
                    'regime_multiplier': float,
                    'loss_multiplier': float,
                    'drawdown_multiplier': float
                }
            }
        """
        # Step 1: Base Kelly calculation
        kelly_risk_pct, kelly_fraction, kelly_reason = self.calculate_kelly_risk(
            winrate, avg_win, avg_loss, base_risk_pct, trade_count
        )

        # Step 2: Regime adjustment
        regime_multiplier = self.REGIME_RISK_MULTIPLIERS.get(unified_regime, 0.50)
        regime_adjusted = kelly_risk_pct * regime_multiplier

        # Step 3: Consecutive loss adjustment
        capped_losses = min(consecutive_losses, 5)
        loss_multiplier = self.CONSECUTIVE_LOSS_MULTIPLIERS.get(capped_losses, 0.25)
        loss_adjusted = regime_adjusted * loss_multiplier

        # Step 4: Drawdown adjustment
        drawdown_multiplier = 1.0
        for threshold, mult in self.DRAWDOWN_RISK_THRESHOLDS:
            if current_drawdown_pct >= threshold:
                drawdown_multiplier = mult
        drawdown_adjusted = loss_adjusted * drawdown_multiplier

        # Apply minimum risk floor
        min_risk_pct = 0.1
        final_risk_pct = max(drawdown_adjusted, min_risk_pct)

        # Apply maximum risk cap
        final_risk_pct = min(final_risk_pct, self.max_risk_pct)

        return {
            'final_risk_pct': final_risk_pct,
            'kelly_risk_pct': kelly_risk_pct,
            'kelly_fraction': kelly_fraction,
            'reason': kelly_reason,
            'adjustments': {
                'regime_multiplier': regime_multiplier,
                'loss_multiplier': loss_multiplier,
                'drawdown_multiplier': drawdown_multiplier,
                'total_multiplier': regime_multiplier * loss_multiplier * drawdown_multiplier
            }
        }

    # =========================================================================
    # DIAGNOSTICS & SUMMARY
    # =========================================================================

    def get_kelly_summary(self, winrate: float, avg_win: float,
                           avg_loss: float, trade_count: int) -> Dict:
        """
        Get comprehensive Kelly diagnostics summary.

        Args:
            winrate: Win rate as decimal (0.0 - 1.0)
            avg_win: Average winning trade amount (USD)
            avg_loss: Average losing trade amount (USD, positive)
            trade_count: Number of trades in the sample

        Returns:
            Dict with Kelly diagnostics
        """
        # Calculate components
        if winrate > 0 and winrate < 1 and avg_win > 0 and avg_loss > 0:
            p = winrate
            q = 1.0 - winrate
            b = avg_win / avg_loss

            full_kelly = (p * b - q) / b
            half_kelly = full_kelly / 2.0

            profit_factor = (winrate * avg_win) / ((1 - winrate) * avg_loss)
            expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)
        else:
            p = q = b = 0.0
            full_kelly = half_kelly = 0.0
            profit_factor = expectancy = 0.0

        # Determine data sufficiency
        data_sufficient = trade_count >= self.min_trades

        # Determine edge
        has_edge = profit_factor >= self.min_profit_factor and winrate >= self.min_winrate

        return {
            'inputs': {
                'winrate': winrate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'trade_count': trade_count,
            },
            'kelly_components': {
                'p': p,
                'q': q,
                'b': b,
                'full_kelly': full_kelly,
                'half_kelly': half_kelly,
                'kelly_pct': (half_kelly if self.use_half_kelly else full_kelly) * 100,
            },
            'strategy_metrics': {
                'profit_factor': profit_factor,
                'expectancy_usd': expectancy,
                'has_edge': has_edge,
            },
            'data_status': {
                'trade_count': trade_count,
                'min_trades_required': self.min_trades,
                'data_sufficient': data_sufficient,
                'trades_remaining': max(0, self.min_trades - trade_count),
            },
            'configuration': {
                'use_half_kelly': self.use_half_kelly,
                'max_risk_pct': self.max_risk_pct,
                'min_winrate': self.min_winrate,
                'min_profit_factor': self.min_profit_factor,
            }
        }

    def format_kelly_log(self, summary: Dict) -> str:
        """
        Format Kelly summary as concise log string.

        Args:
            summary: Result from get_kelly_summary()

        Returns:
            Formatted log string
        """
        if summary is None:
            return "[KELLY] No data available"

        inputs = summary.get('inputs', {})
        kelly = summary.get('kelly_components', {})
        metrics = summary.get('strategy_metrics', {})
        data = summary.get('data_status', {})

        return (
            f"[KELLY] Trades: {inputs.get('trade_count', 0)}/{data.get('min_trades_required', 50)} | "
            f"WR: {inputs.get('winrate', 0):.1%} | "
            f"PF: {metrics.get('profit_factor', 0):.2f} | "
            f"Exp: ${metrics.get('expectancy_usd', 0):.2f} | "
            f"Kelly: {kelly.get('kelly_pct', 0):.2f}% | "
            f"Edge: {'YES' if metrics.get('has_edge', False) else 'NO'} | "
            f"Data: {'OK' if data.get('data_sufficient', False) else 'INSUFFICIENT'}"
        )

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def validate_strategy_edge(self, winrate: float, avg_win: float,
                                avg_loss: float, trade_count: int) -> Dict:
        """
        Validate whether a strategy has sufficient edge for Kelly sizing.

        Args:
            winrate: Win rate as decimal
            avg_win: Average winning trade amount (USD)
            avg_loss: Average losing trade amount (USD, positive)
            trade_count: Number of trades

        Returns:
            Dict with validation results:
            {
                'has_sufficient_data': bool,
                'has_positive_edge': bool,
                'has_sufficient_winrate': bool,
                'has_sufficient_profit_factor': bool,
                'kelly_applicable': bool,
                'recommendations': List[str]
            }
        """
        recommendations = []

        # Check data sufficiency
        has_sufficient_data = trade_count >= self.min_trades
        if not has_sufficient_data:
            recommendations.append(
                f"Need {self.min_trades - trade_count} more trades "
                f"before Kelly can be applied"
            )

        # Check winrate
        has_sufficient_winrate = winrate >= self.min_winrate
        if not has_sufficient_winrate:
            recommendations.append(
                f"Winrate {winrate:.1%} below minimum {self.min_winrate:.1%}"
            )

        # Check profit factor
        if avg_loss > 0:
            profit_factor = (winrate * avg_win) / ((1 - winrate) * avg_loss)
        else:
            profit_factor = 0.0

        has_sufficient_profit_factor = profit_factor >= self.min_profit_factor
        if not has_sufficient_profit_factor:
            recommendations.append(
                f"Profit factor {profit_factor:.2f} below minimum "
                f"{self.min_profit_factor:.2f}"
            )

        # Check positive edge
        if avg_loss > 0:
            expectancy = (winrate * avg_win) - ((1 - winrate) * avg_loss)
        else:
            expectancy = 0.0

        has_positive_edge = expectancy > 0

        if not has_positive_edge:
            recommendations.append(
                f"Negative expectancy: ${expectancy:.2f} per trade"
            )

        # Determine if Kelly is applicable
        kelly_applicable = (
            has_sufficient_data and
            has_sufficient_winrate and
            has_sufficient_profit_factor and
            has_positive_edge
        )

        if kelly_applicable:
            recommendations.append("Kelly Criterion is applicable")

        return {
            'has_sufficient_data': has_sufficient_data,
            'has_positive_edge': has_positive_edge,
            'has_sufficient_winrate': has_sufficient_winrate,
            'has_sufficient_profit_factor': has_sufficient_profit_factor,
            'kelly_applicable': kelly_applicable,
            'profit_factor': profit_factor,
            'expectancy_usd': expectancy,
            'recommendations': recommendations
        }

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def get_configuration(self) -> Dict:
        """
        Get current Kelly configuration.

        Returns:
            Dict with all configuration parameters
        """
        return {
            'min_trades': self.min_trades,
            'max_risk_pct': self.max_risk_pct,
            'use_half_kelly': self.use_half_kelly,
            'min_winrate': self.min_winrate,
            'min_profit_factor': self.min_profit_factor,
            'regime_multipliers': self.REGIME_RISK_MULTIPLIERS.copy(),
            'consecutive_loss_multipliers': self.CONSECUTIVE_LOSS_MULTIPLIERS.copy(),
            'drawdown_thresholds': self.DRAWDOWN_RISK_THRESHOLDS.copy(),
        }

    def update_configuration(self, min_trades: int = None,
                              max_risk_pct: float = None,
                              use_half_kelly: bool = None,
                              min_winrate: float = None,
                              min_profit_factor: float = None):
        """
        Update Kelly configuration parameters.

        Args:
            min_trades: New minimum trade count (optional)
            max_risk_pct: New maximum risk cap (optional)
            use_half_kelly: New Half Kelly flag (optional)
            min_winrate: New minimum winrate (optional)
            min_profit_factor: New minimum profit factor (optional)
        """
        if min_trades is not None:
            self.min_trades = min_trades

        if max_risk_pct is not None:
            self.max_risk_pct = max_risk_pct

        if use_half_kelly is not None:
            self.use_half_kelly = use_half_kelly

        if min_winrate is not None:
            self.min_winrate = min_winrate

        if min_profit_factor is not None:
            self.min_profit_factor = min_profit_factor

        self.logger.info(
            f"[KELLY] Configuration updated | min_trades: {self.min_trades} | "
            f"max_risk: {self.max_risk_pct}% | "
            f"half_kelly: {self.use_half_kelly} | "
            f"min_winrate: {self.min_winrate} | "
            f"min_pf: {self.min_profit_factor}"
        )