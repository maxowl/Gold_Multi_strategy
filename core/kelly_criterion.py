"""
Kelly Criterion Engine.
Calculates optimal position sizing based on historical performance.
Includes 6 circuit breakers for safety.
"""
import logging
from typing import Tuple


class KellyCriterionEngine:
    def __init__(self, min_trades: int = 30, max_risk_pct: float = 3.0):
        """
        Initialize Kelly Criterion engine.
        
        Args:
            min_trades: Minimum trades required for Kelly calculation
            max_risk_pct: Maximum risk percentage cap
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.min_trades = min_trades
        self.max_risk_pct = max_risk_pct

    def calculate_kelly_risk(
        self, winrate: float, avg_win: float, avg_loss: float,
        base_risk_pct: float, trade_count: int
    ) -> Tuple[float, float, str]:
        """
        Calculate Kelly Criterion risk percentage.
        
        Half-Kelly Formula: f* = 0.5 * (bp - q) / b
        where:
            b = avg_win / avg_loss (odds ratio)
            p = winrate
            q = 1 - p
        
        Circuit Breakers:
        1. Min 30 trades (Cold Start Fallback)
        2. 100% Win Rate Cap (Max 3% risk)
        3. Zero Avg Win/Loss Block (Breakeven Trap)
        4. Corrupted Data Block
        5. Negative Expectancy Block
        6. Hard Cap 3% Max Risk
        
        Returns:
            Tuple of (kelly_risk_pct, kelly_fraction, reason)
        """
        # Circuit Breaker 1: Min trades
        if trade_count < self.min_trades:
            return base_risk_pct, 0.0, f"Cold Start: {trade_count} < {self.min_trades} trades"
        
        # Circuit Breaker 4: Corrupted data
        if winrate < 0 or winrate > 1 or avg_win < 0 or avg_loss <= 0:
            return base_risk_pct, 0.0, f"Corrupted data: wr={winrate:.2f}, win={avg_win:.2f}, loss={avg_loss:.2f}"
        
        # Circuit Breaker 2: 100% win rate cap
        if winrate >= 1.0:
            capped_risk = min(self.max_risk_pct, base_risk_pct * 2.0)
            return capped_risk, 0.0, f"100% Win Rate Cap: {capped_risk:.2f}%"
        
        # Circuit Breaker 3: Zero avg win OR zero avg loss (prevent division by zero)
        if avg_win == 0 or avg_loss == 0:
            return base_risk_pct, 0.0, "Zero Avg Win/Loss: Breakeven trap detected"
        
        # Calculate odds ratio (safe now - avg_loss != 0)
        b = avg_win / avg_loss
        p = winrate
        q = 1.0 - p
        
        # Full Kelly fraction
        full_kelly = (b * p - q) / b
        
        # Circuit Breaker 5: Negative expectancy
        if full_kelly <= 0:
            return base_risk_pct, full_kelly, f"Negative Expectancy: Kelly={full_kelly:.4f}"
        
        # Half-Kelly for safety
        half_kelly = full_kelly * 0.5
        
        # Convert to percentage
        kelly_risk_pct = half_kelly * 100.0
        
        # Circuit Breaker 6: Hard cap
        if kelly_risk_pct > self.max_risk_pct:
            kelly_risk_pct = self.max_risk_pct
            reason = f"Hard Cap: {kelly_risk_pct:.2f}% (calculated {half_kelly * 100:.2f}%)"
        else:
            reason = f"Kelly: {kelly_risk_pct:.2f}% (fraction={half_kelly:.4f})"
        
        return kelly_risk_pct, half_kelly, reason