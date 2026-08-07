"""
Risk Manager - Micro-Account-Only Edition (REVISED).
Handles position sizing, risk validation, and daily loss limits.
Simplified for Micro-Account Mode ($500-$3000 portfolio).

REVISION LOG:
  [REV-001] ADDED Regime-kill check (CHECK 0) per rule:
            "Regime-kill: ห้ามเทรด PARABOLIC_RALLY/PANIC_CAPITULATION/
             VOLATILE_CHOP/WHIPSAW_MARKET"
  [REV-002] FIXED calculate_position_size() now returns adjusted_sl.
            Previously, SL distance was capped for volume calculation
            but the actual SL remained wider, causing effective risk
            to exceed 0.5%.
  [REV-003] ADDED Same-direction exposure limit (anti-overtrade).
  [REV-004] ADDED Opposite-direction pending order check (anti-hedging)
            per rule: "ห้ามมี hedging, grid".
  [REV-005] ADDED Auto daily reset detection (prevents stale counters
            if event_loop forgets to call reset_daily).
  [REV-006] MOVED hardcoded limits to config with getattr fallback:
            - max_daily_trades
            - circuit_breaker_max_consec_loss
            - minimum_trading_balance
  [REV-007] ADDED regime_name parameter to validate_new_trade().

Features:
  - Risk-based position sizing (0.5% per trade)
  - Lot size clamping (0.01-0.03)
  - Maximum position limits
  - Daily loss limit (2%)
  - Regime-kill enforcement
  - Duplicate signal detection
  - Same-direction exposure limit
  - Anti-hedging protection
  - Spread validation
  - Consecutive loss tracking
"""
import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import config


class RiskManager:
    """
    Manages trading risk for Micro-Account Mode.

    Position sizing is based on:
      - Risk per trade: 0.5% of equity
      - SL distance: config.sl_distance_usd (fixed for XAUUSD)
      - Lot size: Clamped to config.min_lot_size - config.max_lot_size

    Validation Pipeline (12 checks):
      CHECK 0:  Regime-kill [REV-001]
      CHECK 1:  Max open positions
      CHECK 2:  Max pending orders
      CHECK 3:  Daily loss limit
      CHECK 4:  Duplicate signal detection
      CHECK 4b: Same-direction exposure [REV-003]
      CHECK 4c: Opposite-direction pending (anti-hedging) [REV-004]
      CHECK 5:  Spread validation
      CHECK 6:  Consecutive loss limit
      CHECK 7:  Daily trade count limit
      CHECK 8:  Minimum balance
      CHECK 9:  Position sizing (with adjusted_sl) [REV-002]
    """

    # =========================================================================
    # REGIME CONSTANTS [REV-001]
    # =========================================================================
    REGIME_KILL = [
        'PARABOLIC_RALLY', 'PANIC_CAPITULATION',
        'VOLATILE_CHOP', 'WHIPSAW_MARKET'
    ]

    # [REV-003] Max same-direction positions (anti-overtrade)
    MAX_SAME_DIRECTION = 2

    def __init__(self, risk_per_trade_pct: float = 0.5, max_open_positions: int = 2,
                 max_pending_orders: int = 2, max_daily_loss_pct: float = 2.0,
                 symbol: str = "XAUUSDm"):
        """
        Initialize RiskManager.

        Args:
            risk_per_trade_pct: Risk per trade as % of equity
            max_open_positions: Maximum concurrent open positions
            max_pending_orders: Maximum concurrent pending orders
            max_daily_loss_pct: Maximum daily loss as % of equity
            symbol: Trading symbol
        """
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_open_positions = max_open_positions
        self.max_pending_orders = max_pending_orders
        self.max_daily_loss_pct = max_daily_loss_pct
        self.symbol = symbol
        self.daily_start_capital = 0.0
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.logger = logging.getLogger(self.__class__.__name__)

        # [REV-005] Auto daily reset tracking
        self._last_reset_date = None

        self.logger.info(
            f"[RISK_MGR] Initialized | Risk: {self.risk_per_trade_pct:.2f}% | "
            f"Max Positions: {self.max_open_positions} | "
            f"Max Daily Loss: {self.max_daily_loss_pct:.2f}%"
        )
        self._symbol_info_cache = None
        self._symbol_info_cache_time = 0

    # =========================================================================
    # DAILY RESET & TRACKING
    # =========================================================================

    def reset_daily(self, current_balance: float):
        """
        Reset daily counters at start of new day.

        Args:
            current_balance: Current account balance
        """
        self.daily_start_capital = current_balance
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self._last_reset_date = datetime.now().date()
        self.logger.info(f"[RISK_MGR] Daily reset | Capital: ${current_balance:.2f}")

    def _auto_daily_reset(self):
        """
        [REV-005] Auto-detect new day and reset counters.

        Prevents stale daily counters if event_loop forgets to call
        reset_daily() explicitly.
        """
        today = datetime.now().date()
        if self._last_reset_date != today:
            acc = mt5.account_info()
            if acc:
                self.reset_daily(acc.balance)
            else:
                self._last_reset_date = today

    def update_daily_pnl(self, pnl: float):
        """
        Update daily PnL after a trade closes.

        Args:
            pnl: Trade profit/loss in USD
        """
        self.daily_pnl += pnl
        self.daily_trade_count += 1
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        self.logger.info(
            f"[RISK_MGR] Trade PnL: ${pnl:+.2f} | "
            f"Daily PnL: ${self.daily_pnl:+.2f} | "
            f"Trades today: {self.daily_trade_count} | "
            f"Consecutive losses: {self.consecutive_losses}"
        )

    # =========================================================================
    # POSITION SIZING [REV-002]
    # =========================================================================

    def calculate_position_size(self, entry_price: float, sl_price: float,
                                 account_balance: float, position_multiplier: float = 1.0,
                                 risk_pct: float = None) -> Tuple[float, str, float]:
        """
        Calculate position size based on risk percentage.

        Formula:
          risk_amount = balance * risk_pct / 100
          stop_distance = |entry_price - sl_price|
          raw_volume = risk_amount / (stop_distance * contract_size)
          volume = raw_volume * position_multiplier
          volume = clamp(volume, min_lot, max_lot)

        [REV-002] If stop_distance exceeds config.sl_distance_usd,
        the SL is ADJUSTED (not just capped in calculation) to ensure
        effective risk matches the calculated volume.

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            account_balance: Current account balance
            position_multiplier: Multiplier from signal scoring
            risk_pct: Override risk percentage (optional)

        Returns:
            Tuple of (volume, reason_string, adjusted_sl_price)
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade_pct

        adjusted_sl = sl_price  # [REV-002] Default: unchanged

        if entry_price <= 0 or sl_price <= 0 or account_balance <= 0:
            return 0.0, "[FAIL] Invalid prices or balance", adjusted_sl

        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return 0.0, f"[FAIL] Symbol {self.symbol} not found", adjusted_sl

        contract_size = getattr(symbol_info, 'trade_contract_size', 100)
        volume_min = getattr(symbol_info, 'volume_min', 0.01)
        volume_max = getattr(symbol_info, 'volume_max', 100.0)
        volume_step = getattr(symbol_info, 'volume_step', 0.01)

        # Calculate stop distance
        stop_distance = abs(entry_price - sl_price)
        if stop_distance == 0:
            return 0.0, "[FAIL] Stop distance is zero", adjusted_sl

        # Micro-Account: Override SL distance if configured
        max_sl_distance = getattr(config, 'sl_distance_usd', 16.0)
        if stop_distance > max_sl_distance:
            # [REV-002] ADJUST the actual SL to match capped distance
            if sl_price < entry_price:  # BUY: SL below entry
                adjusted_sl = entry_price - max_sl_distance
            else:  # SELL: SL above entry
                adjusted_sl = entry_price + max_sl_distance

            self.logger.info(
                f"[RISK_MGR] SL adjusted from {sl_price:.2f} to {adjusted_sl:.2f} "
                f"(max distance: {max_sl_distance:.2f} USD)"
            )
            stop_distance = max_sl_distance

        # Calculate risk amount
        risk_amount = account_balance * (risk_pct / 100.0)

        # Calculate raw volume
        raw_volume = risk_amount / (stop_distance * contract_size)

        # Apply position multiplier
        adjusted_volume = raw_volume * max(0.1, min(position_multiplier, 3.0))

        # Round to volume step
        volume = round(adjusted_volume / volume_step) * volume_step

        # Apply broker volume constraints
        volume = max(volume_min, min(volume, volume_max))

        # =========================================================================
        # MICRO-ACCOUNT LOT SIZE CLAMPING
        # =========================================================================
        micro_min_lot = getattr(config, 'min_lot_size', 0.01)
        micro_max_lot = getattr(config, 'max_lot_size', 0.03)
        original_volume = volume

        if volume < micro_min_lot:
            volume = micro_min_lot
            new_effective_risk = (volume * stop_distance * contract_size / account_balance) * 100.0
            self.logger.warning(
                f"[RISK_MGR] Calculated volume {original_volume:.2f} below minimum, "
                f"using {micro_min_lot} (effective risk: {new_effective_risk:.2f}%)"
            )
        elif volume > micro_max_lot:
            volume = micro_max_lot
            new_effective_risk = (volume * stop_distance * contract_size / account_balance) * 100.0
            self.logger.info(
                f"[RISK_MGR] Calculated volume {original_volume:.2f} above maximum, "
                f"capped at {micro_max_lot} (reduced risk: {new_effective_risk:.2f}%)"
            )
        else:
            effective_risk_pct = (volume * stop_distance * contract_size / account_balance) * 100.0
            self.logger.info(
                f"[RISK_MGR] Volume {volume:.2f} within bounds [{micro_min_lot}-{micro_max_lot}] "
                f"(risk: {effective_risk_pct:.2f}%)"
            )

        return (
            volume,
            f"[OK] Vol={volume:.2f} (Risk={risk_pct:.2f}%, SL_Dist={stop_distance:.2f})",
            adjusted_sl
        )

    # =========================================================================
    # RISK VALIDATION
    # =========================================================================

    def validate_new_trade(self, signal: dict, current_positions: List[Dict],
                            current_pending_orders: List[Dict], account_balance: float,
                            is_pending_order: bool = False, dynamic_risk_pct: float = None,
                            regime_name: str = 'UNKNOWN') -> Dict:
        """
        Validate whether a new trade should be placed.

        Checks:
          0.  Regime-kill [REV-001]
          1.  Max open positions limit
          2.  Max pending orders limit
          3.  Daily loss limit
          4.  Duplicate signal detection
          4b. Same-direction exposure [REV-003]
          4c. Opposite-direction pending (anti-hedging) [REV-004]
          5.  Spread validation
          6.  Consecutive loss limit
          7.  Daily trade count limit
          8.  Minimum balance
          9.  Position sizing (with adjusted_sl) [REV-002]

        Args:
            signal: Signal dict from strategy
            current_positions: List of current active positions
            current_pending_orders: List of current pending orders
            account_balance: Current account balance
            is_pending_order: Whether this is a pending order
            dynamic_risk_pct: Dynamic risk percentage from Kelly
            regime_name: [REV-007] Current unified regime name

        Returns:
            Dict with 'allowed', 'reason', 'suggested_volume', 'adjusted_sl'
        """
        result = {
            'allowed': False,
            'reason': 'Unknown error',
            'suggested_volume': 0.0,
            'adjusted_sl': 0.0  # [REV-002]
        }
        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')

        # [REV-005] Auto daily reset before validation
        self._auto_daily_reset()

        # =========================================================================
        # CHECK 0: REGIME-KILL [REV-001]
        # =========================================================================
        if regime_name in self.REGIME_KILL:
            result['reason'] = f"[FAIL] Regime-kill: {regime_name} is no-trade zone"
            self.logger.info(
                f"[RISK_MGR] {strategy_name} rejected by regime-kill: {regime_name}"
            )
            return result

        # =========================================================================
        # CHECK 1: Max Open Positions
        # =========================================================================
        if len(current_positions) >= self.max_open_positions:
            result['reason'] = (
                f"[FAIL] Max positions reached "
                f"({len(current_positions)}/{self.max_open_positions})"
            )
            return result

        # =========================================================================
        # CHECK 2: Max Pending Orders
        # =========================================================================
        if is_pending_order and len(current_pending_orders) >= self.max_pending_orders:
            result['reason'] = (
                f"[FAIL] Max pending orders reached "
                f"({len(current_pending_orders)}/{self.max_pending_orders})"
            )
            return result

        # =========================================================================
        # CHECK 3: Daily Loss Limit
        # =========================================================================
        if self.daily_start_capital > 0:
            daily_loss_pct = (self.daily_pnl / self.daily_start_capital) * 100.0
            if daily_loss_pct <= -self.max_daily_loss_pct:
                result['reason'] = f"[FAIL] Daily loss limit reached ({daily_loss_pct:.2f}%)"
                return result

        # =========================================================================
        # CHECK 4: Duplicate Signal Detection
        # =========================================================================
        duplicate_result = self._check_duplicate_signal(
            signal, current_positions, current_pending_orders
        )
        if duplicate_result['is_duplicate']:
            result['reason'] = f"[FAIL] Duplicate signal: {duplicate_result['reason']}"
            return result

        # =========================================================================
        # CHECK 4b: SAME-DIRECTION EXPOSURE [REV-003]
        # =========================================================================
        exposure_result = self._check_same_direction_exposure(signal, current_positions)
        if exposure_result['blocked']:
            result['reason'] = f"[FAIL] {exposure_result['reason']}"
            return result

        # =========================================================================
        # CHECK 4c: OPPOSITE-DIRECTION PENDING (ANTI-HEDGING) [REV-004]
        # =========================================================================
        hedge_result = self._check_opposite_direction(signal, current_pending_orders)
        if hedge_result['blocked']:
            result['reason'] = f"[FAIL] {hedge_result['reason']}"
            return result

        # =========================================================================
        # CHECK 5: Spread Validation
        # =========================================================================
        spread_result = self._check_spread()
        if not spread_result['valid']:
            result['reason'] = f"[FAIL] Spread check failed: {spread_result['reason']}"
            return result

        # =========================================================================
        # CHECK 6: Consecutive Loss Limit [REV-006]
        # =========================================================================
        max_consecutive_losses = getattr(config, 'circuit_breaker_max_consec_loss', 4)
        if self.consecutive_losses >= max_consecutive_losses:
            result['reason'] = (
                f"[FAIL] Consecutive losses limit reached "
                f"({self.consecutive_losses}/{max_consecutive_losses})"
            )
            return result

        # =========================================================================
        # CHECK 7: Daily Trade Count Limit [REV-006]
        # =========================================================================
        max_daily_trades = getattr(config, 'max_daily_trades', 10)
        if self.daily_trade_count >= max_daily_trades:
            result['reason'] = (
                f"[FAIL] Daily trade limit reached "
                f"({self.daily_trade_count}/{max_daily_trades})"
            )
            return result

        # =========================================================================
        # CHECK 8: Minimum Balance [REV-006]
        # =========================================================================
        minimum_balance = getattr(config, 'minimum_trading_balance', 100.0)
        if account_balance < minimum_balance:
            result['reason'] = (
                f"[FAIL] Balance ${account_balance:.2f} below minimum "
                f"${minimum_balance:.2f}"
            )
            return result

        # =========================================================================
        # CHECK 9: Position Sizing [REV-002]
        # =========================================================================
        entry_price = meta.get('entry_price', 0)
        sl_price = meta.get('sl_price', 0)
        position_multiplier = meta.get('position_multiplier', 1.0)

        if entry_price <= 0 or sl_price <= 0:
            result['reason'] = "[FAIL] Invalid entry or SL price"
            return result

        risk_pct = dynamic_risk_pct if dynamic_risk_pct else self.risk_per_trade_pct

        # [REV-002] Unpack 3-tuple including adjusted_sl
        volume, calc_reason, adjusted_sl = self.calculate_position_size(
            entry_price, sl_price, account_balance, position_multiplier, risk_pct
        )

        if volume <= 0:
            result['reason'] = f"[FAIL] Position sizing failed: {calc_reason}"
            return result

        result['allowed'] = True
        result['reason'] = calc_reason
        result['suggested_volume'] = volume
        result['adjusted_sl'] = adjusted_sl  # [REV-002] Pass back to order_manager

        self.logger.info(
            f"[RISK_MGR] {strategy_name} | Trade approved | "
            f"Volume: {volume:.2f} | {calc_reason}"
        )
        return result

    # =========================================================================
    # DUPLICATE SIGNAL DETECTION
    # =========================================================================

    def _check_duplicate_signal(self, signal: dict, current_positions: List[Dict],
                                 current_pending_orders: List[Dict]) -> Dict:
        """
        Check if signal is a duplicate of existing positions/orders.

        Args:
            signal: Signal dict
            current_positions: List of current positions
            current_pending_orders: List of current pending orders

        Returns:
            Dict with 'is_duplicate' and 'reason'
        """
        meta = signal.get('meta', {})
        strategy_name = meta.get('strategy', 'Unknown')

        # Check active positions
        for pos in current_positions:
            pos_meta = pos.get('meta_data', {})
            if pos_meta.get('strategy') == strategy_name:
                return {
                    'is_duplicate': True,
                    'reason': f"Active position with same strategy ({strategy_name})"
                }

        # Check pending orders
        for order in current_pending_orders:
            order_meta = order.get('meta_data', {})
            if order_meta.get('strategy') == strategy_name:
                return {
                    'is_duplicate': True,
                    'reason': f"Pending order with same strategy ({strategy_name})"
                }

        return {'is_duplicate': False, 'reason': ''}

    # =========================================================================
    # [REV-003] SAME-DIRECTION EXPOSURE CHECK
    # =========================================================================

    def _check_same_direction_exposure(self, signal: dict,
                                        current_positions: List[Dict]) -> Dict:
        """
        [REV-003] Check same-direction exposure limit (anti-overtrade).

        Prevents stacking too many positions in the same direction,
        which amplifies risk during regime shifts.

        Args:
            signal: Signal dict
            current_positions: List of current positions

        Returns:
            Dict with 'blocked' and 'reason'
        """
        signal_type = signal.get('signal', '')
        is_buy = 'BUY' in signal_type

        same_direction_count = 0
        for pos in current_positions:
            pos_type = pos.get('position_type', '')
            if (pos_type == 'BUY' and is_buy) or (pos_type == 'SELL' and not is_buy):
                same_direction_count += 1

        if same_direction_count >= self.MAX_SAME_DIRECTION:
            return {
                'blocked': True,
                'reason': (
                    f"Max same-direction positions reached "
                    f"({same_direction_count}/{self.MAX_SAME_DIRECTION})"
                )
            }

        return {'blocked': False, 'reason': ''}

    # =========================================================================
    # [REV-004] OPPOSITE-DIRECTION CHECK (ANTI-HEDGING)
    # =========================================================================

    def _check_opposite_direction(self, signal: dict,
                                   current_pending_orders: List[Dict]) -> Dict:
        """
        [REV-004] Check for opposite-direction pending orders (anti-hedging).

        Per rule: "ห้ามมี hedging, grid" - opening a position while an
        opposite pending order exists would create unintended hedging.

        Args:
            signal: Signal dict
            current_pending_orders: List of current pending orders

        Returns:
            Dict with 'blocked' and 'reason'
        """
        signal_type = signal.get('signal', '')
        is_buy = 'BUY' in signal_type

        for order in current_pending_orders:
            order_type = str(order.get('order_type', '')).upper()
            order_is_buy = 'BUY' in order_type

            if order_is_buy != is_buy:
                return {
                    'blocked': True,
                    'reason': (
                        f"Opposite-direction pending order exists "
                        f"(anti-hedging rule)"
                    )
                }

        return {'blocked': False, 'reason': ''}

    # =========================================================================
    # SPREAD VALIDATION
    # =========================================================================

    def _check_spread(self) -> Dict:
        """
        Check if current spread is within acceptable limits.

        Note: FrictionFilter performs deeper friction analysis.
        This check is defense-in-depth for extreme spread spikes.

        Returns:
            Dict with 'valid' and 'reason'
        """
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return {'valid': False, 'reason': 'Cannot get tick data'}

            symbol_info = self._get_symbol_info()
            if not symbol_info:
                return {'valid': False, 'reason': 'Cannot get symbol info'}

            current_spread = tick.ask - tick.bid
            spread_points = current_spread / symbol_info.point
            max_spread = getattr(config, 'max_spread_points', 30)

            if spread_points > max_spread:
                return {
                    'valid': False,
                    'reason': f'Spread {spread_points:.0f} pts > max {max_spread} pts'
                }

            return {'valid': True, 'reason': ''}

        except Exception as e:
            return {'valid': False, 'reason': f'Spread check error: {str(e)}'}

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_symbol_info(self):
        """Get symbol info with 5-second cache."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_cache_time < 5.0):
            return self._symbol_info_cache

        info = mt5.symbol_info(self.symbol)
        if info is None:
            if mt5.symbol_select(self.symbol, True):
                info = mt5.symbol_info(self.symbol)

        if info:
            self._symbol_info_cache = info
            self._symbol_info_cache_time = current_time
        return info

    def get_risk_summary(self) -> Dict:
        """
        Get current risk status summary.

        Returns:
            Dict with risk metrics
        """
        daily_loss_pct = 0.0
        if self.daily_start_capital > 0:
            daily_loss_pct = (self.daily_pnl / self.daily_start_capital) * 100.0

        return {
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_loss_pct': round(daily_loss_pct, 2),
            'daily_trade_count': self.daily_trade_count,
            'consecutive_losses': self.consecutive_losses,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_open_positions': self.max_open_positions,
            'max_pending_orders': self.max_pending_orders,
        }