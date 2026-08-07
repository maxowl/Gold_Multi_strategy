"""
Friction Filter - Micro-Account-Only Edition (REVISED).
Validates whether a trade is worth taking based on execution friction costs.
For micro-accounts, friction (spread + slippage + commission) represents a
significant percentage of potential profit.

REVISION LOG:
  [REV-001] FIXED Unit-mixing bug in friction cost calculation.
            Previously, commission (USD/oz) was summed directly with
            spread/slippage (price units), producing incorrect totals.
            All friction components are now computed in price units,
            then converted to USD using volume * contract_size.
  [REV-002] ADDED volume parameter to validate_entry().
            Without volume, min_profit_usd and edge/friction checks
            operated on price units instead of USD.
  [REV-003] FIXED CHECK 4/5/6 now operate entirely in USD.
  [REV-004] MOVED config access to getattr with safe fallbacks.
  [REV-005] ADDED strict_mode logging for audit trail.

Friction Components (per round-trip):
  - Spread: Bid-ask spread (typically 20-30 points for XAUUSD)
  - Slippage: Price movement between signal and execution
  - Commission: Broker commission per lot (typically $7/lot round-turn)

Validation Criteria (Micro-Account):
  1. Spread must be within limit (config.max_spread_points)
  2. Spread-to-SL ratio must be < 5%
  3. Net R:R must be >= min_rr_ratio after friction deduction
  4. Net profit must be >= min_profit_usd USD after friction
  5. Edge-to-friction ratio must be >= min_edge_to_friction

Integration Note:
  For accurate USD-based checks, call validate_entry() AFTER risk
  sizing so the actual volume is available. If volume is not provided,
  config.min_lot_size is used as a conservative estimate.
"""
import MetaTrader5 as mt5
import logging
import time
from typing import Dict, Optional

from config import config


class FrictionFilter:
    """
    Filters trades based on execution friction costs.

    For Micro-Account ($500-$3000), friction costs are critical:
      - Spread 25 points = $0.25 per 0.01 lot
      - Slippage 10 points = $0.10 per 0.01 lot
      - Commission $7/lot = $0.07 per 0.01 lot
      - Total friction ~$0.42 per 0.01 lot per round-trip
    """

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize FrictionFilter.

        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # [REV-004] Friction parameters from config with safe fallbacks
        self.max_spread_points = getattr(config, 'max_spread_points', 30)
        # Fallback chain: max_slippage_points_filter -> max_slippage_points -> 10
        self.max_slippage_points = getattr(
            config, 'max_slippage_points_filter',
            getattr(config, 'max_slippage_points', 10)
        )
        self.min_rr_ratio = getattr(config, 'min_rr_ratio', 1.3)
        self.min_profit_usd = getattr(config, 'min_profit_usd', 4.0)
        self.min_edge_to_friction = getattr(config, 'min_edge_to_friction', 2.5)

        # Micro-account default volume for conservative estimates [REV-002]
        self.default_volume = getattr(config, 'min_lot_size', 0.01)

        # Estimated commission per lot (round-trip)
        self.commission_per_lot = getattr(config, 'commission_per_lot', 7.0)

        # Spread-to-SL ratio limit
        self.max_spread_to_sl_ratio = getattr(config, 'max_spread_to_sl_ratio', 0.05)

        # Cache
        self._symbol_info_cache = None
        self._symbol_info_cache_time = 0

        self.logger.info(
            f"[FRICTION] Initialized | Max Spread: {self.max_spread_points} pts | "
            f"Min R:R: {self.min_rr_ratio} | Min Profit: ${self.min_profit_usd} | "
            f"Min Edge/Friction: {self.min_edge_to_friction}"
        )

    # =========================================================================
    # ENTRY VALIDATION
    # =========================================================================

    def validate_entry(self, signal: dict, current_atr: float = 0.0,
                       strict_mode: bool = False, volume: float = None) -> Dict:
        """
        Validate whether a trade is worth taking after friction costs.

        Checks:
          1. Spread within limit
          2. Spread-to-SL ratio acceptable
          3. Friction cost calculation
          4. Net R:R >= minimum after friction (USD-based) [REV-003]
          5. Net profit >= minimum USD after friction [REV-003]
          6. Edge-to-friction ratio >= minimum [REV-003]

        Args:
            signal: Signal dict with entry_price, sl_price, tp_price
            current_atr: Current ATR for volatility context (reserved)
            strict_mode: Whether to use stricter thresholds
            volume: [REV-002] Position volume for USD conversion.
                    If None, config.min_lot_size is used.

        Returns:
            Dict with 'valid', 'reason', 'spread_points', 'net_rr',
            'friction_usd', 'net_profit_usd', 'edge_to_friction'
        """
        meta = signal.get('meta', {})
        entry_price = meta.get('entry_price', 0)
        sl_price = meta.get('sl_price', 0)
        tp_price = meta.get('tp_price', 0)
        strategy_name = meta.get('strategy', 'Unknown')

        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            return {'valid': False, 'reason': 'Invalid prices'}

        # [REV-002] Use provided volume or fall back to min_lot_size
        if volume is None or volume <= 0:
            volume = self.default_volume
            self.logger.debug(
                f"[FRICTION] {strategy_name} | No volume provided, "
                f"using default {volume:.2f} (conservative estimate)"
            )

        # [REV-005] Log strict mode activation for audit trail
        if strict_mode:
            self.logger.info(
                f"[FRICTION] {strategy_name} | strict_mode ACTIVE "
                f"(thresholds tightened)"
            )

        # =========================================================================
        # CHECK 1: Spread Validation
        # =========================================================================
        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return {'valid': False, 'reason': 'Cannot get symbol info'}

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return {'valid': False, 'reason': 'Cannot get tick data'}

        current_spread = tick.ask - tick.bid
        point = getattr(symbol_info, 'point', 0.01)
        spread_points = current_spread / point if point > 0 else 0

        max_spread = self.max_spread_points
        if strict_mode:
            max_spread = int(max_spread * 0.8)  # 20% stricter

        if spread_points > max_spread:
            return {
                'valid': False,
                'reason': f'Spread too wide: {spread_points:.0f} pts (max: {max_spread})',
                'spread_points': spread_points
            }

        # =========================================================================
        # CHECK 2: Spread-to-SL Ratio
        # =========================================================================
        sl_distance = abs(entry_price - sl_price)
        if sl_distance > 0 and current_spread > 0:
            spread_to_sl_ratio = current_spread / sl_distance
            if spread_to_sl_ratio > self.max_spread_to_sl_ratio:
                return {
                    'valid': False,
                    'reason': (
                        f'Spread/SL ratio too high: {spread_to_sl_ratio:.2%} '
                        f'(max: {self.max_spread_to_sl_ratio:.0%})'
                    ),
                    'spread_points': spread_points
                }

        # =========================================================================
        # CHECK 3: Calculate Friction Cost [REV-001]
        # =========================================================================
        friction = self._calculate_friction_cost(symbol_info, current_spread, volume)
        friction_usd = friction['total_usd']

        # =========================================================================
        # CHECK 4: Net R:R After Friction (USD-based) [REV-003]
        # =========================================================================
        contract_size = friction['contract_size']

        gross_risk_usd = abs(entry_price - sl_price) * volume * contract_size
        gross_reward_usd = abs(tp_price - entry_price) * volume * contract_size

        net_risk_usd = gross_risk_usd + friction_usd
        net_reward_usd = gross_reward_usd - friction_usd

        if net_reward_usd <= 0:
            return {
                'valid': False,
                'reason': f'Negative net reward: ${net_reward_usd:.2f}',
                'spread_points': spread_points,
                'friction_usd': friction_usd
            }

        net_rr = net_reward_usd / net_risk_usd if net_risk_usd > 0 else 0

        min_rr = self.min_rr_ratio
        if strict_mode:
            min_rr = min_rr * 1.2  # 20% stricter

        if net_rr < min_rr:
            return {
                'valid': False,
                'reason': f'Net R:R too low: {net_rr:.2f} (min: {min_rr:.2f})',
                'spread_points': spread_points,
                'net_rr': net_rr,
                'friction_usd': friction_usd
            }

        # =========================================================================
        # CHECK 5: Minimum Profit After Friction (USD) [REV-003]
        # =========================================================================
        if net_reward_usd < self.min_profit_usd:
            return {
                'valid': False,
                'reason': (
                    f'Net profit too small: ${net_reward_usd:.2f} '
                    f'(min: ${self.min_profit_usd})'
                ),
                'spread_points': spread_points,
                'net_rr': net_rr,
                'friction_usd': friction_usd
            }

        # =========================================================================
        # CHECK 6: Edge-to-Friction Ratio [REV-003]
        # =========================================================================
        edge_to_friction = (
            net_reward_usd / friction_usd if friction_usd > 0 else float('inf')
        )

        required_ratio = self.min_edge_to_friction
        if strict_mode:
            required_ratio *= 1.3  # 30% stricter

        if edge_to_friction < required_ratio:
            return {
                'valid': False,
                'reason': (
                    f'Edge/Friction {edge_to_friction:.2f} < {required_ratio:.2f}'
                ),
                'spread_points': spread_points,
                'net_rr': net_rr,
                'friction_usd': friction_usd
            }

        # =========================================================================
        # ALL CHECKS PASSED
        # =========================================================================
        return {
            'valid': True,
            'reason': (
                f'OK | Spread: {spread_points:.0f} pts | '
                f'Net R:R: {net_rr:.2f} | '
                f'Net Profit: ${net_reward_usd:.2f} | '
                f'Friction: ${friction_usd:.2f}'
            ),
            'spread_points': spread_points,
            'net_rr': net_rr,
            'friction_usd': friction_usd,
            'net_profit_usd': net_reward_usd,
            'edge_to_friction': edge_to_friction,
            'volume_used': volume
        }

    # =========================================================================
    # FRICTION COST CALCULATION [REV-001]
    # =========================================================================

    def _calculate_friction_cost(self, symbol_info, current_spread: float,
                                  volume: float) -> Dict:
        """
        Calculate total friction cost for a round-trip trade.

        [REV-001] All components are first computed in PRICE UNITS,
        then converted to USD using (volume * contract_size).
        This eliminates the previous unit-mixing bug.

        Components:
          - Spread cost: current_spread (paid once on entry, price units)
          - Slippage cost: estimated slippage (price units)
          - Commission cost: USD per unit of contract (price units)

        Args:
            symbol_info: MT5 symbol info
            current_spread: Current spread in price units
            volume: Position volume in lots

        Returns:
            Dict with:
              - spread_cost_pu, slippage_cost_pu, commission_cost_pu (price units)
              - total_price_units (sum in price units)
              - contract_size
              - total_usd (total friction in USD for given volume)
        """
        point = getattr(symbol_info, 'point', 0.01)
        contract_size = getattr(symbol_info, 'trade_contract_size', 100)

        # --- Component 1: Spread (already in price units) ---
        spread_cost_pu = current_spread

        # --- Component 2: Slippage (price units) ---
        slippage_cost_pu = self.max_slippage_points * point

        # --- Component 3: Commission (convert USD/lot to price units) ---
        # $7 per lot, 1 lot = contract_size units
        # => price units per unit = commission_per_lot / contract_size
        commission_cost_pu = self.commission_per_lot / contract_size if contract_size > 0 else 0.0

        # --- Total in price units ---
        total_price_units = spread_cost_pu + slippage_cost_pu + commission_cost_pu

        # --- Convert to USD for the given volume ---
        # USD = price_units * volume * contract_size
        total_usd = total_price_units * volume * contract_size

        return {
            'spread_cost_pu': spread_cost_pu,
            'slippage_cost_pu': slippage_cost_pu,
            'commission_cost_pu': commission_cost_pu,
            'total_price_units': total_price_units,
            'contract_size': contract_size,
            'volume': volume,
            'total_usd': total_usd
        }

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

    def get_friction_summary(self, volume: float = None) -> Dict:
        """
        Get current friction cost summary.

        Args:
            volume: Position volume for USD conversion (optional)

        Returns:
            Dict with friction components in both price units and USD
        """
        if volume is None or volume <= 0:
            volume = self.default_volume

        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return {'valid': False, 'reason': 'Cannot get symbol info'}

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return {'valid': False, 'reason': 'Cannot get tick data'}

        current_spread = tick.ask - tick.bid
        point = getattr(symbol_info, 'point', 0.01)
        spread_points = current_spread / point if point > 0 else 0

        friction = self._calculate_friction_cost(symbol_info, current_spread, volume)

        return {
            'valid': True,
            'spread_points': spread_points,
            'spread_cost_pu': friction['spread_cost_pu'],
            'slippage_cost_pu': friction['slippage_cost_pu'],
            'commission_cost_pu': friction['commission_cost_pu'],
            'total_price_units': friction['total_price_units'],
            'total_usd': friction['total_usd'],
            'volume': volume,
            'contract_size': friction['contract_size'],
            'max_spread_points': self.max_spread_points,
        }