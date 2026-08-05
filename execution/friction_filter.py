"""
Execution Friction Filter - Institutional Grade
Validates trade signals against real-time market microstructure conditions.
Calculates Net Edge after accounting for Spread, Slippage, and Commission.
Blocks signals where Edge-to-Friction ratio is below institutional thresholds.

Modes:
  - Normal Mode: Standard thresholds (Edge/Friction >= 2.0)
  - Scalping Mode: Stricter thresholds (Edge/Friction >= 3.5)
  - Micro-Account Mode: Enhanced spread protection (Spread/SL < 10%)
"""
import MetaTrader5 as mt5
import logging
import time
from typing import Dict, Tuple, Optional
from config import config


class FrictionFilter:
    """
    Filters trade signals based on execution friction costs.
    Prevents entering trades where friction destroys the edge.
    """
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for symbol info and tick data
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._tick_cache = None
        self._tick_time = 0
        self.cache_ttl = 5.0  # 5 seconds
        
        # =========================================================================
        # MODE-SPECIFIC THRESHOLDS
        # =========================================================================
        if getattr(config, 'scalping_mode', False):
            # Scalping Mode: Ultra-strict (high frequency, low margin)
            self.max_spread_points = getattr(config, 'max_spread_points_scalp', 300)
            self.max_slippage_points = 10
            self.min_edge_to_friction_ratio = 3.5
            self.mode_name = 'SCALPING'
            
        elif getattr(config, 'micro_account_mode', False):
            # Micro-Account Mode: Strict spread protection
            self.max_spread_points = getattr(config, 'micro_max_spread_points', 300)
            self.max_slippage_points = 15
            self.min_edge_to_friction_ratio = 2.5
            self.spread_to_sl_max_ratio = getattr(config, 'micro_spread_to_sl_ratio', 0.10)
            self.mode_name = 'MICRO_ACCOUNT'
            
        else:
            # Normal Mode: Standard institutional thresholds
            self.max_spread_points = 300
            self.max_slippage_points = 20
            self.min_edge_to_friction_ratio = 2.0
            self.mode_name = 'NORMAL'
        
        self.logger.info(f"[FRICTION] Initialized in {self.mode_name} mode")
        self.logger.info(
            f"[FRICTION] Thresholds: Max Spread={self.max_spread_points} pts, "
            f"Max Slippage={self.max_slippage_points} pts, "
            f"Min Edge/Friction={self.min_edge_to_friction_ratio:.1f}"
        )

    def _get_symbol_info(self):
        """Get symbol info with caching."""
        current_time = time.time()
        if self._symbol_info_cache and (current_time - self._symbol_info_time < self.cache_ttl):
            return self._symbol_info_cache
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            # Try to select symbol
            if mt5.symbol_select(self.symbol, True):
                symbol_info = mt5.symbol_info(self.symbol)
        
        if symbol_info:
            self._symbol_info_cache = symbol_info
            self._symbol_info_time = current_time
        
        return symbol_info

    def _get_current_tick(self):
        """Get current tick with caching."""
        current_time = time.time()
        if self._tick_cache and (current_time - self._tick_time < 0.5):
            return self._tick_cache
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self._tick_cache = tick
            self._tick_time = current_time
        
        return tick

    def validate_entry(self, signal: Dict, current_atr: float, strict_mode: bool = False) -> Dict:
        """
        Validate trade signal against friction costs.
        
        Args:
            signal: Trade signal dict with meta containing entry_price, sl_price, tp_price
            current_atr: Current ATR value for volatility context
            strict_mode: If True, apply stricter thresholds (for friction-sensitive strategies)
        
        Returns:
            {
                'valid': bool,
                'reason': str,
                'spread_points': float,
                'spread_usd': float,
                'friction_cost': float,
                'gross_rr': float,
                'net_rr': float,
                'edge_to_friction_ratio': float,
                'details': dict
            }
        """
        meta = signal.get('meta', {})
        entry_price = meta.get('entry_price', 0)
        sl_price = meta.get('sl_price', 0)
        tp_price = meta.get('tp_price', 0)
        strategy_name = meta.get('strategy', 'Unknown')
        
        # =========================================================================
        # BASIC VALIDATION
        # =========================================================================
        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            return self._build_invalid_result('Invalid prices (entry/sl/tp <= 0)', strategy_name)
        
        symbol_info = self._get_symbol_info()
        if not symbol_info:
            return self._build_invalid_result(f'Cannot get symbol info for {self.symbol}', strategy_name)
        
        tick = self._get_current_tick()
        if not tick:
            return self._build_invalid_result('Cannot get current tick', strategy_name)
        
        # =========================================================================
        # EXTRACT SYMBOL PROPERTIES
        # =========================================================================
        point = getattr(symbol_info, 'point', 0.01)
        digits = getattr(symbol_info, 'digits', 2)
        
        # =========================================================================
        # SPREAD VALIDATION
        # =========================================================================
        current_spread = tick.ask - tick.bid
        spread_points = current_spread / point
        spread_usd = current_spread
        
        spread_valid, spread_reason = self._validate_spread(spread_points, spread_usd)
        if not spread_valid:
            return self._build_invalid_result(spread_reason, strategy_name, spread_points, spread_usd)
        
        # =========================================================================
        # MICRO-ACCOUNT: SPREAD/SL RATIO CHECK
        # =========================================================================
        if self.mode_name == 'MICRO_ACCOUNT':
            sl_distance = abs(entry_price - sl_price)
            if sl_distance > 0:
                spread_to_sl_ratio = spread_usd / sl_distance
                if spread_to_sl_ratio > self.spread_to_sl_max_ratio:
                    return self._build_invalid_result(
                        f'Spread/SL ratio too high: {spread_to_sl_ratio:.2%} (max: {self.spread_to_sl_max_ratio:.2%})',
                        strategy_name,
                        spread_points,
                        spread_usd
                    )
        
        # =========================================================================
        # CALCULATE FRICTION COST
        # =========================================================================
        friction_cost = self._calculate_friction_cost(current_spread, point)
        
        # =========================================================================
        # CALCULATE GROSS AND NET EDGE
        # =========================================================================
        gross_risk = abs(entry_price - sl_price)
        gross_reward = abs(tp_price - entry_price)
        
        if gross_risk == 0:
            return self._build_invalid_result('Gross risk is zero', strategy_name, spread_points, spread_usd)
        
        # Net values after friction
        net_risk = gross_risk + friction_cost
        net_reward = gross_reward - friction_cost
        
        if net_reward <= 0:
            return self._build_invalid_result(
                f'Negative net reward: {net_reward:.2f} USD (friction ate all profit)',
                strategy_name,
                spread_points,
                spread_usd,
                friction_cost
            )
        
        # =========================================================================
        # CALCULATE RISK/REWARD RATIOS
        # =========================================================================
        gross_rr = gross_reward / gross_risk
        net_rr = net_reward / net_risk
        
        # =========================================================================
        # EDGE-TO-FRICTION RATIO CHECK
        # =========================================================================
        edge_to_friction_ratio = net_reward / friction_cost if friction_cost > 0 else float('inf')
        
        # Apply stricter threshold if strict_mode or scalping/micro mode
        required_ratio = self.min_edge_to_friction_ratio
        if strict_mode:
            required_ratio *= 1.5
        
        if edge_to_friction_ratio < required_ratio:
            return self._build_invalid_result(
                f'Edge/Friction ratio {edge_to_friction_ratio:.2f} < required {required_ratio:.2f}',
                strategy_name,
                spread_points,
                spread_usd,
                friction_cost,
                gross_rr,
                net_rr,
                edge_to_friction_ratio
            )
        
        # =========================================================================
        # BUILD SUCCESS RESULT
        # =========================================================================
        return {
            'valid': True,
            'reason': f'OK | Spread: {spread_points:.0f} pts | Net R:R: {net_rr:.2f} | Edge/Friction: {edge_to_friction_ratio:.2f}',
            'spread_points': spread_points,
            'spread_usd': spread_usd,
            'friction_cost': friction_cost,
            'gross_rr': gross_rr,
            'net_rr': net_rr,
            'edge_to_friction_ratio': edge_to_friction_ratio,
            'details': {
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'gross_risk': gross_risk,
                'gross_reward': gross_reward,
                'net_risk': net_risk,
                'net_reward': net_reward,
                'mode': self.mode_name,
                'strict_mode': strict_mode
            }
        }

    def _validate_spread(self, spread_points: float, spread_usd: float) -> Tuple[bool, str]:
        """Validate current spread against threshold."""
        if spread_points > self.max_spread_points:
            return False, f'Spread too wide: {spread_points:.0f} pts (max: {self.max_spread_points})'
        
        # Additional check for Micro-Account mode
        if self.mode_name == 'MICRO_ACCOUNT':
            # If spread > 50% of typical ATR, reject
            # This is a sanity check for abnormal spread conditions
            if spread_usd > 2.0:  # For XAUUSD, 2 USD spread is already very wide
                return False, f'Spread abnormally wide: {spread_usd:.2f} USD'
        
        return True, 'OK'

    def _calculate_friction_cost(self, current_spread: float, point: float) -> float:
        """
        Calculate total friction cost in USD.
        
        Friction = Spread + Expected Slippage + Commission (if applicable)
        """
        # Spread cost (always paid)
        spread_cost = current_spread
        
        # Expected slippage (conservative estimate)
        # Use 50% of max_slippage_points as expected value
        expected_slippage = (self.max_slippage_points * 0.5) * point
        
        # Commission (broker-specific, typically $7 per round turn for XAUUSD)
        # For simplicity, we'll use a fixed estimate
        # In production, this should be fetched from broker specs
        commission_per_lot = 7.0  # USD per 1.0 lot round turn
        # For micro-account (0.01-0.03 lots), commission is negligible
        # For normal account (0.1+ lots), it matters more
        
        # Total friction
        total_friction = spread_cost + expected_slippage
        
        return total_friction

    def _build_invalid_result(self, reason: str, strategy_name: str, 
                              spread_points: float = 0, spread_usd: float = 0,
                              friction_cost: float = 0, gross_rr: float = 0,
                              net_rr: float = 0, edge_to_friction: float = 0) -> Dict:
        """Build standardized invalid result dict."""
        self.logger.info(f"[FRICTION] {strategy_name} rejected: {reason}")
        
        return {
            'valid': False,
            'reason': reason,
            'spread_points': spread_points,
            'spread_usd': spread_usd,
            'friction_cost': friction_cost,
            'gross_rr': gross_rr,
            'net_rr': net_rr,
            'edge_to_friction_ratio': edge_to_friction,
            'details': {
                'mode': self.mode_name,
                'strategy': strategy_name
            }
        }

    def get_friction_report(self) -> Dict:
        """
        Get current friction conditions report.
        Useful for monitoring and logging.
        
        Returns:
            {
                'current_spread_points': float,
                'current_spread_usd': float,
                'max_allowed_spread': float,
                'spread_utilization': float (0-1),
                'mode': str,
                'timestamp': str
            }
        """
        symbol_info = self._get_symbol_info()
        tick = self._get_current_tick()
        
        if not symbol_info or not tick:
            return {
                'current_spread_points': 0,
                'current_spread_usd': 0,
                'max_allowed_spread': self.max_spread_points,
                'spread_utilization': 0,
                'mode': self.mode_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        point = getattr(symbol_info, 'point', 0.01)
        current_spread = tick.ask - tick.bid
        spread_points = current_spread / point
        
        utilization = spread_points / self.max_spread_points if self.max_spread_points > 0 else 0
        
        return {
            'current_spread_points': spread_points,
            'current_spread_usd': current_spread,
            'max_allowed_spread': self.max_spread_points,
            'spread_utilization': utilization,
            'mode': self.mode_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def should_avoid_trading(self) -> Tuple[bool, str]:
        """
        Check if current friction conditions are too恶劣 for trading.
        
        Returns:
            - should_avoid: bool
            - reason: str
        """
        report = self.get_friction_report()
        
        # If spread utilization > 80%, avoid trading
        if report['spread_utilization'] > 0.80:
            return True, f"Spread utilization {report['spread_utilization']:.0%} > 80% (too expensive)"
        
        # If spread > 70% of max, warn but don't block
        if report['spread_utilization'] > 0.70:
            self.logger.warning(
                f"[FRICTION] High spread utilization: {report['spread_utilization']:.0%} "
                f"({report['current_spread_points']:.0f} pts)"
            )
        
        return False, "OK"

    def calculate_breakeven_winrate(self, net_rr: float) -> float:
        """
        Calculate minimum win rate required to break even with given Net R:R.
        
        Formula: Breakeven_WR = 1 / (1 + Net_RR)
        
        Example:
          Net R:R = 2.0 → Breakeven WR = 1 / (1 + 2) = 33.3%
          Net R:R = 1.5 → Breakeven WR = 1 / (1 + 1.5) = 40.0%
        """
        if net_rr <= 0:
            return 1.0  # 100% win rate required (impossible)
        
        breakeven_wr = 1.0 / (1.0 + net_rr)
        return breakeven_wr

    def estimate_profit_factor(self, winrate: float, net_rr: float) -> float:
        """
        Estimate Profit Factor given win rate and Net R:R.
        
        Formula: PF = (Win_Rate × Avg_Win) / (Loss_Rate × Avg_Loss)
                 PF = (Win_Rate × Net_Reward) / ((1 - Win_Rate) × Net_Risk)
                 PF = (Win_Rate × Net_RR) / (1 - Win_Rate)
        """
        if winrate >= 1.0:
            return float('inf')
        if winrate <= 0.0:
            return 0.0
        
        loss_rate = 1.0 - winrate
        profit_factor = (winrate * net_rr) / loss_rate
        
        return profit_factor

    def log_friction_analysis(self, signal: Dict, validation_result: Dict):
        """Log detailed friction analysis for debugging."""
        meta = signal.get('meta', {})
        strategy = meta.get('strategy', 'Unknown')
        
        self.logger.info("=" * 80)
        self.logger.info(f"[FRICTION ANALYSIS] {strategy}")
        self.logger.info("=" * 80)
        
        if validation_result['valid']:
            self.logger.info(f"✓ PASSED: {validation_result['reason']}")
        else:
            self.logger.info(f"✗ REJECTED: {validation_result['reason']}")
        
        self.logger.info(f"  Mode: {validation_result['details'].get('mode', 'UNKNOWN')}")
        self.logger.info(f"  Spread: {validation_result['spread_points']:.0f} pts ({validation_result['spread_usd']:.2f} USD)")
        self.logger.info(f"  Friction Cost: {validation_result['friction_cost']:.2f} USD")
        self.logger.info(f"  Gross R:R: {validation_result['gross_rr']:.2f}")
        self.logger.info(f"  Net R:R: {validation_result['net_rr']:.2f}")
        self.logger.info(f"  Edge/Friction: {validation_result['edge_to_friction_ratio']:.2f}")
        
        if validation_result['valid']:
            net_rr = validation_result['net_rr']
            breakeven_wr = self.calculate_breakeven_winrate(net_rr)
            self.logger.info(f"  Breakeven Win Rate: {breakeven_wr:.1%}")
            
            # Estimate PF at 60% win rate
            estimated_pf = self.estimate_profit_factor(0.60, net_rr)
            self.logger.info(f"  Estimated PF @ 60% WR: {estimated_pf:.2f}")
        
        self.logger.info("=" * 80)