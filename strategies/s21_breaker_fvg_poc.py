"""
S21_BreakerFVGPOC - Breaker + FVG + POC Confluence Strategy.

Smart Money Concepts strategy that combines Breaker Blocks, Fair Value
Gaps (FVG), and Point of Control (POC) for high-probability entries.

Strategy Logic:
  1. Detect Breaker Blocks (failed Order Blocks)
  2. Detect Fair Value Gaps (FVGs)
  3. Detect Point of Control (POC) from Volume Profile
  4. Find confluence zones where all three align
  5. Generate entry signal at confluence

Breaker Block Definition:
  A Breaker is an Order Block that was broken and now acts as
  the opposite type of support/resistance.
  - Broken Bullish OB → Acts as Resistance
  - Broken Bearish OB → Acts as Support

Fair Value Gap (FVG) Definition:
  A three-candle pattern where the middle candle creates an
  imbalance between candle 1 and candle 3, leaving a gap
  that price tends to fill.

Point of Control (POC) Definition:
  The price level with the highest volume in the Volume Profile.
  Acts as a magnet for price and strong support/resistance.

Confluence Logic:
  When Breaker + FVG + POC align at the same price level,
  it creates a high-probability entry zone.

Used Engines:
  - SMCStructuralEngine: Breaker and FVG detection
  - BreakerVPEngine: POC detection
  - AdaptiveTPEngine: Dynamic TP calculation

Strategy Category: SMC
Timeframe: M15 (primary), M5 (confirmation)
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from core.base_strategy import BaseStrategy
from core.smc_engine import SMCStructuralEngine
from core.breaker_vp_engine import BreakerVPEngine
from core.adaptive_tp_engine import AdaptiveTPEngine


class S21_BreakerFVGPOC(BaseStrategy):
    """
    Breaker + FVG + POC Confluence Strategy.
    
    This strategy combines Breaker Blocks, Fair Value Gaps, and
    Point of Control for high-probability entries.
    
    Breaker Block Definition:
      A Breaker is an Order Block that was broken and now acts
      as the opposite type of support/resistance.
      
    FVG Definition:
      A three-candle pattern with an imbalance that creates
      a gap price tends to fill.
      
    POC Definition:
      The price level with the highest volume in the Volume Profile.
      
    Entry Criteria:
      - Breaker Block detected
      - FVG nearby or overlapping
      - POC at the same level
      - Confirmation from M5 timeframe
    """

    def __init__(self):
        """Initialize S21_BreakerFVGPOC strategy."""
        super().__init__(
            strategy_name='S21_BreakerFVGPOC',
            strategy_category='SMC',
            timeframes=['M15', 'M5'],
            risk_per_trade_pct=0.5,
            min_rr_ratio=1.8,
            max_spread_points=30,
            trailing_enabled=True,
            partial_close_enabled=True,
            requires_dynamic_exit=False
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize engines
        self.smc_engine = SMCStructuralEngine()
        self.vp_engine = BreakerVPEngine()
        self.adaptive_tp_engine = AdaptiveTPEngine()

        # Strategy parameters
        self.breaker_lookback = 100  # Lookback for breaker detection
        self.fvg_lookback = 50  # Lookback for FVG detection
        self.poc_bins = 100  # Volume Profile bins
        self.confluence_tolerance = 0.003  # Confluence tolerance (0.3%)

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        df_m15: pd.DataFrame,
        df_m5: pd.DataFrame = None,
        regime_context: Dict = None
    ) -> Dict:
        """
        Main analysis method for S21_BreakerFVGPOC.
        
        Args:
            df_m15: M15 DataFrame
            df_m5: M5 DataFrame (optional)
            regime_context: Current regime information
            
        Returns:
            Signal dict with entry/exit information
        """
        # Default neutral signal
        default_signal = self._create_neutral_signal()

        # Validate input
        if df_m15 is None or df_m15.empty or len(df_m15) < 100:
            return default_signal

        try:
            # Check regime compatibility
            if regime_context and not self._is_regime_compatible(regime_context):
                return default_signal

            # =========================================================================
            # STEP 1: Detect Breaker Blocks
            # =========================================================================
            breakers = self.smc_engine.detect_breaker_blocks(df_m15, lookback=self.breaker_lookback)

            if not breakers:
                return default_signal

            # =========================================================================
            # STEP 2: Detect FVGs
            # =========================================================================
            fvgs = self.smc_engine.detect_fvg(df_m15, lookback=self.fvg_lookback)

            if not fvgs:
                return default_signal

            # =========================================================================
            # STEP 3: Detect POC
            # =========================================================================
            poc_result = self._detect_poc(df_m15)

            if poc_result is None:
                return default_signal

            # =========================================================================
            # STEP 4: Find Confluence
            # =========================================================================
            confluence = self._find_confluence(breakers, fvgs, poc_result, df_m15)

            if confluence is None:
                return default_signal

            # =========================================================================
            # STEP 5: M5 Confirmation
            # =========================================================================
            if df_m5 is not None and not df_m5.empty:
                if not self._confirm_m5(df_m5, confluence):
                    return default_signal

            # =========================================================================
            # STEP 6: Generate Signal
            # =========================================================================
            signal = self._generate_signal(df_m15, confluence, regime_context)

            return signal

        except Exception as e:
            self.logger.error(f"[S21_BFP] Analysis error: {e}")
            return default_signal

    # =========================================================================
    # POC DETECTION
    # =========================================================================

    def _detect_poc(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Point of Control from Volume Profile.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            POC result dict or None
        """
        try:
            vp_result = self.vp_engine.calculate_volume_profile_levels(
                df, bins=self.poc_bins
            )

            if vp_result is None:
                return None

            poc = vp_result.get('poc', 0)
            vah = vp_result.get('vah', 0)
            val = vp_result.get('val', 0)

            if poc <= 0:
                return None

            return {
                'poc': float(poc),
                'vah': float(vah),
                'val': float(val),
                'has_poc': True
            }

        except Exception as e:
            self.logger.debug(f"[S21_BFP] POC detection error: {e}")
            return None

    # =========================================================================
    # CONFLUENCE DETECTION
    # =========================================================================

    def _find_confluence(
        self, breakers: List[Dict], fvgs: List[Dict], poc_result: Dict, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Find confluence zones where Breaker + FVG + POC align.
        
        Args:
            breakers: List of Breaker Blocks
            fvgs: List of FVGs
            poc_result: POC result
            df: DataFrame with OHLCV data
            
        Returns:
            Confluence dict or None
        """
        try:
            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            current_price = close[-1]

            poc = poc_result.get('poc', 0)

            best_confluence = None
            best_score = 0.0

            # Check each breaker for confluence
            for breaker in breakers:
                breaker_high = breaker.get('high', 0)
                breaker_low = breaker.get('low', 0)
                breaker_type = breaker.get('type', 'UNKNOWN')

                # Check if breaker is near POC
                breaker_mid = (breaker_high + breaker_low) / 2
                poc_distance = abs(breaker_mid - poc) / poc * 100

                if poc_distance > self.confluence_tolerance * 100:
                    continue  # Breaker too far from POC

                # Check if FVG is near breaker
                for fvg in fvgs:
                    fvg_top = fvg.get('top', 0)
                    fvg_bottom = fvg.get('bottom', 0)
                    fvg_type = fvg.get('type', 'UNKNOWN')
                    fvg_filled = fvg.get('filled', False)

                    if fvg_filled:
                        continue  # Skip filled FVGs

                    fvg_mid = (fvg_top + fvg_bottom) / 2
                    breaker_distance = abs(fvg_mid - breaker_mid) / breaker_mid * 100

                    if breaker_distance > self.confluence_tolerance * 100:
                        continue  # FVG too far from breaker

                    # Calculate confluence score
                    score = self._calculate_confluence_score(
                        breaker, fvg, poc_result, current_price
                    )

                    if score > best_score:
                        best_score = score

                        # Determine direction
                        direction = self._determine_direction(breaker_type, fvg_type, current_price, poc)

                        best_confluence = {
                            'breaker': breaker,
                            'fvg': fvg,
                            'poc': poc_result,
                            'direction': direction,
                            'zone_high': max(breaker_high, fvg_top),
                            'zone_low': min(breaker_low, fvg_bottom),
                            'score': score,
                            'poc_distance': poc_distance,
                            'breaker_distance': breaker_distance
                        }

            return best_confluence

        except Exception as e:
            self.logger.debug(f"[S21_BFP] Confluence detection error: {e}")
            return None

    def _calculate_confluence_score(
        self, breaker: Dict, fvg: Dict, poc_result: Dict, current_price: float
    ) -> float:
        """Calculate confluence zone score."""
        try:
            score = 0.0

            # Breaker quality
            breaker_quality = breaker.get('quality', 50) / 100
            score += breaker_quality * 0.3

            # FVG freshness
            fvg_gap_pct = fvg.get('gap_pct', 0)
            if fvg_gap_pct > 0.2:
                score += 0.2
            else:
                score += 0.1

            # POC proximity
            poc = poc_result.get('poc', 0)
            zone_mid = (breaker.get('high', 0) + breaker.get('low', 0)) / 2
            poc_distance = abs(zone_mid - poc) / poc
            if poc_distance < 0.002:  # Within 0.2%
                score += 0.3
            elif poc_distance < 0.005:  # Within 0.5%
                score += 0.2
            else:
                score += 0.1

            # Price proximity
            distance = abs(current_price - zone_mid) / current_price
            if distance < 0.003:  # Within 0.3%
                score += 0.2
            elif distance < 0.005:  # Within 0.5%
                score += 0.1

            return min(1.0, score)

        except Exception:
            return 0.5

    def _determine_direction(
        self, breaker_type: str, fvg_type: str, current_price: float, poc: float
    ) -> str:
        """Determine trading direction based on confluence."""
        # Breaker type determines direction
        if 'BULLISH_BREAKER' in breaker_type:
            return 'BUY'
        elif 'BEARISH_BREAKER' in breaker_type:
            return 'SELL'

        # Fallback: Use price vs POC
        if current_price < poc:
            return 'BUY'
        else:
            return 'SELL'

    # =========================================================================
    # M5 CONFIRMATION
    # =========================================================================

    def _confirm_m5(self, df_m5: pd.DataFrame, confluence: Dict) -> bool:
        """
        Confirm signal on M5 timeframe.
        
        Args:
            df_m5: M5 DataFrame
            confluence: Confluence dict
            
        Returns:
            True if confirmed
        """
        if df_m5 is None or df_m5.empty or len(df_m5) < 20:
            return True  # Skip confirmation if no M5 data

        try:
            close = df_m5['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))

            direction = confluence.get('direction', 'BUY')

            # Check M5 momentum aligns with confluence direction
            recent_close = close[-10:]
            momentum = recent_close[-1] - recent_close[0]

            if direction == 'BUY':
                return momentum > 0  # Bullish momentum on M5
            else:
                return momentum < 0  # Bearish momentum on M5

        except Exception:
            return True  # Skip confirmation on error

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _generate_signal(
        self, df: pd.DataFrame, confluence: Dict, regime_context: Dict = None
    ) -> Dict:
        """
        Generate trading signal.
        
        Args:
            df: DataFrame with OHLCV data
            confluence: Confluence dict
            regime_context: Current regime information
            
        Returns:
            Signal dict
        """
        try:
            direction = confluence.get('direction', 'BUY')
            zone_high = confluence.get('zone_high', 0)
            zone_low = confluence.get('zone_low', 0)

            close = df['close'].values.astype(float)
            close = np.nan_to_num(close, nan=np.nanmean(close))
            entry_price = close[-1]

            if entry_price <= 0 or zone_high <= 0 or zone_low <= 0:
                return self._create_neutral_signal()

            # Calculate Stop Loss
            if direction == 'BUY':
                # SL below zone low
                sl_buffer = abs(zone_high - zone_low) * 0.3
                sl_price = zone_low - sl_buffer
            else:  # SELL
                # SL above zone high
                sl_buffer = abs(zone_high - zone_low) * 0.3
                sl_price = zone_high + sl_buffer

            # Validate SL
            if sl_price <= 0 or sl_price == entry_price:
                return self._create_neutral_signal()

            # Calculate Take Profit
            tp_result = self.adaptive_tp_engine.calculate_adaptive_tp(
                df, entry_price, sl_price, direction == 'BUY',
                regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            )

            if tp_result and tp_result.get('tp_price', 0) > 0:
                tp_price = tp_result['tp_price']
            else:
                # Fallback: Fixed R:R
                risk = abs(entry_price - sl_price)
                if direction == 'BUY':
                    tp_price = entry_price + risk * 2.0
                else:
                    tp_price = entry_price - risk * 2.0

            # Calculate confidence
            score = confluence.get('score', 0.5)
            confidence = min(1.0, 0.3 + score * 0.7)

            # Build signal
            signal = {
                'signal': f'{direction}_LIMIT',
                'meta': {
                    'strategy': self.strategy_name,
                    'strategy_category': self.strategy_category,
                    'entry_price': round(entry_price, 2),
                    'sl_price': round(sl_price, 2),
                    'tp_price': round(tp_price, 2),
                    'confidence': confidence,
                    'breaker_type': confluence.get('breaker', {}).get('type', 'UNKNOWN'),
                    'fvg_type': confluence.get('fvg', {}).get('type', 'UNKNOWN'),
                    'poc': confluence.get('poc', {}).get('poc', 0),
                    'confluence_score': score,
                    'trailing_enabled': self.trailing_enabled,
                    'partial_close_enabled': self.partial_close_enabled,
                    'requires_dynamic_exit': self.requires_dynamic_exit
                }
            }

            self.logger.info(
                f"[S21_BFP] Signal generated: {direction} | "
                f"Breaker: {confluence.get('breaker', {}).get('type')} | "
                f"FVG: {confluence.get('fvg', {}).get('type')} | "
                f"POC: {confluence.get('poc', {}).get('poc', 0):.2f} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"[S21_BFP] Signal generation error: {e}")
            return self._create_neutral_signal()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_regime_compatible(self, regime_context: Dict) -> bool:
        """
        Check if current regime is compatible with this strategy.
        
        Args:
            regime_context: Current regime information
            
        Returns:
            True if compatible
        """
        regime_name = regime_context.get('regime_name', 'UNKNOWN')

        # SMC strategies work best in trending and consolidation regimes
        compatible_regimes = [
            'HEALTHY_UPTREND', 'HEALTHY_DOWNTREND',
            'QUIET_RALLY', 'SLOW_BLEED',
            'CONSOLIDATING_BULL', 'CONSOLIDATING_BEAR',
            'PRE_BREAKOUT', 'CLASSIC_RANGE'
        ]

        return regime_name in compatible_regimes

    def _create_neutral_signal(self) -> Dict:
        """Create neutral signal."""
        return {
            'signal': 'NEUTRAL',
            'meta': {
                'strategy': self.strategy_name,
                'strategy_category': self.strategy_category,
                'confidence': 0.0
            }
        }