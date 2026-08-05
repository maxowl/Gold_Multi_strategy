"""
Data Manager - Institutional Master Release
Fetches OHLCV data from MT5 for all timeframes with:
  - Auto-Reconnect Logic (Fault Tolerance)
  - Wilder's ATR Smoothing (Industry Standard)
  - Volume Validation (Fallback to real_volume)
  - Spread Column Injection (For Friction Filter)
  - Graceful Degradation (Per-TF Error Handling)
  - Symbol Info Cache (Latency Optimization)
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Optional
from config import config


class DataManager:
    """Manages data fetching from MT5 terminal with institutional-grade reliability."""
    
    # MT5 timeframe mapping
    TF_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    def __init__(self, symbol: str = None):
        self.symbol = symbol or config.symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connected = False
        
        # Symbol Info Cache (5-second TTL)
        self._symbol_info_cache = None
        self._symbol_info_time = 0
        self._symbol_info_ttl = 5.0

    # =========================================================================
    # MT5 CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """Initialize MT5 connection with optional credentials."""
        try:
            init_params = {
                "login": config.mt5_login,
                "password": config.mt5_password,
                "server": config.mt5_server,
                "path": config.mt5_path,
                "timeout": 60000,
                "portable": True
            }
            # Filter out empty/zero params (use current terminal session)
            init_params = {k: v for k, v in init_params.items() if v not in (0, "", None)}
            
            if not mt5.initialize(**init_params):
                self.logger.error(f"[FAIL] MT5 init failed: {mt5.last_error()}")
                return False
            
            # Verify symbol is available
            symbol_info = self._get_symbol_info(force_refresh=True)
            if symbol_info is None:
                self.logger.error(f"[FAIL] Symbol {self.symbol} not found in Market Watch")
                mt5.shutdown()
                return False
            
            account_info = mt5.account_info()
            if account_info:
                self.logger.info(
                    f"[OK] Connected to MT5 | Account: {account_info.login} | "
                    f"Symbol: {self.symbol} | Balance: ${account_info.balance:.2f}"
                )
            
            self.connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"[FAIL] MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Shutdown MT5 connection."""
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("[OK] Disconnected from MT5")
        except Exception as e:
            self.logger.error(f"[FAIL] Disconnect error: {e}")

    def _check_connection(self) -> bool:
        """Check if MT5 is still connected, auto-reconnect if needed."""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None or not terminal_info.connected:
                self.logger.warning("[DATA] MT5 disconnected. Attempting to reconnect...")
                self.connected = False
                return self.connect()
            return True
        except Exception as e:
            self.logger.error(f"[DATA] Connection check error: {e}")
            return False

    def health_check(self) -> Dict:
        """Check MT5 connection health."""
        try:
            account_info = mt5.account_info()
            return {
                'mt5_connected': account_info is not None,
                'account_login': account_info.login if account_info else None,
                'balance': account_info.balance if account_info else 0.0
            }
        except Exception:
            return {'mt5_connected': False, 'account_login': None, 'balance': 0.0}

    def get_account_info(self) -> Optional[Dict]:
        """Get current account info."""
        try:
            info = mt5.account_info()
            if info:
                return {
                    'login': info.login,
                    'balance': info.balance,
                    'equity': info.equity,
                    'margin': info.margin,
                    'free_margin': info.margin_free,
                    'profit': info.profit
                }
        except Exception as e:
            self.logger.error(f"[FAIL] Get account info error: {e}")
        return None

    # =========================================================================
    # SYMBOL INFO CACHE
    # =========================================================================

    def _get_symbol_info(self, force_refresh: bool = False):
        """Get symbol info with 5-second cache to reduce API calls."""
        current_time = time.time()
        
        if not force_refresh and self._symbol_info_cache and (current_time - self._symbol_info_time < self._symbol_info_ttl):
            return self._symbol_info_cache
        
        try:
            info = mt5.symbol_info(self.symbol)
            if info is None:
                if mt5.symbol_select(self.symbol, True):
                    info = mt5.symbol_info(self.symbol)
            
            if info:
                self._symbol_info_cache = info
                self._symbol_info_time = current_time
            
            return info
        except Exception as e:
            self.logger.error(f"[DATA] Symbol info error: {e}")
            return None

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_single_timeframe(self, tf_name: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single timeframe.
        Returns DataFrame with columns: time, open, high, low, close, tick_volume, spread, real_volume, atr
        """
        if tf_name not in self.TF_MAP:
            self.logger.warning(f"[DATA] Unknown timeframe: {tf_name}")
            return None
        
        # Auto-reconnect if needed
        if not self._check_connection():
            return None
        
        try:
            tf_code = self.TF_MAP[tf_name]
            rates = mt5.copy_rates_from_pos(self.symbol, tf_code, 0, bars)
            
            if rates is None or len(rates) == 0:
                self.logger.warning(f"[DATA] No data returned for {tf_name}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            
            # Ensure required columns exist
            required_cols = ['time', 'open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(f"[DATA] Missing column {col} in {tf_name}")
                    return None
            
            # =========================================================================
            # VOLUME VALIDATION & NORMALIZATION
            # =========================================================================
            # Some brokers provide tick_volume = 0, fallback to real_volume
            if 'tick_volume' in df.columns:
                if (df['tick_volume'] == 0).all() and 'real_volume' in df.columns:
                    self.logger.debug(f"[DATA] {tf_name}: tick_volume is all zeros, using real_volume")
                    df['tick_volume'] = df['real_volume']
                elif (df['tick_volume'] == 0).all():
                    df['tick_volume'] = 1  # Last resort fallback
            else:
                df['tick_volume'] = df.get('real_volume', 1)
            
            # =========================================================================
            # SPREAD COLUMN INJECTION (For Friction Filter)
            # =========================================================================
            if 'spread' not in df.columns:
                # MT5 provides spread in points, convert to price
                symbol_info = self._get_symbol_info()
                if symbol_info:
                    point = getattr(symbol_info, 'point', 0.01)
                    # Use current spread for all historical bars (approximation)
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick:
                        current_spread_price = tick.ask - tick.bid
                        df['spread'] = current_spread_price
                    else:
                        df['spread'] = 0.0
                else:
                    df['spread'] = 0.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"[DATA] Fetch error for {tf_name}: {e}")
            return None

    def fetch_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all timeframes and inject ATR column.
        Uses Wilder's Smoothing for ATR (industry standard).
        """
        data = {}
        required_tfs = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
        # Add primary timeframe if not already in list
        if config.primary_timeframe not in required_tfs:
            required_tfs.append(config.primary_timeframe)
        
        for tf_name in required_tfs:
            try:
                df = self._fetch_single_timeframe(tf_name, bars=500)
                
                if df is not None and not df.empty and len(df) >= 20:
                    # =========================================================================
                    # WILDER'S ATR SMOOTHING (Industry Standard)
                    # =========================================================================
                    high = df['high'].to_numpy()
                    low = df['low'].to_numpy()
                    close = df['close'].to_numpy()
                    
                    # Calculate True Range
                    tr1 = high - low
                    tr2 = np.abs(high - np.roll(close, 1))
                    tr3 = np.abs(low - np.roll(close, 1))
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    tr[0] = tr1[0]  # First bar has no previous close
                    
                    # Wilder's Smoothing (EMA with alpha = 1/14)
                    atr = np.zeros(len(tr))
                    atr[13] = np.mean(tr[:14])  # First ATR is simple average
                    
                    for i in range(14, len(tr)):
                        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                    
                    df['atr'] = atr
                    
                    data[tf_name] = df
                    
                    # Log ATR for primary timeframe (debug)
                    if tf_name == config.primary_timeframe or tf_name == 'M15':
                        last_atr = atr[-1]
                        if not np.isnan(last_atr):
                            self.logger.debug(
                                f"[DATA] {tf_name} ATR(14) = {last_atr:.3f} USD | Bars: {len(df)}"
                            )
                else:
                    self.logger.warning(f"[DATA] {tf_name}: Insufficient data (bars: {len(df) if df is not None else 0})")
            
            except Exception as e:
                self.logger.error(f"[DATA] Failed to fetch {tf_name}: {e}")
                continue  # Graceful degradation: skip this TF, continue with others
        
        if not data:
            self.logger.critical("[DATA] No timeframes fetched successfully. Check MT5 connection.")
        
        return data

    def get_single_timeframe(self, tf_name: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch data for a specific timeframe on demand."""
        df = self._fetch_single_timeframe(tf_name, bars)
        
        if df is not None and not df.empty and len(df) >= 20:
            # Inject ATR
            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            close = df['close'].to_numpy()
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            tr[0] = tr1[0]
            
            atr = np.zeros(len(tr))
            atr[13] = np.mean(tr[:14])
            for i in range(14, len(tr)):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            
            df['atr'] = atr
        
        return df

    # =========================================================================
    # PRICE & TICK DATA
    # =========================================================================

    def get_current_price(self) -> Optional[Dict]:
        """Get current bid/ask prices."""
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': tick.time,
                    'spread': tick.ask - tick.bid
                }
        except Exception as e:
            self.logger.error(f"[FAIL] Get current price error: {e}")
        return None

    def get_recent_ticks(self, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get recent tick data."""
        try:
            ticks = mt5.copy_ticks_from(self.symbol, mt5.datetime_from_utc(pd.Timestamp.utcnow()), count, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                return df
        except Exception as e:
            self.logger.error(f"[FAIL] Get ticks error: {e}")
        return None