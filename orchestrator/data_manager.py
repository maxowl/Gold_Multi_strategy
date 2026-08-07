"""
Data Manager - MT5 Data Ingestion (REVISED).
Responsible for fetching and caching market data from MetaTrader 5.
Provides multi-timeframe OHLCV data for strategy evaluation.

REVISION LOG:
  [REV-001] ADDED Timezone documentation and helper methods.
            All data is stored in UTC. Use get_server_time_offset()
            to convert to local time for session detection.
  [REV-002] ADDED Spread validation in _validate_data().
            Abnormal spreads (> 200 points) are flagged as warnings.
  [REV-003] ADDED Convenience methods: get_latest_price(),
            get_current_spread_points(), get_server_time_offset(),
            get_current_server_time().
  [REV-004] FIXED M30 is now optional (configurable via
            ENABLE_M30 flag). Default: disabled (no strategy uses M30).
  [REV-005] ADDED Reconnection logic in _fetch_from_mt5().
            If MT5 terminal disconnects during fetch, attempts
            to reinitialize before retrying.
  [REV-006] ADDED Thread safety for cache operations using
            threading.Lock(). Prevents race conditions if cache
            is accessed from multiple threads.
  [REV-007] ADDED Optional timeframes parameter to
            fetch_all_timeframes() for selective fetching.
  [REV-008] ADDED Stale data detection. If cached data's last
            bar is older than expected, cache is invalidated.

Features:
  - Multi-timeframe data fetching (M1, M5, M15, H1, H4, D1)
  - Data caching with TTL to reduce MT5 API calls
  - Data validation before returning (including spread)
  - UTC timezone normalization with local time helpers
  - Retry logic with automatic reconnection
  - Thread-safe cache operations
  - Health check integration
  - Stale data detection

Timezone Policy:
  All 'time' columns in returned DataFrames are UTC-aware
  (pd.Timestamp with tzinfo=UTC). For session detection or
  local-time comparisons, use get_server_time_offset() to
  calculate the offset between server (UTC) and local time.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta


class DataManager:
    """
    Manages market data ingestion from MetaTrader 5.
    Provides cached, validated OHLCV data for all timeframes.

    Thread Safety:
      All cache operations are protected by a threading.Lock()
      to prevent race conditions in multi-threaded scenarios.

    Timezone Policy:
      All data is stored in UTC. Use helper methods for local
      time conversion when needed for session detection.
    """

    # =========================================================================
    # TIMEFRAME CONFIGURATION
    # =========================================================================

    # Timeframe mapping: name -> MT5 constant
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    # Number of bars to fetch per timeframe
    BARS_TO_FETCH = {
        'M1': 500,
        'M5': 500,
        'M15': 500,
        'M30': 300,
        'H1': 300,
        'H4': 200,
        'D1': 200,
    }

    # Cache TTL in seconds (how long cached data is valid)
    CACHE_TTL = {
        'M1': 30,     # 30 seconds
        'M5': 60,     # 1 minute
        'M15': 120,   # 2 minutes
        'M30': 300,   # 5 minutes
        'H1': 600,    # 10 minutes
        'H4': 1800,   # 30 minutes
        'D1': 3600,   # 1 hour
    }

    # Expected bar duration in seconds (for stale data detection)
    BAR_DURATION = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400,
    }

    # [REV-004] M30 is optional (no strategy uses M30 in ROUTE_MAP)
    ENABLE_M30 = False

    # Maximum reasonable spread in points (for validation)
    MAX_REASONABLE_SPREAD_POINTS = 200

    def __init__(self, symbol: str = "XAUUSDm"):
        """
        Initialize DataManager.

        Args:
            symbol: Trading symbol (e.g., 'XAUUSDm')
        """
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)

        # Data cache: {timeframe: (timestamp, DataFrame)}
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

        # [REV-006] Thread lock for cache operations
        self._cache_lock = threading.Lock()

        # Connection state
        self._connected = False

        # [REV-004] Build active timeframe list
        self._active_timeframes = self._get_active_timeframes()

        self.logger.info(
            f"[DATA_MGR] Initialized | Symbol: {self.symbol} | "
            f"Active timeframes: {', '.join(self._active_timeframes)}"
        )

    def _get_active_timeframes(self) -> List[str]:
        """
        [REV-004] Get list of active timeframes based on configuration.

        Returns:
            List of active timeframe names
        """
        active = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
        if self.ENABLE_M30:
            active.insert(3, 'M30')
        return active

    # =========================================================================
    # MT5 CONNECTION MANAGEMENT
    # =========================================================================

    def connect(self) -> bool:
        """
        Establish MT5 connection.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if not mt5.initialize():
                self.logger.error(
                    f"[DATA_MGR] MT5 initialize() failed: {mt5.last_error()}"
                )
                return False

            # Verify symbol is available
            if not mt5.symbol_select(self.symbol, True):
                self.logger.error(
                    f"[DATA_MGR] Failed to select symbol {self.symbol}"
                )
                mt5.shutdown()
                return False

            self._connected = True
            self.logger.info(
                f"[DATA_MGR] Connected to MT5 | Symbol: {self.symbol}"
            )
            return True

        except Exception as e:
            self.logger.error(f"[DATA_MGR] Connection error: {e}")
            return False

    def disconnect(self):
        """Shutdown MT5 connection."""
        try:
            mt5.shutdown()
            self._connected = False
            self.logger.info("[DATA_MGR] Disconnected from MT5")
        except Exception as e:
            self.logger.error(f"[DATA_MGR] Disconnect error: {e}")

    def health_check(self) -> Dict:
        """
        Check MT5 connection health.

        Returns:
            Dict with 'mt5_connected', 'symbol_available', 'last_error'
        """
        result = {
            'mt5_connected': False,
            'symbol_available': False,
            'last_error': None
        }

        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                result['last_error'] = 'Terminal info is None'
                return result

            result['mt5_connected'] = terminal_info.connected

            if result['mt5_connected']:
                symbol_info = mt5.symbol_info(self.symbol)
                result['symbol_available'] = symbol_info is not None
                if not result['symbol_available']:
                    result['last_error'] = (
                        f'Symbol {self.symbol} not available'
                    )

        except Exception as e:
            result['last_error'] = str(e)

        return result

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def fetch_all_timeframes(
        self, timeframes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for specified timeframes (or all active).

        [REV-007] Added optional timeframes parameter for selective
        fetching. If None, fetches all active timeframes.

        Uses caching to reduce MT5 API calls.

        Args:
            timeframes: List of timeframe names to fetch (optional).
                        If None, fetches all active timeframes.

        Returns:
            Dict of timeframe -> DataFrame with columns:
            [time, open, high, low, close, tick_volume, volume, spread]
        """
        if timeframes is None:
            timeframes = self._active_timeframes

        data = {}
        for tf_name in timeframes:
            if tf_name not in self.TIMEFRAME_MAP:
                self.logger.warning(
                    f"[DATA_MGR] Unknown timeframe requested: {tf_name}"
                )
                continue

            df = self.fetch_timeframe(tf_name)
            if df is not None and not df.empty:
                data[tf_name] = df

        if not data:
            self.logger.warning(
                "[DATA_MGR] No data fetched for any timeframe"
            )

        return data

    def fetch_timeframe(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a specific timeframe.
        Uses caching with TTL to reduce MT5 API calls.

        [REV-008] Added stale data detection.

        Args:
            timeframe: Timeframe name (M1, M5, M15, M30, H1, H4, D1)

        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        # [REV-006] Thread-safe cache check
        with self._cache_lock:
            if timeframe in self._cache:
                cache_time, cached_df = self._cache[timeframe]
                cache_ttl = self.CACHE_TTL.get(timeframe, 60)

                if time.time() - cache_time < cache_ttl:
                    # [REV-008] Check for stale data
                    if not self._is_data_stale(timeframe, cached_df):
                        return cached_df
                    else:
                        self.logger.debug(
                            f"[DATA_MGR] Cache stale for {timeframe}, "
                            f"invalidating"
                        )
                        del self._cache[timeframe]

        # Fetch from MT5
        df = self._fetch_from_mt5(timeframe)

        if df is not None and not df.empty:
            # Validate data
            if self._validate_data(df):
                # [REV-006] Thread-safe cache update
                with self._cache_lock:
                    self._cache[timeframe] = (time.time(), df)
                return df
            else:
                self.logger.warning(
                    f"[DATA_MGR] Data validation failed for {timeframe}"
                )
                return None

        return None

    def _is_data_stale(self, timeframe: str, df: pd.DataFrame) -> bool:
        """
        [REV-008] Check if cached data is stale.

        Data is considered stale if the last bar's timestamp is
        older than expected based on the timeframe's bar duration.

        Args:
            timeframe: Timeframe name
            df: Cached DataFrame

        Returns:
            True if data is stale, False otherwise
        """
        try:
            if df is None or df.empty or 'time' not in df.columns:
                return True

            last_bar_time = df['time'].iloc[-1]
            if not isinstance(last_bar_time, pd.Timestamp):
                return False  # Cannot determine, assume not stale

            # Calculate expected bar age
            bar_duration = self.BAR_DURATION.get(timeframe, 60)
            now_utc = pd.Timestamp.now(tz='UTC')

            # Allow 2x bar duration before considering stale
            max_age = timedelta(seconds=bar_duration * 2)
            bar_age = now_utc - last_bar_time

            return bar_age > max_age

        except Exception:
            return False  # Cannot determine, assume not stale

    def _fetch_from_mt5(
        self, timeframe: str, max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from MT5 with retry logic.

        [REV-005] Added reconnection logic. If MT5 terminal
        disconnects during fetch, attempts to reinitialize.

        Args:
            timeframe: Timeframe name
            max_retries: Maximum number of retry attempts

        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        if timeframe not in self.TIMEFRAME_MAP:
            self.logger.error(f"[DATA_MGR] Unknown timeframe: {timeframe}")
            return None

        mt5_timeframe = self.TIMEFRAME_MAP[timeframe]
        bars_to_fetch = self.BARS_TO_FETCH.get(timeframe, 500)

        for attempt in range(max_retries):
            try:
                rates = mt5.copy_rates_from_pos(
                    self.symbol,
                    mt5_timeframe,
                    0,
                    bars_to_fetch
                )

                if rates is None or len(rates) == 0:
                    # [REV-005] Check if terminal is still connected
                    terminal_info = mt5.terminal_info()
                    if terminal_info is None or not terminal_info.connected:
                        self.logger.warning(
                            f"[DATA_MGR] MT5 disconnected during fetch "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Attempting reconnection..."
                        )
                        if self._attempt_reconnect():
                            continue  # Retry after reconnection
                        else:
                            self.logger.error(
                                f"[DATA_MGR] Reconnection failed"
                            )
                            return None

                    self.logger.warning(
                        f"[DATA_MGR] No data for {timeframe} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(1)
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(rates)

                # Ensure required columns exist
                required_cols = [
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'volume', 'spread'
                ]
                for col in required_cols:
                    if col not in df.columns:
                        self.logger.error(
                            f"[DATA_MGR] Missing column: {col}"
                        )
                        return None

                # Convert time to datetime (UTC)
                # [REV-001] All data is stored in UTC
                df['time'] = pd.to_datetime(
                    df['time'], unit='s', utc=True
                )

                # Select and reorder columns
                df = df[required_cols].copy()

                # Remove any rows with NaN in OHLC
                df = df.dropna(subset=['open', 'high', 'low', 'close'])

                # Remove any rows with zero/negative prices
                df = df[
                    (df['open'] > 0) & (df['high'] > 0) &
                    (df['low'] > 0) & (df['close'] > 0)
                ]

                # Reset index
                df = df.reset_index(drop=True)

                return df

            except Exception as e:
                self.logger.error(
                    f"[DATA_MGR] Fetch error for {timeframe} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        return None

    def _attempt_reconnect(self) -> bool:
        """
        [REV-005] Attempt to reconnect to MT5 terminal.

        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            # Shutdown existing connection
            try:
                mt5.shutdown()
            except Exception:
                pass

            time.sleep(1)

            # Reinitialize
            if mt5.initialize():
                if mt5.symbol_select(self.symbol, True):
                    self._connected = True
                    self.logger.info(
                        f"[DATA_MGR] Reconnected to MT5 | "
                        f"Symbol: {self.symbol}"
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error(f"[DATA_MGR] Reconnection error: {e}")
            return False

    # =========================================================================
    # DATA VALIDATION [REV-002]
    # =========================================================================

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame has sufficient and correct data.

        [REV-002] Added spread validation. Abnormal spreads
        (> MAX_REASONABLE_SPREAD_POINTS) are flagged as warnings.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False

        # Check minimum bars
        if len(df) < 50:
            return False

        # Check required columns
        required_cols = ['time', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                return False

        # Check for NaN in OHLC
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            return False

        # Check for zero/negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False

        # Check OHLC consistency (high >= low)
        if (df['high'] < df['low']).any():
            return False

        # [REV-002] Check spread is reasonable (if spread column exists)
        if 'spread' in df.columns:
            max_spread = df['spread'].max()
            if max_spread > self.MAX_REASONABLE_SPREAD_POINTS:
                self.logger.warning(
                    f"[DATA_MGR] Abnormal spread detected: "
                    f"max={max_spread} points "
                    f"(threshold: {self.MAX_REASONABLE_SPREAD_POINTS})"
                )
                # Don't fail validation, just warn
                # Abnormal spread might be temporary

        return True

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information from MT5.

        Returns:
            Dict with account info, or None if failed
        """
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None

            return {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'currency': account_info.currency,
            }

        except Exception as e:
            self.logger.error(f"[DATA_MGR] Account info error: {e}")
            return None

    # =========================================================================
    # SYMBOL INFO
    # =========================================================================

    def get_symbol_info(self) -> Optional[Dict]:
        """
        Get symbol information from MT5.

        Returns:
            Dict with symbol info, or None if failed
        """
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                return None

            return {
                'name': symbol_info.name,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_contract_size': symbol_info.trade_contract_size,
                'trade_tick_size': symbol_info.trade_tick_size,
                'trade_tick_value': symbol_info.trade_tick_value,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'trade_stops_level': symbol_info.trade_stops_level,
                'trade_freeze_level': symbol_info.trade_freeze_level,
            }

        except Exception as e:
            self.logger.error(f"[DATA_MGR] Symbol info error: {e}")
            return None

    # =========================================================================
    # [REV-003] CONVENIENCE METHODS
    # =========================================================================

    def get_latest_price(self) -> Optional[Dict]:
        """
        [REV-003] Get latest bid/ask prices.

        Returns:
            Dict with 'bid', 'ask', 'spread', 'time', or None if failed
        """
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None

            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': tick.time
            }

        except Exception as e:
            self.logger.error(f"[DATA_MGR] Latest price error: {e}")
            return None

    def get_current_spread_points(self) -> float:
        """
        [REV-003] Get current spread in points.

        Returns:
            Current spread in points, or 0.0 if failed
        """
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            info = mt5.symbol_info(self.symbol)

            if tick and info and info.point > 0:
                return (tick.ask - tick.bid) / info.point

        except Exception as e:
            self.logger.debug(f"[DATA_MGR] Spread points error: {e}")

        return 0.0

    def get_server_time_offset(self) -> timedelta:
        """
        [REV-001] Calculate offset between server time (UTC) and local time.

        Used for session detection and time-based filters that
        require local time comparisons.

        Returns:
            timedelta offset (local_time - utc_time)
        """
        try:
            local_time = datetime.now()
            utc_time = datetime.utcnow()
            offset = local_time - utc_time
            return offset

        except Exception as e:
            self.logger.debug(f"[DATA_MGR] Time offset error: {e}")
            return timedelta(hours=0)

    def get_current_server_time(self) -> datetime:
        """
        [REV-001] Get current server time (UTC).

        Used for consistent time comparisons across components.

        Returns:
            Current server time as UTC datetime
        """
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                return datetime.utcfromtimestamp(tick.time)

        except Exception as e:
            self.logger.debug(f"[DATA_MGR] Server time error: {e}")

        return datetime.utcnow()

    def convert_to_local_time(self, utc_time: pd.Timestamp) -> pd.Timestamp:
        """
        [REV-001] Convert UTC timestamp to local time.

        Args:
            utc_time: UTC-aware timestamp

        Returns:
            Local time timestamp
        """
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.tz_localize('UTC')

            offset = self.get_server_time_offset()
            local_time = utc_time + offset

            return local_time

        except Exception as e:
            self.logger.debug(f"[DATA_MGR] Time conversion error: {e}")
            return utc_time

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def clear_cache(self):
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("[DATA_MGR] Cache cleared")

    def get_cache_status(self) -> Dict:
        """
        Get current cache status.

        Returns:
            Dict with cache info per timeframe
        """
        status = {}
        current_time = time.time()

        with self._cache_lock:
            for tf_name in self._active_timeframes:
                if tf_name in self._cache:
                    cache_time, cached_df = self._cache[tf_name]
                    cache_ttl = self.CACHE_TTL.get(tf_name, 60)
                    age = current_time - cache_time
                    is_valid = age < cache_ttl

                    status[tf_name] = {
                        'cached': True,
                        'age_seconds': round(age, 1),
                        'ttl_seconds': cache_ttl,
                        'is_valid': is_valid,
                        'bars': len(cached_df)
                    }
                else:
                    status[tf_name] = {
                        'cached': False,
                        'age_seconds': 0,
                        'ttl_seconds': self.CACHE_TTL.get(tf_name, 60),
                        'is_valid': False,
                        'bars': 0
                    }

        return status

    def get_active_timeframes(self) -> List[str]:
        """
        Get list of active timeframes.

        Returns:
            List of active timeframe names
        """
        return self._active_timeframes.copy()