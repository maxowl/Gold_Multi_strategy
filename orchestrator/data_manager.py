"""
Data Manager.
Handles MT5 connection and multi-timeframe data fetching.
"""
import MetaTrader5 as mt5
import pandas as pd
import pytz
import logging
import time
from typing import Dict, Optional
from config import config


class DataManager:
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connected = False

    def connect(self) -> bool:
        if self._connected:
            return True
            
        try:
            init_kwargs = {}
            if config.mt5_login and config.mt5_login > 0:
                init_kwargs['login'] = config.mt5_login
                init_kwargs['password'] = config.mt5_password
                init_kwargs['server'] = config.mt5_server
                if config.mt5_path:
                    init_kwargs['path'] = config.mt5_path
            
            if not mt5.initialize(**init_kwargs):
                self.logger.error(f"[FAIL] MT5 initialization failed: {mt5.last_error()}")
                return False
                
            # Ensure symbol is visible in Market Watch
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                self.logger.warning(f"[WARN] Symbol {self.symbol} not found, attempting to select...")
                if not mt5.symbol_select(self.symbol, True):
                    self.logger.error(f"[FAIL] Cannot select symbol {self.symbol}")
                    mt5.shutdown()
                    return False
            
            self._connected = True
            acc_info = mt5.account_info()
            if acc_info:
                self.logger.info(f"[OK] Connected to MT5 | Account: {acc_info.login} | Symbol: {self.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"[FAIL] MT5 connection error: {e}")
            return False

    def disconnect(self):
        if self._connected:
            mt5.shutdown()
            self._connected = False
            self.logger.info("[OK] Disconnected from MT5")

    def health_check(self) -> Dict:
        if not self._connected:
            return {'mt5_connected': False}
        try:
            acc = mt5.account_info()
            return {'mt5_connected': acc is not None}
        except Exception:
            return {'mt5_connected': False}

    def fetch_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf_str in self.TIMEFRAME_MAP.keys():
            df = self._fetch_data(tf_str)
            if df is not None:
                data[tf_str] = df
        return data

    def _fetch_data(self, timeframe_str: str, count: int = 500) -> Optional[pd.DataFrame]:
        tf_enum = self.TIMEFRAME_MAP.get(timeframe_str)
        if not tf_enum:
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, tf_enum, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            df = pd.DataFrame(rates)
            
            # [FIX] Explicitly localize time to UTC to prevent timezone-naive issues downstream
            df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
            
            # Ensure standard column names
            if 'tick_volume' not in df.columns and 'volume' in df.columns:
                df['tick_volume'] = df['volume']
                
            return df
            
        except Exception as e:
            self.logger.error(f"[FAIL] Data fetch error for {timeframe_str}: {e}")
            return None

    def get_account_info(self) -> Optional[Dict]:
        try:
            acc = mt5.account_info()
            if acc:
                return {
                    'balance': acc.balance,
                    'equity': acc.equity,
                    'margin_free': acc.margin_free
                }
            return None
        except Exception as e:
            self.logger.error(f"[FAIL] Account info error: {e}")
            return None