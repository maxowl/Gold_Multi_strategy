"""
Strategy 8: Gaussian Process Regression Volatility (Mean Reversion).
Uses GPR to model price mean and variance, entering on extreme deviations.
[FIX] Implemented Data Normalization to prevent kernel bound warnings and corrected array indexing.
"""
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from core.base_strategy import BaseStrategy
from core.atr_cache import ATRCache


class Strategy8_GPR_Vol(BaseStrategy):
    def __init__(self):
        super().__init__(name="S8_GPR_Vol", strategy_category="MEAN_REVERSION", min_risk_reward=1.5)

    def evaluate(self, df_m15: pd.DataFrame, df_h1: pd.DataFrame = None) -> dict:
        if not self._validate_data(df_m15, 60):
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "insufficient_data"}}

        close = float(df_m15['close'].iloc[-1])
        atr_m15 = ATRCache.get_atr(df_m15, 14).iloc[-1]
        if pd.isna(atr_m15) or atr_m15 == 0:
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "atr_failed"}}

        try:
            # Suppress sklearn convergence warnings to keep production logs clean
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                from sklearn.exceptions import ConvergenceWarning
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                
                # Limit lookback to 50 bars to prevent O(N^3) computational lag in Event Loop
                lookback = 50
                y_raw = df_m15['close'].iloc[-lookback:].to_numpy().reshape(-1, 1)
                X = np.arange(lookback).reshape(-1, 1)
                
                # [FIX] Normalize y to prevent bound issues with high-value assets like XAUUSD
                y_mean = np.mean(y_raw)
                y_std_raw = np.std(y_raw)
                if y_std_raw == 0:
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "zero_std_raw"}}
                    
                y = (y_raw - y_mean) / y_std_raw
                
                # Kernel with bounds suitable for normalized data (mean=0, std=1)
                kernel = RBF(length_scale=10.0, length_scale_bounds=(1.0, 100.0)) + \
                         WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
                         
                gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
                gpr.fit(X, y)
                
                # Predict next step (current bar)
                X_pred = np.array([[lookback - 1]])
                y_pred_norm, y_std_norm = gpr.predict(X_pred, return_std=True)
                
                # [FIX] Inverse transform to actual price scale and correct 1D array indexing
                mean_price = float(y_pred_norm[0] * y_std_raw + y_mean)
                std_dev = float(y_std_norm[0] * y_std_raw)
                
            if std_dev == 0 or np.isnan(std_dev):
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "zero_std"}}
                
            z_score = (close - mean_price) / std_dev
            
            # Volatility Suppressor: Don't trade if ATR is spiking
            atr_series = ATRCache.get_atr(df_m15, 14)
            atr_percentile = pd.Series(atr_series).rolling(50).rank(pct=True).iloc[-1]
            if pd.isna(atr_percentile) or atr_percentile > 0.85:
                return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "high_vol_suppressor"}}

            # Mean Reversion Signals based on GPR Z-Score
            if z_score < -2.0:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(df_m15['low'].iloc[-10:].min()), df_m15, is_buy=True, atr_multiplier=2.0)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                tp_price = mean_price
                risk = abs(entry_price - sl_info['sl_price'])
                if abs(tp_price - entry_price) < risk * self.min_risk_reward:
                    tp_price = entry_price + (risk * self.min_risk_reward)
                    
                signal = self.build_signal(
                    "BUY_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                    extra_meta={'z_score': z_score, 'gpr_mean': mean_price}
                )
                self.log_signal_summary(signal)
                return signal

            elif z_score > 2.0:
                entry_price = close
                sl_info = self.calculate_session_sl(entry_price, float(df_m15['high'].iloc[-10:].max()), df_m15, is_buy=False, atr_multiplier=2.0)
                if not sl_info['valid']: 
                    return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "sl_invalid"}}
                
                tp_price = mean_price
                risk = abs(entry_price - sl_info['sl_price'])
                if abs(entry_price - tp_price) < risk * self.min_risk_reward:
                    tp_price = entry_price - (risk * self.min_risk_reward)
                    
                signal = self.build_signal(
                    "SELL_MARKET", entry_price, sl_info['sl_price'], tp_price, "M15", 0.80,
                    extra_meta={'z_score': z_score, 'gpr_mean': mean_price}
                )
                self.log_signal_summary(signal)
                return signal

        except Exception as e:
            self.logger.error(f"[FAIL] GPR calculation error: {e}")
            return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "gpr_exception"}}

        return {"signal": "NEUTRAL", "meta": {"strategy": self.name, "reason": "z_score_normal"}}