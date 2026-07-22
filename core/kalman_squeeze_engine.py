"""
Kalman Filter Engine for Trend Extraction.
Provides smooth trendline for S24_KalmanMomentum.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict


class KalmanSqueezeEngine:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_kalman_filter(self, series: pd.Series,
                            dt: float = 1.0,
                            process_noise: float = 0.01,
                            measurement_noise: float = 1.0) -> Optional[pd.Series]:
        """
        Apply Kalman filter to extract smooth trend from noisy price data.
        
        State vector: [price, velocity]
        Transition: price_new = price_old + velocity * dt
                   velocity_new = velocity_old
        """
        if series is None or len(series) < 20:
            return None
        
        try:
            x = series.to_numpy().astype(float)
            n = len(x)
            
            # State transition matrix
            F = np.array([[1.0, dt], [0.0, 1.0]])
            
            # Observation matrix
            H = np.array([[1.0, 0.0]])
            
            # Process noise covariance
            Q = np.array([
                [process_noise * dt, 0.0],
                [0.0, process_noise * 10.0]
            ])
            
            # Measurement noise covariance
            R = np.array([[measurement_noise]])
            
            # Initial state
            state = np.array([x[0] if not np.isnan(x[0]) else 0.0, 0.0])
            
            # Initial covariance
            P = np.eye(2) * 1.0
            
            # Storage for filtered values
            filtered = np.zeros(n)
            
            for i in range(n):
                if np.isnan(x[i]):
                    filtered[i] = filtered[i-1] if i > 0 else 0.0
                    continue
                
                # Predict
                state_pred = F @ state
                P_pred = F @ P @ F.T + Q
                
                # Update
                z = np.array([x[i]])
                y = z - H @ state_pred  # Innovation
                S = H @ P_pred @ H.T + R  # Innovation covariance
                
                # Kalman gain
                try:
                    K = P_pred @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    # Fallback for singular matrix
                    K = P_pred @ H.T / (S[0, 0] + 1e-10)
                
                # Update state and covariance
                state = state_pred + K @ y
                P = (np.eye(2) - K @ H) @ P_pred
                
                # Store filtered value
                filtered[i] = state[0]
            
            return pd.Series(filtered, index=series.index)
            
        except Exception as e:
            self.logger.error(f"[FAIL] Kalman filter error: {e}")
            return None

    def apply_kalman_squeeze(self, series: pd.Series,
                             fast_process_noise: float = 0.1,
                             slow_process_noise: float = 0.001) -> Optional[Dict]:
        """
        Apply dual Kalman filters (fast + slow) to detect squeeze breakouts.
        """
        if series is None or len(series) < 30:
            return None
        
        try:
            fast_filter = self.apply_kalman_filter(series, process_noise=fast_process_noise)
            slow_filter = self.apply_kalman_filter(series, process_noise=slow_process_noise)
            
            if fast_filter is None or slow_filter is None:
                return None
            
            # Calculate spread between fast and slow
            spread = (fast_filter - slow_filter).to_numpy()
            
            # Squeeze detection: spread narrows significantly
            spread_std = np.std(spread[-50:]) if len(spread) >= 50 else np.std(spread)
            current_spread = spread[-1]
            prev_spread = spread[-2]
            
            # Squeeze on: absolute spread is small
            squeeze_on = abs(current_spread) < spread_std * 0.5
            
            # Breakout detection
            breakout = None
            if not squeeze_on and abs(prev_spread) < spread_std * 0.5:
                # Just exited squeeze
                if current_spread > 0:
                    breakout = 'BULLISH'
                elif current_spread < 0:
                    breakout = 'BEARISH'
            
            return {
                'fast_filter': fast_filter,
                'slow_filter': slow_filter,
                'spread': pd.Series(spread, index=series.index),
                'squeeze_on': bool(squeeze_on),
                'breakout': breakout
            }
            
        except Exception as e:
            self.logger.error(f"[FAIL] Kalman squeeze error: {e}")
            return None

    def calculate_kalman_velocity(self, kalman_series: pd.Series) -> Optional[pd.Series]:
        """
        Calculate velocity (first derivative) of Kalman-filtered series.
        """
        if kalman_series is None or len(kalman_series) < 5:
            return None
        
        try:
            # Numerical differentiation with smoothing
            velocity = kalman_series.diff().rolling(3).mean()
            return velocity
            
        except Exception as e:
            self.logger.error(f"[FAIL] Kalman velocity calculation error: {e}")
            return None