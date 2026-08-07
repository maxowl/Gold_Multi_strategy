"""
Centralized Configuration Manager - Micro-Account-Only Edition.
Single source of truth for ALL system parameters.

This system is designed EXCLUSIVELY for micro-account trading ($500-$3000).
The user withdraws profits when equity exceeds $3000.
There is NO scalping mode, NO standard mode. Only micro-account mode.

Sections:
  1. MT5 Connection & Symbol Settings
  2. Timeframe Configuration
  3. Micro-Account Risk Management
  4. Auto-Withdrawal & Balance Alerts
  5. SL/TP Configuration (Scaled for XAUUSD @ 4000)
  6. Friction Filter Parameters
  7. Adaptive TP Parameters
  8. Time Stop Parameters
  9. Trailing Methods
  10. Regime Detection Models
  11. Microstructure Predictor (S26)
  12. Drawdown Scaler Thresholds
  13. Equity Circuit Breaker Thresholds
  14. Modification Rate Limiter
  15. Emergency Defense Engine
  16. Session Management
  17. Analytics & Attribution
  18. Event Loop Settings
  19. File Paths
  20. Auto-Scaling
"""
import os


class Config:
    """
    Micro-Account-Only Configuration.
    All parameters are tuned for $500-$3000 portfolio trading XAUUSD.
    """

    # =========================================================================
    # 1. MT5 CONNECTION & SYMBOL SETTINGS
    # =========================================================================
    symbol: str = os.getenv("BOT_SYMBOL", "XAUUSDm")
    magic_number: int = int(os.getenv("BOT_MAGIC", "888888"))
    max_slippage_points: int = int(os.getenv("BOT_SLIPPAGE", "20"))

    # MT5 Credentials (0/empty = attach to currently open terminal)
    mt5_login: int = int(os.getenv("BOT_MT5_LOGIN", "415785591"))
    mt5_password: str = os.getenv("BOT_MT5_PASSWORD", "Nomorefriends@1")
    mt5_server: str = os.getenv("BOT_MT5_SERVER", "Exness-MT5Trial14")
    mt5_path: str = os.getenv("BOT_MT5_PATH", "")

    # =========================================================================
    # 2. TIMEFRAME CONFIGURATION
    # =========================================================================
    primary_timeframe: str = os.getenv("BOT_TF", "M15")

    # =========================================================================
    # 3. MICRO-ACCOUNT RISK MANAGEMENT (Fixed - No Mode Switching)
    # =========================================================================
    # Risk per trade: 0.5% of equity
    risk_per_trade_pct: float = float(os.getenv("BOT_RISK_PCT", "0.5"))

    # Position limits (reduced for micro-account)
    max_open_positions: int = int(os.getenv("BOT_MAX_POS", "2"))
    max_pending_orders: int = int(os.getenv("BOT_MAX_PEND", "2"))

    # Daily loss limit: 2% of equity
    max_daily_loss_pct: float = float(os.getenv("BOT_MAX_DAILY_LOSS", "2.0"))

    # Pending order timeout: 15 minutes (shorter for micro-account)
    pending_order_timeout_minutes: int = int(os.getenv("BOT_TIMEOUT", "15"))

    # Lot size hard cap
    min_lot_size: float = float(os.getenv("BOT_MIN_LOT", "0.01"))
    max_lot_size: float = float(os.getenv("BOT_MAX_LOT", "0.03"))

    # =========================================================================
    # 4. AUTO-WITHDRAWAL & BALANCE ALERTS
    # =========================================================================
    # Alert when equity exceeds this threshold (user withdraws manually)
    withdrawal_alert_balance: float = float(os.getenv("BOT_WITHDRAW_ALERT", "3000.0"))

    # Minimum balance to continue trading
    minimum_trading_balance: float = float(os.getenv("BOT_MIN_BALANCE", "500.0"))

    # =========================================================================
    # 5. SL/TP CONFIGURATION (Scaled for XAUUSD @ 4000)
    # =========================================================================
    # Fixed SL distance: 16 USD
    sl_distance_usd: float = float(os.getenv("BOT_SL_DISTANCE", "16.0"))

    # Regime-Adaptive Breakeven Triggers (USD)
    be_strong_trend_usd: float = float(os.getenv("BOT_BE_STRONG_TREND", "14.0"))
    be_parabolic_usd: float = float(os.getenv("BOT_BE_PARABOLIC", "18.0"))
    be_consolidating_usd: float = float(os.getenv("BOT_BE_CONSOLIDATING", "10.0"))
    be_sideways_usd: float = float(os.getenv("BOT_BE_SIDEWAYS", "7.0"))
    be_choppy_usd: float = float(os.getenv("BOT_BE_CHOPPY", "6.0"))
    be_reversal_usd: float = float(os.getenv("BOT_BE_REVERSAL", "9.0"))

    # Trailing increment: 5 USD
    trail_increment_usd: float = float(os.getenv("BOT_TRAIL_INCREMENT", "5.0"))

    # Partial close: at 10 USD profit, close 50%
    partial_close_trigger_usd: float = float(os.getenv("BOT_PARTIAL_TRIGGER", "10.0"))
    partial_close_percent: float = float(os.getenv("BOT_PARTIAL_PCT", "0.50"))

    # =========================================================================
    # 6. FRICTION FILTER (Stricter for micro-account)
    # =========================================================================
    max_spread_points: int = int(os.getenv("BOT_MAX_SPREAD", "25"))
    max_slippage_points_filter: int = int(os.getenv("BOT_MAX_SLIPPAGE_FILTER", "15"))
    min_rr_ratio: float = float(os.getenv("BOT_MIN_RR", "1.3"))
    min_profit_usd: float = float(os.getenv("BOT_MIN_PROFIT", "4.0"))
    min_edge_to_friction: float = float(os.getenv("BOT_EDGE_FRICTION", "2.5"))

    # =========================================================================
    # 7. ADAPTIVE TP ENGINE (Closer targets for micro-account)
    # =========================================================================
    tp_trend_rr: float = float(os.getenv("BOT_TP_TREND", "1.8"))
    tp_sideway_rr: float = float(os.getenv("BOT_TP_SIDEWAY", "1.3"))
    tp_highvol_rr: float = float(os.getenv("BOT_TP_HIGHVOL", "1.2"))
    tp_reversal_rr: float = float(os.getenv("BOT_TP_REVERSAL", "1.5"))
    min_tp_distance_usd: float = float(os.getenv("BOT_MIN_TP_DIST", "1.0"))

    # =========================================================================
    # 8. TIME STOP (Shorter durations for micro-account)
    # =========================================================================
    time_stop_m1_minutes: int = int(os.getenv("BOT_TS_M1", "30"))
    time_stop_m5_minutes: int = int(os.getenv("BOT_TS_M5", "60"))
    time_stop_m15_minutes: int = int(os.getenv("BOT_TS_M15", "120"))

    # =========================================================================
    # 9. TRAILING METHODS
    # =========================================================================
    trail_fixed_usd_increment: float = float(os.getenv("BOT_TRAIL_USD_INC", "8.0"))
    trail_fixed_usd_threshold: float = float(os.getenv("BOT_TRAIL_USD_THRESH", "8.0"))
    trail_adaptive_volume_enabled: bool = os.getenv("BOT_TRAIL_ADAPTIVE_VOL", "false").lower() == "true"
    trail_low_vol_increment: float = float(os.getenv("BOT_TRAIL_LOW_VOL", "8.0"))
    trail_normal_vol_increment: float = float(os.getenv("BOT_TRAIL_NORMAL_VOL", "12.0"))
    trail_high_vol_increment: float = float(os.getenv("BOT_TRAIL_HIGH_VOL", "15.0"))
    trail_vol_lookback: int = int(os.getenv("BOT_TRAIL_VOL_LB", "100"))
    trail_vol_low_threshold: float = float(os.getenv("BOT_TRAIL_VOL_LOW", "0.40"))
    trail_vol_high_threshold: float = float(os.getenv("BOT_TRAIL_VOL_HIGH", "0.70"))

    # =========================================================================
    # 10. REGIME DETECTION MODELS
    # =========================================================================
    regime_model_path: str = os.getenv("BOT_REGIME_MODEL", "hmm_regime_model.pkl")
    rule_model_path: str = os.getenv("BOT_RULE_MODEL", "regime_model.pkl")
    lightgbm_models_path: str = os.getenv("BOT_LGBM_MODEL", "lightgbm_regime_models.pkl")

    # Regime Direction Filter
    regime_direction_filter: bool = os.getenv("BOT_REGIME_FILTER", "true").lower() == "true"

    # =========================================================================
    # 11. MICROSTRUCTURE PREDICTOR (S26)
    # =========================================================================
    microstructure_enabled: bool = os.getenv("BOT_MICROSTRUCTURE", "true").lower() == "true"
    microstructure_cvd_lookback: int = int(os.getenv("BOT_MICRO_CVD_LB", "20"))
    microstructure_vol_lookback: int = int(os.getenv("BOT_MICRO_VOL_LB", "50"))
    microstructure_z_score: float = float(os.getenv("BOT_MICRO_ZSCORE", "1.8"))
    microstructure_min_volume: int = int(os.getenv("BOT_MICRO_MIN_VOL", "50"))
    microstructure_min_rr: float = float(os.getenv("BOT_MICRO_MIN_RR", "1.2"))

    # =========================================================================
    # 12. DRAWDOWN RISK SCALER
    # =========================================================================
    drawdown_level1_pct: float = float(os.getenv("BOT_DD_LEVEL1", "1.0"))
    drawdown_level2_pct: float = float(os.getenv("BOT_DD_LEVEL2", "2.0"))
    drawdown_level3_pct: float = float(os.getenv("BOT_DD_LEVEL3", "3.0"))
    drawdown_level1_multiplier: float = float(os.getenv("BOT_DD_MULT1", "0.75"))
    drawdown_level2_multiplier: float = float(os.getenv("BOT_DD_MULT2", "0.50"))
    drawdown_level3_multiplier: float = float(os.getenv("BOT_DD_MULT3", "0.00"))

    # =========================================================================
    # 13. EQUITY CIRCUIT BREAKER
    # =========================================================================
    circuit_breaker_min_winrate: float = float(os.getenv("BOT_CB_MIN_WINRATE", "0.40"))
    circuit_breaker_min_pf: float = float(os.getenv("BOT_CB_MIN_PF", "0.80"))
    circuit_breaker_max_consec_loss: int = int(os.getenv("BOT_CB_MAX_CONSEC_LOSS", "5"))
    circuit_breaker_daily_loss_limit: float = float(os.getenv("BOT_CB_DAILY_LOSS", "2.0"))
    circuit_breaker_rolling_window: int = int(os.getenv("BOT_CB_ROLLING_WINDOW", "20"))
    circuit_breaker_winrate_pause_hours: float = float(os.getenv("BOT_CB_WR_PAUSE", "24.0"))
    circuit_breaker_pf_pause_hours: float = float(os.getenv("BOT_CB_PF_PAUSE", "12.0"))
    circuit_breaker_consec_pause_hours: float = float(os.getenv("BOT_CB_CONSEC_PAUSE", "4.0"))

    # =========================================================================
    # 14. MODIFICATION RATE LIMITER
    # =========================================================================
    mod_max_per_position: int = int(os.getenv("BOT_MOD_MAX_PER_POS", "10"))
    mod_max_per_minute: int = int(os.getenv("BOT_MOD_MAX_PER_MIN", "5"))
    mod_cooldown_seconds: float = float(os.getenv("BOT_MOD_COOLDOWN", "3.0"))

    # =========================================================================
    # 15. EMERGENCY DEFENSE ENGINE
    # =========================================================================
    emergency_flash_crash_threshold_pct: float = float(os.getenv("BOT_EMERGENCY_FC_PCT", "0.02"))
    emergency_flash_crash_window_bars: int = int(os.getenv("BOT_EMERGENCY_FC_BARS", "5"))
    emergency_spread_crisis_multiplier: float = float(os.getenv("BOT_EMERGENCY_SPREAD_MULT", "5.0"))
    emergency_max_daily_loss_pct: float = float(os.getenv("BOT_EMERGENCY_MAX_DAILY", "3.0"))
    emergency_position_abnormal_loss_pct: float = float(os.getenv("BOT_EMERGENCY_ABNORMAL", "5.0"))
    emergency_news_buffer_minutes: int = int(os.getenv("BOT_EMERGENCY_NEWS_BUFFER", "30"))

    # =========================================================================
    # 16. SESSION MANAGEMENT (UTC hours)
    # =========================================================================
    session_london_open_utc: tuple = (7, 9)
    session_london_utc: tuple = (9, 12)
    session_ny_open_utc: tuple = (12, 15)
    session_ny_miday_utc: tuple = (15, 18)
    session_asian_utc: tuple = (18, 6)
    session_transition_utc: tuple = (22, 23)

    # Session-specific spread limits (points)
    session_spread_limit_prime: int = int(os.getenv("BOT_SESSION_SPREAD_PRIME", "25"))
    session_spread_limit_active: int = int(os.getenv("BOT_SESSION_SPREAD_ACTIVE", "30"))
    session_spread_limit_asian: int = int(os.getenv("BOT_SESSION_SPREAD_ASIAN", "20"))

    # =========================================================================
    # 17. ANALYTICS & ATTRIBUTION
    # =========================================================================
    attribution_enabled: bool = os.getenv("BOT_ATTRIBUTION", "true").lower() == "true"
    attribution_db_path: str = os.getenv("BOT_ATTRIBUTION_DB", "bot_state.db")
    performance_report_interval_hours: int = int(os.getenv("BOT_PERF_REPORT_HRS", "24"))

    # =========================================================================
    # 18. EVENT LOOP SETTINGS
    # =========================================================================
    event_loop_interval_seconds: float = float(os.getenv("BOT_INTERVAL", "1.0"))
    reconciliation_interval_seconds: int = int(os.getenv("BOT_RECONCILE_INTERVAL", "60"))
    intelligence_check_interval_seconds: int = int(os.getenv("BOT_INTEL_INTERVAL", "300"))
    context_log_interval_seconds: int = int(os.getenv("BOT_CONTEXT_LOG_INTERVAL", "300"))

    # =========================================================================
    # 19. FILE PATHS
    # =========================================================================
    state_db_path: str = os.getenv("BOT_DB_PATH", "bot_state.db")
    log_directory: str = os.getenv("BOT_LOG_DIR", "logs")

    # =========================================================================
    # 20. AUTO-SCALING (Future-proofing for price level changes)
    # =========================================================================
    auto_scale_enabled: bool = os.getenv("BOT_AUTO_SCALE", "false").lower() == "true"
    baseline_price: float = float(os.getenv("BOT_BASELINE_PRICE", "4000.0"))


# Instantiate global config object
config = Config()