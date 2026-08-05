"""
Centralized Configuration Manager.
Stores all system parameters, risk limits, and MT5 settings.
Includes Micro-Account Mode Configuration (Scaled for XAUUSD @ 4000 USD baseline).
"""
import os


class Config:
    """System configuration parameters."""
    
    # =========================================================================
    # MT5 Settings
    # =========================================================================
    symbol: str = os.getenv("BOT_SYMBOL", "XAUUSDm")
    magic_number: int = int(os.getenv("BOT_MAGIC", "888888"))
    max_slippage_points: int = int(os.getenv("BOT_SLIPPAGE", "20"))
    
    # MT5 Connection Credentials (Required for VPS/Headless environments)
    # If left as 0/empty, the bot will attach to the currently open MT5 terminal.
    mt5_login: int = int(os.getenv("BOT_MT5_LOGIN", "415785591"))
    mt5_password: str = os.getenv("BOT_MT5_PASSWORD", "Nomorefriends@1")
    mt5_server: str = os.getenv("BOT_MT5_SERVER", "Exness-MT5Trial14")
    mt5_path: str = os.getenv("BOT_MT5_PATH", "")
    
    # =========================================================================
    # Timeframe Settings
    # =========================================================================
    primary_timeframe: str = os.getenv("BOT_TF", "M15")
    
    # =========================================================================
    # Risk Management (Normal Mode)
    # =========================================================================
    risk_per_trade_pct: float = float(os.getenv("BOT_RISK_PCT", "1.0"))
    max_open_positions: int = int(os.getenv("BOT_MAX_POS", "4"))
    max_pending_orders: int = int(os.getenv("BOT_MAX_PEND", "5"))
    pending_order_timeout_minutes: int = int(os.getenv("BOT_TIMEOUT", "30"))
    max_daily_loss_pct: float = float(os.getenv("BOT_MAX_DAILY_LOSS", "3.0"))
    
    # =========================================================================
    # Event Loop Settings
    # =========================================================================
    event_loop_interval_seconds: float = float(os.getenv("BOT_INTERVAL", "1.0"))
    
    # =========================================================================
    # File Paths
    # =========================================================================
    state_db_path: str = os.getenv("BOT_DB_PATH", "bot_state.db")
    regime_model_path: str = os.getenv("BOT_REGIME_MODEL", "hmm_regime_model.pkl")
    lightgbm_models_path: str = os.getenv("BOT_LGBM_MODEL", "lightgbm_regime_models.pkl")
    log_directory: str = os.getenv("BOT_LOG_DIR", "logs")
    
    # =========================================================================
    # Scalping Mode Configuration
    # =========================================================================
    scalping_mode: bool = os.getenv("BOT_SCALPING_MODE", "false").lower() == "true"
    scalping_primary_tf: str = os.getenv("BOT_SCALP_TF", "M1")
    max_spread_points_scalp: int = int(os.getenv("BOT_MAX_SPREAD_SCALP", "300"))
    max_trades_per_day_scalp: int = int(os.getenv("BOT_MAX_TRADES_SCALP", "20"))
    scalping_regimes_allowed: list = ['TIGHT_RANGE', 'CLASSIC_RANGE', 'QUIET_RALLY', 'SLOW_BLEED']
    scalp_risk_per_trade_pct: float = float(os.getenv("BOT_SCALP_RISK", "0.5"))
    scalp_max_daily_loss_pct: float = float(os.getenv("BOT_SCALP_DAILY_LOSS", "2.0"))
    
    # =========================================================================
    # Fixed Dollar Incremental Trailing Configuration
    # =========================================================================
    trail_fixed_usd_increment: float = float(os.getenv("BOT_TRAIL_USD_INCREMENT", "8.0"))
    trail_fixed_usd_threshold: float = float(os.getenv("BOT_TRAIL_USD_THRESHOLD", "8.0"))
    trail_method_adaptive_volume: bool = os.getenv("BOT_TRAIL_ADAPTIVE_VOL", "true").lower() == "true"
    trail_low_vol_increment: float = float(os.getenv("BOT_TRAIL_LOW_VOL_INC", "8.0"))
    trail_normal_vol_increment: float = float(os.getenv("BOT_TRAIL_NORMAL_VOL_INC", "12.0"))
    trail_high_vol_increment: float = float(os.getenv("BOT_TRAIL_HIGH_VOL_INC", "15.0"))
    trail_vol_lookback: int = int(os.getenv("BOT_TRAIL_VOL_LOOKBACK", "100"))
    trail_vol_low_threshold: float = float(os.getenv("BOT_TRAIL_VOL_LOW", "0.40"))
    trail_vol_high_threshold: float = float(os.getenv("BOT_TRAIL_VOL_HIGH", "0.70"))
    
    # =========================================================================
    # Micro-Account Mode Configuration
    # (Scaled 2x for XAUUSD @ 4000 USD baseline)
    # =========================================================================
    micro_account_mode: bool = os.getenv("BOT_MICRO_ACCOUNT", "true").lower() == "true"
    micro_risk_per_trade_pct: float = float(os.getenv("BOT_MICRO_RISK", "0.5"))
    
    # SL Distance (16 USD = 2x of original 8 USD)
    micro_sl_distance_usd: float = float(os.getenv("BOT_MICRO_SL", "8.0"))
    
    # Regime-Adaptive Breakeven Triggers (Scaled 2x)
    micro_be_strong_trend_usd: float = float(os.getenv("BOT_MICRO_BE_STRONG_TREND", "10.0"))
    micro_be_parabolic_usd: float = float(os.getenv("BOT_MICRO_BE_PARABOLIC", "10.0"))
    micro_be_consolidating_usd: float = float(os.getenv("BOT_MICRO_BE_CONSOLIDATING", "10.0"))
    micro_be_sideways_usd: float = float(os.getenv("BOT_MICRO_BE_SIDEWAYS", "7.0"))
    micro_be_choppy_usd: float = float(os.getenv("BOT_MICRO_BE_CHOPPY", "6.0"))
    micro_be_reversal_usd: float = float(os.getenv("BOT_MICRO_BE_REVERSAL", "9.0"))
    
    # Trailing Increment & Partial Close (Scaled 2x)
    micro_trail_increment_usd: float = float(os.getenv("BOT_MICRO_TRAIL", "5.0"))
    micro_partial_close_trigger_usd: float = float(os.getenv("BOT_MICRO_PARTIAL", "10.0"))
    micro_partial_close_percent: float = float(os.getenv("BOT_MICRO_PARTIAL_PCT", "0.50"))
    
    # Lot Size Bounds (Hard Cap)
    micro_min_lot_size: float = float(os.getenv("BOT_MICRO_MIN_LOT", "0.01"))
    micro_max_lot_size: float = float(os.getenv("BOT_MICRO_MAX_LOT", "0.03"))
    
    # =========================================================================
    # Auto-Scaling Configuration (Optional - For Future Price Levels)
    # =========================================================================
    micro_auto_scale_enabled: bool = os.getenv("BOT_MICRO_AUTO_SCALE", "false").lower() == "true"
    micro_baseline_price: float = float(os.getenv("BOT_MICRO_BASELINE_PRICE", "4000.0"))

    # =========================================================================
    # Direction Lock Configuration (Prevent Hedging)
    # =========================================================================
    direction_lock_enabled: bool = os.getenv("BOT_DIRECTION_LOCK", "true").lower() == "true"
    regime_direction_filter: bool = os.getenv("BOT_REGIME_DIRECTION_FILTER", "true").lower() == "true"

# Instantiate global config object
config = Config()