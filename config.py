"""
Centralized Configuration Manager.
Stores all system parameters, risk limits, and MT5 settings.
"""
import os
from typing import ClassVar, Tuple


class Config:
    """System configuration parameters."""
    
    # MT5 Settings
    symbol: ClassVar[str] = os.getenv("BOT_SYMBOL", "XAUUSDm")
    magic_number: ClassVar[int] = int(os.getenv("BOT_MAGIC", "888888"))
    max_slippage_points: ClassVar[int] = int(os.getenv("BOT_SLIPPAGE", "20"))
    
    # MT5 Connection Credentials
    mt5_login: ClassVar[int] = int(os.getenv("BOT_MT5_LOGIN", "415785591"))
    mt5_password: ClassVar[str] = os.getenv("BOT_MT5_PASSWORD", "Nomorefriends@1")
    mt5_server: ClassVar[str] = os.getenv("BOT_MT5_SERVER", "Exness-MT5Trial14")
    mt5_path: ClassVar[str] = os.getenv("BOT_MT5_PATH", "")

    # Timeframe Settings
    primary_timeframe: ClassVar[str] = os.getenv("BOT_TF", "M15")
    
    # Risk Management
    risk_per_trade_pct: ClassVar[float] = float(os.getenv("BOT_RISK_PCT", "1.0"))
    max_open_positions: ClassVar[int] = int(os.getenv("BOT_MAX_POS", "4"))
    max_pending_orders: ClassVar[int] = int(os.getenv("BOT_MAX_PEND", "5"))
    pending_order_timeout_minutes: ClassVar[int] = int(os.getenv("BOT_TIMEOUT", "30"))
    max_daily_loss_pct: ClassVar[float] = float(os.getenv("BOT_MAX_DAILY_LOSS", "3.0"))
    
    # Event Loop Settings
    event_loop_interval_seconds: ClassVar[float] = float(os.getenv("BOT_INTERVAL", "1.0"))
    
    # File Paths
    state_db_path: ClassVar[str] = os.getenv("BOT_DB_PATH", "bot_state.db")
    regime_model_path: ClassVar[str] = os.getenv("BOT_REGIME_MODEL", "hmm_regime_model.pkl")
    lightgbm_models_path: ClassVar[str] = os.getenv("BOT_LGBM_MODEL", "lightgbm_regime_models.pkl")
    log_directory: ClassVar[str] = os.getenv("BOT_LOG_DIR", "logs")
    
    # Scalping Mode Configuration
    scalping_mode: ClassVar[bool] = os.getenv("BOT_SCALPING_MODE", "false").lower() == "true"
    scalping_primary_tf: ClassVar[str] = os.getenv("BOT_SCALP_TF", "M1")
    max_spread_points_scalp: ClassVar[int] = int(os.getenv("BOT_MAX_SPREAD_SCALP", "300"))
    max_trades_per_day_scalp: ClassVar[int] = int(os.getenv("BOT_MAX_TRADES_SCALP", "20"))
    scalping_regimes_allowed: ClassVar[Tuple[str, ...]] = (
        'TIGHT_RANGE', 'CLASSIC_RANGE', 'QUIET_RALLY', 'SLOW_BLEED'
    )
    scalp_risk_per_trade_pct: ClassVar[float] = float(os.getenv("BOT_SCALP_RISK", "0.5"))
    scalp_max_daily_loss_pct: ClassVar[float] = float(os.getenv("BOT_SCALP_DAILY_LOSS", "2.0"))
    
    # Micro-Account Mode Configuration
    micro_account_mode: ClassVar[bool] = os.getenv("BOT_MICRO_ACCOUNT", "true").lower() == "true"
    micro_account_balance_threshold: ClassVar[float] = float(os.getenv("BOT_MICRO_THRESHOLD", "5000.0"))
    micro_risk_per_trade_pct: ClassVar[float] = float(os.getenv("BOT_MICRO_RISK", "0.5"))
    micro_sl_distance_usd: ClassVar[float] = float(os.getenv("BOT_MICRO_SL", "8.0"))
    micro_breakeven_trigger_usd: ClassVar[float] = float(os.getenv("BOT_MICRO_BE", "3.5"))
    micro_trail_increment_usd: ClassVar[float] = float(os.getenv("BOT_MICRO_TRAIL", "2.5"))
    micro_partial_close_trigger_usd: ClassVar[float] = float(os.getenv("BOT_MICRO_PARTIAL", "5.0"))
    micro_partial_close_percent: ClassVar[float] = float(os.getenv("BOT_MICRO_PARTIAL_PCT", "0.50"))
    micro_min_lot_size: ClassVar[float] = float(os.getenv("BOT_MICRO_MIN_LOT", "0.01"))
    micro_max_lot_size: ClassVar[float] = float(os.getenv("BOT_MICRO_MAX_LOT", "0.03"))

# Instantiate global configuration object
config = Config()