#!/bin/bash
# ============================================================================
# Institutional Trading Bot - VPS Deployment Script (Linux)
# ============================================================================

set -e

echo "=================================================="
echo " Starting Institutional Trading Bot (Micro-Account)"
echo "=================================================="

# 1. Check Python Environment
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment 'venv' not found."
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 2. Activate Virtual Environment
source venv/bin/activate

# 3. Set Environment Variables (Micro-Account Mode)
export BOT_MICRO_ACCOUNT=true
export BOT_MICRO_RISK=0.5
export BOT_MICRO_SL=8.0
export BOT_MICRO_BE=3.5
export BOT_MICRO_TRAIL=2.5
export BOT_MICRO_PARTIAL=5.0
export BOT_MICRO_PARTIAL_PCT=0.50
export BOT_MICRO_MIN_LOT=0.01
export BOT_MICRO_MAX_LOT=0.03
export BOT_LOG_LEVEL=INFO

# 4. Ensure Log Directory Exists
mkdir -p logs

# 5. Run the Bot
echo "[OK] Environment configured. Starting main.py..."
python main.py --symbol XAUUSDm --tf M15 --risk 0.5

# 6. Deactivate on Exit
deactivate
echo "[OK] Bot shutdown complete."