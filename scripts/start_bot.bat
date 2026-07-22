@echo off
REM ============================================================================
REM Institutional Trading Bot - Local Deployment Script (Windows)
REM ============================================================================

echo ==================================================
echo  Starting Institutional Trading Bot (Micro-Account)
echo ==================================================

REM 1. Check Python Environment
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found.
    echo Please run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM 2. Activate Virtual Environment
call venv\Scripts\activate.bat

REM 3. Set Environment Variables (Micro-Account Mode)
set BOT_MICRO_ACCOUNT=true
set BOT_MICRO_RISK=0.5
set BOT_MICRO_SL=8.0
set BOT_MICRO_BE=3.5
set BOT_MICRO_TRAIL=2.5
set BOT_MICRO_PARTIAL=5.0
set BOT_MICRO_PARTIAL_PCT=0.50
set BOT_MICRO_MIN_LOT=0.01
set BOT_MICRO_MAX_LOT=0.03
set BOT_LOG_LEVEL=INFO

REM 4. Ensure Log Directory Exists
IF NOT EXIST "logs" mkdir logs

REM 5. Run the Bot
echo [OK] Environment configured. Starting main.py...
python main.py --symbol XAUUSDm --tf M15 --risk 0.5

REM 6. Deactivate on Exit
call deactivate
echo [OK] Bot shutdown complete.
pause