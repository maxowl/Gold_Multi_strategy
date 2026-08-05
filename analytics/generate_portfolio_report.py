"""
Institutional Portfolio Telemetry & Analytics Dashboard
Extracts data from bot_state.db and generates comprehensive risk/performance reports.
Must be run with Read-Only access to prevent locking the live trading bot.



        python analytics/generate_portfolio_report.py --days 30
        python analytics/generate_portfolio_report.py --days 7 --export
"""
import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = os.getenv("BOT_DB_PATH", "bot_state.db")
EXPORT_DIR = "analytics_exports"

# ============================================================================
# DATA INGESTION
# ============================================================================
def load_and_parse_trades(db_path: str, days: int = 30) -> pd.DataFrame:
    """
    Load trade history from SQLite and parse JSON meta_data.
    Uses Read-Only URI to prevent locking the live bot.
    """
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at {db_path}")
        sys.exit(1)

    try:
        # Read-Only connection to prevent OperationalError (Database Locked)
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        
        query = """
            SELECT ticket, symbol, direction, entry_price, exit_price, volume, 
                   profit, open_time, close_time, strategy, meta_data
            FROM trade_history
            WHERE close_time >= datetime('now', ?)
            ORDER BY close_time DESC
        """
        df = pd.read_sql_query(query, conn, params=(f"-{days} days",))
        conn.close()
        
        if df.empty:
            print(f"[INFO] No trades found in the last {days} days.")
            return pd.DataFrame()

        # Parse JSON meta_data safely
        parsed_meta = []
        for meta_str in df['meta_data']:
            try:
                meta_dict = json.loads(meta_str) if meta_str else {}
            except json.JSONDecodeError:
                meta_dict = {}
            parsed_meta.append(meta_dict)
            
        df['meta'] = parsed_meta
        
        # Extract key analytics fields from meta_data
        df['regime'] = df['meta'].apply(lambda x: x.get('regime', 'UNKNOWN'))
        df['regime_name'] = df['meta'].apply(lambda x: x.get('regime_name', 'UNKNOWN'))
        df['loss_attribution'] = df['meta'].apply(lambda x: x.get('loss_attribution', 'UNKNOWN'))
        df['expert_score'] = df['meta'].apply(lambda x: x.get('expert_score', 0))
        df['kelly_risk_used'] = df['meta'].apply(lambda x: x.get('kelly_risk_pct', 0))
        
        # Calculate R-Multiples (Assuming initial risk is stored in meta, or estimate from SL)
        df['initial_sl'] = df['meta'].apply(lambda x: x.get('sl_price', 0))
        df['risk_distance'] = abs(df['entry_price'] - df['initial_sl'])
        df['r_multiple'] = np.where(
            df['risk_distance'] > 0,
            (df['exit_price'] - df['entry_price']) / df['risk_distance'] * np.where(df['direction'] == 'BUY', 1, -1),
            0
        )
        
        return df

    except sqlite3.OperationalError as e:
        print(f"[ERROR] Database locked or inaccessible: {e}")
        print("[HINT] Ensure StateManager uses WAL mode, or stop the bot before running analytics.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

# ============================================================================
# METRICS CALCULATION ENGINES
# ============================================================================
def calculate_core_metrics(df: pd.DataFrame) -> Dict:
    """Calculate standard portfolio health metrics."""
    if df.empty:
        return {}
        
    total_trades = len(df)
    winners = df[df['profit'] > 0]
    losers = df[df['profit'] <= 0]
    
    win_rate = len(winners) / total_trades if total_trades > 0 else 0
    gross_profit = winners['profit'].sum() if not winners.empty else 0
    gross_loss = abs(losers['profit'].sum()) if not losers.empty else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winners['profit'].mean() if not winners.empty else 0
    avg_loss = abs(losers['profit'].mean()) if not losers.empty else 0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Max Drawdown Calculation (Peak to Trough)
    df_sorted = df.sort_values('close_time')
    cumulative_pnl = df_sorted['profit'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd,
        'net_pnl': df['profit'].sum()
    }

def analyze_regime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Group performance by Unified Regime."""
    if df.empty:
        return pd.DataFrame()
        
    regime_stats = df.groupby('regime').agg(
        trades=('profit', 'count'),
        win_rate=('profit', lambda x: (x > 0).mean()),
        net_pnl=('profit', 'sum'),
        avg_r_multiple=('r_multiple', 'mean')
    ).reset_index()
    
    return regime_stats.sort_values('net_pnl', ascending=False)

def analyze_loss_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize the root causes of losing trades."""
    losers = df[df['profit'] <= 0]
    if losers.empty:
        return pd.DataFrame()
        
    attribution_stats = losers.groupby('loss_attribution').agg(
        count=('profit', 'count'),
        total_loss=('profit', 'sum')
    ).reset_index()
    
    total_losses = attribution_stats['count'].sum()
    attribution_stats['pct_of_losses'] = attribution_stats['count'] / total_losses
    
    return attribution_stats.sort_values('count', ascending=False)

def evaluate_kelly_efficiency(df: pd.DataFrame) -> Dict:
    """Evaluate if Kelly sizing added value over flat sizing."""
    if df.empty or 'kelly_risk_used' not in df.columns:
        return {}
        
    # Filter trades where Kelly was actually applied (risk > 0)
    kelly_trades = df[df['kelly_risk_used'] > 0]
    if kelly_trades.empty:
        return {'status': 'No Kelly trades found'}
        
    # Actual Return (using Kelly sizing)
    actual_return = kelly_trades['profit'].sum()
    
    # Simulated Flat Return (assuming all trades used base 1% risk)
    # We approximate by normalizing profit to a 1% baseline
    base_risk_pct = 1.0
    kelly_trades = kelly_trades.copy()
    kelly_trades['flat_profit'] = kelly_trades['profit'] * (base_risk_pct / kelly_trades['kelly_risk_used'])
    simulated_flat_return = kelly_trades['flat_profit'].sum()
    
    efficiency_ratio = actual_return / simulated_flat_return if simulated_flat_return != 0 else 1.0
    
    return {
        'actual_kelly_pnl': actual_return,
        'simulated_flat_pnl': simulated_flat_return,
        'efficiency_ratio': efficiency_ratio,
        'kelly_trades_count': len(kelly_trades)
    }

# ============================================================================
# TERMINAL DASHBOARD RENDERING
# ============================================================================
def render_terminal_dashboard(core_metrics: Dict, regime_df: pd.DataFrame, 
                              loss_df: pd.DataFrame, kelly_metrics: Dict, days: int):
    """Print institutional-grade ASCII dashboard to console."""
    
    print("\n" + "="*80)
    print(f"  INSTITUTIONAL PORTFOLIO TELEMETRY REPORT | LAST {days} DAYS")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Core Portfolio Health
    print("\n[1] CORE PORTFOLIO HEALTH")
    print("-" * 80)
    if core_metrics:
        print(f"  Total Trades      : {core_metrics['total_trades']}")
        print(f"  Win Rate          : {core_metrics['win_rate']:.2%}")
        print(f"  Profit Factor     : {core_metrics['profit_factor']:.2f}")
        print(f"  Expectancy (EV)   : ${core_metrics['expectancy']:.2f} per trade")
        print(f"  Net PnL           : ${core_metrics['net_pnl']:.2f}")
        print(f"  Max Drawdown      : ${core_metrics['max_drawdown']:.2f}")
        print(f"  Avg Win / Avg Loss: ${core_metrics['avg_win']:.2f} / ${core_metrics['avg_loss']:.2f}")
    else:
        print("  No data available.")
        
    # 2. Regime Performance Matrix
    print("\n[2] REGIME PERFORMANCE MATRIX (Unified)")
    print("-" * 80)
    if not regime_df.empty:
        print(f"  {'Regime':<15} | {'Trades':<8} | {'Win Rate':<10} | {'Net PnL':<12} | {'Avg R-Mult'}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
        for _, row in regime_df.iterrows():
            print(f"  {row['regime']:<15} | {row['trades']:<8} | {row['win_rate']:<10.2%} | ${row['net_pnl']:<11.2f} | {row['avg_r_multiple']:.2f}R")
    else:
        print("  No regime data available.")
        
    # 3. Loss Attribution Breakdown
    print("\n[3] LOSS ATTRIBUTION BREAKDOWN")
    print("-" * 80)
    if not loss_df.empty:
        print(f"  {'Category':<25} | {'Count':<8} | {'% of Losses':<12} | {'Total Lost'}")
        print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")
        for _, row in loss_df.iterrows():
            print(f"  {row['loss_attribution']:<25} | {row['count']:<8} | {row['pct_of_losses']:<12.2%} | ${row['total_loss']:.2f}")
    else:
        print("  No losing trades recorded. (Perfect run or insufficient data)")
        
    # 4. Kelly Criterion Efficiency
    print("\n[4] KELLY CRITERION EFFICIENCY")
    print("-" * 80)
    if kelly_metrics and 'status' not in kelly_metrics:
        print(f"  Trades Sized by Kelly : {kelly_metrics['kelly_trades_count']}")
        print(f"  Actual Kelly PnL      : ${kelly_metrics['actual_kelly_pnl']:.2f}")
        print(f"  Simulated Flat (1%)   : ${kelly_metrics['simulated_flat_pnl']:.2f}")
        eff = kelly_metrics['efficiency_ratio']
        status = "OUTPERFORMING" if eff > 1.0 else "UNDERPERFORMING"
        print(f"  Efficiency Ratio      : {eff:.2f}x ({status})")
        if eff < 0.8:
            print("  [WARNING] Kelly is underperforming flat sizing. Consider increasing min_trades threshold.")
    else:
        print("  Insufficient data to evaluate Kelly efficiency.")
        
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Institutional Portfolio Analytics Dashboard")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    parser.add_argument("--export", action="store_true", help="Export raw data to CSV")
    args = parser.parse_args()
    
    print(f"[INFO] Loading trade history from {DB_PATH} (Last {args.days} days)...")
    df = load_and_parse_trades(DB_PATH, days=args.days)
    
    if df.empty:
        print("[INFO] Exiting. No trades to analyze.")
        return
        
    if args.export:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(EXPORT_DIR, f"trade_data_{timestamp}.csv")
        # Drop the complex 'meta' dict column for clean CSV export
        df_export = df.drop(columns=['meta'])
        df_export.to_csv(export_path, index=False)
        print(f"[OK] Raw data exported to {export_path}")

    # Calculate Metrics
    core_metrics = calculate_core_metrics(df)
    regime_df = analyze_regime_performance(df)
    loss_df = analyze_loss_attribution(df)
    kelly_metrics = evaluate_kelly_efficiency(df)
    
    # Render Dashboard
    render_terminal_dashboard(core_metrics, regime_df, loss_df, kelly_metrics, args.days)

if __name__ == "__main__":
    main()