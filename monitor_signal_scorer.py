#!/usr/bin/env python3
"""
Expert Signal Scorer Monitor.
Analyzes the correlation between Signal Score and Trade Profitability.
"""
import sqlite3
import pandas as pd
import json
import logging
import sys
import numpy as np
from config import config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    logger = logging.getLogger("MonitorScorer")
    
    logger.info(f"[START] Analyzing Scorer Performance from {config.state_db_path}")
    
    try:
        conn = sqlite3.connect(config.state_db_path)
        # Fetch all closed trades
        query = """
            SELECT profit, meta_data, strategy, close_time 
            FROM trade_history 
            WHERE is_pending = 0 AND profit IS NOT NULL
            ORDER BY close_time DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("[WARN] No trade history found.")
            return
            
        logger.info(f"[INFO] Loaded {len(df)} trades.")
        
        # [FIX] Parse JSON meta_data safely
        def extract_score(meta_str):
            if not meta_str: return 50.0 # Default score
            try:
                meta = json.loads(meta_str)
                return float(meta.get('expert_score', 50.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                return 50.0
                
        df['score'] = df['meta_data'].apply(extract_score)
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0.0)
        df['is_win'] = df['profit'] > 0
        
        # Analysis Buckets
        buckets = [
            (90, 100, 'A+'), (80, 89, 'A'), (70, 79, 'B+'), 
            (60, 69, 'B'), (50, 59, 'C'), (0, 49, 'F')
        ]
        
        print("\n" + "="*60)
        print("EXPERT SIGNAL SCORER PERFORMANCE REPORT")
        print("="*60)
        print(f"{'Grade':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Total PnL':<12} | {'Avg PnL':<10}")
        print("-" * 60)
        
        for min_s, max_s, grade in buckets:
            mask = (df['score'] >= min_s) & (df['score'] <= max_s)
            subset = df[mask]
            
            if subset.empty:
                continue
                
            trades = len(subset)
            win_rate = subset['is_win'].mean() * 100
            total_pnl = subset['profit'].sum()
            avg_pnl = subset['profit'].mean()
            
            print(f"{grade:<10} | {trades:<8} | {win_rate:<10.1f}% | ${total_pnl:<11.2f} | ${avg_pnl:<9.2f}")
            
        print("="*60)
        
        # Correlation
        corr = df['score'].corr(df['profit'])
        print(f"\n[METRIC] Pearson Correlation (Score vs Profit): {corr:.3f}")
        
        if corr < 0.1:
            print("[ADVICE] Correlation is weak. Scorer weights may need tuning.")
        elif corr > 0.3:
            print("[ADVICE] Strong positive correlation. Scorer is effective.")
        else:
            print("[ADVICE] Moderate correlation. Monitor for stability.")
            
    except Exception as e:
        logger.error(f"[FAIL] Analysis error: {e}", exc_info=True)

if __name__ == "__main__":
    main()