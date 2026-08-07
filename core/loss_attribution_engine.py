"""
Loss Attribution Engine - Root Cause Analysis.

Analyzes the root causes of losing trades to improve system performance.
Provides detailed categorization and pattern detection for continuous improvement.

Loss Categories:
  1. STOP_LOSS: Price hit SL (normal loss)
  2. TIME_STOP: Position closed due to time limit
  3. EDGE_DECAY: Setup invalidated
  4. REGIME_CONFLICT: Position contradicted new regime
  5. CHOPPY_EXIT: Closed due to choppy market
  6. REVERSAL: Closed due to reversal signal
  7. EMERGENCY: Emergency close (flash crash, etc.)
  8. DYNAMIC_EXIT: Strategy-specific exit triggered
  9. MANUAL: Manually closed by user

Analysis Dimensions:
  - Strategy performance by regime
  - Loss distribution by category
  - Time-of-day loss patterns
  - Recurring loss patterns
  - Consecutive loss analysis
"""
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from config import config


class LossAttributionEngine:
    """
    Analyzes root causes of losing trades for continuous improvement.
    
    Provides:
      - Detailed loss categorization
      - Root cause analysis
      - Pattern detection
      - Strategy-Regime performance analysis
      - Time-based loss patterns
    """

    # Loss categories
    CATEGORY_STOP_LOSS = 'STOP_LOSS'
    CATEGORY_TIME_STOP = 'TIME_STOP'
    CATEGORY_EDGE_DECAY = 'EDGE_DECAY'
    CATEGORY_REGIME_CONFLICT = 'REGIME_CONFLICT'
    CATEGORY_CHOPPY_EXIT = 'CHOPPY_EXIT'
    CATEGORY_REVERSAL = 'REVERSAL'
    CATEGORY_EMERGENCY = 'EMERGENCY'
    CATEGORY_DYNAMIC_EXIT = 'DYNAMIC_EXIT'
    CATEGORY_MANUAL = 'MANUAL'
    CATEGORY_UNKNOWN = 'UNKNOWN'

    def __init__(self, db_path: str = None):
        """
        Initialize LossAttributionEngine.
        
        Args:
            db_path: Path to SQLite database (defaults to config.state_db_path)
        """
        if db_path is None:
            db_path = config.state_db_path
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)

        self._create_tables()
        self.logger.info("[LOSS_ATTR] Initialized")

    # =========================================================================
    # TABLE CREATION
    # =========================================================================

    def _create_tables(self):
        """Create loss_attribution table if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loss_attribution (
                    ticket INTEGER PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    strategy_category TEXT,
                    regime_name TEXT,
                    unified_regime TEXT,
                    loss_category TEXT NOT NULL,
                    loss_amount REAL NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    duration_minutes REAL,
                    exit_reason TEXT,
                    root_cause_analysis TEXT,
                    time_of_day INTEGER,
                    day_of_week INTEGER,
                    session TEXT,
                    choppy_score REAL,
                    active_killers TEXT,
                    recorded_at TEXT NOT NULL
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_loss_attribution_strategy
                ON loss_attribution(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_loss_attribution_regime
                ON loss_attribution(regime_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_loss_attribution_category
                ON loss_attribution(loss_category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_loss_attribution_recorded
                ON loss_attribution(recorded_at)
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Table creation error: {e}")

    # =========================================================================
    # LOSS CATEGORIZATION
    # =========================================================================

    def categorize_loss(self, exit_reason: str, meta: Dict = None) -> str:
        """
        Categorize a loss based on exit reason.
        
        Args:
            exit_reason: Exit reason string
            meta: Position metadata
            
        Returns:
            Loss category string
        """
        if not exit_reason:
            return self.CATEGORY_UNKNOWN

        exit_reason_lower = exit_reason.lower()

        # Categorize based on keywords
        if 'stop loss' in exit_reason_lower or 'sl' in exit_reason_lower:
            return self.CATEGORY_STOP_LOSS

        if 'time stop' in exit_reason_lower or 'time limit' in exit_reason_lower:
            return self.CATEGORY_TIME_STOP

        if 'edge decay' in exit_reason_lower or 'decay' in exit_reason_lower:
            return self.CATEGORY_EDGE_DECAY

        if 'regime conflict' in exit_reason_lower:
            return self.CATEGORY_REGIME_CONFLICT

        if 'choppy' in exit_reason_lower:
            return self.CATEGORY_CHOPPY_EXIT

        if 'reversal' in exit_reason_lower:
            return self.CATEGORY_REVERSAL

        if 'emergency' in exit_reason_lower or 'flash crash' in exit_reason_lower:
            return self.CATEGORY_EMERGENCY

        if 'dynamic exit' in exit_reason_lower:
            return self.CATEGORY_DYNAMIC_EXIT

        if 'manual' in exit_reason_lower:
            return self.CATEGORY_MANUAL

        return self.CATEGORY_UNKNOWN

    # =========================================================================
    # ROOT CAUSE ANALYSIS
    # =========================================================================

    def analyze_root_cause(self, pos: Dict, exit_price: float,
                            exit_reason: str, regime_context: Dict = None) -> Dict:
        """
        Perform deep root cause analysis for a losing trade.
        
        Args:
            pos: Position dict
            exit_price: Exit price
            exit_reason: Exit reason string
            regime_context: Current regime information
            
        Returns:
            Dict with detailed analysis
        """
        meta = pos.get('meta_data', {})
        strategy_name = meta.get('strategy', 'Unknown')
        strategy_category = meta.get('strategy_category', 'GENERAL')
        entry_price = pos.get('entry_price', 0)
        sl_price = pos.get('sl', 0)
        tp_price = pos.get('tp', 0)
        position_type = pos.get('position_type', 'BUY')

        is_buy = position_type == 'BUY'
        loss_amount = (exit_price - entry_price) if is_buy else (entry_price - exit_price)

        # Categorize loss
        category = self.categorize_loss(exit_reason, meta)

        # Calculate additional metrics
        risk = abs(entry_price - sl_price) if sl_price > 0 else 0
        reward = abs(tp_price - entry_price) if tp_price > 0 else 0
        rr_achieved = abs(exit_price - entry_price) / risk if risk > 0 else 0

        # Setup time and duration
        setup_time_str = pos.get('setup_time') or pos.get('open_time')
        duration_minutes = 0
        if setup_time_str:
            try:
                setup_time = datetime.fromisoformat(setup_time_str)
                duration_minutes = (datetime.now() - setup_time).total_seconds() / 60.0
            except Exception:
                pass

        # Root cause analysis
        root_cause = self._determine_root_cause(
            category, entry_price, exit_price, sl_price, tp_price,
            is_buy, duration_minutes, regime_context
        )

        # Time analysis
        time_analysis = self._analyze_time_factors(setup_time_str)

        # Market context
        market_context = self._analyze_market_context(meta, regime_context)

        analysis = {
            'ticket': pos.get('ticket', 0),
            'strategy': strategy_name,
            'strategy_category': strategy_category,
            'category': category,
            'loss_amount': round(loss_amount, 2),
            'risk': round(risk, 2),
            'reward_target': round(reward, 2),
            'rr_achieved': round(rr_achieved, 2),
            'duration_minutes': round(duration_minutes, 1),
            'exit_reason': exit_reason,
            'root_cause': root_cause,
            'time_analysis': time_analysis,
            'market_context': market_context,
            'recommendations': self._generate_recommendations(
                category, root_cause, strategy_category, duration_minutes
            )
        }

        return analysis

    def _determine_root_cause(self, category: str, entry_price: float,
                                exit_price: float, sl_price: float,
                                tp_price: float, is_buy: bool,
                                duration_minutes: float,
                                regime_context: Dict = None) -> str:
        """Determine root cause based on category and context."""
        distance_to_sl = abs(exit_price - sl_price) if sl_price > 0 else 0
        distance_to_tp = abs(exit_price - tp_price) if tp_price > 0 else 0

        if category == self.CATEGORY_STOP_LOSS:
            if distance_to_sl < 1.0:
                return "Price hit SL precisely - setup was invalid or market moved against thesis"
            else:
                return "Price moved significantly against position - thesis invalidated by market action"

        elif category == self.CATEGORY_TIME_STOP:
            if duration_minutes < 30:
                return "Position closed early due to time stop - setup failed to develop quickly"
            else:
                return "Position held too long without progress - market was range-bound or thesis was wrong"

        elif category == self.CATEGORY_EDGE_DECAY:
            return "Setup edge decayed over time - original thesis no longer valid"

        elif category == self.CATEGORY_REGIME_CONFLICT:
            regime_name = regime_context.get('regime_name', 'UNKNOWN') if regime_context else 'UNKNOWN'
            return f"Market regime shifted to {regime_name} which contradicted position direction"

        elif category == self.CATEGORY_CHOPPY_EXIT:
            return "Market became too choppy - volatility prevented setup from working"

        elif category == self.CATEGORY_REVERSAL:
            return "Reversal signals detected - market direction changed against position"

        elif category == self.CATEGORY_EMERGENCY:
            return "Emergency close triggered - extreme market conditions"

        elif category == self.CATEGORY_DYNAMIC_EXIT:
            return "Strategy-specific exit condition triggered"

        elif category == self.CATEGORY_MANUAL:
            return "Position manually closed by user"

        return "Unknown root cause"

    def _analyze_time_factors(self, setup_time_str: str) -> Dict:
        """Analyze time-related factors."""
        result = {
            'time_of_day': 0,
            'day_of_week': 0,
            'session': 'UNKNOWN',
            'is_prime_session': False
        }

        if not setup_time_str:
            return result

        try:
            setup_time = datetime.fromisoformat(setup_time_str)
            result['time_of_day'] = setup_time.hour
            result['day_of_week'] = setup_time.weekday()

            # Determine session
            hour = setup_time.hour
            if 7 <= hour < 9:
                result['session'] = 'LONDON_OPEN'
                result['is_prime_session'] = True
            elif 9 <= hour < 12:
                result['session'] = 'LONDON'
                result['is_prime_session'] = True
            elif 12 <= hour < 15:
                result['session'] = 'NY_OPEN'
                result['is_prime_session'] = True
            elif 15 <= hour < 18:
                result['session'] = 'NY_MIDDAY'
            elif 18 <= hour < 22:
                result['session'] = 'ASIAN'
            else:
                result['session'] = 'OTHER'

        except Exception:
            pass

        return result

    def _analyze_market_context(self, meta: Dict, regime_context: Dict = None) -> Dict:
        """Analyze market context at time of loss."""
        return {
            'regime_name': meta.get('regime_name', 'UNKNOWN'),
            'unified_regime': meta.get('regime', 'UNKNOWN'),
            'choppy_score': meta.get('choppy_score', 0),
            'active_killers': meta.get('active_killers', []),
            'entry_confidence': meta.get('confidence', 0),
            'expert_score': meta.get('expert_score', 0)
        }

    def _generate_recommendations(self, category: str, root_cause: str,
                                   strategy_category: str,
                                   duration_minutes: float) -> List[str]:
        """Generate actionable recommendations based on loss analysis."""
        recommendations = []

        if category == self.CATEGORY_STOP_LOSS:
            recommendations.append("Review entry timing - may be entering too early or late")
            recommendations.append("Check if SL placement is appropriate for current volatility")

        elif category == self.CATEGORY_TIME_STOP:
            if duration_minutes < 20:
                recommendations.append("Setup may not be suitable for current market conditions")
            else:
                recommendations.append("Consider reducing time stop threshold for this strategy")

        elif category == self.CATEGORY_EDGE_DECAY:
            recommendations.append("Setup has short edge lifespan - consider faster timeframes")

        elif category == self.CATEGORY_REGIME_CONFLICT:
            recommendations.append("Improve regime detection to avoid counter-regime entries")
            recommendations.append("Add regime filter to strategy entry conditions")

        elif category == self.CATEGORY_CHOPPY_EXIT:
            recommendations.append("Add stricter choppy filter before entry")
            recommendations.append("Avoid trading this strategy in choppy regimes")

        elif category == self.CATEGORY_REVERSAL:
            recommendations.append("Consider tightening trailing stop to protect profits")
            recommendations.append("Add reversal detection to entry conditions")

        return recommendations

    # =========================================================================
    # LOSS RECORDING
    # =========================================================================

    def record_loss_attribution(self, analysis: Dict):
        """
        Record loss attribution to database.
        
        Args:
            analysis: Analysis dict from analyze_root_cause
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO loss_attribution
                (ticket, strategy, strategy_category, regime_name, unified_regime,
                 loss_category, loss_amount, entry_price, exit_price, sl_price, tp_price,
                 duration_minutes, exit_reason, root_cause_analysis, time_of_day,
                 day_of_week, session, choppy_score, active_killers, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.get('ticket', 0),
                analysis.get('strategy', 'Unknown'),
                analysis.get('strategy_category', 'GENERAL'),
                analysis.get('market_context', {}).get('regime_name', 'UNKNOWN'),
                analysis.get('market_context', {}).get('unified_regime', 'UNKNOWN'),
                analysis.get('category', self.CATEGORY_UNKNOWN),
                analysis.get('loss_amount', 0),
                analysis.get('entry_price', 0),
                analysis.get('exit_price', 0),
                analysis.get('sl_price', 0),
                analysis.get('tp_price', 0),
                analysis.get('duration_minutes', 0),
                analysis.get('exit_reason', ''),
                json.dumps({
                    'root_cause': analysis.get('root_cause', ''),
                    'recommendations': analysis.get('recommendations', [])
                }),
                analysis.get('time_analysis', {}).get('time_of_day', 0),
                analysis.get('time_analysis', {}).get('day_of_week', 0),
                analysis.get('time_analysis', {}).get('session', 'UNKNOWN'),
                analysis.get('market_context', {}).get('choppy_score', 0),
                json.dumps(analysis.get('market_context', {}).get('active_killers', [])),
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

            self.logger.info(
                f"[LOSS_ATTR] Recorded loss | "
                f"Ticket: {analysis.get('ticket')} | "
                f"Strategy: {analysis.get('strategy')} | "
                f"Category: {analysis.get('category')} | "
                f"Loss: ${analysis.get('loss_amount', 0):.2f}"
            )

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Recording error: {e}")

    # =========================================================================
    # PATTERN DETECTION
    # =========================================================================

    def detect_loss_patterns(self, days: int = 30) -> Dict:
        """
        Detect recurring loss patterns.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with detected patterns
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT strategy, strategy_category, regime_name, loss_category,
                       loss_amount, time_of_day, session
                FROM loss_attribution
                WHERE recorded_at >= datetime('now', ?)
                ORDER BY recorded_at DESC
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {'patterns': [], 'total_losses': 0}

            patterns = []

            # Pattern 1: Strategy-Regime combinations with high loss
            strategy_regime_loss = defaultdict(lambda: {'count': 0, 'total_loss': 0})
            for row in rows:
                key = (row[0], row[2])  # (strategy, regime)
                strategy_regime_loss[key]['count'] += 1
                strategy_regime_loss[key]['total_loss'] += row[4]

            for (strategy, regime), stats in strategy_regime_loss.items():
                if stats['count'] >= 3:
                    avg_loss = stats['total_loss'] / stats['count']
                    patterns.append({
                        'type': 'STRATEGY_REGIME',
                        'strategy': strategy,
                        'regime': regime,
                        'occurrences': stats['count'],
                        'total_loss': round(stats['total_loss'], 2),
                        'avg_loss': round(avg_loss, 2),
                        'recommendation': f"Disable {strategy} in {regime} regime"
                    })

            # Pattern 2: Time-based patterns
            time_loss = defaultdict(lambda: {'count': 0, 'total_loss': 0})
            for row in rows:
                hour = row[5]
                time_loss[hour]['count'] += 1
                time_loss[hour]['total_loss'] += row[4]

            for hour, stats in time_loss.items():
                if stats['count'] >= 5 and stats['total_loss'] / stats['count'] > 5:
                    patterns.append({
                        'type': 'TIME_OF_DAY',
                        'hour': hour,
                        'occurrences': stats['count'],
                        'total_loss': round(stats['total_loss'], 2),
                        'avg_loss': round(stats['total_loss'] / stats['count'], 2),
                        'recommendation': f"Avoid trading during hour {hour}:00"
                    })

            # Pattern 3: Category patterns
            category_loss = defaultdict(lambda: {'count': 0, 'total_loss': 0})
            for row in rows:
                category = row[3]
                category_loss[category]['count'] += 1
                category_loss[category]['total_loss'] += row[4]

            for category, stats in category_loss.items():
                if stats['count'] >= 5:
                    patterns.append({
                        'type': 'LOSS_CATEGORY',
                        'category': category,
                        'occurrences': stats['count'],
                        'total_loss': round(stats['total_loss'], 2),
                        'avg_loss': round(stats['total_loss'] / stats['count'], 2),
                        'recommendation': f"Investigate high {category} losses"
                    })

            return {
                'patterns': patterns,
                'total_losses': len(rows),
                'analysis_period_days': days
            }

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Pattern detection error: {e}")
            return {'patterns': [], 'total_losses': 0}

    # =========================================================================
    # STRATEGY-REGIME ANALYTICS
    # =========================================================================

    def get_strategy_regime_stats(self, strategy_name: str = None,
                                    regime_name: str = None,
                                    days: int = 30) -> Dict:
        """
        Get loss statistics per strategy-regime combination.
        
        Args:
            strategy_name: Filter by strategy (optional)
            regime_name: Filter by regime (optional)
            days: Number of days to analyze
            
        Returns:
            Dict with stats
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
                SELECT strategy, regime_name, loss_category,
                       COUNT(*) as loss_count,
                       SUM(loss_amount) as total_loss,
                       AVG(loss_amount) as avg_loss
                FROM loss_attribution
                WHERE recorded_at >= datetime('now', ?)
            """
            params = [f"-{days} days"]

            if strategy_name:
                query += " AND strategy = ?"
                params.append(strategy_name)

            if regime_name:
                query += " AND regime_name = ?"
                params.append(regime_name)

            query += " GROUP BY strategy, regime_name, loss_category ORDER BY total_loss DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            stats = []
            for row in rows:
                stats.append({
                    'strategy': row[0],
                    'regime': row[1],
                    'category': row[2],
                    'loss_count': row[3],
                    'total_loss': round(row[4], 2),
                    'avg_loss': round(row[5], 2)
                })

            return {
                'stats': stats,
                'total_records': sum(s['loss_count'] for s in stats)
            }

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Strategy-regime stats error: {e}")
            return {'stats': [], 'total_records': 0}

    # =========================================================================
    # LOSS BREAKDOWN
    # =========================================================================

    def get_loss_breakdown(self, days: int = 30) -> Dict:
        """
        Get loss distribution by category.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with breakdown
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT loss_category, COUNT(*) as count,
                       SUM(loss_amount) as total_loss,
                       AVG(loss_amount) as avg_loss
                FROM loss_attribution
                WHERE recorded_at >= datetime('now', ?)
                GROUP BY loss_category
                ORDER BY total_loss DESC
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            conn.close()

            breakdown = []
            total_count = 0
            total_loss = 0

            for row in rows:
                breakdown.append({
                    'category': row[0],
                    'count': row[1],
                    'total_loss': round(row[2], 2),
                    'avg_loss': round(row[3], 2)
                })
                total_count += row[1]
                total_loss += row[2]

            # Calculate percentages
            for item in breakdown:
                item['count_pct'] = round(item['count'] / total_count * 100, 1) if total_count > 0 else 0
                item['loss_pct'] = round(item['total_loss'] / total_loss * 100, 1) if total_loss > 0 else 0

            return {
                'breakdown': breakdown,
                'total_count': total_count,
                'total_loss': round(total_loss, 2),
                'analysis_period_days': days
            }

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Loss breakdown error: {e}")
            return {'breakdown': [], 'total_count': 0, 'total_loss': 0}

    # =========================================================================
    # CONSECUTIVE LOSS ANALYSIS
    # =========================================================================

    def analyze_consecutive_losses(self, days: int = 7) -> Dict:
        """
        Analyze consecutive loss streaks.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with streak analysis
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT strategy, loss_amount, recorded_at
                FROM loss_attribution
                WHERE recorded_at >= datetime('now', ?)
                ORDER BY recorded_at DESC
            """, (f"-{days} days",))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {'max_streak': 0, 'current_streak': 0}

            # Count current streak
            current_streak = 0
            max_streak = 0
            streak = 0

            for row in rows:
                streak += 1
                max_streak = max(max_streak, streak)

            # Current streak is from most recent
            current_streak = len(rows)  # All are losses in this table

            return {
                'max_streak': max_streak,
                'current_streak': current_streak,
                'total_losses_in_period': len(rows),
                'analysis_period_days': days
            }

        except Exception as e:
            self.logger.error(f"[LOSS_ATTR] Consecutive loss analysis error: {e}")
            return {'max_streak': 0, 'current_streak': 0}

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def generate_loss_report(self, days: int = 30) -> str:
        """
        Generate comprehensive loss report.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Formatted report string
        """
        breakdown = self.get_loss_breakdown(days)
        patterns = self.detect_loss_patterns(days)
        consecutive = self.analyze_consecutive_losses(days)

        report_lines = [
            "=" * 60,
            f"LOSS ATTRIBUTION REPORT (Last {days} days)",
            "=" * 60,
            "",
            f"Total Losses: {breakdown['total_count']}",
            f"Total Loss Amount: ${breakdown['total_loss']:.2f}",
            "",
            "LOSS BREAKDOWN BY CATEGORY:",
            "-" * 40
        ]

        for item in breakdown['breakdown']:
            report_lines.append(
                f"  {item['category']}: {item['count']} losses "
                f"(${item['total_loss']:.2f}, {item['loss_pct']:.1f}%)"
            )

        report_lines.extend([
            "",
            "DETECTED PATTERNS:",
            "-" * 40
        ])

        if patterns['patterns']:
            for pattern in patterns['patterns'][:5]:  # Top 5
                if pattern['type'] == 'STRATEGY_REGIME':
                    report_lines.append(
                        f"  {pattern['strategy']} in {pattern['regime']}: "
                        f"{pattern['occurrences']} losses (${pattern['total_loss']:.2f})"
                    )
                    report_lines.append(f"    → {pattern['recommendation']}")
                elif pattern['type'] == 'TIME_OF_DAY':
                    report_lines.append(
                        f"  Hour {pattern['hour']}:00: "
                        f"{pattern['occurrences']} losses (${pattern['total_loss']:.2f})"
                    )
                    report_lines.append(f"    → {pattern['recommendation']}")
        else:
            report_lines.append("  No significant patterns detected")

        report_lines.extend([
            "",
            "CONSECUTIVE LOSSES:",
            "-" * 40,
            f"  Max Streak: {consecutive['max_streak']}",
            f"  Current Streak: {consecutive['current_streak']}",
            "",
            "=" * 60
        ])

        return "\n".join(report_lines)