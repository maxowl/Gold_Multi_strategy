"""
Trade Recorder - Thread-Safe Database Writer.

Separates database I/O from the main execution thread to prevent
latency spikes. Uses a queue and background worker thread.

Features:
  - Thread-safe trade recording
  - Asynchronous database writes
  - Batch processing
  - Error handling

Used by:
  - OrderManager (trade recording)
"""
import sqlite3
import json
import logging
import threading
from typing import Dict, Any
from queue import Queue, Empty


class TradeRecorder:
    """
    Trade Recorder with thread-safe database writes.
    
    Features:
      - Thread-safe queue
      - Background worker thread
      - Asynchronous database writes
      - Graceful shutdown
    """

    def __init__(self, db_path: str = "bot_state.db"):
        """
        Initialize TradeRecorder.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)

        self._queue = Queue()
        self._lock = threading.Lock()
        self._running = True

        # Start background worker thread
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

        # Statistics
        self._records_written = 0
        self._errors = 0

    def enqueue_trade(self, trade_data: Dict[str, Any]):
        """
        Add a trade record to the queue for asynchronous writing.
        
        Args:
            trade_data: Trade data dict
        """
        self._queue.put(trade_data)

    def _process_queue(self):
        """Background worker that processes the queue and writes to SQLite."""
        while self._running:
            try:
                # Block for up to 1 second waiting for new items
                trade_data = self._queue.get(timeout=1.0)
                self._write_to_db(trade_data)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"[RECORDER] Worker error: {e}")
                self._errors += 1

    def _write_to_db(self, trade_data: Dict[str, Any]):
        """Execute the actual SQLite insert."""
        try:
            # Serialize meta_data if it's a dict
            meta_str = json.dumps(trade_data.get('meta_data', {})) if isinstance(
                trade_data.get('meta_data'), dict
            ) else str(trade_data.get('meta_data', '{}'))

            with self._lock:
                with sqlite3.connect(self.db_path, timeout=10) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO trade_history
                        (ticket, symbol, strategy, direction, entry_price, exit_price,
                         sl_price, tp_price, volume, profit, commission, swap,
                         open_time, close_time, entry_reason, exit_reason,
                         is_pending, order_type, expected_entry, meta_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_data.get('ticket'),
                        trade_data.get('symbol'),
                        trade_data.get('strategy'),
                        trade_data.get('direction'),
                        trade_data.get('entry_price'),
                        trade_data.get('exit_price'),
                        trade_data.get('sl_price'),
                        trade_data.get('tp_price'),
                        trade_data.get('volume'),
                        trade_data.get('profit'),
                        trade_data.get('commission'),
                        trade_data.get('swap'),
                        str(trade_data.get('open_time', '')),
                        str(trade_data.get('close_time', '')),
                        trade_data.get('entry_reason'),
                        trade_data.get('exit_reason'),
                        int(trade_data.get('is_pending', 0)),
                        trade_data.get('order_type'),
                        trade_data.get('expected_entry'),
                        meta_str
                    ))
                    conn.commit()

            self._records_written += 1

        except Exception as e:
            self.logger.error(f"[RECORDER] Write error for ticket {trade_data.get('ticket')}: {e}")
            self._errors += 1

    def shutdown(self):
        """Gracefully shutdown the worker thread."""
        self._running = False
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def get_statistics(self) -> Dict:
        """Get trade recorder statistics."""
        return {
            'records_written': self._records_written,
            'errors': self._errors,
            'queue_size': self._queue.qsize()
        }