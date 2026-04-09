import duckdb
import pandas as pd
import threading
from pathlib import Path


class DuckDBStore:
    """DuckDB数据存储类（线程安全）"""
    
    def __init__(self, db_path: str = "data/astock_full.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 自动清理残留WAL锁文件（防止上次进程被kill后锁不释放）
        self._cleanup_wal_lock()
        self.conn = duckdb.connect(str(self.db_path))
        # 限制DuckDB内存使用（防止OOM）
        self.conn.execute("PRAGMA threads=2")
        self.conn.execute("PRAGMA max_memory='4GB'")
        self._lock = threading.RLock()  # 保护所有DB操作

    def _cleanup_wal_lock(self):
        """启动时检查并清理残留的WAL/lock文件"""
        for suffix in ['.wal', '.lock', '.temp']:
            stale = self.db_path.with_suffix(self.db_path.suffix + suffix)
            if stale.exists():
                try:
                    stale.unlink()
                    print(f"[DuckDB] 🧹 已清理残留文件: {stale.name}")
                except Exception:
                    pass
    
    def execute(self, sql: str, params=None):
        """执行SQL语句（线程安全）"""
        with self._lock:
            if params:
                return self.conn.execute(sql, params)
            return self.conn.execute(sql)

    def df(self, sql: str, params=None) -> pd.DataFrame:
        """执行SQL并返回DataFrame（线程安全）"""
        with self._lock:
            if params:
                return self.conn.execute(sql, params).df()
            return self.conn.sql(sql).df()

    def insert(self, table: str, df: pd.DataFrame, chunk_size: int = 10000):
        """插入数据"""
        with self._lock:
            self.conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df")

    def checkpoint(self):
        """手动触发WAL checkpoint（释放锁）"""
        with self._lock:
            self.conn.execute("CHECKPOINT")

    def close(self):
        """安全关闭连接（自动checkpoint）"""
        try:
            self.checkpoint()
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
    
    def close(self):
        """关闭连接"""
        self.conn.close()
    
    def init_tables(self):
        """初始化所有表"""
        sqls = [
            """CREATE TABLE IF NOT EXISTS stock_list (
                ts_code VARCHAR PRIMARY KEY, symbol VARCHAR, name VARCHAR,
                list_date DATE, delist_date DATE, industry VARCHAR, market_cap BIGINT)""",
            """CREATE TABLE IF NOT EXISTS stock_daily (
                ts_code VARCHAR, trade_date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
                close DOUBLE, vol DOUBLE, amount DOUBLE, pct_chg DOUBLE,
                limit_up BOOLEAN, limit_down BOOLEAN, is_st BOOLEAN, suspended BOOLEAN,
                PRIMARY KEY (ts_code, trade_date))""",
            """CREATE TABLE IF NOT EXISTS fundamentals (
                ts_code VARCHAR, ann_date DATE, end_date DATE,
                pe DOUBLE, pb DOUBLE, roe DOUBLE, revenue_growth DOUBLE, profit_growth DOUBLE,
                PRIMARY KEY (ts_code, end_date, ann_date))""",
            """CREATE TABLE IF NOT EXISTS hkt_data (
                ts_code VARCHAR, trade_date DATE, hold_ratio DOUBLE, net_flow_20d DOUBLE,
                PRIMARY KEY (ts_code, trade_date))""",
            """CREATE TABLE IF NOT EXISTS factors (
                factor_name VARCHAR, ts_code VARCHAR, trade_date DATE,
                value DOUBLE, zscore DOUBLE,
                PRIMARY KEY (factor_name, ts_code, trade_date))""",
            """CREATE TABLE IF NOT EXISTS factor_pool (
                factor_name VARCHAR PRIMARY KEY, avg_ic DOUBLE, avg_ir DOUBLE,
                ic_series JSON, rank INT, status VARCHAR, updated_at TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS strategy_pool (
                strategy_id VARCHAR PRIMARY KEY, strategy_name VARCHAR,
                factors JSON, parameters JSON, metrics JSON,
                rank INT, status VARCHAR, created_at TIMESTAMP, updated_at TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR, strategy_id VARCHAR, ts_code VARCHAR,
                trade_date DATE, direction VARCHAR, price DOUBLE, quantity INT,
                amount DOUBLE, signal_reason VARCHAR,
                PRIMARY KEY (trade_id))""",
            """CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY,
                alert_type VARCHAR,
                message VARCHAR,
                created_at TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS backtest_nav (
                date VARCHAR PRIMARY KEY,
                nav DOUBLE,
                updated_at TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS backtest_positions (
                date VARCHAR,
                code VARCHAR,
                position_ratio DOUBLE,
                updated_at TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS factor_ic (
                factor_name VARCHAR,
                date VARCHAR,
                ic DOUBLE,
                rank_ic DOUBLE,
                PRIMARY KEY (factor_name, date))""",
            """CREATE SEQUENCE IF NOT EXISTS alerts_seq START 1""",
        ]
        for sql in sqls:
            self.execute(sql)
