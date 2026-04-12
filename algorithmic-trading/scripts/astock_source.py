"""
PyBroker A股数据源 - 从本地DuckDB读取数据
"""

import pandas as pd
from datetime import datetime
from typing import Union, Iterable, Optional, Any
from pybroker.data import DataSource
from pybroker.common import to_datetime, verify_date_range


class AStockDataSource(DataSource):
    """
    A股本地数据源 - 从DuckDB读取数据

    使用方法:
        from scripts.astock_source import AStockDataSource

        data_source = AStockDataSource(
            db_path='~/.openclaw/my-skills/algorithmic-trading/data/astock_full.duckdb'
            # 或使用绝对路径: '/home/hyh/.openclaw/my-skills/algorithmic-trading/data/astock_full.duckdb'
        )
        strategy = Strategy(data_source, start_date, end_date, config)
    """

    def __init__(self, db_path: str):
        import duckdb
        import os
        super().__init__()
        self.db_path = os.path.expanduser(db_path)  # 支持 ~ 路径扩展
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def _fetch_data(self, symbols: frozenset, start_date: datetime,
                    end_date: datetime, timeframe: str,
                    adjust: Optional[Any]) -> pd.DataFrame:
        """
        从DuckDB获取数据，实现DataSource抽象方法
        """
        # A股代码处理
        ts_codes = []
        for s in symbols:
            if isinstance(s, str) and len(s) == 6 and s.isdigit():
                ts_codes.append(s)
            elif isinstance(s, str) and '.' in s:
                # 处理如 000001.XSHG 格式
                code = s.split('.')[0]
                ts_codes.append(code)
            else:
                ts_codes.append(s)

        if not ts_codes:
            return pd.DataFrame()

        placeholders = ','.join([f"'{c}'" for c in ts_codes])
        query = f"""
            SELECT
                trade_date as date,
                ts_code as symbol,
                open,
                high,
                low,
                close,
                vol as volume
            FROM stock_daily
            WHERE ts_code IN ({placeholders})
              AND trade_date >= '{start_date.date()}'
              AND trade_date <= '{end_date.date()}'
            ORDER BY ts_code, trade_date
        """

        try:
            df = self.conn.execute(query).df()
            if df.empty:
                return df

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])

            # 确保列顺序正确
            df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]

            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_bars(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        直接获取bar数据（不通过PyBroker）

        Args:
            symbols: 股票代码列表，如 ['000001', '600000']
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume
        """
        df = self._fetch_data(
            frozenset(symbols),
            to_datetime(start_date),
            to_datetime(end_date),
            '', None
        )
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self.conn.execute("SELECT * FROM stock_list").df()

    def get_trade_dates(self, start_date: str, end_date: str) -> list:
        """获取交易日列表"""
        query = f"""
            SELECT DISTINCT trade_date
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        result = self.conn.execute(query).fetchall()
        return [r[0] for r in result]

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
