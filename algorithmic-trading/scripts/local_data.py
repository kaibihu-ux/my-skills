"""
本地DuckDB数据源 - 替代YFinance从本地数据库读取股票数据
"""

import duckdb
import pandas as pd
from datetime import date, datetime
from typing import Optional, Union

class LocalDataSource:
    """
    从本地DuckDB数据库读取A股数据
    数据路径: ~/.openclaw/skills/algorithmic-trading/data/astock_full.duckdb
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def fetch_bars(self, symbols: list, start_date: Union[str, date],
                   end_date: Union[str, date]) -> pd.DataFrame:
        """
        获取指定股票和日期范围的日线数据

        Args:
            symbols: 股票代码列表，如 ['000001', '600000']
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        # A股代码格式处理（如果传入的是6位纯数字，加0前缀）
        processed_symbols = []
        for s in symbols:
            if len(s) == 6 and s.isdigit():
                processed_symbols.append(s)
            else:
                processed_symbols.append(s)

        placeholders = ','.join(['?' for _ in processed_symbols])
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
              AND trade_date >= '{start_date}'
              AND trade_date <= '{end_date}'
            ORDER BY ts_code, trade_date
        """

        df = self.conn.execute(query, processed_symbols).df()
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """获取所有股票列表"""
        return self.conn.execute("SELECT * FROM stock_list").df()

    def get_trade_dates(self, start_date: str, end_date: str) -> list:
        """获取指定日期范围内的所有交易日"""
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


# PyBroker兼容的数据源适配器
class PyBrokerLocalData:
    """
    适配PyBroker的数据源接口
    用于替代YFinance
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local_ds = LocalDataSource(db_path)

    def bars(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        返回PyBroker格式的bar数据
        列名格式符合PyBroker的DateDataSource要求
        """
        df = self.local_ds.fetch_bars(symbols, start_date, end_date)
        # 重命名列为PyBroker期望的格式
        df = df.rename(columns={
            'date': 'date',
            'symbol': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        return df


if __name__ == '__main__':
    # 测试代码
    import os
    db_path = os.path.expanduser('~/.openclaw/skills/algorithmic-trading/data/astock_full.duckdb')

    if os.path.exists(db_path):
        ds = LocalDataSource(db_path)

        # 测试获取数据
        df = ds.fetch_bars(['000001', '600000'], '2024-01-01', '2024-03-31')
        print(f"获取 {len(df)} 条记录")
        print(df.head())

        # 测试股票列表
        stocks = ds.get_stock_list()
        print(f"\n股票总数: {len(stocks)}")

        ds.close()
    else:
        print(f"数据库文件不存在: {db_path}")
