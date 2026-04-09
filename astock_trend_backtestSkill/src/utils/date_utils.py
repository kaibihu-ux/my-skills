import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


def get_trade_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
    
    Returns:
        交易日列表
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # 生成所有工作日（排除周末）
    dates = pd.date_range(start, end, freq='B')
    
    return [d.strftime('%Y%m%d') for d in dates]


def is_trade_date(date: str) -> bool:
    """
    判断是否为交易日
    
    Args:
        date: 日期，格式YYYYMMDD
    
    Returns:
        是否为交易日
    """
    dt = pd.to_datetime(date)
    # 周末不是交易日
    if dt.dayofweek >= 5:
        return False
    return True


def next_trade_date(date: str, n: int = 1) -> str:
    """
    获取下一个交易日
    
    Args:
        date: 当前日期，格式YYYYMMDD
        n: 跳过的交易日数量
    
    Returns:
        下一个交易日
    """
    dt = pd.to_datetime(date)
    result = dt
    
    while n > 0:
        result += timedelta(days=1)
        if result.dayofweek < 5:  # 不是周末
            n -= 1
    
    return result.strftime('%Y%m%d')


def prev_trade_date(date: str, n: int = 1) -> str:
    """
    获取上一个交易日
    
    Args:
        date: 当前日期，格式YYYYMMDD
        n: 跳过的交易日数量
    
    Returns:
        上一个交易日
    """
    dt = pd.to_datetime(date)
    result = dt
    
    while n > 0:
        result -= timedelta(days=1)
        if result.dayofweek < 5:  # 不是周末
            n -= 1
    
    return result.strftime('%Y%m%d')


def get_date_range(n_days: int, end_date: Optional[str] = None) -> tuple:
    """
    获取日期范围
    
    Args:
        n_days: 天数
        end_date: 结束日期，默认今天
    
    Returns:
        (start_date, end_date)
    """
    if end_date is None:
        end = datetime.now()
    else:
        end = pd.to_datetime(end_date)
    
    start = end - timedelta(days=n_days * 2)  # 预留更多天数用于过滤
    
    # 找到第一个交易日
    while start.dayofweek >= 5:
        start += timedelta(days=1)
    
    return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')


def format_date(date_str: str, fmt: str = '%Y-%m-%d') -> str:
    """
    格式化日期字符串
    
    Args:
        date_str: 日期字符串，格式YYYYMMDD
        fmt: 目标格式
    
    Returns:
        格式化后的日期字符串
    """
    dt = pd.to_datetime(date_str)
    return dt.strftime(fmt)


def parse_date(date_str: str) -> datetime:
    """
    解析日期字符串
    
    Args:
        date_str: 日期字符串
    
    Returns:
        datetime对象
    """
    return pd.to_datetime(date_str)
