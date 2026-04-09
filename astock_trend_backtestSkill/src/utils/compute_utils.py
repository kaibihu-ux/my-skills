"""
多进程并行计算工具
用于因子计算、数据获取等 CPU 密集型任务的并行加速
"""
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Any
import warnings

# 抑制 Windows multiprocessing 警告
warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_factor_parallel(
    stock_codes: List[str],
    compute_func: Callable[[str], pd.DataFrame],
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    多进程并行计算因子（按股票并行）

    Args:
        stock_codes: 股票代码列表
        compute_func: 单只股票计算函数，输入 ts_code，返回 DataFrame
        n_workers: worker 数量，默认为 cpu_count - 1

    Returns:
        合并后的 DataFrame
    """
    if not stock_codes:
        return pd.DataFrame()

    n_workers = n_workers or max(1, cpu_count() - 1)
    n_workers = min(n_workers, len(stock_codes))

    with Pool(n_workers) as pool:
        results = pool.map(compute_func, stock_codes)

    valid_results = [r for r in results if r is not None and not r.empty]
    if not valid_results:
        return pd.DataFrame()
    return pd.concat(valid_results, ignore_index=True)


def batch_compute_indicator(
    stock_df: pd.DataFrame,
    indicator_func: Callable[[pd.DataFrame], pd.Series],
    group_col: str = 'ts_code',
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    按股票分组并行计算指标

    Args:
        stock_df: 包含 group_col 列的 DataFrame（已排序）
        indicator_func: 单只股票 DataFrame -> pd.Series，计算函数
        group_col: 分组列名，默认 'ts_code'
        n_workers: worker 数量

    Returns:
        结果 DataFrame，包含 group_col 和计算结果
    """
    n_workers = n_workers or max(1, cpu_count() - 1)

    stock_codes = stock_df[group_col].unique().tolist()

    def compute_one(code: str) -> Optional[pd.DataFrame]:
        try:
            df = stock_df[stock_df[group_col] == code].sort_values('trade_date')
            if df.empty:
                return None
            result = indicator_func(df)
            if not isinstance(result, pd.DataFrame):
                result = result.to_frame(name=result.name or 'value')
            result = result.copy()
            result[group_col] = code
            # 附加原始索引信息
            if 'trade_date' not in result.columns and 'trade_date' in df.columns:
                result['trade_date'] = df['trade_date'].values[:len(result)]
            return result
        except Exception:
            return None

    with Pool(n_workers) as pool:
        results = pool.map(compute_one, stock_codes)

    valid_results = [r for r in results if r is not None and not r.empty]
    if not valid_results:
        return pd.DataFrame()
    return pd.concat(valid_results, ignore_index=True)


def parallel_apply(
    df: pd.DataFrame,
    func: Callable[[Any], Any],
    n_workers: Optional[int] = None,
    chunk_size: int = 1000,
) -> List[Any]:
    """
    将 DataFrame 行或块并行分发到多个 worker

    Args:
        df: 输入 DataFrame
        func: 处理函数
        n_workers: worker 数量
        chunk_size: 每块行数

    Returns:
        结果列表（与 df 行对应）
    """
    n_workers = n_workers or max(1, cpu_count() - 1)

    # 分块
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def process_chunk(chunk: pd.DataFrame) -> List[Any]:
        return [func(row) for _, row in chunk.iterrows()]

    with Pool(n_workers) as pool:
        chunk_results = pool.map(process_chunk, chunks)

    # 合并结果
    flat = []
    for chunk_res in chunk_results:
        flat.extend(chunk_res)
    return flat


def starmap_parallel(
    func: Callable,
    args_list: List[tuple],
    n_workers: Optional[int] = None,
) -> List[Any]:
    """
    多进程 starmap（批量提交带多个参数的任務）

    Args:
        func: 多参数函数
        args_list: 参数元组列表，[(arg1, arg2, ...), ...]
        n_workers: worker 数量

    Returns:
        结果列表
    """
    n_workers = n_workers or max(1, cpu_count() - 1)
    with Pool(n_workers) as pool:
        results = pool.starmap(func, args_list)
    return results


def get_optimal_workers(task_type: str = 'cpu_bound') -> int:
    """
    根据任务类型返回推荐 worker 数量

    Args:
        task_type: 'cpu_bound' 或 'io_bound'
            - cpu_bound: 推荐 cpu_count - 1（保留一个核心）
            - io_bound: 可用 cpu_count * 2 或 cpu_count
    """
    total = cpu_count()
    if task_type == 'cpu_bound':
        return max(1, total - 1)
    elif task_type == 'io_bound':
        return max(1, total * 2)
    return total
