#!/usr/bin/env python3
"""
下载全部A股主板日线数据（2020-01-01 ~ 2026-03-28）
支持断点续传：DuckDB中已有数据的股票不会重复下载
"""
import baostock as bs
import pandas as pd
import numpy as np
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.duckdb_store import DuckDBStore
from src.core.data_manager import AShareMainBoardFilter
from src.utils.logger import Logger

START_DATE = "2020-01-01"
END_DATE = "2026-03-28"
DB_PATH = Path(__file__).parent.parent / "data" / "astock_full.duckdb"
FETCH_INTERVAL = 5       # 每只股票间隔5秒
MAX_RETRIES = 3          # 最大重试次数
RETRY_BASE_DELAY = 3     # 重试间隔基数（秒），递增

logger = Logger("fetch_all", log_dir=str(Path(__file__).parent.parent / "logs"))
_main_filter = AShareMainBoardFilter()


def is_main_board(bs_code: str) -> bool:
    if not bs_code.startswith(("sh.", "sz.")):
        return False
    code = bs_code.split(".", 1)[1]
    return _main_filter.is_main_board(code)


def get_all_stocks() -> list:
    bs.login()
    rs = bs.query_all_stock(day="2026-03-27")
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(data_list, columns=rs.fields)
    # baostock 返回字段: ['code', 'tradeStatus', 'code_name']
    df = df[df['tradeStatus'] == '1']
    df = df[df['code'].apply(is_main_board)]
    return df.to_dict('records')


def get_downloaded_codes() -> set:
    try:
        store = DuckDBStore(str(DB_PATH))
        df = store.df("SELECT DISTINCT ts_code FROM stock_daily")
        store.close()
        return set(df['ts_code'].tolist())
    except:
        return set()


def fetch_stock_with_retry(bs_code: str) -> pd.DataFrame:
    """带重试的股票数据获取"""
    for attempt in range(MAX_RETRIES):
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,code,open,high,low,close,volume,amount,pctChg',
                start_date=START_DATE,
                end_date=END_DATE,
                frequency='d',
                adjustflag='3'
            )
            
            if rs.error_code != '0':
                raise Exception(f"baostock error: {rs.error_msg}")
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()
            
            if not data_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df.rename(columns={
                'date': 'trade_date',
                'volume': 'vol',
                'pctChg': 'pct_chg'
            })
            df['ts_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['limit_up'] = df['pct_chg'] >= 9.9
            df['limit_down'] = df['pct_chg'] <= -9.9
            df['is_st'] = False
            df['suspended'] = df['vol'] == 0
            cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'vol', 'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
            cols = [c for c in cols if c in df.columns]
            return df[cols]
            
        except Exception as e:
            bs.logout()
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                logger.warn(f"{bs_code} 第{attempt+1}次失败: {e}，{delay}秒后重试...")
                time.sleep(delay)
            else:
                logger.warn(f"{bs_code} 全部重试失败: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()


def main():
    os.makedirs(DB_PATH.parent, exist_ok=True)
    os.makedirs(Path(__file__).parent.parent / "logs", exist_ok=True)
    
    store = DuckDBStore(str(DB_PATH))
    store.init_tables()
    
    all_stocks = get_all_stocks()
    already = get_downloaded_codes()
    
    to_download = []
    for item in all_stocks:
        bs_code = item['code']
        ts_code = bs_code.replace('sh.', '').replace('sz.', '')
        if ts_code not in already:
            to_download.append({'bs_code': bs_code, 'ts_code': ts_code, 'name': item.get('name', '')})
    
    total = len(to_download)
    done = 0
    success = 0
    buffer = []
    BATCH = 50
    
    logger.info(f"总待下载: {total} 只股票")
    logger.info(f"已跳过: {len(already)} 只")
    logger.info(f"间隔: {FETCH_INTERVAL}秒, 最大重试: {MAX_RETRIES}次")
    
    for item in to_download:
        bs_code = item['bs_code']
        
        # 获取数据（含重试）
        df = fetch_stock_with_retry(bs_code)
        if not df.empty:
            buffer.append(df)
            success += 1
        
        done += 1
        
        # 每20只报告进度
        if done % 20 == 0 or done == total:
            logger.info(f"进度: {done}/{total}, 成功: {success}, 缓冲: {len(buffer)}")
        
        # 批量写入
        if len(buffer) >= BATCH:
            combined = pd.concat(buffer, ignore_index=True)
            try:
                store.conn.execute("INSERT INTO stock_daily BY NAME SELECT * FROM combined")
            except Exception as e:
                logger.error(f"写入失败: {e}")
            buffer = []
        
        # 每只间隔5秒（避免触发限流）
        if done < total:  # 最后一只不需要等
            time.sleep(FETCH_INTERVAL)
    
    # 剩余写入
    if buffer:
        combined = pd.concat(buffer, ignore_index=True)
        try:
            store.conn.execute("INSERT INTO stock_daily BY NAME SELECT * FROM combined")
        except Exception as e:
            logger.error(f"写入失败: {e}")
    
    store.close()
    logger.info(f"=== 完成: {success}/{total} 只 ===")


if __name__ == "__main__":
    main()
