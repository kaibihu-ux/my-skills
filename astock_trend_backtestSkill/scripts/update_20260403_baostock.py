#!/usr/bin/env python3
"""增量更新 2026-04-03 股票数据（baostock版，无代理）"""
import os, sys

# 取消代理
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.skill_api import get_instance
import time
import baostock as bs
import pandas as pd

PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

def fetch_one(code, start_fmt, end_fmt):
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=start_fmt, end_date=end_fmt,
            frequency='d', adjustflag='3'
        )
        if rs.error_code != '0':
            return None
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['ts_code'] = df['code'].str.replace('sh.','').str.replace('sz.','')
        df['trade_date'] = df['date']
        df.rename(columns={'volume':'vol','amount':'amount','pctChg':'pct_chg'}, inplace=True)
        for col in ['open','high','low','close','vol','amount','pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        cols = ['ts_code','trade_date','open','high','low','close','vol','amount',
                'pct_chg','limit_up','limit_down','is_st','suspended']
        return df[cols]
    except Exception:
        return None

def main():
    api = get_instance()
    store = api.store
    target_date = '2026-04-03'

    stocks = store.df("SELECT ts_code FROM stock_list ORDER BY ts_code")
    all_codes = stocks['ts_code'].tolist()

    existing = store.df(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{target_date}'")
    existing_codes = set(existing['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in existing_codes]

    print(f"总数: {len(all_codes)}, 已有: {len(existing_codes)}, 待取: {len(to_fetch)}")

    bs.login()
    success = 0
    fail = 0
    fail_codes = []

    for i, code in enumerate(to_fetch):
        df = fetch_one(code, target_date, target_date)
        if df is not None and not df.empty:
            try:
                store.conn.execute("DELETE FROM stock_daily WHERE ts_code = ? AND trade_date = ?", [code, target_date])
                store.conn.execute(f"INSERT INTO stock_daily BY NAME SELECT * FROM df")
                success += 1
            except Exception:
                fail += 1
                fail_codes.append(code)
        else:
            fail += 1
            fail_codes.append(code)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(to_fetch)} | 成功 {success} | 失败 {fail}")

        time.sleep(0.5)

    bs.logout()
    print(f"首轮: 成功 {success} | 失败 {fail}")

    if fail_codes:
        print(f"重试 {len(fail_codes)} 只（间隔2秒）...")
        time.sleep(5)
        retry_ok = 0
        for code in fail_codes[:]:
            df = fetch_one(code, target_date, target_date)
            if df is not None and not df.empty:
                try:
                    store.conn.execute("DELETE FROM stock_daily WHERE ts_code = ? AND trade_date = ?", [code, target_date])
                    store.conn.execute(f"INSERT INTO stock_daily BY NAME SELECT * FROM df")
                    retry_ok += 1
                    fail_codes.remove(code)
                except:
                    pass
            time.sleep(2.0)
        print(f"重试成功: {retry_ok}")

    final = store.df(f"SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date = '{target_date}'")
    print(f"最终: {final['cnt'].iloc[0]} 条")

if __name__ == '__main__':
    main()
