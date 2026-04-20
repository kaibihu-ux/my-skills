#!/usr/bin/env python3
"""增量更新 2026-04-03 股票数据（无代理、无间隔、断点续算）"""
import os, sys, time, json, duckdb, pandas as pd

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403.log'

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

def load_ckpt():
    if os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE) as f:
                return json.load(f)
        except:
            pass
    return None

def save_ckpt(data):
    os.makedirs(os.path.dirname(CKPT_FILE), exist_ok=True)
    tmp = CKPT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.rename(tmp, CKPT_FILE)

TARGET_DATE = '2026-04-03'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

import baostock as bs

def fetch_one(code):
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=TARGET_DATE, end_date=TARGET_DATE,
            frequency='d', adjustflag='3'
        )
        if rs.error_code != '0':
            return (code, None)
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return (code, None)
        df = pd.DataFrame(data, columns=rs.fields)
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
        return (code, df[cols])
    except:
        return (code, None)

def write_one(code, df, con):
    if df is not None and not df.empty:
        try:
            con.execute("DELETE FROM stock_daily WHERE ts_code = ? AND trade_date = ?", [code, TARGET_DATE])
            con.register('dft', df)
            con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
            con.unregister('dft')
            return True
        except:
            return False
    return False

if __name__ == '__main__':
    log("=" * 50)
    log(f"开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    con = duckdb.connect(DB_PATH, read_only=False)

    # 获取待取列表
    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing = con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist()
    existing_set = set(existing)
    to_fetch = [c for c in all_codes if c not in existing_set]
    log(f"总数: {len(all_codes)}, 已有: {len(existing)}, 待取: {len(to_fetch)}")

    # 断点续算：从数据库确认哪些真正写入了
    ckpt = load_ckpt()
    confirmed_done = set()
    if ckpt and 'done' not in ckpt:
        # 只信任数据库中实际存在的记录
        confirmed = con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist()
        confirmed_done = set(confirmed)
        ckpt_success = ckpt.get('success', 0)
        ckpt_codes = len(ckpt.get('done_codes', []))
        log(f"断点参考: {ckpt_codes} 条, 实际确认: {len(confirmed_done)} 条")

    # 最终待取列表（排除已确认写入的）
    to_fetch = [c for c in all_codes if c not in confirmed_done]
    log(f"本轮实际待取: {len(to_fetch)} 条")

    bs.login()

    total_success = 0
    total_fail = 0
    last_save = time.time()

    for i, code in enumerate(to_fetch):
        code2, df = fetch_one(code)
        ok = write_one(code2, df, con)
        if ok:
            confirmed_done.add(code2)
            total_success += 1
        else:
            total_fail += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - last_save
            if elapsed >= 30:
                save_ckpt({'done_codes': list(confirmed_done), 'success': total_success, 'fail': total_fail})
                log(f"  {i+1}/{len(to_fetch)} | 成功 {total_success} | 失败 {total_fail}")
                last_save = time.time()

    bs.logout()

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条")
    save_ckpt({'done': True, 'final_count': final, 'success': total_success, 'fail': total_fail})
    con.close()
    log("完成!")
