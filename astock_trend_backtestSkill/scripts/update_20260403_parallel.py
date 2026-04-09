#!/usr/bin/env python3
"""并行增量更新 2026-04-03 股票数据（joblib多线程+每线程独立baostock）"""
import os, sys, time, json, duckdb, pandas as pd
from joblib import Parallel, delayed

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403_parallel.log'

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

def fetch_batch(codes):
    """每线程独立抓取一批股票（每线程维护自己的baostock连接）"""
    import baostock as bs
    results = []
    bs.login()
    try:
        for code in codes:
            bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    'date,code,open,high,low,close,volume,amount,pctChg',
                    start_date=TARGET_DATE, end_date=TARGET_DATE,
                    frequency='d', adjustflag='3'
                )
                if rs.error_code != '0':
                    results.append((code, None))
                    continue
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                if not data:
                    results.append((code, None))
                    continue
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
                results.append((code, df[cols]))
            except:
                results.append((code, None))
    finally:
        bs.logout()
    return results

def write_one(code, df, con):
    if df is not None and not df.empty:
        try:
            con.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{TARGET_DATE}'")
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

    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing = con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist()
    confirmed_done = set(existing)
    to_fetch = [c for c in all_codes if c not in confirmed_done]
    log(f"总数: {len(all_codes)}, 已有确认: {len(confirmed_done)}, 待取: {len(to_fetch)}")

    ckpt = load_ckpt()
    if ckpt and 'done' not in ckpt:
        db_confirmed = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
        to_fetch = [c for c in all_codes if c not in db_confirmed]
        log(f"断点续算: 本轮剩余 {len(to_fetch)} 条")

    N_JOBS = 6
    CHUNK = 30  # 每线程每次抓30只
    log(f"并行: {N_JOBS} 线程, 每块 {CHUNK} 只")

    all_results = []
    total_success = 0
    total_fail = 0
    last_save = time.time()

    # 将股票列表分成 N_JOBS * CHUNK 大小的块，parallel 会分配给各线程
    chunks = [to_fetch[i:i+CHUNK] for i in range(0, len(to_fetch), CHUNK)]
    log(f"分成 {len(chunks)} 个块并行抓取")

    # 并行抓取（每块分配一个线程）
    batch_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(fetch_batch)(chunk) for chunk in chunks
    )

    # 展平结果
    for batch in batch_results:
        for code, df in batch:
            ok = write_one(code, df, con)
            if ok:
                confirmed_done.add(code)
                total_success += 1
            else:
                total_fail += 1

    elapsed = time.time() - last_save
    save_ckpt({'done_codes': list(confirmed_done), 'success': total_success, 'fail': total_fail})
    log(f"抓取完成: 成功 {total_success} | 失败 {total_fail}")

    # 重试失败的
    fail_codes = [c for c in to_fetch if c not in confirmed_done]
    if fail_codes:
        log(f"重试 {len(fail_codes)} 只...")
        retry = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(fetch_batch)([code]) for code in fail_codes
        )
        retry_ok = 0
        for batch in retry:
            for code, df in batch:
                ok = write_one(code, df, con)
                if ok:
                    retry_ok += 1
        log(f"重试成功: {retry_ok}")

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条记录")
    save_ckpt({'done': True, 'final_count': final, 'success': total_success, 'fail': total_fail})
    con.close()
    log("完成!")
