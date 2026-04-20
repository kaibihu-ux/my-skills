#!/usr/bin/env python3
"""并行增量更新 2026-04-03（进程池隔离baostock连接）"""
import os, sys, time, json, duckdb, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403_parallel.log'

TARGET_DATE = '2026-04-03'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

def fetch_one_process(code):
    """独立进程：自己login/logout，安全隔离"""
    import baostock as bs
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=TARGET_DATE, end_date=TARGET_DATE,
            frequency='d', adjustflag='3'
        )
        bs.logout()
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
        try:
            bs.logout()
        except:
            pass
        return (code, None)

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

    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in existing]
    log(f"总数: {len(all_codes)}, 已有: {len(existing)}, 待取: {len(to_fetch)}")

    ckpt = load_ckpt()
    if ckpt and 'done' not in ckpt:
        confirmed = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
        to_fetch = [c for c in all_codes if c not in confirmed]
        log(f"断点续算: 本轮剩余 {len(to_fetch)} 条")

    N_WORKERS = 6
    BATCH = 50
    log(f"进程池: {N_WORKERS} workers, 每批 {BATCH} 只提交")

    total_success = 0
    total_fail = 0
    confirmed = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
    last_save = time.time()

    for batch_start in range(0, len(to_fetch), BATCH):
        batch = to_fetch[batch_start:batch_start + BATCH]
        
        # 进程池并行提交
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(fetch_one_process, code): code for code in batch}
            for future in as_completed(futures):
                code, df = future.result()
                ok = write_one(code, df, con)
                if ok:
                    confirmed.add(code)
                    total_success += 1
                else:
                    total_fail += 1

        elapsed = time.time() - last_save
        if elapsed >= 30 or batch_start + BATCH >= len(to_fetch):
            save_ckpt({'done_codes': list(confirmed), 'success': total_success, 'fail': total_fail})
            log(f"  {batch_start + len(batch)}/{len(to_fetch)} | 成功 {total_success} | 失败 {total_fail}")
            last_save = time.time()

    # 重试
    fail_codes = [c for c in to_fetch if c not in confirmed]
    if fail_codes:
        log(f"重试 {len(fail_codes)} 只...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(fetch_one_process, code): code for code in fail_codes}
            for future in as_completed(futures):
                code, df = future.result()
                ok = write_one(code, df, con)
                if ok:
                    total_success += 1
        log(f"重试完成")

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条记录")
    save_ckpt({'done': True, 'final_count': final, 'success': total_success, 'fail': total_fail})
    con.close()
    log("完成!")
