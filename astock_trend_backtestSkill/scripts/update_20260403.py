#!/usr/bin/env python3
"""并行增量更新 2026-04-03（multiprocessing.Pool + pool.map，进程隔离baostock）"""
import os, sys, time, json, duckdb, pandas as pd
from multiprocessing import Pool, cpu_count

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403.log'

TARGET_DATE = '2026-04-03'
BS_DATE = '2026-04-03'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

def fetch_one(code):
    """每进程独立函数，自己管理baostock连接"""
    import baostock as bs
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=BS_DATE, end_date=BS_DATE,
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

def save_ckpt(data):
    os.makedirs(os.path.dirname(CKPT_FILE), exist_ok=True)
    tmp = CKPT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.rename(tmp, CKPT_FILE)

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
    confirmed = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in confirmed]
    log(f"总数: {len(all_codes)}, 已有: {len(confirmed)}, 待取: {len(to_fetch)}")

    # 加载断点
    if os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE) as f:
                ckpt = json.load(f)
            if not ckpt.get('done'):
                confirmed2 = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
                to_fetch = [c for c in all_codes if c not in confirmed2]
                log(f"断点续算: 本轮剩余 {len(to_fetch)} 条")
        except:
            pass

    N_WORKERS = 3
    log(f"进程池: {N_WORKERS} workers (pool.map 稳定版)")

    total_success = 0
    total_fail = 0
    last_save = time.time()

    # pool.map 稳定，完整收集结果后再写DB
    with Pool(N_WORKERS, maxtasksperchild=100) as pool:
        results = pool.map(fetch_one, to_fetch, chunksize=1)

    # 整理结果并写DB
    done_list = []
    for code, df in results:
        ok = write_one(code, df, con)
        if ok:
            confirmed.add(code)
            done_list.append(code)
            total_success += 1
        else:
            total_fail += 1

        # 每100只保存一次
        if len(done_list) % 100 == 0:
            save_ckpt({'done_codes': list(confirmed), 'success': total_success, 'fail': total_fail})
            elapsed = time.time() - last_save
            log(f"  {len(done_list)}/{len(to_fetch)} | 成功 {total_success} | 失败 {total_fail}")
            last_save = time.time()

    save_ckpt({'done_codes': list(confirmed), 'success': total_success, 'fail': total_fail})
    log(f"抓取完成: 成功 {total_success} | 失败 {total_fail}")

    # 重试（单进程）
    fail_codes = [c for c in to_fetch if c not in confirmed]
    if fail_codes:
        log(f"重试 {len(fail_codes)} 只（单进程）...")
        for code in fail_codes:
            _, df = fetch_one(code)
            ok = write_one(code, df, con)
            if ok:
                total_success += 1
            else:
                total_fail += 1
            time.sleep(0.5)

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条记录")
    save_ckpt({'done': True, 'final_count': final, 'success': total_success, 'fail': total_fail})
    con.close()
    log("完成!")
