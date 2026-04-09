#!/usr/bin/env python3
"""增量更新 2026-04-03 股票数据（无代理、串行、0.5s间隔、断点续算）"""
import os, sys, time, json, duckdb, pandas as pd

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403_safe.log'

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

log("=" * 50)
log(f"开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 打开写连接
con = duckdb.connect(DB_PATH, read_only=False)
# 允许并发写，减少死锁
con.execute("PRAGMA threads=1")
con.execute("CHECKPOINT")

target_date = '2026-04-03'

# 获取待取列表（用同一连接，SQLite兼容模式）
all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
existing = con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{target_date}'").fetchdf()['ts_code'].tolist()
existing_set = set(existing)
to_fetch = [c for c in all_codes if c not in existing_set]
log(f"总数: {len(all_codes)}, 已有: {len(existing)}, 待取: {len(to_fetch)}")

ckpt = load_ckpt()
done = set()
if ckpt and 'done' not in ckpt:
    done = set(ckpt.get('done_codes', []))
    to_fetch = [c for c in to_fetch if c not in done]
    log(f"断点续算: 已完成 {len(done)} 条, 剩余 {len(to_fetch)} 条")

import baostock as bs
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

def fetch_one(code):
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=target_date, end_date=target_date,
            frequency='d', adjustflag='3'
        )
        if rs.error_code != '0':
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
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
        return df[cols]
    except Exception as e:
        sys.stderr.write(f"[ERR fetch {code}] {e}\n")
        return None

bs.login()
success = 0
fail = 0
fail_codes = []
last_ckpt_time = time.time()

for i, code in enumerate(to_fetch):
    df = fetch_one(code)
    if df is not None and not df.empty:
        try:
            con.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{target_date}'")
            con.register('df_to_insert', df)
            con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM df_to_insert")
            con.unregister('df_to_insert')
            success += 1
            done.add(code)
        except Exception as e:
            fail += 1
            fail_codes.append(code)
            sys.stderr.write(f"[ERR insert {code}] {e}\n")
    else:
        fail += 1
        fail_codes.append(code)

    # 每25只或每60秒保存一次断点
    elapsed = time.time() - last_ckpt_time
    if (i+1) % 25 == 0 or elapsed >= 60:
        save_ckpt({'done_codes': list(done), 'success': success, 'fail': fail})
        log(f"  {i+1}/{len(to_fetch)} | 成功 {success} | 失败 {fail}")
        last_ckpt_time = time.time()

    time.sleep(0)

bs.logout()

# 重试失败
if fail_codes:
    log(f"重试 {len(fail_codes)} 只（间隔1秒）...")
    time.sleep(3)
    retry_ok = 0
    for code in fail_codes[:]:
        df = fetch_one(code)
        if df is not None and not df.empty:
            try:
                con.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{target_date}'")
                con.register('df_to_insert', df)
                con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM df_to_insert")
                con.unregister('df_to_insert')
                retry_ok += 1
                done.add(code)
                fail_codes.remove(code)
            except Exception as e:
                sys.stderr.write(f"[ERR retry {code}] {e}\n")
        time.sleep(0.3)
    log(f"重试成功: {retry_ok}")

final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{target_date}'").fetchone()[0]
log(f"最终 {target_date}: {final} 条记录")
save_ckpt({'done': True, 'final_count': final, 'success': success, 'fail': len(fail_codes)})
con.close()
log("完成!")
