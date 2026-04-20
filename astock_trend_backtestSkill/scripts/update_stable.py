#!/usr/bin/env python3
"""单线程增量更新（禁止删除+断点续算+每50只报告）"""
import os, sys, time, json, duckdb, pandas as pd

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_stable_ckpt.json'
LOG_FILE = '/tmp/update_stable.log'

TARGET_DATE = '2026-04-03'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

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

def fetch_with_retry(code, max_retries=3, delay=2.0):
    import baostock as bs
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    for attempt in range(max_retries):
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
                time.sleep(delay)
                continue
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
        except Exception:
            try:
                bs.logout()
            except:
                pass
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None

def write_one(code, df, con):
    """只插入，不删除——保护已有数据"""
    if df is not None and not df.empty:
        try:
            # 先检查是否已存在
            exists = con.execute(
                "SELECT 1 FROM stock_daily WHERE ts_code = ? AND trade_date = ? LIMIT 1"
            ).fetchone()
            if exists:
                return True  # 已存在，跳过
            con.register('dft', df)
            con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
            con.unregister('dft')
            return True
        except Exception as e:
            return False
    return False

if __name__ == '__main__':
    log("=" * 50)
    log(f"开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    con = duckdb.connect(DB_PATH, read_only=False)

    # 获取数据库已有记录（永远不删除！）
    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing_in_db = set(con.execute(
        f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'"
    ).fetchdf()['ts_code'].tolist())
    log(f"总数: {len(all_codes)}, 数据库已有: {len(existing_in_db)}, 待取: {len(all_codes) - len(existing_in_db)}")

    # 断点续算：从检查点恢复已完成列表
    ckpt = load_ckpt()
    done = set(existing_in_db)  # 从数据库已有的开始
    if ckpt and 'done' not in ckpt:
        done = set(ckpt.get('done_codes', [])) | existing_in_db
        log(f"断点续算: 已完成 {len(done)} 条")

    success = 0
    fail = 0
    last_save = time.time()
    start_time = time.time()

    for i, code in enumerate(all_codes):
        if code in done:
            continue

        df = fetch_with_retry(code, max_retries=3, delay=2.0)
        ok = write_one(code, df, con)
        if ok:
            done.add(code)
            success += 1
        else:
            fail += 1

        elapsed = time.time() - start_time
        if (i + 1) % 10 == 0:
            rate = success / elapsed * 60 if elapsed > 0 else 0
            eta = (len(all_codes) - len(done)) / max(rate, 1) if rate > 0 else 0
            log(f"  {i+1}/{len(all_codes)} | 成功 {success} | 失败 {fail} | {rate:.0f}只/分钟 | 剩余约{eta:.0f}分钟")

        if (i + 1) % 50 == 0 or time.time() - last_save >= 60:
            save_ckpt({'done_codes': list(done), 'success': success, 'fail': fail})
            last_save = time.time()

    # 最终保存
    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    elapsed = time.time() - start_time
    rate = success / elapsed * 60 if elapsed > 0 else 0
    log(f"最终 {TARGET_DATE}: {final} 条 | 新增 {success} | 失败 {fail} | {rate:.1f}只/分钟 | 耗时 {elapsed/60:.1f}分钟")
    save_ckpt({'done': True, 'done_codes': list(done), 'success': success, 'fail': fail})
    con.close()
    log("完成!")
