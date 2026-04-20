#!/usr/bin/env python3
"""10进程并行增量更新：baostock优先，失败切换akshare，结果统一写DuckDB"""
import os, sys, time, json, duckdb, pandas as pd
from multiprocessing import Pool, cpu_count

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints'
LOG_FILE = '/tmp/update_10proc.log'
TARGET_DATE = '2026-04-03'
TARGET_DATE_AK = '20260403'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}

os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

# ========================================================================
# 每进程独立函数：尝试 baostock，失败自动切换 akshare
# ========================================================================
def fetch_with_fallback(code):
    """返回 (code, df) 或 (code, None)"""
    # ---- 方法1: baostock ----
    try:
        import baostock as bs
        bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=TARGET_DATE, end_date=TARGET_DATE,
            frequency='d', adjustflag='3'
        )
        bs.logout()
        if rs.error_code == '0':
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if data:
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

    # ---- 方法2: akshare (带代理) ----
    try:
        http_proxy = os.environ.get('HTTP_PROXY', '')
        https_proxy = os.environ.get('HTTPS_PROXY', '')
        # 不使用代理（akshare 直连）
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=code, start_date=TARGET_DATE_AK, end_date=TARGET_DATE_AK, adjust='qfq')
        if df is not None and not df.empty:
            df = df.rename(columns={
                '日期':'trade_date','股票代码':'ts_code','开盘':'open','收盘':'close',
                '最高':'high','最低':'low','成交量':'vol','成交额':'amount',
                '涨跌幅':'pct_chg'
            })
            df['ts_code'] = code
            df['limit_up'] = df['pct_chg'] >= 9.9
            df['limit_down'] = df['pct_chg'] <= -9.9
            df['is_st'] = False
            df['suspended'] = df['vol'] == 0
            cols = ['ts_code','trade_date','open','high','low','close','vol','amount',
                    'pct_chg','limit_up','limit_down','is_st','suspended']
            return (code, df[cols])
    except:
        pass

    return (code, None)

# ========================================================================
# 主进程：分配任务 → 收集结果 → 写入 DuckDB
# ========================================================================
if __name__ == '__main__':
    log("=" * 50)
    log(f"启动 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    con = duckdb.connect(DB_PATH, read_only=False)

    # 数据库已有记录
    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing = set(con.execute(
        f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'"
    ).fetchdf()['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in existing]
    log(f"总数: {len(all_codes)}, 已有: {len(existing)}, 待取: {len(to_fetch)}")

    # 断点续算
    ckpt_file = f'{CKPT_DIR}/update_10proc_ckpt.json'
    done = set(existing)
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            if not ckpt.get('done'):
                done = set(ckpt.get('done_codes', [])) | existing
                to_fetch = [c for c in all_codes if c not in done]
                log(f"断点续算: 剩余 {len(to_fetch)} 条")
        except:
            pass

    N_WORKERS = min(8, max(2, cpu_count() - 2))
    CHUNK = 10  # 每进程每次处理10只
    log(f"进程池: {N_WORKERS} workers, chunksize={CHUNK}")

    total_success = 0
    total_fail = 0
    last_save = time.time()

    # 分块并行
    for batch_start in range(0, len(to_fetch), CHUNK * N_WORKERS):
        batch = to_fetch[batch_start:batch_start + CHUNK * N_WORKERS]
        with Pool(N_WORKERS) as pool:
            results = pool.map(fetch_with_fallback, batch, chunksize=CHUNK)

        # 收集结果并写 DB
        for code, df in results:
            if df is not None and not df.empty:
                try:
                    exists = con.execute(
                        "SELECT 1 FROM stock_daily WHERE ts_code = ? AND trade_date = ? LIMIT 1"
                    ).fetchone()
                    if not exists:
                        con.register('dft', df)
                        con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
                        con.unregister('dft')
                        total_success += 1
                        done.add(code)
                    else:
                        total_success += 1
                        done.add(code)
                except:
                    total_fail += 1
            else:
                total_fail += 1

        elapsed = time.time() - last_save
        if elapsed >= 30 or batch_start + CHUNK * N_WORKERS >= len(to_fetch):
            with open(ckpt_file + '.tmp', 'w') as f:
                json.dump({'done_codes': list(done), 'success': total_success, 'fail': total_fail}, f)
            os.rename(ckpt_file + '.tmp', ckpt_file)
            rate = total_success / elapsed * 60 if elapsed > 0 else 0
            log(f"  {batch_start + len(batch)}/{len(to_fetch)} | 成功 {total_success} | 失败 {total_fail} | {rate:.0f}只/分钟")
            last_save = time.time()

    # 最终
    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    elapsed_total = time.time()
    log(f"最终 {TARGET_DATE}: {final} 条 | 新增 {total_success} | 失败 {total_fail}")
    with open(ckpt_file, 'w') as f:
        json.dump({'done': True, 'done_codes': list(done), 'success': total_success, 'fail': total_fail, 'final': final}, f)
    con.close()
    log("完成!")
