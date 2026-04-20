#!/usr/bin/env python3
"""增量更新 2026-04-03 股票数据（akshare + 代理 + 单线程 + 0.3s间隔）"""
import os, sys, time, json, duckdb, pandas as pd

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
CKPT_FILE = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints/update_20260403_ckpt.json'
LOG_FILE = '/tmp/update_20260403.log'
TARGET_DATE = '2026-04-03'

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

def fetch_one(code):
    import akshare as ak
    try:
        df = ak.stock_zh_a_hist(symbol=code, start_date='20260403', end_date='20260403', adjust='qfq')
        if df is None or df.empty:
            return (code, None)
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
    except Exception:
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
    log(f"开始 {time.strftime('%Y-%m-%d %H:%M:%S')} (akshare + 代理)")

    con = duckdb.connect(DB_PATH, read_only=False)

    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    confirmed = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in confirmed]
    log(f"总数: {len(all_codes)}, 已有确认: {len(confirmed)}, 待取: {len(to_fetch)}")

    ckpt = load_ckpt()
    if ckpt and 'done' not in ckpt:
        confirmed2 = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
        to_fetch = [c for c in all_codes if c not in confirmed2]
        log(f"断点续算: 本轮剩余 {len(to_fetch)} 条")

    total_success = 0
    total_fail = 0
    last_save = time.time()

    for i, code in enumerate(to_fetch):
        code2, df = fetch_one(code)
        ok = write_one(code2, df, con)
        if ok:
            confirmed.add(code2)
            total_success += 1
        else:
            total_fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - last_save
            if elapsed >= 30:
                save_ckpt({'done_codes': list(confirmed), 'success': total_success, 'fail': total_fail})
                log(f"  {i+1}/{len(to_fetch)} | 成功 {total_success} | 失败 {total_fail} | 速度 {50/elapsed:.1f}只/秒")
                last_save = time.time()

        time.sleep(0.3)

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条记录")
    save_ckpt({'done': True, 'final_count': final, 'success': total_success, 'fail': total_fail})
    con.close()
    log("完成!")
