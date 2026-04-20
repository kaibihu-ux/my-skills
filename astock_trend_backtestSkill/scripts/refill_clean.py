#!/usr/bin/env python3
"""
干净的历史数据补抓脚本：逐日期顺序执行，无文件冲突。
每次只处理一个日期，完成后才写入下一日期的批次。
"""
import os, sys, json, subprocess, time
import duckdb, pandas as pd
from datetime import date, timedelta

SKILL_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
DB_PATH   = f'{SKILL_DIR}/data/astock_full.duckdb'
CKPT_DIR  = f'{SKILL_DIR}/checkpoints'
FETCH_SCR = f'{SKILL_DIR}/scripts/fetch_daily_batch.py'

# 要处理的日期（交易日）
DATES = ['2026-03-27', '2026-03-30', '2026-03-31', '2026-04-01', '2026-04-02', '2026-04-03']

os.makedirs(CKPT_DIR, exist_ok=True)

def get_missing(target_date):
    """返回该日期缺失的股票代码列表"""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        have = set(r[0] for r in con.execute(
            'SELECT DISTINCT ts_code FROM stock_daily WHERE trade_date = ?', (target_date,)
        ).fetchall())
    except:
        have = set()
    stock_list = set(r[0] for r in con.execute('SELECT ts_code FROM stock_list').fetchall())
    con.close()
    merged = have | stock_list
    return sorted(merged - have), len(have), len(merged)

def process_date(target_date):
    """抓取并写入单个日期数据"""
    print(f"\n{'='*50}")
    print(f"处理: {target_date}")
    sys.stdout.flush()

    to_fetch, n_have, n_merged = get_missing(target_date)
    print(f"  已有 {n_have} | 合并基准 {n_merged} | 待抓 {len(to_fetch)}")
    sys.stdout.flush()

    if not to_fetch:
        print("  无需更新")
        return 0

    # 写入批次文件
    n = len(to_fetch)
    nw = 4
    q = max(1, n // nw)
    for i in range(nw):
        start = i * q
        end = (i + 1) * q if i < nw - 1 else n
        with open(f'{CKPT_DIR}/batch_{i}.json', 'w') as f:
            json.dump(to_fetch[start:end], f)

    # 逐个启动 fetch 进程（不用 nohup，直接等待）
    print(f"  启动 {nw} 个抓取进程...")
    sys.stdout.flush()
    procs = []
    for i in range(nw):
        # 直接 Popen 并等待，不经过 shell
        p = subprocess.Popen(
            ['python3', FETCH_SCR, str(i), target_date],
            cwd=SKILL_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append((i, p))
        print(f"    组{i} PID={p.pid}")
        sys.stdout.flush()

    # 等待每个进程完成
    for i, p in procs:
        p.wait()
        print(f"    组{i} PID={p.pid} 完成 (rc={p.returncode})")
        sys.stdout.flush()

    # 读结果写 DuckDB
    print("  写入 DuckDB...")
    sys.stdout.flush()
    con = duckdb.connect(DB_PATH, read_only=False)
    total = 0
    for i in range(nw):
        rf = f'{CKPT_DIR}/batch_{i}_results.json'
        try:
            with open(rf) as f:
                results = json.load(f)
        except:
            print(f"    组{i}: 结果文件读取失败")
            continue

        written = 0
        for item in results:
            code = item['code']
            row = item['data']
            try:
                df = pd.DataFrame([row])
                cols = ['ts_code','trade_date','open','high','low','close','vol',
                        'amount','pct_chg','limit_up','limit_down','is_st','suspended']
                df = df[cols]
                con.execute("DELETE FROM stock_daily WHERE ts_code = ? AND trade_date = ?", [code, target_date])
                con.register('dft', df)
                con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
                con.unregister('dft')
                written += 1
            except:
                pass
        print(f"    组{i}: 写入 {written} 条")
        total += written
        sys.stdout.flush()

    cnt = con.execute(
        'SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (target_date,)
    ).fetchone()[0]
    print(f"  {target_date} 最终: {cnt} 条 (新增 {total})")
    con.close()
    return total

def main():
    print(f"开始补抓: {DATES}")
    print(f"SKILL_DIR: {SKILL_DIR}")

    total_all = 0
    for d in DATES:
        written = process_date(d)
        total_all += written

    print(f"\n{'='*50}")
    print(f"全部完成！新增 {total_all} 条")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 汇总
    con = duckdb.connect(DB_PATH, read_only=True)
    print("\n各天最终数据量:")
    for d in DATES:
        cnt = con.execute(
            'SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (d,)
        ).fetchone()[0]
        print(f"  {d}: {cnt}")
    con.close()

if __name__ == '__main__':
    main()
