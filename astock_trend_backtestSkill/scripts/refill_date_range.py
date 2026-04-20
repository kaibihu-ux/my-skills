#!/usr/bin/env python3
"""
给定日期区间，逐天抓取缺失的股票数据。

每个日期的逻辑：
    1. 读取 stock_list 与该日期历史已有股票的并集（覆盖退市股）
    2. 计算缺失股票列表，分4组写入 checkpoints/batch_N.json
    3. 逐个启动4个 fetch_daily_batch.py 进程并等待其完成
    4. 读取 batch_N_results.json，写入 DuckDB

Usage:
    python3 scripts/refill_date_range.py 2026-03-27 2026-04-03
"""
import os, sys, time, json, subprocess
import duckdb, pandas as pd
from multiprocessing import Process
from datetime import date, timedelta

SKILL_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
DB_PATH = f'{SKILL_DIR}/data/astock_full.duckdb'
CKPT_DIR = f'{SKILL_DIR}/checkpoints'
FETCH_SCRIPT = f'{SKILL_DIR}/scripts/fetch_daily_batch.py'

os.makedirs(CKPT_DIR, exist_ok=True)

START_DATE = sys.argv[1] if len(sys.argv) > 1 else '2026-03-27'
END_DATE = sys.argv[2] if len(sys.argv) > 2 else '2026-04-03'

def get_dates(start, end):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    dates = []
    while d <= e:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates

def get_stock_lists_for_date(target_date):
    """获取该日期需要抓取的股票列表（已有 + stock_list 并集）"""
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        have = set([r[0] for r in con.execute(
            'SELECT DISTINCT ts_code FROM stock_daily WHERE trade_date = ?', (target_date,)
        ).fetchall()])
    except:
        have = set()
    stock_list = set([r[0] for r in con.execute('SELECT ts_code FROM stock_list').fetchall()])
    con.close()
    merged = have | stock_list  # 并集：保留历史退市股
    to_fetch = sorted(merged - have)
    return to_fetch, len(have), len(merged)

def fetch_and_write(target_date, num_workers=4):
    """抓取单个日期缺失股票并写入 DuckDB"""
    print(f"\n{'='*50}")
    print(f"处理日期: {target_date}")

    to_fetch, n_have, n_merged = get_stock_lists_for_date(target_date)
    print(f"  已有: {n_have} | 合并基准: {n_merged} | 待抓: {len(to_fetch)}")

    if not to_fetch:
        print("  无需更新。")
        return 0

    # 写入 batch 文件（确保在启动子进程前完成）
    n = len(to_fetch)
    q = max(1, n // num_workers)
    batch_files = []
    for i in range(num_workers):
        start = i * q
        end = (i + 1) * q if i < num_workers - 1 else n
        batch_stocks = to_fetch[start:end]
        bf = f'{CKPT_DIR}/batch_{i}.json'
        with open(bf, 'w') as f:
            json.dump(batch_stocks, f)
        batch_files.append(bf)
    print(f"  批次文件已写入: {n} 只分 {num_workers} 组")

    # 等待子进程完成的辅助函数
    import subprocess
    procs = []
    for i in range(num_workers):
        p = subprocess.Popen(
            ['python3', FETCH_SCRIPT, str(i), target_date],
            cwd=SKILL_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append((i, p))
        print(f"    启动组{i} PID={p.pid}")

    # 等待所有子进程完成
    for i, p in procs:
        p.wait()
        print(f"    组{i} PID={p.pid} 结束 (returncode={p.returncode})")

    # 读取结果并写入 DuckDB
    print(f"  写入 DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=False)
    total_written = 0

    for i in range(num_workers):
        rf = f'{CKPT_DIR}/batch_{i}_results.json'
        try:
            with open(rf) as f:
                results = json.load(f)
        except Exception as e:
            print(f"    组{i}: 结果文件读取失败 ({e})")
            continue

        written = 0
        for item in results:
            code = item['code']
            row = item['data']
            try:
                df = pd.DataFrame([row])
                cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol',
                        'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
                df = df[cols]
                con.execute("DELETE FROM stock_daily WHERE ts_code = ? AND trade_date = ?", [code, target_date])
                con.register('dft', df)
                con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
                con.unregister('dft')
                written += 1
            except Exception as e:
                pass
        print(f"    组{i}: 写入 {written} 条")
        total_written += written

    # 验证
    cnt = con.execute(
        'SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (target_date,)
    ).fetchone()[0]
    print(f"  {target_date} 最终: {cnt} 条 (本次新增 {total_written})")
    con.close()

    return total_written

def main():
    dates = get_dates(START_DATE, END_DATE)
    print(f"区间: {START_DATE} ~ {END_DATE} ({len(dates)} 天)")
    print(f"SKILL_DIR: {SKILL_DIR}")
    print(f"DB: {DB_PATH}")

    total_all = 0
    for d in dates:
        written = fetch_and_write(d, num_workers=4)
        total_all += written

    print(f"\n{'='*50}")
    print(f"全部完成！共处理 {len(dates)} 天，新增 {total_all} 条")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 最终汇总
    con = duckdb.connect(DB_PATH, read_only=True)
    print("\n各天最终数据量:")
    for d in dates:
        cnt = con.execute('SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (d,)).fetchone()[0]
        print(f"  {d}: {cnt}")
    con.close()

if __name__ == '__main__':
    main()
