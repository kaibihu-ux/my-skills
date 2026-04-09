#!/usr/bin/env python3
"""主控脚本：启动4个并行抓取进程，等待完成后统一写入 DuckDB"""
import os, sys, time, json, duckdb, pandas as pd
from multiprocessing import Process

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

CKPT_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints'
DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
LOG_FILE = '/tmp/update_20260403.log'
TARGET_DATE = '2026-04-03'

os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

def run_fetcher(group_id):
    import subprocess
    p = subprocess.run(
        ['python3', 'scripts/fetch_batch.py', str(group_id)],
        capture_output=True, text=True
    )
    return group_id, p.returncode

def main():
    log("=" * 50)
    log(f"主进程启动 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 连接 DuckDB 获取待取列表
    con = duckdb.connect(DB_PATH, read_only=True)
    all_codes = con.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchdf()['ts_code'].tolist()
    existing = set(con.execute(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchdf()['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in existing]
    con.close()
    log(f"总数: {len(all_codes)}, 已有: {len(existing)}, 待取: {len(to_fetch)}")

    # 将待取列表分成4组
    n = len(to_fetch)
    q = n // 4
    groups = [to_fetch[0:q], to_fetch[q:q*2], to_fetch[q*2:q*3], to_fetch[q*3:]]
    for i, g in enumerate(groups):
        with open(f'{CKPT_DIR}/batch_{i}.json', 'w') as f:
            json.dump(g, f)
        log(f"组{i}: {len(g)} 只")

    # 启动4个抓取进程
    log("启动4个并行抓取进程...")
    procs = []
    for i in range(4):
        p = Process(target=run_fetcher, args=(i,))
        p.start()
        procs.append(p)
        log(f"  启动组{i} PID={p.pid}")

    # 等待所有抓取进程完成
    for p in procs:
        p.join()
        log(f"  组{p.pid} 结束")

    # 收集所有结果并写入 DuckDB
    log("所有抓取完成，开始写入 DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=False)
    total_written = 0

    for i in range(4):
        result_file = f'{CKPT_DIR}/batch_{i}_results.json'
        if not os.path.exists(result_file):
            log(f"  组{i} 结果文件不存在")
            continue

        with open(result_file) as f:
            results = json.load(f)

        log(f"  组{i}: 处理 {len(results)} 条数据...")
        for item in results:
            code = item['code']
            row = item['data']
            try:
                df = pd.DataFrame([row])
                df['trade_date'] = TARGET_DATE
                df['limit_up'] = df['pct_chg'] >= 9.9
                df['limit_down'] = df['pct_chg'] <= -9.9
                df['is_st'] = False
                df['suspended'] = df['vol'] == 0
                cols = ['ts_code','trade_date','open','high','low','close','vol','amount',
                        'pct_chg','limit_up','limit_down','is_st','suspended']
                df = df[cols]
                con.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{TARGET_DATE}'")
                con.register('dft', df)
                con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
                con.unregister('dft')
                total_written += 1
            except Exception as e:
                log(f"    写入失败 {code}: {e}")

        log(f"  组{i} 完成")

    final = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{TARGET_DATE}'").fetchone()[0]
    log(f"最终 {TARGET_DATE}: {final} 条记录 (本次新增 {total_written})")
    con.close()
    log("全部完成!")

if __name__ == '__main__':
    main()
