#!/usr/bin/env python3
"""历史股票数据更新主控脚本

核心逻辑（2026-04-09 修改）：
    用 baostock.query_all_stock(day=target_date) 获取该日期所有在交易股票，
    再补抓 DuckDB 中缺失的股票。

Usage:
    python3 scripts/update_history_daily.py [target_date] [num_workers]
    python3 scripts/update_history_daily.py 2026-04-07 4         # 更新单天
    python3 scripts/update_history_daily.py 2026-03-27 2026-04-03 4  # 更新区间

功能：
    1. 用 baostock.query_all_stock() 获取目标日期的全部股票代码
    2. 过滤出 DuckDB 中该日期缺失的股票
    3. 将待取列表分组，启动 N 个并行进程抓取
    4. 等待完成后统一写入 DuckDB
"""
import os, sys, time, json, duckdb, pandas as pd
from multiprocessing import Process
import datetime

# 清除代理
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
    os.environ.pop(k, None)

SKILL_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
CKPT_DIR = f'{SKILL_DIR}/checkpoints'
DB_PATH = f'{SKILL_DIR}/data/astock_full.duckdb'
LOG_FILE = '/tmp/update_history_daily.log'

if len(sys.argv) == 3:
    START_DATE = sys.argv[1]
    END_DATE = sys.argv[2]
    NUM_WORKERS = 4
elif len(sys.argv) == 4:
    START_DATE = sys.argv[1]
    END_DATE = sys.argv[2]
    NUM_WORKERS = int(sys.argv[3])
elif len(sys.argv) == 2:
    START_DATE = sys.argv[1]
    END_DATE = sys.argv[1]
    NUM_WORKERS = 4
else:
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    START_DATE = yesterday.strftime('%Y-%m-%d')
    END_DATE = yesterday.strftime('%Y-%m-%d')
    NUM_WORKERS = 4

os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

def get_dates_between(start, end):
    """生成 start 到 end 之间的所有日期（含两端）"""
    from datetime import timedelta
    d = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    dates = []
    while d <= e:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates

def get_stock_codes_for_date(target_date):
    """获取目标日期需要抓取的股票代码列表

    策略（2026-04-09 新逻辑）：
        用 baostock.query_all_stock(day=target_date) 获取该日期所有在交易股票，
        再过滤掉 DuckDB 中该日期已有数据的股票，避免重复抓取。
    """
    import baostock as bs

    # 用 baostock 获取该日期的全部股票
    bs.login()
    rs = bs.query_all_stock(day=target_date)
    bs.logout()

    if rs.error_code != '0':
        log(f"  [ERROR] query_all_stock 失败: {rs.error_msg}")
        return []

    codes = []
    while rs.next():
        row = rs.get_row_data()
        # row: [code, code_name, is_hs, ...]
        code = row[0]  # e.g. "sh.600000" 或 "sz.000001"
        codes.append(code)
    log(f"  baostock.query_all_stock({target_date}): 共获取 {len(codes)} 只")

    # 去重 + 过滤 ST/退市（可选）
    # baostock 返回格式是 "sh.600000" / "sz.000001"，需要去掉前缀转成 "600000"
    codes = sorted(set(codes))
    codes = [c.replace('sh.', '').replace('sz.', '') for c in codes]

    # 过滤：只保留主板股票（沪：600/601/603，深：000/001）
    main_board = {c for c in codes if c[:3] in {'600', '601', '603', '000', '001'}}
    log(f"  主板过滤后: {len(main_board)} 只（原始 {len(codes)} 只）")

    return sorted(main_board)

def run_fetcher(group_id, target_date):
    """启动一个 fetch_daily_batch.py 子进程"""
    import subprocess
    p = subprocess.run(
        ['python3', 'scripts/fetch_daily_batch.py', str(group_id), target_date],
        capture_output=True, text=True,
        cwd=SKILL_DIR
    )
    return group_id, p.returncode

def fetch_and_write(target_date, num_workers=4):
    """抓取并写入单个日期的数据"""
    log(f"\n{'='*50}")
    log(f"处理日期: {target_date}")

    # 1. 获取该日期已有数据的股票
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        existing = set([r[0] for r in con.execute(
            'SELECT DISTINCT ts_code FROM stock_daily WHERE trade_date = ?', (target_date,)
        ).fetchall()])
    except Exception:
        existing = set()
    con.close()

    # 2. 获取需要抓取的股票（用 baostock.query_all_stock 获取基准）
    all_codes = get_stock_codes_for_date(target_date)
    to_fetch = [c for c in all_codes if c not in existing]

    log(f"  baostock基准: {len(all_codes)} | 已有数据: {len(existing)} | 待抓: {len(to_fetch)}")

    if not to_fetch:
        log("  无需更新，数据已是最新。")
        return 0

    # 3. 分组
    n = len(to_fetch)
    q = max(1, n // num_workers)
    groups = []
    for i in range(num_workers):
        start = i * q
        end = (i + 1) * q if i < num_workers - 1 else n
        groups.append(to_fetch[start:end])

    for i, g in enumerate(groups):
        with open(f'{CKPT_DIR}/batch_{i}.json', 'w') as f:
            json.dump(g, f)
        log(f"  组{i}: {len(g)} 只")

    # 4. 启动并行抓取
    log(f"  启动 {num_workers} 个并行抓取进程...")
    procs = []
    for i in range(num_workers):
        p = Process(target=run_fetcher, args=(i, target_date))
        p.start()
        procs.append(p)
        log(f"    启动组{i} PID={p.pid}")

    # 5. 等待完成
    for p in procs:
        p.join()
        log(f"    组 PID={p.pid} 结束")

    # 6. 写入 DuckDB
    log("  抓取完成，开始写入 DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=False)
    total_written = 0

    for i in range(num_workers):
        result_file = f'{CKPT_DIR}/batch_{i}_results.json'
        if not os.path.exists(result_file):
            log(f"    组{i} 结果文件不存在")
            continue

        with open(result_file) as f:
            results = json.load(f)

        log(f"    组{i}: 处理 {len(results)} 条数据...")
        for item in results:
            code = item['code']
            row = item['data']
            try:
                df = pd.DataFrame([row])
                cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol',
                        'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
                df = df[cols]
                con.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{target_date}'")
                con.register('dft', df)
                con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
                con.unregister('dft')
                total_written += 1
            except Exception as e:
                pass

        log(f"    组{i} 完成")

    # 7. 验证
    try:
        final = con.execute(
            'SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (target_date,)
        ).fetchone()[0]
        log(f"  {target_date} 最终: {final} 条 (本次新增 {total_written})")
    except Exception:
        pass

    con.close()
    log(f"  {target_date} 处理完毕！")

    return total_written

def main():
    log("=" * 60)
    log(f"历史数据更新启动 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"区间: {START_DATE} ~ {END_DATE} | worker数: {NUM_WORKERS}")

    dates = get_dates_between(START_DATE, END_DATE)
    log(f"共 {len(dates)} 个日期")

    total_all = 0
    for d in dates:
        written = fetch_and_write(d, NUM_WORKERS)
        total_all += written

    log(f"\n全部完成！共处理 {len(dates)} 天，新增 {total_all} 条记录")
    log(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 最终验证
    con = duckdb.connect(DB_PATH, read_only=True)
    for d in dates:
        cnt = con.execute('SELECT COUNT(*) FROM stock_daily WHERE trade_date = ?', (d,)).fetchone()[0]
        log(f"  {d}: {cnt} 条")
    con.close()

if __name__ == '__main__':
    main()
