#!/usr/bin/env python3
"""每日 stock_list 自动更新脚本

功能：
    1. 从 baostock 获取最新主板股票列表（600/601/603 + 000/001）
    2. 对比当前 stock_list，增量更新（新增/退市）
    3. 输出新增和退市股票列表

Usage:
    python3 scripts/update_stock_list.py
    python3 scripts/update_stock_list.py --dry-run   # 不写入，只对比
    python3 scripts/update_stock_list.py --date 2026-04-07  # 指定日期

依赖：baostock, pandas, duckdb
"""
import os, sys, argparse, json
import baostock as bs
import pandas as pd
import duckdb
from datetime import date, timedelta

SKILL_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
DB_PATH = f'{SKILL_DIR}/data/astock_full.duckdb'
CKPT_DIR = f'{SKILL_DIR}/checkpoints'

# 主板代码前缀（上交所 + 深交所主板）
MAIN_BOARD_PATTERNS = ['600', '601', '603', '000', '001']


def fetch_baostock_list():
    """从 baostock 获取最新股票列表"""
    bs.login()
    rs = bs.query_stock_basic(code_name='')
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data, columns=rs.fields)

    # 只保留主板：type=1 且 代码前缀符合，且仍在交易（outDate为空字符串）
    active = df[
        (df['type'] == '1') &
        (df['outDate'].str.strip() == '')
    ].copy()

    active['ts_code'] = active['code'].str.replace('sh.', '').str.replace('sz.', '')
    active = active[active['ts_code'].str.match(r'^(' + '|'.join(MAIN_BOARD_PATTERNS) + ')')]
    active['symbol'] = active['code'].str.replace('.', '')
    active['name'] = active['code_name']
    active['list_date'] = active['ipoDate']

    return active[['ts_code', 'symbol', 'name', 'list_date']]


def update_stock_list(dry_run=False, target_date=None):
    """对比并更新 stock_list"""
    today = date.today()
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    day_str = target_date or yesterday

    print(f"[{today}] stock_list 更新检查 | 目标日期: {day_str}")

    # 1. 获取最新列表
    new_df = fetch_baostock_list()
    new_codes = set(new_df['ts_code'])
    print(f"  baostock 主板在交易: {len(new_codes)} 只")

    # 2. 读取当前列表
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        current_codes = set([r[0] for r in con.execute('SELECT ts_code FROM stock_list').fetchall()])
    except Exception:
        current_codes = set()
    con.close()

    # 3. 对比
    added = new_codes - current_codes
    removed = current_codes - new_codes
    same = new_codes & current_codes

    print(f"  当前 stock_list: {len(current_codes)} 只")
    print(f"  新增（需加入）: {len(added)} 只")
    print(f"  退市（需移除）: {len(removed)} 只")
    print(f"  共同: {len(same)} 只")

    if added:
        print(f"  新增示例: {sorted(added)[:10]}")

    if dry_run:
        print("  [DRY RUN] 跳过写入")
        return {'added': sorted(added), 'removed': sorted(removed), 'same': sorted(same)}

    # 4. 重建 stock_list（删除旧表，插入新表）
    con = duckdb.connect(DB_PATH, read_only=False)
    con.execute('DELETE FROM stock_list')

    inserted = 0
    for _, row in new_df.iterrows():
        try:
            con.execute(
                'INSERT INTO stock_list (ts_code, symbol, name, list_date) VALUES (?, ?, ?, ?)',
                (row['ts_code'], row['symbol'], row['name'],
                 row['list_date'] if row['list_date'] else None)
            )
            inserted += 1
        except Exception as e:
            pass

    final_count = con.execute('SELECT COUNT(*) FROM stock_list').fetchone()[0]
    con.close()

    print(f"  stock_list 已更新: {final_count} 只（实际插入: {inserted}）")

    # 5. 保存新增股票列表（供后续补抓历史数据用）
    if added:
        added_list = sorted(added)
        # 默认取最近30个交易日
        import subprocess
        result = subprocess.run(
            ['python3', '-c', f'''
import duckdb
con = duckdb.connect("{DB_PATH}", read_only=True)
# 取最近有数据的日期
latest = con.execute("SELECT MAX(trade_date) FROM stock_daily").fetchone()[0]
print(latest)
con.close()
'''],
            capture_output=True, text=True
        )
        latest_date = result.stdout.strip() if result.returncode == 0 else None

        with open(f'{CKPT_DIR}/new_stocks.json', 'w') as f:
            json.dump({'date': day_str, 'codes': added_list, 'latest_data_date': latest_date}, f)
        print(f"  新增股票列表已保存: checkpoints/new_stocks.json ({len(added_list)} 只)")

    return {'added': sorted(added), 'removed': sorted(removed), 'same': sorted(same)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='更新 stock_list 主牌股票列表')
    parser.add_argument('--dry-run', action='store_true', help='只对比不写入')
    parser.add_argument('--date', type=str, help='目标日期 YYYY-MM-DD')
    args = parser.parse_args()

    result = update_stock_list(dry_run=args.dry_run, target_date=args.date)
    print(f"\n完成: 新增{len(result['added'])} 退市{len(result['removed'])}")
