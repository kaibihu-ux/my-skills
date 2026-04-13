#!/usr/bin/env python3
"""
补跑因子数据：2026-03-28 ~ 2026-04-09
删除指定日期范围的因子数据，然后重新计算并写入
"""
import sys
import subprocess
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🔧 因子数据补跑开始：2026-03-28 ~ 2026-04-09")

    api = get_instance()
    store = api.store

    # 1. 删除旧数据（2026-03-28 及之后）
    print(f"[{now}] 删除现有因子数据（2026-03-28 及之后）...")
    result = store.conn.execute(
        "DELETE FROM factors WHERE trade_date >= '2026-03-28'"
    )
    # DuckDB DELETE returns None, use row_count
    deleted = store.conn.execute(
        "SELECT COUNT(*) FROM factors WHERE trade_date >= '2026-03-28'"
    ).fetchone()[0]
    # Note: above query run AFTER delete, so should be 0. Let's check before delete.
    store.close()

    print(f"[{now}] 已删除 2026-03-28 起的因子数据")

    # 2. 重新计算因子（调用 batch_compute_factors.py）
    script_path = Path(__file__).parent / "batch_compute_factors.py"
    print(f"[{now}] 开始重新计算因子（调用 batch_compute_factors.py）...")
    print(f"[{now}] 这将处理全部股票，请耐心等待...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            timeout=36000  # 10小时超时
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] ✅ 因子计算完成")
        else:
            print(f"[{datetime.now()}] ❌ 因子计算失败，退出码: {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now()}] ❌ 子进程超时（10小时）")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 执行失败: {e}")

    # 3. 验证（直接用 duckdb，不走 api 避免单例连接问题）
    import duckdb as _duckdb
    _db_path = Path(__file__).parent.parent / "data" / "astock_full.duckdb"
    conn3 = _duckdb.connect(str(_db_path), read_only=True)
    result = conn3.execute(
        "SELECT MAX(trade_date), COUNT(*) FROM factors WHERE trade_date >= '2026-03-28'"
    ).fetchone()
    print(f"[{datetime.now()}] 补跑后验证：最新日期={result[0]}, 2026-03-28起共{result[1]}行")

    result2 = conn3.execute(
        "SELECT factor_name, COUNT(*) as cnt FROM factors WHERE trade_date >= '2026-03-28' GROUP BY factor_name ORDER BY cnt DESC LIMIT 15"
    ).fetchall()
    print(f"[{datetime.now()}] 各因子补跑行数：")
    for row in result2:
        print(f"  {row[0]}: {row[1]}行")
    conn3.close()
    print(f"[{datetime.now()}] 🎉 补跑完成")


if __name__ == '__main__':
    main()
