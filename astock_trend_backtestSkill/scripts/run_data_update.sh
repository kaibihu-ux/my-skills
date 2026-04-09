#!/bin/bash
# 停止 scheduler 进程
pkill -f "scheduler/tasks" 2>/dev/null
sleep 2

# 杀掉所有持有 DuckDB 锁的进程
for pid in $(lsof -t /home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb 2>/dev/null); do
    kill -9 $pid 2>/dev/null && echo "已杀掉 $pid"
done
sleep 1

# 运行数据更新
cd /home/hyh/.openclaw/my-skills/astock_trend_backtestSkill
python3 scripts/update_20260403_baostock.py >> /tmp/update_20260403.log 2>&1
echo "数据更新完成: $(date)" >> /tmp/update_20260403.log
