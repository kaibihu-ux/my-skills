#!/bin/bash
# 全量优化启动脚本 - 今晚22:00运行
cd ~/.openclaw/my-skills/algorithmic-trading

# 获取所有股票代码
STOCKS=$(python3 -c "
import duckdb
conn = duckdb.connect('~/.openclaw/my-skills/algorithmic-trading/data/astock_full.duckdb', read_only=True)
stocks = conn.execute('SELECT ts_code FROM stock_list').fetchall()
conn.close()
# 输出空格分隔的股票代码
echo ' '.join([s[0] for s in stocks])
")

echo "📊 股票总数: $(echo $STOCKS | wc -w)"

# 精简参数网格 (快速模式)
nohup python3 scripts/optimizer.py \
  --symbols $STOCKS \
  --start 2020-01-01 \
  --end 2026-03-31 \
  --quick \
  --top 100 \
  > output/full_optimization_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "🚀 全量优化已启动!"
echo "   PID: $!"
echo "   股票数量: $(echo $STOCKS | wc -w)"
echo "   日志: output/full_optimization_*.log"
