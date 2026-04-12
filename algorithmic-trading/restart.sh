#!/bin/bash
cd ~/.openclaw/my-skills/algorithmic-trading
nohup python3 scripts/optimizer.py --symbols AAPL MSFT TSLA GOOGL NVDA AMZN --start 2022-01-01 --end 2023-12-31 --quick --top 50 > output/full_optimization.log 2>&1 &
echo "优化已重新启动 (PID: $!)"
