#!/bin/bash
cd ~/.openclaw/skills/algorithmic-trading
nohup python3 scripts/full_batch_optimizer.py \
  --batch-size 30 \
  --max-stocks 200 \
  --start 2020-01-01 \
  --end 2026-03-31 \
  > output/full_batch_optimization_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "启动完成，PID: $!"
