#!/bin/bash
# 恢复被暂停的优化进程
PID=191910
if ps -p $PID > /dev/null 2>&1; then
    kill -CONT $PID
    echo "[$(date)] 优化进程已恢复 (PID: $PID)"
else
    echo "[$(date)] 进程不存在，需要重新启动"
fi
