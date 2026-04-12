# RL 分批续训 + Eval 回测改造总结

## 改造日期
2026-04-12

## 改造目标
将 Step3_RL 分成 4 批次（batch0-3），每批 5 episodes，共 20 episodes。每批后执行 Eval 回测，结果增量保存。

## 改造内容

### 1. rl_optimizer.py 修改

#### 新增参数
```python
def optimize(self, ..., batch_id: int = 0, daily_reset: bool = False):
```

#### batch0 逻辑
- `batch_id == 0` 或 `daily_reset=True` 时：清除旧数据（q_table, training_history, eval_results）
- 其他情况：加载 checkpoint 续训

#### 新增方法
1. **load_checkpoint()**: 加载检查点续训
2. **run_eval_backtest()**: 使用 Q-table 执行 Eval 回测
3. **load_all_data_to_memory()**: 预加载 539 天数据到内存 DataFrame
4. **backtest_with_qtable(data)**: 使用训练好的 Q-table 执行回测

#### 增量保存
Checkpoint 包含：
- `rl_batch_id`: 批次 ID
- `rl_episodes_done`: 已完成 episodes 数 = (batch_id + 1) * 5
- `q_table`: Q 表
- `eval_results`: Eval 回测结果列表（追加）
- `training_history`: 训练历史（追加）
- `rl_best_sharpe`: 最优 Sharpe（保持兼容）

### 2. tasks_split.py 修改

#### 新增参数
```python
def job_step3_rl(force_restart=False, batch_id=None, daily_reset=False):
```

#### batch0 逻辑
```python
if batch_id == 0:
    daily_reset = True  # 清除旧数据
```

#### 调用 RL
```python
if batch_id is not None:
    rl_result = rl_opt.optimize(..., batch_id=batch_id, daily_reset=daily_reset)
    eval_result = rl_opt.run_eval_backtest()
    rl_result['eval_results'].append(eval_result)
```

### 3. Cron 配置

#### 非交易日
- 17:30 Step3_RL_batch0 (清除旧数据)
- 18:00 Step3_RL_batch1
- 18:30 Step3_RL_batch2
- 19:00 Step3_RL_batch3
- 19:30 Step4_Bayes
- 20:30 因子相关性
- 20:45 Step5_Final

#### 交易日
- 20:00 Step3_RL_batch0
- 20:30 Step3_RL_batch1
- 21:00 Step3_RL_batch2
- 21:30 Step3_RL_batch3
- 22:00 Step4_Bayes
- 23:00 Step5_Final
- 23:10 发送报告

### 4. 兼容性保持
- `rl_best_sharpe`: 保持兼容
- `ga_best_factors`: 保持兼容
- `ga_best_params`: 保持兼容
- 报告人维持拉姆

## 测试验证

### 测试脚本
```bash
cd /home/hyh/.openclaw/my-skills/astock_trend_backtestSkill
python3 test_rl_batch.py
```

### 测试内容
1. Batch 0 (daily_reset=True) - 清除旧数据重新开始
2. Eval 回测 - 验证 run_eval_backtest() 功能
3. Batch 1 (续训) - 验证 checkpoint 加载
4. Eval 回测 - 验证 eval_results 追加

## 文件清单
- ✅ `src/core/rl_optimizer.py` - 已修改
- ✅ `scheduler/tasks_split.py` - 已修改
- ✅ `crontab` - 已更新
- ✅ `test_rl_batch.py` - 测试脚本（新建）
- ✅ `docs/RL_BATCH_CHANGES.md` - 本文档（新建）

## 注意事项
1.  cron 配置同时包含交易日和非交易日任务，实际执行时需根据市场状态选择
2.  建议后续添加交易日判断逻辑，避免在非交易日执行交易日任务
3.  Eval 回测使用 Q-table 进行决策，不使用 RL 训练时的探索策略
4.  所有 checkpoint 保存在 `checkpoints/rl_checkpoint.json`

## 下一步
1. 运行测试脚本验证功能
2. 观察首次完整运行（4 个批次）
3. 确认 Eval 回测结果正确保存
4. 优化 cron 配置（添加交易日判断）
