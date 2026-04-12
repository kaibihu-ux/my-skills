#!/usr/bin/env python3
"""
RL 分批训练测试脚本
测试 batch_id 和 daily_reset 参数是否正常工作
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.core.rl_optimizer import RLOptimizer
from src.core.backtester import BacktestExecutor
from src.api.skill_api import get_instance

def test_rl_batch():
    """测试 RL 分批训练功能"""
    print("=" * 60)
    print("RL 分批训练功能测试")
    print("=" * 60)
    
    api = get_instance()
    store = api.store
    logger = api.logger
    
    # 创建 BacktestExecutor
    bt_executor = BacktestExecutor(store, logger, api.config)
    
    # 创建 RLOptimizer
    rl_opt = RLOptimizer(
        bt_executor, logger,
        start_date='20240101',
        end_date='20260327',
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1,
        n_episodes=5,
        lookback_days=20,
    )
    
    # 测试策略
    rl_strategy = {
        'strategy_id': 'rl_position',
        'strategy_name': 'rl_position',
        'factors': ['momentum_20', 'volatility_20', 'size'],
    }
    rl_params = {
        'holding_period': 10,
        'stop_loss': 0.10,
        'take_profit': 0.20,
    }
    
    print("\n[TEST] Batch 0 (daily_reset=True)...")
    result0 = rl_opt.optimize(
        rl_strategy, rl_params,
        use_rl_position=True,
        batch_id=0,
        daily_reset=True
    )
    print(f"✓ Batch 0 完成 | Episodes done: {result0.get('rl_episodes_done', 0)}")
    print(f"  Q-table states: {result0.get('n_states', 0)}")
    print(f"  Sharpe: {result0.get('final_sharpe', 0):.4f}")
    
    # 测试 Eval 回测
    print("\n[TEST] Running Eval backtest...")
    eval_result = rl_opt.run_eval_backtest()
    print(f"✓ Eval 回测完成 | Sharpe: {eval_result.get('sharpe', 0):.4f}")
    print(f"  Total return: {eval_result.get('total_return', 0):.4f}")
    print(f"  Max drawdown: {eval_result.get('max_drawdown', 0):.4f}")
    
    print("\n[TEST] Batch 1 (续训)...")
    result1 = rl_opt.optimize(
        rl_strategy, rl_params,
        use_rl_position=True,
        batch_id=1,
        daily_reset=False
    )
    print(f"✓ Batch 1 完成 | Episodes done: {result1.get('rl_episodes_done', 0)}")
    print(f"  Q-table states: {result1.get('n_states', 0)}")
    print(f"  Sharpe: {result1.get('final_sharpe', 0):.4f}")
    
    # 测试 Eval 回测
    print("\n[TEST] Running Eval backtest (Batch 1)...")
    eval_result1 = rl_opt.run_eval_backtest()
    print(f"✓ Eval 回测完成 | Sharpe: {eval_result1.get('sharpe', 0):.4f}")
    
    print(f"\n✓ 所有测试通过!")
    print(f"  Eval results count: {len(rl_opt.eval_results)}")
    
    return True

if __name__ == '__main__':
    try:
        test_rl_batch()
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
