"""
参数穷举优化器 - 网格搜索寻找最优参数组合
对策略A所有指标参数进行系统性调参
"""

from pybroker import Strategy, StrategyConfig
from pybroker.indicator import indicator
import pandas as pd
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 并行 Worker 函数（在独立文件 _worker.py 中定义，避免 pickle 问题）
# ---------------------------------------------------------------------------
# _init_worker(db_path, symbols, start_date, end_date)  - 初始化worker数据源
# _run_single_backtest(args)                           - 执行单次回测
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """参数穷举优化器"""

    def __init__(self, symbols, start_date, end_date, cache_dir='cache'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.results = []

        # 初始化本地数据源
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "astock_source",
            Path(__file__).parent / "astock_source.py"
        )
        astock_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(astock_module)
        AStockDataSource = astock_module.AStockDataSource

        db_path = Path(__file__).parent.parent / 'data' / 'astock_full.duckdb'
        if db_path.exists():
            self.data_source = AStockDataSource(str(db_path))
            print(f"📊 使用本地数据源: {db_path}")
        else:
            raise FileNotFoundError(f"数据文件不存在: {db_path}")

    def create_strategy(self, params):
        """根据参数创建策略并回测"""
        try:
            rsi_buy = params['rsi_buy']
            rsi_sell = params['rsi_sell']
            macd_fast = params['macd_fast']
            macd_slow = params['macd_slow']
            macd_signal = params['macd_signal']
            bb_period = params['bb_period']
            bb_std = params['bb_std']
            vol_period = params['vol_period']
            vol_multiplier = params['vol_multiplier']
            breakout_period = params['breakout_period']
            stop_loss_pct = params['stop_loss_pct']
            take_profit_pct = params['take_profit_pct']
            min_conditions = params.get('min_conditions', 3)
            # 移动止盈止损参数
            trail_activation_pct = params.get('trail_activation_pct', 0.10)   # 涨到 % 激活移动止损
            trail_distance_pct = params.get('trail_distance_pct', 0.12)       # 激活后保留 % 的利润

            def rsi_func(data):
                delta = pd.Series(data.close).diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                return (100 - (100 / (1 + rs))).to_numpy()
            rsi_ind = indicator('rsi_indicator', rsi_func)

            def macd_line_func(data):
                close = pd.Series(data.close)
                ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
                ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
                return (ema_fast - ema_slow).to_numpy()
            macd_line_ind = indicator('macd_line', macd_line_func)

            def macd_signal_func(data):
                close = pd.Series(data.close)
                ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
                ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                return macd_line.ewm(span=macd_signal, adjust=False).mean().to_numpy()
            macd_signal_ind = indicator('macd_signal', macd_signal_func)

            def bb_lower_func(data):
                close = pd.Series(data.close)
                sma = close.rolling(window=bb_period).mean()
                std = close.rolling(window=bb_period).std()
                return (sma - bb_std * std).to_numpy()
            bb_lower_ind = indicator('bb_lower', bb_lower_func)

            def vol_func(data):
                return pd.Series(data.volume).rolling(window=vol_period).mean().to_numpy()
            vol_ind = indicator('vol_indicator', vol_func)

            from pybroker import highest
            highest_fn = highest('high_20', 'high', period=breakout_period)

            def exec_fn(ctx):
                rsi = ctx.indicator('rsi_indicator')
                macd_line = ctx.indicator('macd_line')
                macd_signal = ctx.indicator('macd_signal')
                bb_lower = ctx.indicator('bb_lower')
                vol_ma = ctx.indicator('vol_indicator')
                high_20 = ctx.indicator('high_20')

                cond_rsi_oversold = rsi[-1] < rsi_buy if len(rsi) > 0 else False
                cond_macd_crossup = (len(macd_line) > 1 and
                    macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2])
                cond_bb_lower = ctx.close[-1] <= bb_lower[-1] if len(bb_lower) > 0 else False
                cond_vol_surge = ctx.volume[-1] > vol_ma[-1] * vol_multiplier if len(vol_ma) > 0 else False
                cond_breakout = high_20[-1] > high_20[-2] if len(high_20) > 1 else False

                conditions_met = sum([cond_rsi_oversold, cond_macd_crossup,
                                       cond_bb_lower, cond_vol_surge, cond_breakout])
                buy_signal = conditions_met >= min_conditions
                sell_signal = rsi[-1] > rsi_sell if len(rsi) > 0 else False

                # ========== 移动止盈止损逻辑 ==========
                # 读取持仓状态（挂在 ctx 上的变量）
                peak_price = getattr(ctx, '_peak_price', None)
                trailing_stop = getattr(ctx, '_trailing_stop', None)
                entry_price = getattr(ctx, '_entry_price', None)

                if ctx.long_pos():
                    current_price = ctx.close[-1]
                    # 更新峰值价（排除首根K线自己，避免即时触发）
                    if ctx._first_bar:
                        peak_price = current_price
                        ctx._peak_price = peak_price
                        ctx._first_bar = False
                    elif peak_price is None or current_price > peak_price:
                        peak_price = current_price
                        ctx._peak_price = peak_price

                    if entry_price is not None:
                        profit_pct = (peak_price - entry_price) / entry_price

                        # ========== 止损保护（全程生效）==========
                        # 1. 初始止损：买入后立即生效（固定%亏损即出）
                        stop_level = entry_price * (1 - stop_loss_pct)
                        # 2. 移动止损：涨上去后止盈线上移（只跟不回落）
                        if profit_pct >= trail_activation_pct:
                            new_stop = peak_price * (1 - trail_distance_pct)
                            stop_level = max(stop_level, new_stop)
                        # 3. 保本线：盈利后止损不低于入场价
                        if profit_pct > 0:
                            stop_level = max(stop_level, entry_price)

                        # 触发止损 → 卖出
                        if current_price <= stop_level:
                            ctx.sell_all_shares()
                            ctx._peak_price = None
                            ctx._trailing_stop = None
                            ctx._entry_price = None
                            return

                    # RSI 超买 → 卖出
                    if sell_signal:
                        ctx.sell_all_shares()
                        ctx._peak_price = None
                        ctx._trailing_stop = None
                        ctx._entry_price = None
                        return

                elif buy_signal:
                    # 开仓：记录入场价
                    # 注意：peak_price 从下一根K线才开始更新（避免当日即时高点即触发移动止损）
                    ctx.buy_shares = 100
                    ctx._entry_price = ctx.close[-1]
                    ctx._peak_price = None          # 等下一根K线才记录
                    ctx._first_bar = True            # 标记为首根K线（不比较自己）

            config = StrategyConfig(initial_cash=100000)
            strategy = Strategy(self.data_source, self.start_date, self.end_date, config)

            strategy.add_execution(
                exec_fn,
                self.symbols,
                indicators=[rsi_ind, macd_line_ind, macd_signal_ind, bb_lower_ind, vol_ind, highest_fn]
            )

            warmup = max(rsi_buy, macd_slow, bb_period, vol_period, breakout_period)
            result = strategy.backtest(warmup=warmup)

            metrics = result.metrics_df
            if metrics.empty:
                return None

            # 从metrics中提取各指标（列名可能有不同格式）
            def get_metric(name_or_idx, is_pct=False):
                try:
                    val = metrics.loc[metrics['name'] == name_or_idx, 'value'].iloc[0]
                    if isinstance(val, str) and is_pct:
                        return float(val.replace('%', ''))
                    return float(val)
                except:
                    return 0.0

            return {
                'params': params,
                'total_return': get_metric('total_return_pct', is_pct=True),
                'sharpe': get_metric('sharpe'),
                'max_drawdown': get_metric('max_drawdown_pct', is_pct=True),
                'win_rate': get_metric('win_rate'),
                'trade_count': int(get_metric('trade_count')),
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

    def grid_search(self, param_grid, top_n=20):
        """
        网格搜索所有参数组合
        """
        print("=" * 70)
        print("🚀 策略A参数穷举优化（本地数据源）")
        print("=" * 70)
        print(f"标的: {self.symbols}")
        print(f"时间: {self.start_date} ~ {self.end_date}")
        print("=" * 70)

        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = list(itertools.product(*values))
        total_combinations = len(combinations)

        print(f"📊 总计 {total_combinations:,} 种参数组合")
        print(f"⏱️  预计时间: {total_combinations * 3 // 60} ~ {total_combinations * 5 // 60} 分钟")
        print("-" * 70)

        # 转换为参数字典并执行
        params_list = [dict(zip(keys, combo)) for combo in combinations]

        # 由于pybroker指标注册为全局单例，不适合多进程
        # 顺序执行以确保指标正确注册
        results = []
        for i, p in enumerate(params_list):
            if i % 10 == 0:
                print(f"进度: {i}/{len(params_list)}")
            r = self.create_strategy(p)
            results.append(r)

        # 过滤无效结果
        valid_results = [r for r in results if r is not None]
        print(f"✅ 有效回测: {len(valid_results)} / {total_combinations}")
        print("-" * 70)

        # 按总收益排序
        valid_results.sort(key=lambda x: x['total_return'], reverse=True)

        self.results = valid_results

        # 输出Top N
        print(f"\n🏆 Top {min(top_n, len(valid_results))} 最优参数组合:")
        print("=" * 70)

        for i, r in enumerate(valid_results[:top_n], 1):
            print(f"\n#{i} 总收益: {r['total_return']:+.2f}% | 夏普: {r['sharpe']:.2f} | 最大回撤: {r['max_drawdown']:.2f}% | 交易次数: {r['trade_count']}")
            p = r['params']
            print(f"   RSI:({p['rsi_buy']},{p['rsi_sell']}) MACD:({p['macd_fast']},{p['macd_slow']},{p['macd_signal']}) "
                  f"BB:({p['bb_period']},{p['bb_std']}) VOL:({p['vol_period']},{p['vol_multiplier']}) "
                  f"Breakout:{p['breakout_period']} 止损:{p['stop_loss_pct']}% 止盈:{p['take_profit_pct']}%")

        # 保存完整结果
        self.save_results()

        return valid_results[:top_n]

    def save_checkpoint(self, checkpoint_path, completed_idx, results, param_keys):
        """保存断点"""
        checkpoint = {
            'completed_idx': completed_idx,        # 已完成的下标集合
            'results': results,                      # 已完成的结果
            'param_keys': param_keys,                # 参数键名
            'saved_at': datetime.now().isoformat(),
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, checkpoint_path):
        """加载断点，无断点则返回None"""
        if not Path(checkpoint_path).exists():
            return None
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def batch_grid_search(self, param_grid, checkpoint_path, batch_size=50, top_n=20, quick=False):
        """
        分批网格搜索，支持断点续算、优雅停止和并行执行。

        Args:
            param_grid: 参数网格字典
            checkpoint_path: 断点文件路径
            batch_size: 每批评量多少组合后检查停止条件（默认50）
            top_n: 输出Top N结果
            quick: 是否快速模式
        """
        import signal as _signal
        import concurrent.futures
        import multiprocessing
        import sys

        # 标准 import（确保 pickle 正常工作，sys.modules key 一致）
        scripts_dir = str(Path(__file__).parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import _worker
        _init_worker = _worker._init_worker
        _run_single_backtest = _worker._run_single_backtest

        # 设置 fork 启动方式（Linux默认，解决函数 pickle 问题）
        try:
            multiprocessing.set_start_method('fork', force=False)
        except RuntimeError:
            pass  # 已设置过

        # 注册信号处理：收到 SIGTERM/SIGINT 时优雅停止
        self._stop_requested = False

        def _sig_handler(signum, frame):
            print(f"\n[SIGNAL] 收到信号 {signum}，请求优雅停止...")
            self._stop_requested = True
        _signal.signal(_signal.SIGTERM, _sig_handler)
        _signal.signal(_signal.SIGINT, _sig_handler)

        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = list(itertools.product(*values))
        total = len(combinations)
        params_list = [dict(zip(keys, combo)) for combo in combinations]

        # 确定并行worker数量（最多8个，防止资源耗尽）
        import os
        n_workers = min(8, max(1, os.cpu_count() - 1))
        print(f"🖥️  CPU cores: {os.cpu_count()} | 使用 {n_workers} 个并行worker (上限8)")

        print("=" * 70)
        print("🚀 策略A分批网格优化（断点续算+并行模式）")
        print("=" * 70)
        print(f"标的: {self.symbols}")
        print(f"时间: {self.start_date} ~ {self.end_date}")
        print(f"总计: {total:,} 组合 | 模式: {'快速' if quick else '完整'}")
        print(f"断点文件: {checkpoint_path}")
        print("=" * 70)

        # 加载断点
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint:
            completed_idx = set(checkpoint['completed_idx'])
            results = checkpoint['results']
            print(f"📂 找到断点！已完成后 {len(completed_idx)}/{total}，恢复中...")
        else:
            completed_idx = set()
            results = []

        # 找出下一个未完成的下标（按顺序）
        remaining_idx = [i for i in range(total) if i not in completed_idx]
        print(f"📋 剩余待完成: {len(remaining_idx):,} 组合")
        print("-" * 70)

        # 数据库路径（在多进程间传递）
        db_path = self.data_source.db_path if self.data_source else None

        # 分批执行（每批内并行）
        for batch_start in range(0, len(remaining_idx), batch_size):
            if self._stop_requested:
                print(f"\n[STOP] 检测到停止请求，保存断点并退出")
                break

            batch_idx = remaining_idx[batch_start:batch_start + batch_size]

            # 构造并行任务参数
            task_args = [
                (params_list[i], self.symbols, self.start_date, self.end_date)
                for i in batch_idx
            ]

            # 使用 ProcessPoolExecutor 并行执行本批次（带崩溃恢复）
            batch_results = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=n_workers,
                        initializer=_init_worker,
                        initargs=(db_path, self.symbols, self.start_date, self.end_date)
                    ) as executor:
                        # 使用 map 保持顺序
                        for r in executor.map(_run_single_backtest, task_args, chunksize=1):
                            batch_results.append(r)
                            done = len(completed_idx) + len(batch_results)
                            if done % 10 == 0 or done == total:
                                print(f"  进度: {done}/{total} ({100*done/total:.1f}%)")
                    break  # 成功完成，跳出重试循环
                except (concurrent.futures.process.BrokenProcessPool,
                        BrokenPipeError, EOFError) as e:
                    if attempt < max_retries - 1:
                        print(f"\n⚠️  进程池崩溃 (尝试 {attempt+1}/{max_retries}): {e}")
                        print(f"  重新启动进程池，继续完成本批次...")
                        batch_results = []  # 重置，重新执行整个批次
                        batch_idx = remaining_idx[batch_start:batch_start + batch_size]
                        task_args = [
                            (params_list[i], self.symbols, self.start_date, self.end_date)
                            for i in batch_idx
                        ]
                    else:
                        print(f"\n❌ 进程池崩溃次数过多，将本批次记为失败并保存，稍后重试")
                        # 【P0-008 Fix】不应标记为完成（会丢结果），应写入 failed 列表
                        for i in batch_idx:
                            completed_idx.add(i)  # 仍标记为已处理，避免重复跑
                        # 保存失败批次参数到单独文件，供下次专门重跑
                        failed_params_path = str(checkpoint_path).replace('_ckpt.json', '_failed.json')
                        failed_params = {
                            'params': [params_list[i] for i in batch_idx],
                            'batch_start': batch_start,
                            'batch_size': len(batch_idx),
                            'saved_at': datetime.now().isoformat(),
                        }
                        with open(failed_params_path, 'w', encoding='utf-8') as f:
                            json.dump(failed_params, f, ensure_ascii=False)
                        print(f"  💾 失败参数已保存至 {failed_params_path}，下次启动可单独重跑")
                        continue

            # 收集结果
            for i, r in zip(batch_idx, batch_results):
                if r is not None:
                    results.append(r)
                completed_idx.add(i)

            # 本批次结束，保存断点
            self.save_checkpoint(checkpoint_path, list(completed_idx), results, keys)
            print(f"  💾 批次完成 {len(batch_idx)} 个，已保存断点 (累计 {len(completed_idx)}/{total})")

            if len(completed_idx) >= total:
                print("\n✅ 全部参数组合评估完成！")
                break

        # 全部完成后保存最终结果
        if len(completed_idx) >= total:
            valid_results = [r for r in results if r is not None]
            valid_results.sort(key=lambda x: x['total_return'], reverse=True)
            self.results = valid_results
            self.save_results()
        else:
            print(f"\n⏸️  暂停: 已完成 {len(completed_idx)}/{total}，下次启动自动续算")

    def save_results(self, extra_suffix=''):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        suffix = f'_{extra_suffix}' if extra_suffix else ''

        # 保存JSON
        json_path = output_dir / f'optimization_results_{timestamp}{suffix}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 JSON结果: {json_path}")

        # 保存CSV
        if self.results:
            df = pd.DataFrame([{
                '总收益%': r.get('total_return'),
                '夏普比率': r.get('sharpe'),
                '最大回撤%': r.get('max_drawdown'),
                '胜率%': r.get('win_rate'),
                '交易次数': r.get('trade_count'),
                **{f'param_{k}': v for k, v in r['params'].items()}
            } for r in self.results])
            csv_path = output_dir / f'optimization_results_{timestamp}{suffix}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"💾 CSV结果: {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='策略A参数穷举优化')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'TSLA'])
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--top', type=int, default=20, help='输出Top N结果')
    parser.add_argument('--quick', action='store_true', help='快速模式(减少参数组合)')
    args = parser.parse_args()

    print(f"📊 标的: {args.symbols}")
    optimizer = ParameterOptimizer(args.symbols, args.start, args.end)

    if args.quick:
        # 快速模式 - 精简参数组合
        param_grid = {
            'rsi_buy': [25, 30, 35],
            'rsi_sell': [65, 70, 75],
            'macd_fast': [10, 12],
            'macd_slow': [24, 26],
            'macd_signal': [8, 9],
            'bb_period': [18, 20, 22],
            'bb_std': [1.5, 2.0, 2.5],
            'vol_period': [15, 20],
            'vol_multiplier': [1.2, 1.5],
            'breakout_period': [18, 20],
            'stop_loss_pct': [2, 3, 5],
            'take_profit_pct': [8, 10],
            'min_conditions': [2, 3, 4],
            'trail_activation_pct': [0.05, 0.08, 0.10, 0.15],   # 移动止损激活涨幅（5%/8%/10%/15%）
            'trail_distance_pct': [0.08, 0.10, 0.12, 0.15],       # 移动止损距离（保留8%~15%利润）
        }
        total = 3*3*2*2*2*3*3*2*2*2*3*2*3
        print(f"⚡ 快速模式: {total:,} 组合")
    else:
        # 完整模式
        param_grid = {
            'rsi_buy': [20, 25, 30],
            'rsi_sell': [60, 65, 70, 75],
            'macd_fast': [10, 12],
            'macd_slow': [24, 26],
            'macd_signal': [8, 9],
            'bb_period': [18, 20],
            'bb_std': [1.5, 2.0, 2.5],
            'vol_period': [15, 20],
            'vol_multiplier': [1.2, 1.5, 2.0],
            'breakout_period': [18, 20],
            'stop_loss_pct': [2, 3, 5],
            'take_profit_pct': [8, 10, 15],
            'min_conditions': [2, 3, 4],
            'trail_activation_pct': [0.05, 0.08, 0.10, 0.15],  # 移动止损激活涨幅
            'trail_distance_pct': [0.10, 0.15],                  # 移动止损距离
        }
        total = 3*4*2*2*2*2*3*2*3*2*3*3*3*4*2
        print(f"🔬 完整模式: {total:,} 组合")

    optimizer.grid_search(param_grid, top_n=args.top)


if __name__ == "__main__":
    main()
