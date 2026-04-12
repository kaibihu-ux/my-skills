#!/usr/bin/env python3
"""
全量优化脚本 - 支持所有股票、所有参数组合
分批处理，支持断点续传

使用方法:
    python3 scripts/full_batch_optimizer.py --batch-size 50 --stocks-per-run 100
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pybroker import Strategy, StrategyConfig
from pybroker.indicator import indicator
import pandas as pd
import itertools
import importlib.util


class FullBatchOptimizer:
    """
    全量分批优化器

    策略：分批处理股票，每批使用相同的参数网格搜索
    """

    def __init__(self, db_path: str, start_date: str, end_date: str, output_dir: str = 'output'):
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 状态文件
        self.state_file = self.output_dir / 'full_optimization_state.json'

        # 初始化数据源
        spec = importlib.util.spec_from_file_location("astock_source", Path(__file__).parent / "astock_source.py")
        astock_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(astock_module)
        self.AStockDataSource = astock_module.AStockDataSource

        # 获取所有股票
        self.all_stocks = self._get_all_stocks()
        print(f"📊 数据库中股票总数: {len(self.all_stocks)}")

    def _get_all_stocks(self):
        """获取所有股票列表"""
        import duckdb
        conn = duckdb.connect(self.db_path, read_only=True)
        stocks = conn.execute("SELECT ts_code FROM stock_list").fetchall()
        conn.close()
        return [s[0] for s in stocks]

    def _get_parameter_grid(self):
        """完整的参数网格"""
        return {
            # RSI 参数
            'rsi_buy': [20, 25, 30, 35, 40],
            'rsi_sell': [60, 65, 70, 75, 80],
            # MACD 参数
            'macd_fast': [8, 10, 12, 15],
            'macd_slow': [20, 24, 26, 30],
            'macd_signal': [7, 9, 11],
            # 布林带参数
            'bb_period': [15, 18, 20, 22, 25],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            # 成交量参数
            'vol_period': [10, 15, 20, 25],
            'vol_multiplier': [1.0, 1.3, 1.5, 2.0],
            # 突破参数
            'breakout_period': [15, 18, 20, 25],
            # 风控参数
            'stop_loss_pct': [2, 3, 5, 7],
            'take_profit_pct': [5, 8, 10, 15],
            # 条件数
            'min_conditions': [2, 3],
        }

    def _calculate_total_combinations(self, param_grid):
        """计算总组合数"""
        total = 1
        for k, v in param_grid.items():
            total *= len(v)
        return total

    def _create_strategy(self, params, data_source):
        """创建策略"""
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
        min_conditions = params.get('min_conditions', 3)

        # RSI指标
        def rsi_func(data):
            delta = pd.Series(data.close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).to_numpy()
        rsi_ind = indicator('rsi', rsi_func)

        # MACD指标
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

        # 布林带指标
        def bb_lower_func(data):
            close = pd.Series(data.close)
            sma = close.rolling(window=bb_period).mean()
            std = close.rolling(window=bb_period).std()
            return (sma - bb_std * std).to_numpy()
        bb_lower_ind = indicator('bb_lower', bb_lower_func)

        # 成交量指标
        def vol_func(data):
            return pd.Series(data.volume).rolling(window=vol_period).mean().to_numpy()
        vol_ind = indicator('vol_indicator', vol_func)

        # 最高价突破
        from pybroker import highest
        highest_fn = highest('high_20', 'high', period=breakout_period)

        # 执行函数
        def exec_fn(ctx):
            rsi = ctx.indicator('rsi')
            macd_line = ctx.indicator('macd_line')
            macd_signal_val = ctx.indicator('macd_signal')
            bb_lower = ctx.indicator('bb_lower')
            vol_ma = ctx.indicator('vol_indicator')
            high_20 = ctx.indicator('high_20')

            cond_rsi_oversold = rsi[-1] < rsi_buy if len(rsi) > 0 else False
            cond_macd_crossup = (len(macd_line) > 1 and
                macd_line[-1] > macd_signal_val[-1] and macd_line[-2] <= macd_signal_val[-2])
            cond_bb_lower = ctx.close[-1] <= bb_lower[-1] if len(bb_lower) > 0 else False
            cond_vol_surge = ctx.volume[-1] > vol_ma[-1] * vol_multiplier if len(vol_ma) > 0 else False
            cond_breakout = high_20[-1] > high_20[-2] if len(high_20) > 1 else False

            conditions_met = sum([cond_rsi_oversold, cond_macd_crossup,
                                   cond_bb_lower, cond_vol_surge, cond_breakout])

            buy_signal = conditions_met >= min_conditions
            sell_signal = rsi[-1] > rsi_sell if len(rsi) > 0 else False

            if buy_signal and not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = stop_loss_pct
                ctx.hold_bars = 20
            elif sell_signal and ctx.long_pos():
                ctx.sell_all_shares()

        config = StrategyConfig(initial_cash=100000)
        strategy = Strategy(data_source, self.start_date, self.end_date, config)

        strategy.add_execution(
            exec_fn,
            [],  # symbols will be set later
            indicators=[rsi_ind, macd_line_ind, macd_signal_ind, bb_lower_ind, vol_ind, highest_fn]
        )

        return strategy

    def _get_metric(self, metrics, name):
        """提取指标"""
        try:
            val = metrics.loc[metrics['name'] == name, 'value'].iloc[0]
            if isinstance(val, str):
                return float(val.replace('%', ''))
            return float(val)
        except:
            return 0.0

    def run(self, batch_size: int = 50, max_stocks: int = None, resume: bool = True):
        """
        运行全量优化

        Args:
            batch_size: 每批处理的股票数量
            max_stocks: 最多使用的股票数量（None=全部）
            resume: 是否从断点继续
        """
        param_grid = self._get_parameter_grid()
        total_combinations = self._calculate_total_combinations(param_grid)

        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combinations = list(itertools.product(*values))
        param_list = [dict(zip(keys, combo)) for combo in all_combinations]

        stocks_to_use = self.all_stocks[:max_stocks] if max_stocks else self.all_stocks
        total_stocks = len(stocks_to_use)

        print("=" * 70)
        print("🚀 全量优化开始")
        print("=" * 70)
        print(f"📊 股票总数: {total_stocks}")
        print(f"📊 参数组合: {total_combinations:,}")
        print(f"📊 时间范围: {self.start_date} ~ {self.end_date}")
        print(f"📊 每批大小: {batch_size}")
        print(f"⏱️ 预计时间: {(total_stocks * total_combinations * 2) / 3600:.1f} 小时")
        print("=" * 70)

        # 加载或初始化状态
        if resume and self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            start_batch = state.get('completed_batches', 0)
            all_results = state.get('results', [])
            print(f"📂 从断点恢复，已完成 {start_batch} 批")
        else:
            start_batch = 0
            all_results = []

        # 分批处理
        for batch_idx in range(start_batch, len(stocks_to_use), batch_size):
            batch_stocks = stocks_to_use[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size
            total_batches = (len(stocks_to_use) + batch_size - 1) // batch_size

            print(f"\n{'='*70}")
            print(f"📦 批次 {current_batch + 1}/{total_batches}")
            print(f"📊 股票: {batch_stocks}")
            print(f"📊 数量: {len(batch_stocks)} 只")
            print(f"{'='*70}")

            # 初始化数据源
            data_source = self.AStockDataSource(self.db_path)

            # 测试数据可用性
            test_df = data_source.fetch_bars(batch_stocks[:1], self.start_date, self.end_date)
            if test_df.empty:
                print(f"⚠️ 批次 {current_batch + 1} 数据为空，跳过")
                continue

            # 对每个参数组合运行回测
            batch_results = []
            for i, params in enumerate(param_list):
                if i % 100 == 0:
                    elapsed = i * len(batch_stocks) * 2  # 粗略估算
                    eta = (total_combinations - i) * len(batch_stocks) * 2
                    print(f"  参数进度: {i}/{total_combinations} (ETA: {eta//60}分钟)")

                try:
                    strategy = self._create_strategy(params, data_source)
                    warmup = max(params['rsi_buy'], params['macd_slow'],
                                 params['bb_period'], params['vol_period'], params['breakout_period'])
                    result = strategy.backtest(warmup=warmup)
                    metrics = result.metrics_df

                    if not metrics.empty:
                        metrics_dict = {
                            'params': params,
                            'total_return': self._get_metric(metrics, 'total_return_pct'),
                            'sharpe': self._get_metric(metrics, 'sharpe'),
                            'max_drawdown': self._get_metric(metrics, 'max_drawdown_pct'),
                            'win_rate': self._get_metric(metrics, 'win_rate'),
                            'trade_count': int(self._get_metric(metrics, 'trade_count')),
                            'stocks': batch_stocks,
                        }
                        batch_results.append(metrics_dict)
                except Exception as e:
                    pass  # 跳过失败的组合

            # 合并批次结果
            all_results.extend(batch_results)

            # 保存断点
            state = {
                'completed_batches': current_batch + 1,
                'results': all_results,
                'last_update': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)

            print(f"  ✅ 批次 {current_batch + 1} 完成，有效结果: {len(batch_results)}")

            data_source.close()

        # 最终汇总
        print("\n" + "=" * 70)
        print("🏆 全量优化完成!")
        print("=" * 70)

        # 按收益排序
        all_results.sort(key=lambda x: x['total_return'], reverse=True)

        # 保存最终结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f'full_optimization_{timestamp}.json'
        csv_path = self.output_dir / f'full_optimization_{timestamp}.csv'

        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        if all_results:
            df = pd.DataFrame([{
                '总收益%': r['total_return'],
                '夏普比率': r['sharpe'],
                '最大回撤%': r['max_drawdown'],
                '胜率%': r['win_rate'],
                '交易次数': r['trade_count'],
                '涉及股票': ','.join(r['stocks'][:5]) + ('...' if len(r['stocks']) > 5 else ''),
                **{f'param_{k}': v for k, v in r['params'].items()}
            } for r in all_results])
            df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"💾 结果已保存:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"📊 总有效结果: {len(all_results)}")

        # Top 20
        print(f"\n🏆 Top 20 最优组合:")
        for i, r in enumerate(all_results[:20], 1):
            p = r['params']
            print(f"#{i} 收益:{r['total_return']:+.2f}% 夏普:{r['sharpe']:.3f} "
                  f"RSI:({p['rsi_buy']},{p['rsi_sell']}) 条件≥{p['min_conditions']}/5")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='全量优化脚本')
    parser.add_argument('--batch-size', type=int, default=30, help='每批股票数量')
    parser.add_argument('--max-stocks', type=int, default=None, help='最多使用的股票数')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2026-03-31', help='结束日期')
    parser.add_argument('--no-resume', action='store_true', help='不复位继续')
    args = parser.parse_args()

    db_path = os.path.expanduser('~/.openclaw/skills/algorithmic-trading/data/astock_full.duckdb')

    optimizer = FullBatchOptimizer(
        db_path=db_path,
        start_date=args.start,
        end_date=args.end,
        output_dir='output'
    )

    optimizer.run(
        batch_size=args.batch_size,
        max_stocks=args.max_stocks,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
