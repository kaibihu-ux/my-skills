"""
PyBroker Backtest - 算法交易回测工具
支持本地DuckDB数据源（A股数据）
"""

from pybroker import Strategy, highest, lowest, StrategyConfig, param
from pybroker.indicator import indicator
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import os

def get_data_source(use_local=True):
    """获取数据源"""
    if use_local:
        # 直接导入同一目录下的模块
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
            return AStockDataSource(str(db_path))
        else:
            print(f"⚠️ 本地数据不存在: {db_path}")
            print("   切换到YFinance...")
            from pybroker import YFinance
            return YFinance()
    else:
        from pybroker import YFinance
        return YFinance()

class BacktestRunner:
    """回测运行器"""

    def __init__(self, cache_dir='cache', use_local=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_local = use_local
        self.data_source = None

    def _get_data_source(self):
        if self.data_source is None:
            self.data_source = get_data_source(self.use_local)
        return self.data_source

    def run_basic_strategy(self, symbols, start_date, end_date, initial_cash=100000):
        """运行基础突破策略"""

        def exec_fn(ctx):
            high_20 = ctx.indicator('high_20')

            if not ctx.long_pos() and high_20[-1] > high_20[-2]:
                ctx.buy_shares = 100
                ctx.hold_bars = 5
                ctx.stop_loss_pct = 2

        config = StrategyConfig(initial_cash=initial_cash)
        strategy = Strategy(self._get_data_source(), start_date, end_date, config)

        strategy.add_execution(
            exec_fn,
            symbols,
            indicators=highest('high_20', 'high', period=20)
        )

        result = strategy.backtest(warmup=20)
        return result
    
    def run_rsi_strategy(self, symbols, start_date, end_date, initial_cash=100000):
        """运行 RSI 策略"""

        def rsi_fn(data):
            delta = pd.Series(data.close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).to_numpy()

        rsi_indicator = indicator('rsi', rsi_fn)

        def exec_fn(ctx):
            rsi = ctx.indicator('rsi')
            
            if rsi[-1] < 30 and not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 3
            elif rsi[-1] > 70 and ctx.long_pos():
                ctx.sell_all_shares()
        
        config = StrategyConfig(initial_cash=initial_cash)
        strategy = Strategy(self._get_data_source(), start_date, end_date, config)

        strategy.add_execution(exec_fn, symbols, indicators=rsi_indicator)

        result = strategy.backtest(warmup=20)
        return result

    def run_comprehensive_strategy(self, symbols, start_date, end_date, initial_cash=100000,
                                    rsi_buy=30, rsi_sell=70,
                                    macd_fast=12, macd_slow=26, macd_signal=9,
                                    bb_period=20, bb_std=2,
                                    vol_period=20, vol_multiplier=1.5,
                                    breakout_period=20,
                                    stop_loss_pct=3, take_profit_pct=10,
                                    min_conditions=3):
        """
        综合策略A：结合 RSI + MACD + 布林带 + 成交量 + 20日突破

        买入条件（min_conditions个满足即可，默认为3/5）：
        1. RSI < rsi_buy（超卖）
        2. MACD 金叉（MACD线上穿信号线）
        3. 价格触及布林带下轨（强势超卖信号）
        4. 成交量 > vol_period日均量 * vol_multiplier（放量确认）
        5. 20日最高价突破（趋势启动确认）

        卖出条件：
        - RSI > rsi_sell（超买）
        - 或触发止盈/止损
        """
        def rsi_func(data):
            delta = pd.Series(data.close).diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).to_numpy()
        rsi_indicator = indicator('rsi_indicator', rsi_func)

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

        def bb_sma_func(data):
            return pd.Series(data.close).rolling(window=bb_period).mean().to_numpy()
        bb_sma_ind = indicator('bb_sma', bb_sma_func)

        def bb_upper_func(data):
            close = pd.Series(data.close)
            sma = close.rolling(window=bb_period).mean()
            std = close.rolling(window=bb_period).std()
            return (sma + bb_std * std).to_numpy()
        bb_upper_ind = indicator('bb_upper', bb_upper_func)

        def vol_func(data):
            return pd.Series(data.volume).rolling(window=vol_period).mean().to_numpy()
        vol_indicator = indicator('vol_indicator', vol_func)

        def exec_fn(ctx):
            rsi = ctx.indicator('rsi_indicator')
            macd_line = ctx.indicator('macd_line')
            macd_signal = ctx.indicator('macd_signal')
            bb_lower = ctx.indicator('bb_lower')
            vol_ma = ctx.indicator('vol_indicator')
            high_20 = ctx.indicator('high_20')

            # 计算各条件满足情况
            cond_rsi_oversold = rsi[-1] < rsi_buy if len(rsi) > 0 else False
            cond_macd_crossup = (len(macd_line) > 1 and
                                 macd_line[-1] > macd_signal[-1] and macd_line[-2] <= macd_signal[-2])
            cond_bb_lower = ctx.close[-1] <= bb_lower[-1] if len(bb_lower) > 0 else False
            cond_vol_surge = ctx.volume[-1] > vol_ma[-1] * vol_multiplier if len(vol_ma) > 0 else False
            cond_breakout = high_20[-1] > high_20[-2] if len(high_20) > 1 else False

            # 统计满足的条件数量
            conditions_met = sum([cond_rsi_oversold, cond_macd_crossup,
                                   cond_bb_lower, cond_vol_surge, cond_breakout])

            buy_signal = conditions_met >= min_conditions

            # 卖出条件
            sell_signal = rsi[-1] > rsi_sell if len(rsi) > 0 else False

            if buy_signal and not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = stop_loss_pct
                ctx.hold_bars = 20

            elif sell_signal and ctx.long_pos():
                ctx.sell_all_shares()

        config = StrategyConfig(initial_cash=initial_cash)
        strategy = Strategy(self._get_data_source(), start_date, end_date, config)

        strategy.add_execution(
            exec_fn,
            symbols,
            indicators=[
                rsi_indicator,
                macd_line_ind,
                macd_signal_ind,
                bb_lower_ind,
                vol_indicator,
                highest('high_20', 'high', period=breakout_period)
            ]
        )

        result = strategy.backtest(warmup=max(rsi_buy, macd_slow, bb_period, vol_period, breakout_period))
        return result

    def print_metrics(self, result):
        """打印回测指标"""
        metrics = result.metrics_df

        print("\n" + "="*60)
        print("回测结果")
        print("="*60)

        if not metrics.empty:
            for col in metrics.columns:
                print(f"{col}: {metrics[col].iloc[0]}")

        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="PyBroker Backtest")
    parser.add_argument('--strategy', choices=['basic', 'rsi', 'comprehensive'], default='comprehensive')
    parser.add_argument('--symbols', nargs='+', default=['AAPL'])
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2023-12-31')
    parser.add_argument('--cash', type=float, default=100000)
    parser.add_argument('--report', help='输出报告路径')
    parser.add_argument('--rsi-buy', type=float, default=30, help='RSI 超卖阈值（默认30）')
    parser.add_argument('--rsi-sell', type=float, default=70, help='RSI 超买阈值（默认70）')
    parser.add_argument('--vol-mult', type=float, default=1.5, help='成交量倍数（默认1.5）')
    parser.add_argument('--stop-loss', type=float, default=3, help='止损百分比（默认3%%）')
    parser.add_argument('--min-conditions', type=int, default=3, help='最少满足条件数（默认3/5）')

    args = parser.parse_args()

    runner = BacktestRunner()

    print(f"[运行] 策略: {args.strategy}")
    print(f"[运行] 标的: {args.symbols}")
    print(f"[运行] 时间: {args.start} - {args.end}")

    if args.strategy == 'basic':
        result = runner.run_basic_strategy(
            args.symbols, args.start, args.end, args.cash
        )
    elif args.strategy == 'rsi':
        result = runner.run_rsi_strategy(
            args.symbols, args.start, args.end, args.cash
        )
    elif args.strategy == 'comprehensive':
        result = runner.run_comprehensive_strategy(
            args.symbols, args.start, args.end, args.cash,
            rsi_buy=args.rsi_buy, rsi_sell=args.rsi_sell,
            vol_multiplier=args.vol_mult,
            stop_loss_pct=args.stop_loss,
            min_conditions=args.min_conditions
        )
        print(f"[参数] RSI超卖:<{args.rsi_buy} | RSI超买:>{args.rsi_sell}")
        print(f"[参数] 成交量放大: >{args.vol_mult}x | 止损:{args.stop_loss}% | 条件数≥{args.min_conditions}/5")

    runner.print_metrics(result)

    if args.report:
        print(f"[保存] 报告: {args.report}")

if __name__ == "__main__":
    main()
