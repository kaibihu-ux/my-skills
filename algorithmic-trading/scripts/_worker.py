"""
打板策略并行Worker
- 每个进程独立初始化 pybroker 数据源
- 接收 (params, symbols, start_date, end_date) 执行单次回测
- 避免跨进程的 pybroker 全局状态冲突
"""

import os
import sys
from pathlib import Path

# 全局变量：每个worker进程独立初始化
_worker_data_source = None


def _init_worker(db_path, symbols, start_date, end_date):
    """每个worker进程启动时初始化自己的数据源"""
    global _worker_data_source

    skill_dir = Path(__file__).parent.parent
    scripts_dir = skill_dir / 'scripts'

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "astock_source",
        scripts_dir / "astock_source.py"
    )
    astock_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(astock_module)
    AStockDataSource = astock_module.AStockDataSource

    _worker_data_source = AStockDataSource(str(db_path))


def _run_single_backtest(args):
    """
    在worker进程中运行单次回测。
    必须作为顶层函数（模块级别），才能被 ProcessPoolExecutor pickle。
    """
    params, symbols, start_date, end_date = args
    global _worker_data_source

    try:
        import pandas as pd
        from pybroker import Strategy, StrategyConfig
        from pybroker.indicator import indicator
        from pybroker import highest

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

        # 定义指标
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

        highest_fn = highest('high_20', 'high', period=breakout_period)

        def exec_fn(ctx):
            rsi = ctx.indicator('rsi_indicator')
            macd_line = ctx.indicator('macd_line')
            macd_sig = ctx.indicator('macd_signal')
            bb_lower = ctx.indicator('bb_lower')
            vol_ma = ctx.indicator('vol_indicator')
            high_20 = ctx.indicator('high_20')

            cond_rsi_oversold = rsi[-1] < rsi_buy if len(rsi) > 0 else False
            cond_macd_crossup = (len(macd_line) > 1 and
                macd_line[-1] > macd_sig[-1] and macd_line[-2] <= macd_sig[-2])
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
        strategy = Strategy(_worker_data_source, start_date, end_date, config)
        strategy.add_execution(
            exec_fn,
            symbols,
            indicators=[rsi_ind, macd_line_ind, macd_signal_ind, bb_lower_ind, vol_ind, highest_fn]
        )
        warmup = max(rsi_buy, macd_slow, bb_period, vol_period, breakout_period)
        result = strategy.backtest(warmup=warmup)
        metrics = result.metrics_df
        if metrics.empty:
            return None

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
        return None
