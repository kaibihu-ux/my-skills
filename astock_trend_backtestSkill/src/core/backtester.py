import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor


class BacktestExecutor:
    """回测执行器"""
    
    def __init__(self, store, logger, config: Dict):
        self.store = store
        self.logger = logger
        self.config = config
        self.initial_cash = config['backtest']['initial_cash']
        self.commission = config['backtest']['commission']
        self.slippage = config['backtest']['slippage']
        self.top_n = config['backtest'].get('top_n_stocks', 20)
        
        # 初始化时计算并存储基础因子
        self._ensure_factors_stored()
        # 初始化中性化器
        self._init_neutralizer()
        # 数据预加载缓存（由 preload_data 填充，避免重复 SQL 查询）
        self._price_cache = {}   # {(ts_code, trade_date): close_price}
        self._factor_cache = {}  # {(factor_name, ts_code, trade_date): value}
        self._price_by_date = {} # {trade_date: {ts_code: close_price}}  # 日期索引，加速 _get_signals
        self._factor_by_date = {}  # {trade_date: {factor_name: {ts_code: value}}}  # 因子缓存
        self._avg_vol_cache = {}  # {(ts_code, trade_date): avg_vol_20}  # 日均成交量缓存（用于中性化市值估算）
        self._industry_cache = {}  # {ts_code: industry}  # 行业缓存（静态数据，初始化时加载一次）

    def _init_neutralizer(self):
        """初始化因子中性化器"""
        self._neutralizer = None
        ncfg = self.config.get('neutralization', {})
        if ncfg.get('enabled', False):
            try:
                from src.core.neutralizer import Neutralizer
                self._neutralizer = Neutralizer(self.store, self.logger, self.config)
                self.logger.info("[Backtester] 中性化器已启用")
            except ImportError:
                self.logger.warning("[Backtester] 无法导入 Neutralizer")

    def preload_data(self, start_date: str, end_date: str, factor_names: list = None):
        """
        预加载回测期间所有股票数据到内存，大幅加速多次回测。

        将 DuckDB 中的 stock_daily 和 factors 表数据一次性加载到内存缓存，
        后续 _get_close_price、_get_signals 等方法全部走内存查表，
        避免 240 组合 × 539 天 的重复 SQL 查询。
        
        factor_names: 可选，指定要加载的因子名称列表。如果为 None，则加载所有因子。
        """
        import time
        t0 = time.time()
        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)

        # 清空旧缓存
        self._price_cache.clear()
        self._price_by_date.clear()
        self._factor_cache.clear()
        self._factor_by_date.clear()
        self._avg_vol_cache.clear()

        # ---- 加载行业数据（静态数据，初始化一次）----
        try:
            industry_df = self.store.df(
                "SELECT ts_code, industry FROM stock_list WHERE industry IS NOT NULL AND industry != ''"
            )
            self._industry_cache = {}
            for _, row in industry_df.iterrows():
                self._industry_cache[str(row['ts_code'])] = str(row['industry'])
        except Exception:
            self._industry_cache = {}

        # ---- 加载价格数据 ----
        price_df = self.store.df(
            f"SELECT ts_code, trade_date, close FROM stock_daily "
            f"WHERE trade_date >= '{start_fmt}' AND trade_date <= '{end_fmt}' AND close > 0"
        )
        for _, row in price_df.iterrows():
            ts_code = row['ts_code']
            trade_date = str(row['trade_date'])
            close = float(row['close'])
            self._price_cache[(ts_code, trade_date)] = close
            if trade_date not in self._price_by_date:
                self._price_by_date[trade_date] = {}
            self._price_by_date[trade_date][ts_code] = close

        # ---- 加载日均成交量数据（用于市值估算）----
        try:
            avg_vol_df = self.store.df(f"""
                SELECT ts_code, trade_date,
                       AVG(vol) OVER (
                           PARTITION BY ts_code
                           ORDER BY trade_date
                           ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                       ) as avg_vol_20
                FROM stock_daily
                WHERE trade_date >= '{start_fmt}' AND trade_date <= '{end_fmt}' AND vol > 0
            """)
            for _, row in avg_vol_df.iterrows():
                ts_code = row['ts_code']
                trade_date = str(row['trade_date'])
                avg_vol = float(row['avg_vol_20'])
                self._avg_vol_cache[(ts_code, trade_date)] = avg_vol
        except Exception as e:
            self.logger.warning(f"[Preload] 日均成交量加载失败: {e}")

        # ---- 加载因子数据（可指定因子列表）----
        if factor_names and len(factor_names) > 0:
            factor_list = "', '".join(factor_names)
            factor_filter = f"AND factor_name IN ('{factor_list}')"
        else:
            factor_filter = ""
        factor_df = self.store.df(
            f"SELECT factor_name, ts_code, trade_date, value FROM factors "
            f"WHERE trade_date >= '{start_fmt}' AND trade_date <= '{end_fmt}' AND value IS NOT NULL {factor_filter}"
        )
        for _, row in factor_df.iterrows():
            factor_name = row['factor_name']
            ts_code = row['ts_code']
            trade_date = str(row['trade_date'])
            value = float(row['value'])
            self._factor_cache[(factor_name, ts_code, trade_date)] = value
            if trade_date not in self._factor_by_date:
                self._factor_by_date[trade_date] = {}
            if factor_name not in self._factor_by_date[trade_date]:
                self._factor_by_date[trade_date][factor_name] = {}
            self._factor_by_date[trade_date][factor_name][ts_code] = value

        elapsed = time.time() - t0
        self.logger.info(
            f"[Preload] 价格: {len(self._price_cache):,} 条, "
            f"因子: {len(self._factor_cache):,} 条, "
            f"预加载耗时: {elapsed:.1f}s"
        )

    def _has_cached_data(self) -> bool:
        """判断缓存是否已加载"""
        return len(self._price_cache) > 0

    def _ensure_factors_stored(self):
        """确保基础因子已存储到 factors 表"""
        # 检查 factors 表是否已有 momentum_20 数据
        try:
            count = self.store.df(
                "SELECT COUNT(*) as cnt FROM factors WHERE factor_name = 'momentum_20'"
            ).iloc[0]['cnt']
            if count > 0:
                self.logger.info(f"因子已存在 (momentum_20: {count} 条)，跳过计算")
                return
        except Exception:
            pass
        
        self.logger.info("开始批量计算并存储基础因子...")
        self._batch_calculate_and_store_factors()
    
    def _batch_calculate_and_store_factors(self):
        """批量计算并存储基础因子到 factors 表（SQL 高效模式）"""
        import time
        start_time = time.time()
        
        conn = self.store.conn
        
        # 检查是否已有因子数据
        try:
            count = self.store.df(
                "SELECT COUNT(*) as cnt FROM factors WHERE factor_name = 'momentum_20'"
            ).iloc[0]['cnt']
            if count > 10000:
                self.logger.info(f"因子已存在 (momentum_20: {count} 条)，跳过计算")
                return
            else:
                # 清空旧数据重新计算
                conn.execute("DELETE FROM factors WHERE factor_name = 'momentum_20'")
                conn.execute("DELETE FROM factors WHERE factor_name LIKE 'momentum_%'")
                conn.execute("DELETE FROM factors WHERE factor_name = 'volatility_20'")
                conn.execute("DELETE FROM factors WHERE factor_name = 'volume_ratio_20'")
                self.logger.info("已清空旧因子数据，重新计算")
        except Exception:
            pass
        
        self.logger.info("开始使用 SQL 计算基础因子...")
        
        # ===== 1. 计算动量因子 (SQL 窗口函数，使用 CTE) =====
        for period in [5, 10, 20, 60]:
            sql = f"""
            WITH price_lagged AS (
                SELECT 
                    ts_code,
                    trade_date,
                    close,
                    LAG(close, {period}) OVER (
                        PARTITION BY ts_code 
                        ORDER BY trade_date
                    ) as lag_close
                FROM stock_daily
            )
            INSERT INTO factors (factor_name, ts_code, trade_date, value)
            SELECT 
                'momentum_{period}' as factor_name,
                ts_code,
                trade_date,
                (close - lag_close) / lag_close as value
            FROM price_lagged
            WHERE lag_close IS NOT NULL 
              AND lag_close > 0
              AND ABS((close - lag_close) / lag_close) < 10
            """
            try:
                conn.execute(sql)
                self.logger.info(f"momentum_{period} 计算完成")
            except Exception as e:
                self.logger.error(f"momentum_{period} 计算失败: {e}")
        
        # ===== 2. 计算波动率因子 =====
        sql_vol = """
        WITH daily_returns_cte AS (
            SELECT 
                ts_code,
                trade_date,
                close,
                (close - LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date)) 
                    / LAG(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as daily_return
            FROM stock_daily
        ),
        volatility_cte AS (
            SELECT
                ts_code,
                trade_date,
                STDDEV_POP(daily_return) OVER (
                    PARTITION BY ts_code 
                    ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as std_dev
            FROM daily_returns_cte
        )
        INSERT INTO factors (factor_name, ts_code, trade_date, value)
        SELECT 
            'volatility_20' as factor_name,
            ts_code,
            trade_date,
            std_dev as value
        FROM volatility_cte
        WHERE std_dev IS NOT NULL AND std_dev < 5
        """
        try:
            conn.execute(sql_vol)
            self.logger.info("volatility_20 计算完成")
        except Exception as e:
            self.logger.error(f"volatility_20 计算失败: {e}")
        
        # ===== 3. 计算量比因子 =====
        sql_vol_ratio = """
        INSERT INTO factors (factor_name, ts_code, trade_date, value)
        SELECT 
            'volume_ratio_20' as factor_name,
            ts_code,
            trade_date,
            vol / avg_vol_20 as value
        FROM (
            SELECT 
                ts_code,
                trade_date,
                vol,
                AVG(vol) OVER (
                    PARTITION BY ts_code 
                    ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) as avg_vol_20
            FROM stock_daily
        ) t
        WHERE avg_vol_20 IS NOT NULL AND avg_vol_20 > 0 AND vol / avg_vol_20 < 100
        """
        try:
            conn.execute(sql_vol_ratio)
            self.logger.info("volume_ratio_20 计算完成")
        except Exception as e:
            self.logger.error(f"volume_ratio_20 计算失败: {e}")
        
        # 统计
        try:
            total = self.store.df("SELECT COUNT(*) as cnt FROM factors").iloc[0]['cnt']
            self.logger.info(f"因子计算完成，共 {total} 条记录，耗时 {time.time() - start_time:.1f}s")
        except Exception:
            pass
    
    def run(self, strategy: Dict, params: Dict, start_date: str, end_date: str) -> Dict:
        """运行回测"""
        # 转换日期格式
        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)
        
        cash = self.initial_cash
        positions = {}  # {ts_code: {'shares': int, 'cost': float, 'entry_date': str}}
        trades = []
        nav_history = []
        
        # 维护当前持仓的动量排名
        current_holdings = {}  # {ts_code: momentum_rank at entry}
        
        # 获取回测日期列表
        dates = self._get_trade_dates(start_fmt, end_fmt)
        
        # 每年1月的第一个交易日
        rebalance_dates = set()
        prev_year = None
        for d in dates:
            date_str = str(d)[:10]  # 确保格式为 YYYY-MM-DD
            year = date_str[:4]
            if year != prev_year:
                rebalance_dates.add(date_str)
                prev_year = year
        
        self.logger.info(f"回测期: {start_fmt} ~ {end_fmt}, 共 {len(dates)} 个交易日")
        self.logger.info(f"再平衡日期: {sorted(rebalance_dates)}")
        
        for i, date in enumerate(dates):
            # 获取当日信号
            signals = self._get_signals(date, strategy, params, positions, current_holdings, rebalance_dates)

            # 止损/止盈/排名淘汰/得分下滑 已在 _get_signals 中统一处理

            # 仓位权重逻辑（用于buy信号分配）
            target_weights = {}
            weight_scheme = params.get('weight_scheme', 'equal')
            if weight_scheme != 'equal' and positions:
                # 计算当前总市值
                total_pos_value = sum(
                    pos['shares'] * self._get_close_price(ts, date)
                    for ts, pos in positions.items()
                    if self._get_close_price(ts, date) > 0
                )
                portfolio_value = cash + total_pos_value

                if weight_scheme == 'ic_weighted':
                    # 按 IC 分配（直接用动量因子值作为权重，走缓存）
                    for ts_code, pos in positions.items():
                        date_str = str(date)
                        ic_val = self._factor_cache.get(('momentum_20', ts_code, date_str), None)
                        if ic_val is not None:
                            target_weights[ts_code] = max(0, ic_val + 1e-6)
                        else:
                            target_weights[ts_code] = 1.0
                    # 归一化
                    total_w = sum(target_weights.values())
                    if total_w > 0:
                        target_weights = {k: v / total_w for k, v in target_weights.items()}

                elif weight_scheme == 'volatility_inverse':
                    # 按波动率倒数分配，走缓存
                    for ts_code, pos in positions.items():
                        date_str = str(date)
                        vol_val = self._factor_cache.get(('volatility_20', ts_code, date_str), None)
                        if vol_val is not None:
                            target_weights[ts_code] = 1.0 / max(vol_val + 1e-6, 1e-8)
                        else:
                            target_weights[ts_code] = 1.0
                    # 归一化
                    total_w = sum(target_weights.values())
                    if total_w > 0:
                        target_weights = {k: v / total_w for k, v in target_weights.items()}

            # 执行交易
            for signal in signals:
                ts_code = signal['ts_code']
                direction = signal['direction']
                price = signal['price']
                
                if direction == 'buy' and cash > 0:
                    # 计算当前总市值
                    portfolio_value = cash
                    for ts, pos in positions.items():
                        close_p = self._get_close_price(ts, date)
                        if close_p > 0:
                            portfolio_value += pos['shares'] * close_p

                    # 仓位权重分配
                    n_target_stocks = self.top_n
                    if target_weights and ts_code in target_weights:
                        # 按权重分配（该股票应占总权益的比例）
                        target_pct = target_weights[ts_code] / max(1, len(target_weights))
                    else:
                        # 平均分配
                        target_pct = 1.0 / max(1, n_target_stocks)

                    # 风控：单股最大仓位不超过 max_position_per_stock，默认20%
                    max_pos_pct = params.get('max_position_per_stock', 0.2)
                    max_shares_by_pos = int(portfolio_value * max_pos_pct / (price * (1 + self.slippage)))
                    # 按权重分配的目标买入金额
                    target_value = portfolio_value * target_pct
                    max_shares_by_weight = int(target_value / (price * (1 + self.slippage)))
                    # 同时限制现金：最多用 15% 现金
                    max_shares_by_cash = int(cash * 0.15 / (price * (1 + self.slippage)))
                    shares = min(max_shares_by_pos, max_shares_by_weight, max_shares_by_cash)
                    if shares > 0:
                        cost = shares * price * (1 + self.slippage + self.commission)
                        cash -= cost
                        positions[ts_code] = {
                            'shares': shares,
                            'cost': price,
                            'entry_date': date,
                            'entry_nav': cash + sum(
                                positions[p]['shares'] * self._get_close_price(p, date)
                                for p in positions
                            )
                        }
                        trades.append({
                            'trade_id': f"{date}_{ts_code}_buy",
                            'strategy_id': strategy.get('strategy_id', 'default'),
                            'ts_code': ts_code,
                            'trade_date': date,
                            'direction': 'buy',
                            'price': price,
                            'quantity': shares,
                            'amount': cost
                        })
                
                elif direction == 'sell' and ts_code in positions:
                    # 卖出全部
                    pos = positions[ts_code]
                    proceeds = pos['shares'] * price * (1 - self.slippage - self.commission)
                    cash += proceeds
                    trades.append({
                        'trade_id': f"{date}_{ts_code}_sell",
                        'strategy_id': strategy.get('strategy_id', 'default'),
                        'ts_code': ts_code,
                        'trade_date': date,
                        'direction': 'sell',
                        'price': price,
                        'quantity': pos['shares'],
                        'amount': proceeds
                    })
                    del positions[ts_code]
                    if ts_code in current_holdings:
                        del current_holdings[ts_code]
            
            # 计算当日净值
            portfolio_value = cash
            for ts_code, pos in positions.items():
                close_price = self._get_close_price(ts_code, date)
                if close_price > 0:
                    portfolio_value += pos['shares'] * close_price
            
            nav_history.append({'date': date, 'nav': portfolio_value})
        
        return self._calc_metrics(trades, nav_history)

    # ------------------------------------------------------------------
    # 并行回测（8线程）
    # ------------------------------------------------------------------

    def parallel_run(
        self,
        strategy: Dict,
        params_list: List[Dict],
        start_date: str,
        end_date: str,
        max_workers: int = 8,
    ) -> List[Dict]:
        """
        并行运行多个参数组合的回测（使用 ThreadPoolExecutor）
        适用于 Step5 Final 阶段的多参数 Grid Search

        Args:
            strategy: 策略定义（因子列表等）
            params_list: 参数组合列表，如 [{'top_n_stocks': 20, ...}, {...}]
            start_date: 回测开始日期
            end_date: 回测结束日期
            max_workers: 最大并行线程数，默认8

        Returns:
            List[Dict]: 每个参数组合的回测结果
        """
        self.logger.info(
            f"[Backtester] 启动并行回测 | 参数组合数={len(params_list)}, 最大并发={max_workers}"
        )

        def _run_single(params: Dict) -> Dict:
            try:
                result = self.run(
                    strategy=strategy,
                    params=params,
                    start_date=start_date,
                    end_date=end_date,
                )
                result['params'] = params
                return result
            except Exception as e:
                self.logger.warning(f"[Backtester] 参数组合回测失败: {params}, 错误: {e}")
                return {'params': params, 'error': str(e), 'sharpe_ratio': 0.0}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_run_single, params_list))

        self.logger.info(f"[Backtester] 并行回测完成 | 成功={sum(1 for r in results if 'error' not in r)}/{len(results)}")
        return results
    
    def _get_close_price(self, ts_code: str, date: str) -> float:
        """获取指定日期收盘价（优先走缓存）"""
        date_str = str(date)
        if self._has_cached_data():
            return self._price_cache.get((ts_code, date_str), 0.0)
        # 兜底：缓存未加载时查 DuckDB
        df = self.store.df(
            f"SELECT close FROM stock_daily WHERE ts_code = '{ts_code}' AND trade_date = '{date_str}'"
        )
        if len(df) > 0:
            return float(df.iloc[0]['close'])
        return 0.0
    
    def _format_date(self, date_str: str) -> str:
        """统一日期格式：YYYYMMDD -> YYYY-MM-DD"""
        if date_str is None:
            return ''
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s  # 已经是 YYYY-MM-DD 格式则直接返回
    
    def _get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期列表"""
        df = self.store.df(
            f"SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date BETWEEN '{start_date}' AND '{end_date}' ORDER BY trade_date"
        )
        return df['trade_date'].astype(str).tolist()

    def _apply_factor_neutralization(
        self,
        merged: pd.DataFrame,
        factor_names: List[str],
        date: str,
    ) -> pd.DataFrame:
        """
        对因子值进行中性化处理（使用预加载缓存，无重复 SQL 查询）
        """
        try:
            date_str = str(date)

            # 使用预加载的收盘价和日均成交量估算市值代理：close × avg_vol_20
            if date_str in self._price_by_date:
                price_map = self._price_by_date[date_str]
                mcap_list = []
                for ts_code in merged['ts_code'].values:
                    close = price_map.get(ts_code, 0.0)
                    avg_vol = self._avg_vol_cache.get((ts_code, date_str), 0.0)
                    # 市值代理 = close × avg_vol_20（价量乘积作为流动性的代理）
                    mcap_list.append(close * avg_vol if close > 0 and avg_vol > 0 else 1.0)
            else:
                # 兜底：全部用 1.0
                mcap_list = [1.0] * len(merged)

            if 'mcap_proxy' not in merged.columns:
                merged['mcap_proxy'] = mcap_list
            else:
                merged['mcap_proxy'] = mcap_list

            # 使用预加载的行业缓存（静态数据，无需每日查询）
            if self._industry_cache:
                industry_list = [self._industry_cache.get(ts, 'Unknown') for ts in merged['ts_code'].values]
            else:
                industry_list = ['Unknown'] * len(merged)

            if 'industry' not in merged.columns:
                merged['industry'] = industry_list
            else:
                merged['industry'] = industry_list

            # 对每个因子执行中性化
            for fname in factor_names:
                if fname not in merged.columns:
                    continue

                factor_vals = merged[fname].values
                mcap_vals = merged['mcap_proxy'].fillna(1.0).values
                industry_vals = merged['industry'].values

                neutralized = self._neutralizer.neutralize(
                    factor_vals,
                    mcap_vals,
                    industry_vals,
                    method='winsorize',
                )
                merged[fname] = neutralized

            # 删除辅助列
            merged = merged.drop(columns=['mcap_proxy'], errors='ignore')

        except Exception as e:
            self.logger.warning(f"因子中性化失败: {e}")

        return merged

    def _get_signals(
        self, 
        date: str, 
        strategy: Dict, 
        params: Dict,
        positions: Dict,
        current_holdings: Dict,
        rebalance_dates: set
    ) -> List[Dict]:
        """
        获取当日信号 - 多因子量化策略，全市场选股

        策略逻辑：
        1. 按调仓频率（daily/weekly/monthly）再平衡
        2. 多因子等权打分，取综合排名前N只
        3. 持仓：综合排名跌破阈值则卖出
        4. 空仓：买入综合排名最强的股票
        """
        from src.core.strategy_gen import SignalBuilder

        signals = []

        # 获取调仓频率参数
        rebalance_freq = params.get('rebalance_frequency', 'weekly')

        # 检查是否需要调仓
        # 注意：每年首日是强制调仓日，即使不是普通调仓日也要生成信号
        is_forced_rebal = date in rebalance_dates
        if not SignalBuilder.build_rebalance_signal(date, rebalance_freq) and not is_forced_rebal and not positions:
            return signals

        # 判断是否需要再平衡
        is_rebalance_day = date in rebalance_dates
        should_rebalance = is_rebalance_day or not positions

        # ===== 步骤1：获取策略使用的因子列表 =====
        strategy_factors = strategy.get('factors', ['momentum_20'])
        top_n = params.get('top_n_stocks', self.top_n)

        # ===== 步骤2：多因子打分 =====
        # 先获取当日收盘价（全部股票）- 优先从缓存获取
        date_str = str(date)
        if self._has_cached_data() and date_str in self._price_by_date:
            price_map = self._price_by_date[date_str]
            price_df = pd.DataFrame([
                {'ts_code': ts, 'close': p} for ts, p in price_map.items() if p > 0
            ])
        else:
            price_df = self.store.df(
                f"""SELECT ts_code, close
                     FROM stock_daily
                     WHERE trade_date = '{date_str}' AND close > 0"""
            )
        if price_df.empty:
            return signals

        # 逐个因子获取并合并 - 优先从缓存获取
        merged = price_df.copy()
        for factor_name in strategy_factors:
            if self._has_cached_data() and date_str in self._factor_by_date:
                factor_map = self._factor_by_date[date_str].get(factor_name, {})
                if factor_map:
                    factor_col = pd.DataFrame([
                        {'ts_code': ts, factor_name: val} for ts, val in factor_map.items()
                    ])
                else:
                    factor_col = pd.DataFrame(columns=['ts_code', factor_name])
            else:
                factor_col = self.store.df(
                    f"""SELECT ts_code, value
                         FROM factors
                         WHERE factor_name = '{factor_name}'
                           AND trade_date = '{date_str}'
                           AND value IS NOT NULL"""
                )
                if not factor_col.empty:
                    factor_col = factor_col.rename(columns={'value': factor_name})
            if factor_col.empty:
                continue
            if factor_name not in factor_col.columns:
                factor_col = factor_col.rename(columns={'value': factor_name})
            merged = merged.merge(factor_col, on='ts_code', how='left')

        if merged.empty:
            return signals

        # ===== 步骤2.5（可选）：因子中性化 =====
        if self._neutralizer is not None and len(merged) >= 30:
            merged = self._apply_factor_neutralization(merged, strategy_factors, date)

        # ===== 步骤3：计算综合排名得分 =====
        # 对每个因子排名（升序，值越大排名越靠前），然后平均
        rank_cols = []
        for factor_name in strategy_factors:
            if factor_name in merged.columns:
                # 百分位排名：rank / 总数，越大越好
                merged[f'{factor_name}_rank'] = merged[factor_name].rank(ascending=False, pct=True)
                rank_cols.append(f'{factor_name}_rank')

        if not rank_cols:
            return signals

        # 综合得分 = 各因子平均排名（越高越好）
        merged['composite_score'] = merged[rank_cols].mean(axis=1)

        # 综合排名前N只（排除缺失数据的）
        merged = merged.dropna(subset=rank_cols)
        merged = merged.sort_values('composite_score', ascending=False).reset_index(drop=True)

        # ===== 步骤4：处理持仓股票（止损/止盈/排名淘汰）=====
        stop_loss = params.get('stop_loss', 0.05)
        take_profit = params.get('take_profit', 0.20)

        for ts_code, pos_info in list(positions.items()):
            cost = pos_info.get('cost', 0)
            shares = pos_info.get('shares', 0)
            if cost <= 0 or shares <= 0:
                continue

            # 查找该股票当前数据
            row_data = merged[merged['ts_code'] == ts_code]

            if len(row_data) == 0:
                close_price = self._get_close_price(ts_code, date)
                if close_price > 0:
                    signals.append({
                        'ts_code': ts_code,
                        'direction': 'sell',
                        'price': close_price,
                        'shares': shares,
                        'reason': 'no_data'
                    })
                continue

            row = row_data.iloc[0]
            current_price = row['close']
            current_score = row['composite_score']
            current_rank = row.name  # composite_score 排名（0=最好）
            entry_score = current_holdings.get(ts_code, {}).get('entry_score', 1.0)

            pnl_pct = (current_price - cost) / cost

            # 止损
            if pnl_pct <= -stop_loss:
                signals.append({
                    'ts_code': ts_code,
                    'direction': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'reason': f'stop_loss({pnl_pct:.2%})'
                })
                continue

            # 止盈
            if pnl_pct >= take_profit:
                signals.append({
                    'ts_code': ts_code,
                    'direction': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'reason': f'take_profit({pnl_pct:.2%})'
                })
                continue

            # 排名淘汰：综合排名跌破前 N*2 名
            if current_rank >= top_n * 2:
                signals.append({
                    'ts_code': ts_code,
                    'direction': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'reason': f'rank_drop({current_rank})'
                })
                continue

            # 综合得分大幅下滑（超过20%的rank下降）
            score_drop = (entry_score - current_score) / (entry_score + 1e-10)
            if score_drop > 0.20:
                signals.append({
                    'ts_code': ts_code,
                    'direction': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'reason': f'score_drop({score_drop:.2%})'
                })
                continue

        # ===== 步骤5：买入新股票 =====
        if should_rebalance or not positions:
            current_positions_set = set(positions.keys())
            n_to_buy = top_n - len(positions)

            if n_to_buy > 0:
                candidates = merged[~merged['ts_code'].isin(current_positions_set)]
                buy_list = candidates.head(n_to_buy)

                for _, row in buy_list.iterrows():
                    price = row['close']
                    if price <= 0:
                        continue
                    # 不在这里计算shares，由run()方法根据weight_scheme统一计算
                    signals.append({
                        'ts_code': row['ts_code'],
                        'direction': 'buy',
                        'price': price,
                        'shares': 0,  # 由run()根据weight_scheme计算
                        'reason': f'composite_top({row.name})'
                    })
                    current_holdings[row['ts_code']] = {
                        'entry_score': row['composite_score'],
                        'entry_rank': row.name
                    }

        return signals
    
    def _calc_metrics(self, trades: List, nav_history: List) -> Dict:
        """计算绩效指标"""
        if not nav_history:
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'total_return': 0, 'nav_series': []}
        
        nav_df = pd.DataFrame(nav_history)
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df = nav_df.set_index('date').sort_index()
        
        nav_series = nav_df['nav']
        
        if len(nav_series) < 2:
            return {'sharpe_ratio': 0, 'max_drawdown': 0, 'total_return': 0, 'nav_series': nav_df}
        
        nav_df['returns'] = nav_df['nav'].pct_change()
        returns = nav_df['returns'].dropna()
        
        # 总收益率
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
        
        # 年化收益率
        n_days = len(nav_series)
        n_years = n_days / 252
        if n_years > 0:
            annual_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_return = 0
        
        # 夏普比率 (假设无风险利率 3%)
        risk_free = 0.03
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252 - risk_free) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # ===== 计算每笔交易的盈亏 =====
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['pnl'] = 0.0
            
            # 按 ts_code 分组配对买卖
            for ts_code in trades_df['ts_code'].unique():
                code_trades = trades_df[trades_df['ts_code'] == ts_code].sort_values('trade_date')
                buys = code_trades[code_trades['direction'] == 'buy']
                sells = code_trades[code_trades['direction'] == 'sell']
                
                buy_queue = buys.to_dict('records')
                sell_queue = sells.to_dict('records')
                
                # 简单 FIFO：配对买卖计算盈亏
                for buy in buy_queue:
                    buy_cost = buy['price'] * buy['quantity'] * (1 + self.slippage + self.commission)
                    if sell_queue:
                        sell = sell_queue.pop(0)
                        sell_proceeds = sell['price'] * sell['quantity'] * (1 - self.slippage - self.commission)
                        pnl = sell_proceeds - buy_cost
                        trades_df.loc[buy['trade_id'] == trades_df['trade_id'], 'pnl'] = pnl
            
            # 胜率
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
        else:
            win_rate = 0
        
        # 计算过拟合指标（从净值序列提取日收益率）
        overfit_metrics = self._calc_overfit_metrics(returns, sharpe_ratio)

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': len(trades),
            'trades': trades,
            'nav_series': nav_df,
            # 过拟合指标
            'pbo': overfit_metrics.get('pbo'),
            'dsr': overfit_metrics.get('dsr'),
            'cscv': overfit_metrics.get('cscv'),
            'overfit_report': overfit_metrics,
        }

    def _calc_overfit_metrics(self, returns: pd.Series, sharpe: float) -> Dict:
        """
        计算过拟合检测指标

        Args:
            returns: 日收益率序列（pd.Series）
            sharpe: 当前策略夏普比率（用于 DSR 模拟）

        Returns:
            包含 pbo/dsr/cscv 的字典；若数据不足返回空值
        """
        from src.core.overfit_detector import OverfitDetector

        if self.config is None:
            return {}

        ods_cfg = self.config.get('overfit_detection', {})
        if not ods_cfg.get('enabled', False):
            return {}

        oos_ratio = float(ods_cfg.get('oos_ratio', 0.2))
        n_splits = int(ods_cfg.get('n_splits_cscv', 4))

        ret_list = returns.dropna().tolist()
        if len(ret_list) < 60:
            return {}

        split_idx = int(len(ret_list) * (1.0 - oos_ratio))
        if split_idx < 30 or (len(ret_list) - split_idx) < 30:
            return {}

        is_ret = ret_list[:split_idx]
        oos_ret = ret_list[split_idx:]

        detector = OverfitDetector(self.store, self.logger, self.config)

        # PBO：基于收益率序列的样本内/外分割
        pbo = detector.calc_pbo(is_ret, oos_ret, n_portfolios=10)

        # DSR：使用 bootstrap 模拟多次试验来估算 DSR
        n_boots = 100
        boot_sharpes = []
        rng = np.random.default_rng(42)
        for _ in range(n_boots):
            boot_idx = rng.choice(len(ret_list), len(ret_list), replace=True)
            boot_ret = [ret_list[i] for i in boot_idx]
            boot_sr = np.mean(boot_ret) / (np.std(boot_ret) + 1e-10) * np.sqrt(252)
            boot_sharpes.append(float(boot_sr))
        # 加入真实 Sharpe 作为"已知最优"
        boot_sharpes.append(float(sharpe))

        dsr = detector.calc_dsr(boot_sharpes, [boot_sharpes], n_trials=n_boots + 1)

        # CSCV
        cscv = detector.calc_cscv(ret_list, n_splits=n_splits)

        return {
            'pbo': round(float(pbo), 4),
            'dsr': round(float(dsr), 4),
            'cscv': round(float(cscv), 4),
        }
