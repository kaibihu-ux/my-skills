"""
因子单调性检验器（分组单调性回测）

验证因子是否具有单调性：Top组 > Middle组 > Bottom组
所有阈值/参数从 config/settings.yaml 读取
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class MonotonicityTester:
    """
    因子单调性检验器

    核心功能：
    1. 按因子值分 n_groups 组
    2. 每周/月计算各组收益
    3. 检验单调性：Top > Middle > Bottom
    4. 返回单调性得分、收益 Spread、胜率
    """

    def __init__(self, store, logger, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self._load_config(config)

    def _load_config(self, config: Optional[Dict] = None):
        """从配置加载单调性检验参数"""
        if config is not None:
            mcfg = config.get('monotonicity', {})
        else:
            mcfg = {}

        self.enabled = mcfg.get('enabled', True)
        self.n_groups = int(mcfg.get('n_groups', 5))
        self.min_monotonicity = float(mcfg.get('min_monotonicity', 0.6))
        self.future_period = int(mcfg.get('future_period', 20))  # 持仓天数
        self.freq = mcfg.get('frequency', 'monthly')  # monthly / weekly

    def test(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        n_groups: Optional[int] = None,
    ) -> Dict:
        """
        执行分组单调性回测

        Args:
            factor_name: 因子名称
            start_date:  开始日期 YYYYMMDD
            end_date:    结束日期 YYYYMMDD
            n_groups:   分组数（默认从配置读取）

        Returns:
            {
                'factor_name': str,
                'monotonicity_score': float,   # 0~1，越高越单调
                'spread': float,                # Top组 - Bottom组收益
                'win_rate': float,              # Top组跑赢Bottom组的月份比例
                'group_returns': List[float],   # 各组平均收益
                'n_groups': int,
                'passed': bool,                # monotonicity_score >= min_monotonicity
                'details': List[Dict],          # 每期各组收益详情
            }
        """
        if not self.enabled:
            return self._empty_result(factor_name, n_groups or self.n_groups)

        n_groups_cfg = n_groups or self.n_groups
        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)

        # 读取因子数据
        factor_df = self._load_factor(factor_name, start_fmt, end_fmt)
        if factor_df.empty:
            self._log(f"因子 {factor_name} 无数据")
            return self._empty_result(factor_name, n_groups_cfg)

        # 读取价格数据
        price_df = self._load_price(start_fmt, end_fmt)
        if price_df.empty:
            return self._empty_result(factor_name, n_groups_cfg)

        # 计算未来收益率（未来N日）
        period = self.future_period
        price_df[f'return_{period}'] = (
            price_df.groupby('ts_code')['close']
            .pct_change(period)
            .shift(-period)
        )

        # 合并
        merged = factor_df.merge(
            price_df[['ts_code', 'trade_date', f'return_{period}']],
            on=['ts_code', 'trade_date'],
            how='inner'
        )

        if len(merged) < 100:
            return self._empty_result(factor_name, n_groups_cfg)

        # 按调仓频率分批处理
        if self.freq == 'weekly':
            merged = merged.copy()
            merged['period_key'] = pd.to_datetime(merged['trade_date']).dt.to_period('W')
        else:  # monthly
            merged = merged.copy()
            merged['period_key'] = pd.to_datetime(merged['trade_date']).dt.to_period('M')

        # 对每个调仓周期执行分组
        period_results: List[Dict] = []
        group_avg_returns: Dict[int, List[float]] = {g: [] for g in range(n_groups_cfg)}

        for period_key, group_data in merged.groupby('period_key'):
            valid = group_data.dropna(subset=['value', f'return_{period}'])
            if len(valid) < n_groups_cfg * 5:
                continue

            try:
                valid = valid.copy()
                valid['quantile'] = pd.qcut(
                    valid['value'],
                    n_groups_cfg,
                    labels=False,
                    duplicates='drop'
                )
                for q in range(n_groups_cfg):
                    q_data = valid[valid['quantile'] == q][f'return_{period}']
                    if len(q_data) > 0:
                        group_avg_returns[q].append(float(q_data.mean()))
            except Exception as e:
                self._log(f"分组失败 ({period_key}): {e}")
                continue

        # 汇总各组平均收益
        group_returns = []
        for g in range(n_groups_cfg):
            vals = group_avg_returns.get(g, [])
            group_returns.append(float(np.mean(vals)) if vals else 0.0)

        if not group_returns or all(r == 0 for r in group_returns):
            return self._empty_result(factor_name, n_groups_cfg)

        # ===== 计算单调性得分 =====
        # 单调性 = 满足递增或递减的相邻组对数 / 总组对数
        n_pairs = n_groups_cfg - 1
        monotonic_increases = 0
        monotonic_decreases = 0

        for i in range(n_pairs):
            if group_returns[i + 1] > group_returns[i]:
                monotonic_increases += 1
            elif group_returns[i + 1] < group_returns[i]:
                monotonic_decreases += 1
            # equal 不计入

        # 选择单调方向（增加 or 减少）
        if monotonic_increases >= monotonic_decreases:
            # 期望 Top > Bottom（递增）
            monotonic_score = monotonic_increases / n_pairs if n_pairs > 0 else 0.0
            spread = group_returns[-1] - group_returns[0]  # Top - Bottom
        else:
            # 期望 Top < Bottom（递减）
            monotonic_score = monotonic_decreases / n_pairs if n_pairs > 0 else 0.0
            spread = group_returns[0] - group_returns[-1]  # Bottom - Top

        # ===== 计算 Top vs Bottom 胜率 =====
        # 逐期对比 Top 组和 Bottom 组收益
        n_periods = len(list(group_avg_returns.values())[0])
        if n_periods > 0:
            top_wins = sum(
                1 for i in range(n_periods)
                if group_avg_returns[n_groups_cfg - 1][i] > group_avg_returns[0][i]
            )
            win_rate = top_wins / n_periods
        else:
            win_rate = 0.0

        passed = monotonic_score >= self.min_monotonicity

        self._log(
            f"单调性检验 [{factor_name}]: score={monotonic_score:.3f}, "
            f"spread={spread:.4f}, win_rate={win_rate:.2%}, "
            f"groups={group_returns}, passed={passed}"
        )

        return {
            'factor_name': factor_name,
            'monotonicity_score': round(float(monotonic_score), 4),
            'spread': round(float(spread), 6),
            'win_rate': round(float(win_rate), 4),
            'group_returns': [round(float(r), 6) for r in group_returns],
            'n_groups': n_groups_cfg,
            'passed': bool(passed),
            'details': period_results,
        }

    def _load_factor(self, factor_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库加载因子数据"""
        if self.store is None:
            return pd.DataFrame()
        try:
            return self.store.df(
                f"""SELECT ts_code, trade_date, value
                    FROM factors
                    WHERE factor_name = '{factor_name}'
                      AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY trade_date, ts_code"""
            )
        except Exception as e:
            self._log(f"加载因子数据失败: {e}")
            return pd.DataFrame()

    def _load_price(self, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库加载价格数据"""
        if self.store is None:
            return pd.DataFrame()
        try:
            return self.store.df(
                f"""SELECT ts_code, trade_date, close
                    FROM stock_daily
                    WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY ts_code, trade_date"""
            )
        except Exception as e:
            self._log(f"加载价格数据失败: {e}")
            return pd.DataFrame()

    def _empty_result(self, factor_name: str, n_groups: int) -> Dict:
        return {
            'factor_name': factor_name,
            'monotonicity_score': 0.0,
            'spread': 0.0,
            'win_rate': 0.0,
            'group_returns': [0.0] * n_groups,
            'n_groups': n_groups,
            'passed': False,
            'details': [],
        }

    @staticmethod
    def _format_date(date_str: str) -> str:
        """YYYYMMDD -> YYYY-MM-DD"""
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[MonotonicityTester] {msg}")
        else:
            print(f"[MonotonicityTester] {msg}")
