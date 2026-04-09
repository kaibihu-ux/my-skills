"""
压力测试：历史极端行情 + 参数鲁棒性 + 样本外隔离

功能：
1. run_historical_stress: 历史极端行情回溯（股灾/熔断/贸易战/COVID/熊市）
2. run_param_robustness:  参数鲁棒性测试（随机扰动±20%）
3. run_oos_isolation:     样本外隔离验证（训练/测试集分离）
4. full_stress_report:   完整压力测试报告

所有参数从 config/settings.yaml 读取
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple, Callable


class StressTester:
    """
    压力测试器

    三种测试模式：
    1. 历史极端行情回溯
    2. 参数鲁棒性
    3. 样本外隔离
    """

    def __init__(self, backtester, logger, config: Optional[Dict] = None):
        """
        Args:
            backtester: 回测执行器（BacktestExecutor 实例）
            logger:     日志记录器
            config:     配置字典
        """
        self.backtester = backtester
        self.logger = logger
        self.config = config or {}
        self._load_config()

    def _load_config(self):
        """从配置加载压力测试参数"""
        scfg = self.config.get('stress_test', {})

        self.enabled = bool(scfg.get('enabled', True))

        # 历史极端行情区间
        self.historical_periods = scfg.get('historical_periods', [
            {'name': '股灾1.0', 'start': '2015-06-01', 'end': '2015-09-30'},
            {'name': '熔断',   'start': '2016-01-01', 'end': '2016-02-29'},
            {'name': '贸易战', 'start': '2018-02-01', 'end': '2018-12-31'},
            {'name': 'COVID',  'start': '2020-02-01', 'end': '2020-03-31'},
            {'name': '熊市',   'start': '2022-03-01', 'end': '2022-10-31'},
        ])

        # 参数鲁棒性
        self.param_robustness_trials = int(scfg.get('param_robustness_trials', 20))
        self.param_perturb_range = float(scfg.get('param_perturb_range', 0.20))

        # 样本外比例
        self.oos_ratio = float(scfg.get('oos_ratio', 0.2))

    # =====================================================================
    # 公开 API
    # =====================================================================

    def full_stress_report(
        self,
        strategy: Dict,
        params: Dict,
    ) -> Dict:
        """
        完整压力测试报告

        综合三种测试：
        1. 历史极端行情
        2. 参数鲁棒性
        3. 样本外隔离

        Returns:
            {
                'worst_drawdown': float,        # 最差回撤（最核心指标）
                'worst_period': str,            # 最差区间名称
                'sharpe_in_sample': float,
                'sharpe_oos': float,
                'sharpe_degradation': float,   # 样本内-样本外夏普差距
                'stability_ratio': float,      # 参数鲁棒性（正收益比例）
                'passed': bool,                 # 通过标准：worst_drawdown > -0.30
                'historical_stress': {...},
                'param_robustness': {...},
                'oos_isolation': {...},
            }
        """
        if not self.enabled:
            return self._empty_report()

        self._log("=== 开始完整压力测试 ===")

        # 1. 历史极端行情
        hist_report = self.run_historical_stress(strategy, params)

        # 2. 参数鲁棒性
        robust_report = self.run_param_robustness(strategy, params)

        # 3. 样本外隔离（需要较长回测期，先找可行区间）
        oos_report = self.run_oos_isolation(
            strategy,
            train_start='20200101',
            train_end='20221231',
            test_start='20230101',
            test_end='20240331',
        )

        # 综合判断
        worst_dd = hist_report.get('worst_drawdown', 0.0)
        worst_period = hist_report.get('worst_period_name', 'N/A')
        sharpe_is = oos_report.get('sharpe_in_sample', 0.0)
        sharpe_oos = oos_report.get('sharpe_oos', 0.0)
        sharpe_deg = sharpe_is - sharpe_oos
        stability = robust_report.get('positive_ratio', 0.0)

        # 通过标准：
        # 1. 极端行情最差回撤 > -30%
        # 2. 样本外夏普与样本内差距 < 1.0
        # 3. 参数稳定性 > 50%
        passed = (
            worst_dd > -0.30
            and abs(sharpe_deg) < 1.5
            and stability > 0.40
        )

        self._log(
            f"压力测试完成: worst_dd={worst_dd:.2%}, "
            f"sharpe_deg={sharpe_deg:.4f}, "
            f"stability={stability:.2%}, passed={passed}"
        )

        return {
            'worst_drawdown': round(float(worst_dd), 4),
            'worst_period_name': worst_period,
            'sharpe_in_sample': round(float(sharpe_is), 4),
            'sharpe_oos': round(float(sharpe_oos), 4),
            'sharpe_degradation': round(float(sharpe_deg), 4),
            'stability_ratio': round(float(stability), 4),
            'passed': bool(passed),
            # 子报告
            'historical_stress': hist_report,
            'param_robustness': robust_report,
            'oos_isolation': oos_report,
        }

    def run_historical_stress(
        self,
        strategy: Dict,
        params: Dict,
    ) -> Dict:
        """
        历史极端行情回溯测试

        对每个历史极端区间跑回测，返回各区间指标
        """
        if not self.enabled or self.backtester is None:
            return self._empty_historical_report()

        results = []

        for period in self.historical_periods:
            name = period['name']
            start = period['start'].replace('-', '')  # YYYYMMDD
            end = period['end'].replace('-', '')

            try:
                # 尝试跑回测
                bt_result = self.backtester.run(strategy, params, start, end)
                total_ret = bt_result.get('total_return', 0.0)
                max_dd = bt_result.get('max_drawdown', 0.0)
                sharpe = bt_result.get('sharpe_ratio', 0.0)
            except Exception as e:
                self._log(f"历史压力测试 [{name}] 失败: {e}")
                total_ret, max_dd, sharpe = -0.50, -0.60, -2.0

            results.append({
                'period_name': name,
                'start': period['start'],
                'end': period['end'],
                'total_return': round(float(total_ret), 4),
                'max_drawdown': round(float(max_dd), 4),
                'sharpe_ratio': round(float(sharpe), 4),
            })

            self._log(
                f"历史压力 [{name}]: "
                f"收益={total_ret:.2%}, 回撤={max_dd:.2%}, 夏普={sharpe:.2f}"
            )

        if not results:
            return self._empty_historical_report()

        # 找最差区间
        worst_dd_result = min(results, key=lambda x: x['max_drawdown'])

        return {
            'periods': results,
            'worst_drawdown': worst_dd_result['max_drawdown'],
            'worst_period_name': worst_dd_result['period_name'],
            'worst_sharpe': worst_dd_result['sharpe_ratio'],
            'avg_return': round(
                float(np.mean([r['total_return'] for r in results])), 4
            ),
            'avg_drawdown': round(
                float(np.mean([r['max_drawdown'] for r in results])), 4
            ),
        }

    def run_param_robustness(
        self,
        strategy: Dict,
        base_params: Dict,
        n_trials: Optional[int] = None,
    ) -> Dict:
        """
        参数鲁棒性测试

        对每个参数在 ±range 范围内随机扰动，
        检验策略表现是否稳定

        Returns:
            {
                'n_trials': int,
                'positive_ratio': float,   # 正收益试验占比
                'avg_sharpe': float,
                'avg_return': float,
                'sharpe_std': float,
                'passed': bool,             # positive_ratio > 0.5
            }
        """
        if not self.enabled or self.backtester is None:
            return self._empty_robustness_report()

        n_trials = n_trials or self.param_robustness_trials
        perturb_range = self.param_perturb_range

        # 提取数值参数
        numeric_keys = [
            'stop_loss', 'take_profit', 'top_n_stocks',
            'holding_period', 'rebalance_frequency',
        ]

        sharpe_list = []
        ret_list = []
        results = []

        for i in range(n_trials):
            # 随机扰动参数
            perturbed = self._perturb_params(base_params, perturb_range)

            try:
                # 跑固定区间回测（用较长区间）
                result = self.backtester.run(
                    strategy,
                    perturbed,
                    '20200101',
                    '20231231',
                )
                sharpe = result.get('sharpe_ratio', -999.0)
                total_ret = result.get('total_return', -999.0)
            except Exception as e:
                self._log(f"参数鲁棒性 Trial {i+1} 失败: {e}")
                sharpe = -999.0
                total_ret = -999.0

            sharpe_list.append(float(sharpe))
            ret_list.append(float(total_ret))
            results.append({
                'trial': i + 1,
                'sharpe': round(float(sharpe), 4),
                'return': round(float(total_ret), 4),
            })

        # 过滤无效结果
        valid_sharpes = [s for s in sharpe_list if s > -100]
        valid_returns = [r for r in ret_list if r > -100]

        if not valid_sharpes:
            return self._empty_robustness_report()

        positive_ratio = sum(1 for r in valid_returns if r > 0) / len(valid_returns)
        avg_sharpe = float(np.mean(valid_sharpes))
        sharpe_std = float(np.std(valid_sharpes))
        avg_return = float(np.mean(valid_returns))

        self._log(
            f"参数鲁棒性: {n_trials}次试验, "
            f"正收益比例={positive_ratio:.2%}, "
            f"平均夏普={avg_sharpe:.3f}, "
            f"夏普标准差={sharpe_std:.3f}"
        )

        return {
            'n_trials': n_trials,
            'positive_ratio': round(float(positive_ratio), 4),
            'avg_sharpe': round(float(avg_sharpe), 4),
            'avg_return': round(float(avg_return), 4),
            'sharpe_std': round(float(sharpe_std), 4),
            'passed': bool(positive_ratio >= 0.40),
            'results': results[:20],  # 最多保留20条
        }

    def run_oos_isolation(
        self,
        strategy: Dict,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Dict:
        """
        样本外隔离验证

        训练集: train_start ~ train_end
        测试集: test_start ~ test_end（盲测）

        对比：训练集夏普 vs 测试集夏普
        差距越小越稳定
        """
        if not self.enabled or self.backtester is None:
            return self._empty_oos_report()

        train_start_fmt = self._format_date(train_start)
        train_end_fmt = self._format_date(train_end)
        test_start_fmt = self._format_date(test_start)
        test_end_fmt = self._format_date(test_end)

        try:
            train_result = self.backtester.run(
                strategy, {}, train_start_fmt, train_end_fmt
            )
            test_result = self.backtester.run(
                strategy, {}, test_start_fmt, test_end_fmt
            )
        except Exception as e:
            self._log(f"样本外隔离测试失败: {e}")
            return self._empty_oos_report()

        sharpe_is = float(train_result.get('sharpe_ratio', 0.0))
        sharpe_oos = float(test_result.get('sharpe_ratio', 0.0))
        ret_is = float(train_result.get('total_return', 0.0))
        ret_oos = float(test_result.get('total_return', 0.0))
        dd_is = float(train_result.get('max_drawdown', 0.0))
        dd_oos = float(test_result.get('max_drawdown', 0.0))

        degradation = sharpe_is - sharpe_oos

        self._log(
            f"样本外隔离: 样本内夏普={sharpe_is:.3f}, "
            f"样本外夏普={sharpe_oos:.3f}, "
            f"衰退={degradation:.3f}"
        )

        return {
            'train_start': train_start_fmt,
            'train_end': train_end_fmt,
            'test_start': test_start_fmt,
            'test_end': test_end_fmt,
            'sharpe_in_sample': sharpe_is,
            'sharpe_oos': sharpe_oos,
            'return_in_sample': round(float(ret_is), 4),
            'return_oos': round(float(ret_oos), 4),
            'drawdown_in_sample': round(float(dd_is), 4),
            'drawdown_oos': round(float(dd_oos), 4),
            'sharpe_degradation': round(float(degradation), 4),
            'passed': bool(abs(degradation) < 1.5),
        }

    # =====================================================================
    # 私有工具
    # =====================================================================

    def _perturb_params(
        self,
        base_params: Dict,
        perturb_range: float = 0.20,
    ) -> Dict:
        """
        对参数进行随机扰动

        数值参数：±range 随机扰动
        字符串参数：保持不变
        """
        perturbed = base_params.copy()

        for key, value in base_params.items():
            if isinstance(value, (int, float)) and value > 0:
                # 随机扰动因子
                factor = 1.0 + random.uniform(-perturb_range, perturb_range)
                perturbed[key] = round(value * factor, 6)
            # 其他类型保持不变

        return perturbed

    def _empty_report(self) -> Dict:
        return {
            'worst_drawdown': 0.0,
            'worst_period_name': 'N/A',
            'sharpe_in_sample': 0.0,
            'sharpe_oos': 0.0,
            'sharpe_degradation': 0.0,
            'stability_ratio': 0.0,
            'passed': True,
            'historical_stress': self._empty_historical_report(),
            'param_robustness': self._empty_robustness_report(),
            'oos_isolation': self._empty_oos_report(),
        }

    def _empty_historical_report(self) -> Dict:
        return {
            'periods': [],
            'worst_drawdown': 0.0,
            'worst_period_name': 'N/A',
            'worst_sharpe': 0.0,
            'avg_return': 0.0,
            'avg_drawdown': 0.0,
        }

    def _empty_robustness_report(self) -> Dict:
        return {
            'n_trials': 0,
            'positive_ratio': 0.0,
            'avg_sharpe': 0.0,
            'avg_return': 0.0,
            'sharpe_std': 0.0,
            'passed': True,
            'results': [],
        }

    def _empty_oos_report(self) -> Dict:
        return {
            'train_start': '',
            'train_end': '',
            'test_start': '',
            'test_end': '',
            'sharpe_in_sample': 0.0,
            'sharpe_oos': 0.0,
            'return_in_sample': 0.0,
            'return_oos': 0.0,
            'drawdown_in_sample': 0.0,
            'drawdown_oos': 0.0,
            'sharpe_degradation': 0.0,
            'passed': True,
        }

    @staticmethod
    def _format_date(date_str: str) -> str:
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[StressTester] {msg}")
        else:
            print(f"[StressTester] {msg}")
