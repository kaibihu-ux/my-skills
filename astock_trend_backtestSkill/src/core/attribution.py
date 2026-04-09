"""
绩效归因分析：Brinson模型 + Barra风格因子归因

功能：
- Brinson 行业归因：行业配置收益 + 个股选择收益 + 交互收益
- Barra 风格因子归因：市值/价值/成长/动量/波动率/盈利/质量因子贡献
- 统一归因引擎：生成完整归因报告

所有参数从 config/settings.yaml 读取
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple


class BrinsonAttributor:
    """
    Brinson 行业归因

    归因分解公式：
    总收益 = 行业配置收益 + 个股选择收益 + 交互收益

    行业配置收益 = Σ(w_portfolio_i - w_benchmark_i) × r_benchmark_i
    个股选择收益 = Σ w_benchmark_i × (r_portfolio_i - r_benchmark_i)
    交互收益     = Σ(w_portfolio_i - w_benchmark_i) × (r_portfolio_i - r_benchmark_i)

    其中：
    - w_portfolio_i: 组合在行业 i 的权重
    - w_benchmark_i: 基准在行业 i 的权重
    - r_benchmark_i: 基准行业 i 的收益率
    - r_portfolio_i: 组合行业 i 的收益率
    """

    def __init__(self, store=None, logger=None, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger

    def attribute(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        benchmark_returns: Union[pd.Series, List[float]],
        holdings: Dict[str, Dict],      # {stock_code: {'weight': float, 'industry': str, 'return': float}}
        industry_map: Dict[str, str],   # {stock_code: industry_name}
        factor_exposure: Optional[Dict] = None,
    ) -> Dict:
        """
        执行 Brinson 归因

        Args:
            portfolio_returns:  组合日收益率序列
            benchmark_returns:   基准日收益率序列
            holdings:            当前持仓（格式见上）
            industry_map:        股票 → 行业 映射

        Returns:
            {
                'allocation': float,      # 行业配置收益
                'selection': float,        # 个股选择收益
                'interaction': float,      # 交互收益
                'total_active': float,    # 主动收益合计
                'industry_contrib': Dict[str, Dict],  # 各行业贡献明细
            }
        """
        try:
            # 转换为 pandas Series
            port_ret = self._to_series(portfolio_returns)
            bench_ret = self._to_series(benchmark_returns)

            if port_ret.empty or bench_ret.empty:
                return self._empty_result()

            # 对齐索引
            common_idx = port_ret.index.intersection(bench_ret.index)
            port_ret = port_ret[common_idx]
            bench_ret = bench_ret[common_idx]

            # ----- 计算持仓行业权重 -----
            industry_weights, industry_returns = self._calc_industry_weights(
                holdings, industry_map
            )

            # ----- 计算基准行业权重 -----
            bench_industry_weights = self._calc_bench_weights(
                industry_map, common_idx
            )

            # ----- 行业归因计算 -----
            allocation = 0.0
            selection = 0.0
            interaction = 0.0
            industry_contrib = {}

            all_industries = set(industry_weights.keys()) | set(
                bench_industry_weights.keys()
            )

            for ind in all_industries:
                w_p = industry_weights.get(ind, 0.0)
                w_b = bench_industry_weights.get(ind, 0.0)
                r_p = industry_returns.get(ind, 0.0)
                r_b = bench_ret.mean()  # 用基准整体收益作为行业基准代理

                # 行业配置收益
                alloc_contrib = (w_p - w_b) * r_b
                # 个股选择收益
                sel_contrib = w_b * (r_p - r_b)
                # 交互收益
                int_contrib = (w_p - w_b) * (r_p - r_b)

                allocation += alloc_contrib
                selection += sel_contrib
                interaction += int_contrib

                industry_contrib[ind] = {
                    'portfolio_weight': round(float(w_p), 6),
                    'benchmark_weight': round(float(w_b), 6),
                    'portfolio_return': round(float(r_p), 6),
                    'benchmark_return': round(float(r_b), 6),
                    'allocation': round(float(alloc_contrib), 6),
                    'selection': round(float(sel_contrib), 6),
                    'interaction': round(float(int_contrib), 6),
                    'total': round(float(alloc_contrib + sel_contrib + int_contrib), 6),
                }

            total_active = allocation + selection + interaction

            self._log(
                f"Brinson归因: 配置={allocation:.4f}, "
                f"选择={selection:.4f}, 交互={interaction:.4f}, "
                f"主动={total_active:.4f}"
            )

            return {
                'allocation': round(float(allocation), 6),
                'selection': round(float(selection), 6),
                'interaction': round(float(interaction), 6),
                'total_active': round(float(total_active), 6),
                'industry_contrib': industry_contrib,
            }

        except Exception as e:
            self._log(f"Brinson归因失败: {e}")
            return self._empty_result()

    def _calc_industry_weights(
        self,
        holdings: Dict,
        industry_map: Dict,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算组合各行业权重和收益率"""
        if not holdings:
            return {}, {}

        # 计算总持仓市值
        total_value = sum(h.get('value', 0) or 0 for h in holdings.values())
        if total_value <= 0:
            return {}, {}

        industry_weights: Dict[str, float] = {}
        industry_values: Dict[str, float] = {}
        industry_returns_sum: Dict[str, float] = {}

        for stock, info in holdings.items():
            ind = industry_map.get(stock, 'Unknown')
            value = info.get('value', 0) or 0
            ret = info.get('return', 0.0)

            industry_values[ind] = industry_values.get(ind, 0) + value
            industry_returns_sum[ind] = industry_returns_sum.get(ind, 0) + value * ret

        for ind, value in industry_values.items():
            industry_weights[ind] = value / total_value

        # 各行业加权收益率
        industry_returns = {}
        for ind in industry_values:
            if industry_values[ind] > 0:
                industry_returns[ind] = (
                    industry_returns_sum[ind] / industry_values[ind]
                )

        return industry_weights, industry_returns

    def _calc_bench_weights(
        self,
        industry_map: Dict,
        date_index: List,
    ) -> Dict[str, float]:
        """计算基准行业权重（简化：用行业市值占比）"""
        # 简化：假设基准 = 全市场等权，均匀分配到各行业
        unique_inds = set(industry_map.values())
        n_ind = len(unique_inds)
        if n_ind == 0:
            return {}
        return {ind: 1.0 / n_ind for ind in unique_inds}

    def _empty_result(self) -> Dict:
        return {
            'allocation': 0.0,
            'selection': 0.0,
            'interaction': 0.0,
            'total_active': 0.0,
            'industry_contrib': {},
        }

    @staticmethod
    def _to_series(data) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        return pd.Series(dtype=float)

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[BrinsonAttributor] {msg}")
        else:
            print(f"[BrinsonAttributor] {msg}")


# =============================================================================
# Barra 风格因子归因
# =============================================================================

class BarraAttributor:
    """
    Barra 风格因子归因

    归因公式：
    R_portfolio = Σ βᵢ × Fᵢ + α

    其中：
    - βᵢ: 组合在因子 i 的暴露（风格因子 beta）
    - Fᵢ: 因子 i 的收益率（因子收益）
    - α:  特异性收益（残差）

    Barra 风格因子：
    - 市值因子（size）
    - 价值因子（book_to_price）
    - 成长因子（earnings_growth）
    - 动量因子（momentum）
    - 波动率因子（volatility）
    - 盈利因子（profitability）
    - 质量因子（quality/roe）
    - 流动性因子（liquidity）
    """

    # 标准 Barra 风格因子列表
    BARRA_FACTORS = [
        'size',           # 市值因子
        'book_to_price',  # 价值因子
        'earnings_growth',  # 成长因子
        'momentum',       # 动量因子
        'volatility',     # 波动率因子
        'profitability',  # 盈利因子
        'quality',        # 质量因子
        'liquidity',      # 流动性因子
    ]

    def __init__(self, store=None, logger=None, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self._load_config(config)

    def _load_config(self, config: Optional[Dict] = None):
        if config is not None:
            acfg = config.get('attribution', {})
        else:
            acfg = {}

        self.enabled = bool(acfg.get('enabled', True))
        self.factors = acfg.get('factors', self.BARRA_FACTORS)

    def attribute(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        factor_returns: Union[pd.DataFrame, Dict],   # {factor_name: pd.Series}
        factor_exposure: Dict[str, Dict],             # {stock_code: {factor_name: beta}}
    ) -> Dict:
        """
        执行 Barra 风格因子归因

        Args:
            portfolio_returns: 组合日收益率
            factor_returns:    各因子日收益率（DataFrame 或 Dict）
            factor_exposure:   组合在各因子的暴露 {stock_code: {factor: beta}}

        Returns:
            {
                'factor_contrib': Dict[str, float],  # 各因子收益贡献
                'specific_return': float,             # 特异性收益
                'total_return': float,                 # 总收益
                'factor_exposure': Dict,               # 各因子暴露
            }
        """
        if not self.enabled:
            return self._empty_result()

        try:
            port_ret = self._to_series(portfolio_returns)

            # 处理 factor_returns（可能是 DataFrame 或 Dict）
            if isinstance(factor_returns, pd.DataFrame):
                fr_df = factor_returns
            elif isinstance(factor_returns, dict):
                fr_df = pd.DataFrame(factor_returns)
            else:
                fr_df = pd.DataFrame()

            if fr_df.empty:
                return self._empty_result()

            # 对齐索引
            common_idx = port_ret.index.intersection(fr_df.index)
            port_ret = port_ret[common_idx]
            fr_df = fr_df.loc[common_idx]

            # 计算组合在各因子的平均暴露（横截面平均）
            exposure = self._calc_avg_exposure(factor_exposure)

            # 计算各因子收益贡献
            factor_contrib = {}
            for fname in self.factors:
                if fname in fr_df.columns:
                    avg_beta = exposure.get(fname, 0.0)
                    factor_ret = fr_df[fname].mean()  # 因子平均收益
                    contrib = avg_beta * factor_ret
                    factor_contrib[fname] = float(contrib)

            # 计算特异性收益
            total_factor_contrib = sum(factor_contrib.values())
            specific_return = float(port_ret.mean()) - total_factor_contrib

            total_return = float(port_ret.mean())

            self._log(
                f"Barra归因: 总收益={total_return:.4f}, "
                f"因子贡献={total_factor_contrib:.4f}, "
                f"特异性={specific_return:.4f}, "
                f"因子={list(factor_contrib.keys())}"
            )

            return {
                'factor_contrib': {k: round(float(v), 6) for k, v in factor_contrib.items()},
                'specific_return': round(float(specific_return), 6),
                'total_return': round(float(total_return), 6),
                'factor_exposure': {k: round(float(v), 4) for k, v in exposure.items()},
            }

        except Exception as e:
            self._log(f"Barra归因失败: {e}")
            return self._empty_result()

    def _calc_avg_exposure(
        self,
        factor_exposure: Dict[str, Dict],
    ) -> Dict[str, float]:
        """计算组合在各因子的平均暴露"""
        if not factor_exposure:
            return {f: 0.0 for f in self.factors}

        exposure_sum: Dict[str, float] = {f: 0.0 for f in self.factors}
        n = 0

        for stock, betas in factor_exposure.items():
            n += 1
            for fname in self.factors:
                exposure_sum[fname] += float(betas.get(fname, 0.0))

        if n == 0:
            return {f: 0.0 for f in self.factors}

        return {f: exposure_sum[f] / n for f in self.factors}

    def _empty_result(self) -> Dict:
        return {
            'factor_contrib': {f: 0.0 for f in self.factors},
            'specific_return': 0.0,
            'total_return': 0.0,
            'factor_exposure': {f: 0.0 for f in self.factors},
        }

    @staticmethod
    def _to_series(data) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        return pd.Series(dtype=float)

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[BarraAttributor] {msg}")
        else:
            print(f"[BarraAttributor] {msg}")


# =============================================================================
# 统一归因引擎
# =============================================================================

class PerformanceAttributor:
    """
    统一绩效归因引擎

    生成完整归因报告：
    1. Brinson 行业归因
    2. Barra 风格因子归因
    3. 收益来源分解摘要
    """

    def __init__(self, store=None, logger=None, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self.config = config or {}

        self.brinson = BrinsonAttributor(store, logger, config)
        self.barra = BarraAttributor(store, logger, config)

    def full_report(
        self,
        portfolio_returns: Union[pd.Series, List[float]],
        benchmark_returns: Union[pd.Series, List[float]],
        holdings: Dict,
        industry_map: Dict,
        factor_returns: Optional[Union[pd.DataFrame, Dict]] = None,
        factor_exposure: Optional[Dict] = None,
    ) -> Dict:
        """
        生成完整归因报告

        Args:
            portfolio_returns:  组合日收益率
            benchmark_returns:   基准日收益率
            holdings:            当前持仓
            industry_map:        股票 → 行业 映射
            factor_returns:      因子日收益率（可选）
            factor_exposure:     因子暴露（可选）

        Returns:
            {
                'brinson': {...},    # Brinson归因结果
                'barra': {...},      # Barra归因结果
                'summary': {
                    'total_return': float,
                    'active_return': float,
                    'allocation': float,
                    'selection': float,
                    'interaction': float,
                    'factor_contrib': float,
                    'specific_return': float,
                }
            }
        """
        # Brinson 归因
        brinson_result = self.brinson.attribute(
            portfolio_returns,
            benchmark_returns,
            holdings,
            industry_map,
        )

        # Barra 归因
        barra_result = {}
        if factor_returns is not None and factor_exposure is not None:
            barra_result = self.barra.attribute(
                portfolio_returns,
                factor_returns,
                factor_exposure,
            )

        # 汇总摘要
        port_ret = self._to_series(portfolio_returns)
        bench_ret = self._to_series(benchmark_returns)
        total_return = float(port_ret.mean()) if len(port_ret) > 0 else 0.0
        active_return = total_return - float(bench_ret.mean()) if len(bench_ret) > 0 else 0.0

        summary = {
            'total_return': round(total_return, 6),
            'active_return': round(active_return, 6),
            'allocation': brinson_result.get('allocation', 0.0),
            'selection': brinson_result.get('selection', 0.0),
            'interaction': brinson_result.get('interaction', 0.0),
            'factor_contrib': sum(barra_result.get('factor_contrib', {}).values()),
            'specific_return': barra_result.get('specific_return', 0.0),
        }

        self._log(
            f"完整归因报告: 总收益={summary['total_return']:.4f}, "
            f"主动={summary['active_return']:.4f}, "
            f"配置={summary['allocation']:.4f}, "
            f"选择={summary['selection']:.4f}"
        )

        return {
            'brinson': brinson_result,
            'barra': barra_result,
            'summary': summary,
        }

    @staticmethod
    def _to_series(data) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        return pd.Series(dtype=float)

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[PerformanceAttributor] {msg}")
        else:
            print(f"[PerformanceAttributor] {msg}")
