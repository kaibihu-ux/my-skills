"""
因子中性化处理器

功能：
1. 极值处理（Winsorize）：截断上下1%
2. 行业中性化：减去行业均值
3. 市值中性化：市值回归取残差
4. 完整中性化：行业哑变量回归 + 市值回归

所有阈值/参数从 config/settings.yaml 读取
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class Neutralizer:
    """
    因子中性化处理器

    支持三种方法：
    - winsorize: 极值截断
    - zscore:   标准化（减均值除标准差）
    - rank:     因子值排名归一化

    完整中性化流程：
    1. winsorize（可选）
    2. 行业中性化：factor = α + Σβᵢ·industry_i + residual
    3. 市值中性化：residual = γ + δ·ln_mcap + ε
    4. 返回 ε（双重残差即为中性化后因子）
    """

    def __init__(self, store=None, logger=None, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self._load_config(config)

    def _load_config(self, config: Optional[Dict] = None):
        """从配置加载中性化参数"""
        if config is not None:
            ncfg = config.get('neutralization', {})
        else:
            ncfg = {}

        self.enabled = ncfg.get('enabled', True)
        self.winsorize_pct = float(ncfg.get('winsorize_pct', 0.01))
        self.methods = ncfg.get('methods', ['industry', 'market_cap'])

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def neutralize(
        self,
        factor_values: np.ndarray,
        market_cap: np.ndarray,
        industry_codes: np.ndarray,
        method: str = 'winsorize',
    ) -> np.ndarray:
        """
        行业 + 市值中性化

        Args:
            factor_values:   原始因子值（1D array）
            market_cap:      市值数组（1D array，与 factor_values 同长度）
            industry_codes:  行业代码数组（1D array，与 factor_values 同长度）
            method:          预处理方法，'winsorize' | 'zscore' | 'rank'

        Returns:
            中性化后的因子值（1D array）
        """
        if not self.enabled:
            return np.array(factor_values, dtype=np.float64)

        factor = np.array(factor_values, dtype=np.float64)
        mcap = np.array(market_cap, dtype=np.float64)
        industry = np.array(industry_codes)

        # 过滤有效数据
        valid_mask = (
            np.isfinite(factor)
            & np.isfinite(mcap)
            & (mcap > 0)
            & (industry != '')
        )
        if valid_mask.sum() < 30:
            return np.array(factor_values, dtype=np.float64)

        result = factor.copy()

        # 1. 极值处理
        if method == 'winsorize':
            result = self._winsorize(result, self.winsorize_pct)
        elif method == 'zscore':
            result = self._zscore(result)
        elif method == 'rank':
            result = self._rank_transform(result)

        # 2. 行业中性化
        if 'industry' in self.methods:
            result = self.industry_neutralize(result, industry)

        # 3. 市值中性化
        if 'market_cap' in self.methods:
            result = self.market_cap_neutralize(result, mcap)

        return result

    def industry_neutralize(
        self,
        factor: np.ndarray,
        industry_codes: np.ndarray,
    ) -> np.ndarray:
        """
        行业中性化：减去行业均值（O(n) 纯numpy向量化，无Python循环）

        factor_neutral = factor - industry_mean

        Args:
            factor:         因子值（1D array）
            industry_codes: 行业代码（1D array）

        Returns:
            行业中性化后的因子值
        """
        result = np.asarray(factor, dtype=np.float64).copy()
        industry = np.asarray(industry_codes)

        # 过滤空行业码（取子集，不改变原始数组长度）
        valid = ~pd.isna(industry) & (industry != '')
        if valid.sum() < 30:
            return factor

        # 保存原始值（用于还原NaN）
        orig_valid = result[valid].copy()

        # 用0填充NaN以便计算均值
        result[valid] = np.nan_to_num(result[valid], nan=0.0)

        # 行业去重 + 获取映射索引
        unique_ind, inverse = np.unique(industry[valid], return_inverse=True)

        # 一次性累加所有行业的因子值和计数
        n_industry = len(unique_ind)
        industry_sums = np.zeros(n_industry, dtype=np.float64)
        industry_counts = np.zeros(n_industry, dtype=np.int64)

        np.add.at(industry_sums, inverse, result[valid])  # 按行业索引累加
        np.add.at(industry_counts, inverse, 1)              # 计数

        # 计算行业均值（避免除0）
        industry_counts[industry_counts == 0] = 1
        industry_means = industry_sums / industry_counts

        # 广播减均值（向量化）
        result[valid] = result[valid] - industry_means[inverse]

        return result

    def market_cap_neutralize(
        self,
        factor: np.ndarray,
        market_cap: np.ndarray,
    ) -> np.ndarray:
        """
        市值中性化：市值回归取残差

        对 ln(market_cap) 做线性回归：
        factor = α + β * ln(mcap) + residual
        返回 residual

        Args:
            factor:     因子值（1D array）
            market_cap: 市值（1D array）

        Returns:
            市值中性化后的因子值（残差）
        """
        factor = np.asarray(factor, dtype=np.float64)
        mcap = np.asarray(market_cap, dtype=np.float64)

        # 过滤有效数据
        valid_mask = np.isfinite(factor) & (mcap > 0) & np.isfinite(mcap)
        if valid_mask.sum() < 10:
            return factor

        ln_mcap = np.log(mcap[valid_mask])
        y = factor[valid_mask]

        # OLS 一元线性回归（纯 numpy）
        n = len(y)
        x_mean = np.mean(ln_mcap)
        y_mean = np.mean(y)

        cov_xy = np.sum((ln_mcap - x_mean) * (y - y_mean))
        var_x = np.sum((ln_mcap - x_mean) ** 2)

        if abs(var_x) < 1e-10:
            return factor

        beta = cov_xy / var_x
        alpha = y_mean - beta * x_mean

        # 残差 = y - (alpha + beta * x)
        residual = y - (alpha + beta * ln_mcap)

        result = factor.copy()
        result[valid_mask] = residual
        return result

    def full_neutralize(
        self,
        factor: np.ndarray,
        market_cap: np.ndarray,
        industry_codes: np.ndarray,
    ) -> np.ndarray:
        """
        完整中性化（行业哑变量回归 + 市值回归，向量化版本）

        步骤：
        1. winsorize 极值处理
        2. 行业哑变量回归（向量化 groupby，无 Python 循环）
        3. 市值回归残差（O(n) 一遍扫描）
        4. 返回 ε
        """
        factor = np.asarray(factor, dtype=np.float64)
        mcap = np.asarray(market_cap, dtype=np.float64)
        industry = np.asarray(industry_codes)

        valid_mask = (
            np.isfinite(factor)
            & (mcap > 0)
            & np.isfinite(mcap)
            & (industry != '')
        )
        if valid_mask.sum() < 30:
            return factor

        result = factor.copy()

        # Step 1: Winsorize
        result = self._winsorize(result, self.winsorize_pct)

        # Step 2: 行业哑变量回归（使用 pandas groupby 向量化）
        try:
            import pandas as pd
            n = len(result)
            df_temp = pd.DataFrame({'factor': result, 'industry': industry})
            group_means = df_temp.groupby('industry', dropna=False)['factor'].transform('mean')
            result = result - group_means.values
        except Exception:
            # 回退到循环版本
            unique_industries = np.unique(industry[industry != ''])
            for ind in unique_industries:
                mask = industry == ind
                if mask.sum() < 1:
                    continue
                ind_mean = np.mean(result[mask])
                result[mask] = result[mask] - ind_mean

        # Step 3: 市值回归残差
        ln_mcap = np.log(mcap[valid_mask])
        y = result[valid_mask]

        x_mean = np.mean(ln_mcap)
        y_mean = np.mean(y)

        cov_xy = np.sum((ln_mcap - x_mean) * (y - y_mean))
        var_x = np.sum((ln_mcap - x_mean) ** 2)

        if abs(var_x) > 1e-10:
            beta = cov_xy / var_x
            alpha = y_mean - beta * x_mean
            residual = y - (alpha + beta * ln_mcap)
            result[valid_mask] = residual

        return result

    # ------------------------------------------------------------------
    # 私有工具
    # ------------------------------------------------------------------

    @staticmethod
    def _winsorize(arr: np.ndarray, pct: float = 0.01) -> np.ndarray:
        """极值截断：把超出 percentile 的值替换为边界值"""
        arr = np.asarray(arr, dtype=np.float64)
        lower = np.nanpercentile(arr, pct * 100)
        upper = np.nanpercentile(arr, (1 - pct) * 100)
        result = np.clip(arr, lower, upper)
        return result

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        """Z-score 标准化"""
        arr = np.asarray(arr, dtype=np.float64)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std < 1e-10:
            return arr - mean
        return (arr - mean) / std

    @staticmethod
    def _rank_transform(arr: np.ndarray) -> np.ndarray:
        """排名归一化：转为 [0,1] 区间"""
        arr = np.asarray(arr, dtype=np.float64)
        n = len(arr)
        ranks = arr.argsort().argsort() + 1  # 1-indexed rank
        return (ranks - 1) / (n - 1) if n > 1 else arr

    def neutralize_factor_list(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        对因子列表中每个因子做中性化处理
        返回中性化后的因子名列表（格式：原名_neutralized）
        使用 joblib 8进程并行计算（写入串行执行）
        """
        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)

        # ---- 预加载所有因子数据（避免并行 worker 内重复 SQL 查询）----
        factor_df = self.store.df(f"""
            SELECT factor_name, ts_code, trade_date, value
            FROM factors
            WHERE trade_date BETWEEN '{start_fmt}' AND '{end_fmt}'
              AND value IS NOT NULL
            ORDER BY trade_date, ts_code
        """)
        # 按因子名拆分
        factor_by_name: Dict[str, pd.DataFrame] = {}
        for fname in factor_names:
            sub = factor_df[factor_df['factor_name'] == fname].copy()
            if not sub.empty:
                factor_by_name[fname] = sub

        # ---- 预加载市值数据（所有日期，一次性查完）----
        mcap_df = self.store.df(f"""
            SELECT ts_code, trade_date, close * vol as mcap
            FROM stock_daily
            WHERE trade_date BETWEEN '{start_fmt}' AND '{end_fmt}'
              AND close > 0 AND vol > 0
        """)

        def _neutralize_one(fname: str):
            """单个因子中性化（供并行调用，返回待写入 DataFrame 或 None）"""
            try:
                if fname not in factor_by_name:
                    self._log(f"中性化跳过（无数据）: {fname}")
                    return fname, None

                df = factor_by_name[fname].copy()
                merged = df.merge(mcap_df, on=['ts_code', 'trade_date'], how='inner')
                if merged.empty or len(merged) < 100:
                    self._log(f"中性化跳过（合并不足）: {fname}")
                    return fname, None

                neutralized_values = self.neutralize(
                    merged['value'].values,
                    merged['mcap'].values,
                    merged['ts_code'].values,
                    method='full'
                )

                neutralized_name = f"{fname}_neutralized"
                neutralized_df = merged[['ts_code', 'trade_date']].copy()
                neutralized_df['factor_name'] = neutralized_name
                neutralized_df['value'] = neutralized_values
                neutralized_df['zscore'] = 0.0
                return neutralized_name, neutralized_df
            except Exception as e:
                self._log(f"中性化失败 [{fname}]: {e}")
                return fname, None

        n_workers = min(len(factor_names), 8) if HAS_JOBLIB else 1
        if n_workers > 1 and len(factor_names) > 1:
            self._log(f"[并行] 中性化处理 {len(factor_names)} 个因子，workers={n_workers}")
            results = joblib.Parallel(n_jobs=n_workers, prefer="processes", timeout=300)(
                joblib.delayed(_neutralize_one)(fname) for fname in factor_names
            )
        else:
            results = [_neutralize_one(fname) for fname in factor_names]

        # 串行写入 DuckDB（避免锁冲突）
        neutralized = []
        for fname, neutralized_df in results:
            if neutralized_df is not None:
                try:
                    self.store.insert('factors', neutralized_df)
                    self._log(f"中性化完成: {fname} -> {fname}_neutralized, {len(neutralized_df)} 条")
                    neutralized.append(f"{fname}_neutralized")
                except Exception as e:
                    self._log(f"写入失败 [{fname}]: {e}")
                    neutralized.append(fname)
            else:
                neutralized.append(fname)
        return neutralized

    def _format_date(self, date_str: str) -> str:
        """YYYYMMDD -> YYYY-MM-DD"""
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[Neutralizer] {msg}")
        else:
            print(f"[Neutralizer] {msg}")
