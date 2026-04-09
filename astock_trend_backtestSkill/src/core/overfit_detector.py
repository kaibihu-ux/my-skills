"""
过拟合检测器：PBO / DSR / CSCV / 多重检验偏误矫正

完全使用 numpy 实现基础统计（mean/std/norm_cdf/norm_ppf），
不依赖 scipy.stats。所有阈值/参数均从配置读取。
"""

import numpy as np
from typing import List, Dict, Optional


# ------------------------------------------------------------------
# 纯 numpy 实现的标准正态分布函数（避免 scipy.stats 依赖）
# ------------------------------------------------------------------

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    标准正态分布 CDF。
    Φ(x) = 0.5 * erfc(-x/√2)，使用 math.erfc（Python 标准库）。
    对数组使用逐元素计算。
    """
    import math as _math
    x = np.asarray(x, dtype=np.float64)
    sqrty = 1.0 / np.sqrt(2.0)
    # 逐元素计算：0.5 * erfc(-x / sqrt(2))
    result = np.empty_like(x, dtype=np.float64)
    for i, val in enumerate(x.flat):
        result.flat[i] = 0.5 * _math.erfc(-val * sqrty)
    return result


def _norm_ppf(q: float) -> float:
    """
    标准正态分布分位数函数（PPF）。
    使用二分搜索法（bisection），基于 math.erfc 实现。
    精度 1e-10，最多 200 次迭代。
    """
    import math as _math
    if q <= 0.0:
        return -float('inf')
    if q >= 1.0:
        return float('inf')
    if q == 0.5:
        return 0.0

    lo, hi = -12.0, 12.0
    tol = 1e-10
    sqrt2 = np.sqrt(2.0)
    for _ in range(200):
        mid = (lo + hi) * 0.5
        cdf_mid = 0.5 * _math.erfc(-mid / sqrt2)
        if abs(hi - lo) < tol:
            break
        if cdf_mid < q:
            lo = mid
        else:
            hi = mid
    return float(mid)


# ------------------------------------------------------------------
# OverfitDetector 主类
# ------------------------------------------------------------------

class OverfitDetector:
    """
    过拟合检测器

    指标说明:
    - PBO (Probability of Overfitting): Bailey et al. (2017)
      PBO < 0.5 表示过拟合概率低，策略可接受
    - DSR (Deflated Sharpe Ratio): Bailey & Lopez de Prado (2012)
      DSR > 0 表示考虑运气调整后仍显著
    - CSCV (Combinatorial Symmetric Cross-Validation): Lopez de Prado (2015)
      CSCV 越接近 1 越好，CSCV > 0.5 通常可接受
    """

    def __init__(self, store, logger, config: Optional[Dict] = None):
        """
        Args:
            store: 数据存储对象（用于读取配置，可为 None）
            logger: 日志记录器
            config: 配置字典（若为 None，从 store 读取或使用默认值）
        """
        self.store = store
        self.logger = logger
        self._load_config(config)

    def _load_config(self, config: Optional[Dict] = None):
        """从配置加载过拟合检测参数"""
        if config is not None:
            ocfg = config.get('overfit_detection', {})
            mcfg = config.get('multiple_testing', {})
        else:
            ocfg = {}
            mcfg = {}

        self.enabled = ocfg.get('enabled', True)
        self.pbo_threshold = float(ocfg.get('pbo_threshold', 0.5))
        self.dsr_threshold = float(ocfg.get('dsr_threshold', 0.0))
        self.cscv_threshold = float(ocfg.get('cscv_threshold', 0.5))
        self.n_splits_cscv = int(ocfg.get('n_splits_cscv', 4))
        self.oos_ratio = float(ocfg.get('oos_ratio', 0.2))

        self.mt_method = mcfg.get('method', 'bh')
        self.fdr_rate = float(mcfg.get('fdr_rate', 0.05))

    # ------------------------------------------------------------------
    # PBO - Probability of Overfitting
    # ------------------------------------------------------------------

    def calc_pbo(
        self,
        is_returns: List[float],
        oos_returns: List[float],
        n_portfolios: int = 10,
    ) -> float:
        """
        计算回测过拟合概率 (Probability of Overfitting)

        Bailey et al. (2017) 方法：
        1. 把样本内数据分成 n_portfolios 等份
        2. 对每个子样本 i：
           - 以 i 份作为伪"样本外"，其余 n-1 份合并为伪"样本内"
           - 计算样本内/外各自的夏普比率
        3. 统计在样本外仍保持"最优"（高于中位数）的次数
        4. PBO = 1 - (样本外最优次数 / 总次数)
           即：样本外最优策略跑输基准（随机）的概率

        PBO 越低越好：
        - PBO < 0.3: 低过拟合风险
        - 0.3 <= PBO < 0.5: 中等风险
        - PBO >= 0.5: 高过拟合风险
        """
        if not self.enabled:
            return 0.0

        is_arr = np.array(is_returns, dtype=np.float64)
        oos_arr = np.array(oos_returns, dtype=np.float64)

        if len(is_arr) < n_portfolios * 2 or len(oos_arr) < 2:
            self._log("PBO: 数据不足，返回默认值 0.5")
            return 0.5

        n = len(is_arr)
        split_size = n // n_portfolios
        if split_size < 2:
            self._log("PBO: split_size < 2，数据不足，返回默认值 0.5")
            return 0.5

        is_sharpe_list = []
        oos_sharpe_list = []

        for i in range(n_portfolios):
            # 构建伪样本内 mask（排除第 i 份）
            is_mask = np.ones(n, dtype=bool)
            is_mask[i * split_size:(i + 1) * split_size] = False

            is_data = is_arr[is_mask]
            oos_data = is_arr[~is_mask]  # 第 i 份作为伪样本外

            if len(is_data) < 2 or len(oos_data) < 2:
                continue

            is_s = self._sharpe_ratio(is_data)
            oos_s = self._sharpe_ratio(oos_data)

            is_sharpe_list.append(is_s)
            oos_sharpe_list.append(oos_s)

        if not is_sharpe_list:
            return 0.5

        # PBO = 样本外 Top 策略跑输中位数的比例
        median_oos = np.median(oos_sharpe_list)
        best_oos_count = sum(1 for s in oos_sharpe_list if s > median_oos)
        pbo = 1.0 - (best_oos_count / len(oos_sharpe_list))
        return float(np.clip(pbo, 0.0, 1.0))

    # ------------------------------------------------------------------
    # DSR - Deflated Sharpe Ratio
    # ------------------------------------------------------------------

    def calc_dsr(
        self,
        all_sharpe_ratios: List[float],
        all_returns: List[List[float]],
        n_trials: int,
    ) -> float:
        """
        计算通缩夏普比率 (Deflated Sharpe Ratio)

        Bailey & Lopez de Prado (2012) 方法：
        DSR = (SR* - z_q * σ_SR) / √(1 + γ_1 * k + γ_2 * k*(k-1))

        其中：
        - SR* 是样本内最优夏普
        - z_q = norm.ppf(1 - 1 / n_trials) 是 Bonferroni 分位数
        - σ_SR 是所有试验夏普的标准差
        - γ_1 ≈ 0，γ_2 ≈ 0（简化公式：DSR = SR* - z_q * σ_SR / √n_trials）

        DSR > 0 表示考虑多重检验运气调整后，策略仍显著
        """
        if not self.enabled:
            return 0.0

        sr_arr = np.array(all_sharpe_ratios, dtype=np.float64)
        n = len(sr_arr)

        if n < 2:
            self._log("DSR: n_trials < 2，返回默认值 0.0")
            return 0.0

        # 样本外 Sharpe 的标准差
        std_sr = float(np.std(sr_arr, ddof=1)) if n > 1 else 0.0

        # 最优策略的"运气调整"分位数
        # z_q = norm.ppf(1 - 1/n_trials)
        z_q = _norm_ppf(1.0 - 1.0 / max(n_trials, 2))

        # 平均夏普
        mean_sr = float(np.mean(sr_arr))

        # 最优夏普
        best_sr = float(np.max(sr_arr))

        # DSR 简化公式（不依赖 scipy）
        # DSR = SR* - z_q * σ_SR / √n_trials
        # 等价于：考虑多重检验后，最高夏普的"运气调整"
        if std_sr > 1e-10 and n_trials > 0:
            dsr = best_sr - z_q * std_sr / np.sqrt(n_trials)
        else:
            dsr = best_sr

        return float(dsr)

    # ------------------------------------------------------------------
    # CSCV - Combinatorial Symmetric Cross-Validation
    # ------------------------------------------------------------------

    def calc_cscv(
        self,
        strategy_returns: List[float],
        n_splits: int = 4,
    ) -> float:
        """
        计算组合对称交叉验证得分 (Combinatorial Symmetric CV)

        Lopez de Prado (2015) 方法：
        1. 把数据分成 n_splits 个等份
        2. 遍历所有 C(n_splits, n_splits-1) 个子集组合
           即：每次用 n_splits-1 份做样本内，1份做样本外
        3. 计算每个组合的样本内/外夏普
        4. CSCV = mean(oos_sharpe) / mean(is_sharpe)
           越接近 1 越好，越低说明越不稳定（过拟合）
        """
        if not self.enabled:
            return 1.0

        n_splits = int(n_splits or self.n_splits_cscv)
        if n_splits < 2:
            n_splits = 2

        ret_arr = np.array(strategy_returns, dtype=np.float64)
        n = len(ret_arr)

        min_n = n_splits * 2
        if n < min_n:
            self._log(f"CSCV: 数据不足 ({n} < {min_n})，返回默认值 1.0")
            return 1.0

        split_size = n // n_splits
        if split_size < 2:
            self._log("CSCV: split_size < 2，返回默认值 1.0")
            return 1.0

        # 所有 C(n_splits, n_splits-1) 个子集组合
        from itertools import combinations

        oos_sharpes = []
        is_sharpes = []

        indices = list(range(n_splits))

        # 遍历每个作为样本外的组合
        for oos_idx in combinations(indices, n_splits - 1):
            # 构建样本内 mask（排除 oos_idx 指定的份）
            is_mask = np.ones(n, dtype=bool)
            for idx in oos_idx:
                start = idx * split_size
                end = (idx + 1) * split_size if idx < n_splits - 1 else n
                is_mask[start:end] = False

            is_ret = ret_arr[is_mask]
            oos_ret = ret_arr[~is_mask]

            if len(is_ret) < 2 or len(oos_ret) < 2:
                continue

            is_s = self._sharpe_ratio(is_ret)
            oos_s = self._sharpe_ratio(oos_ret)

            is_sharpes.append(is_s)
            oos_sharpes.append(oos_s)

        if not oos_sharpes or not is_sharpes:
            self._log("CSCV: 无有效组合，返回默认值 1.0")
            return 1.0

        mean_oos = float(np.mean(oos_sharpes))
        mean_is = float(np.mean(is_sharpes))

        if abs(mean_is) < 1e-10:
            return 0.0

        cscv = mean_oos / mean_is
        return float(np.clip(cscv, -2.0, 2.0))

    # ------------------------------------------------------------------
    # 多重检验偏误矫正
    # ------------------------------------------------------------------

    def calc_multiple_testing_adjustment(
        self,
        all_sharpe_ratios: List[float],
        n_trials: int,
    ) -> Dict:
        """
        多重检验偏误矫正

        方法：Bonferroni + Benjamini-Hochberg FDR
        1. 计算每个策略的原始 p 值（假设 Sharpe = 0）
           p = 2 * (1 - Φ(|SR|))
        2. Bonferroni 矫正：p_adj = min(1, p * n_trials)
        3. Benjamini-Hochberg FDR 矫正
        4. 返回矫正后显著策略数
        """
        if not self.enabled:
            return {
                'n_significant_bonferroni': 0,
                'n_significant_bh': 0,
                'bonferroni_adjusted': [],
                'bh_adjusted': [],
                'p_values': [],
            }

        sr_arr = np.array(all_sharpe_ratios, dtype=np.float64)
        n = len(sr_arr)

        if n < 1 or n_trials < 1:
            return {
                'n_significant_bonferroni': 0,
                'n_significant_bh': 0,
                'bonferroni_adjusted': [],
                'bh_adjusted': [],
                'p_values': [],
            }

        # 原始 p 值（假设 SR = 0，双尾）
        # p = 2 * (1 - Φ(|SR|))
        abs_sr = np.abs(sr_arr)
        p_values = 2.0 * (1.0 - _norm_cdf(abs_sr))
        p_values = np.clip(p_values, 0.0, 1.0)

        # Bonferroni 矫正
        p_bonf = np.minimum(1.0, p_values * n_trials)

        # Benjamini-Hochberg FDR 矫正
        sorted_indices = np.argsort(p_values)
        bh_adjusted = np.zeros_like(p_values, dtype=np.float64)
        for rank, idx in enumerate(sorted_indices):
            bh_adjusted[idx] = min(1.0, p_values[idx] * n_trials / (rank + 1))

        n_sig_bonf = int(np.sum(p_bonf < self.fdr_rate))
        n_sig_bh = int(np.sum(bh_adjusted < self.fdr_rate))

        return {
            'n_significant_bonferroni': n_sig_bonf,
            'n_significant_bh': n_sig_bh,
            'bonferroni_adjusted': p_bonf.tolist(),
            'bh_adjusted': bh_adjusted.tolist(),
            'p_values': p_values.tolist(),
        }

    # ------------------------------------------------------------------
    # 综合报告
    # ------------------------------------------------------------------

    def full_report(
        self,
        is_returns: List[float],
        oos_returns: List[float],
        all_sharpe_ratios: List[float],
        all_returns: List[List[float]],
        n_trials: int,
    ) -> Dict:
        """
        生成完整过拟合报告

        Returns:
            {
                'pbo': float,
                'dsr': float,
                'cscv': float,
                'multiple_testing': {...},
                'passed': bool,
                'rejection_reasons': [...],
            }
        """
        pbo = self.calc_pbo(is_returns, oos_returns)
        dsr = self.calc_dsr(all_sharpe_ratios, all_returns, n_trials)
        cscv = self.calc_cscv(is_returns)
        mt_result = self.calc_multiple_testing_adjustment(all_sharpe_ratios, n_trials)

        reasons = []
        if pbo >= self.pbo_threshold:
            reasons.append(f"PBO({pbo:.4f}) >= {self.pbo_threshold}")
        if dsr <= self.dsr_threshold:
            reasons.append(f"DSR({dsr:.4f}) <= {self.dsr_threshold}")
        if cscv < self.cscv_threshold:
            reasons.append(f"CSCV({cscv:.4f}) < {self.cscv_threshold}")

        passed = len(reasons) == 0

        self._log(
            f"过拟合报告: PBO={pbo:.4f}(<{self.pbo_threshold}? {'✓' if pbo < self.pbo_threshold else '✗'}), "
            f"DSR={dsr:.4f}(>{self.dsr_threshold}? {'✓' if dsr > self.dsr_threshold else '✗'}), "
            f"CSCV={cscv:.4f}(>{self.cscv_threshold}? {'✓' if cscv > self.cscv_threshold else '✗'}), "
            f"passed={'是' if passed else '否'}"
        )

        return {
            'pbo': float(pbo),
            'dsr': float(dsr),
            'cscv': float(cscv),
            'multiple_testing': mt_result,
            'passed': passed,
            'rejection_reasons': reasons,
            # 阈值快照（方便调试）
            '_thresholds': {
                'pbo_threshold': self.pbo_threshold,
                'dsr_threshold': self.dsr_threshold,
                'cscv_threshold': self.cscv_threshold,
                'fdr_rate': self.fdr_rate,
                'mt_method': self.mt_method,
            }
        }

    def is_acceptable(self, report: Dict) -> bool:
        """
        判断策略是否通过过拟合检测
        只有 passed=True 才接受
        """
        return bool(report.get('passed', False))

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray, risk_free: float = 0.03) -> float:
        """计算夏普比率（年化，假设无风险利率 3%）"""
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret < 1e-10:
            return 0.0
        # 年化（日频 × √252）
        annualized = (mean_ret * 252 - risk_free) / (std_ret * np.sqrt(252))
        return float(annualized)

    def _log(self, msg: str):
        """日志记录"""
        if self.logger:
            self.logger.info(f"[OverfitDetector] {msg}")
        else:
            print(f"[OverfitDetector] {msg}")
