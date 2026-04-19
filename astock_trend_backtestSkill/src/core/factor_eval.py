import pandas as pd
import numpy as np
from typing import Dict, List
import json
import threading
from functools import wraps

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# 延迟导入避免循环依赖
_monotonicity_tester = None


def _timeout_call(timeout_sec: float, default, func, *args, **kwargs):
    """
    用独立线程为 func 设置超时限制。
    超时返回 default，异常返回 default，正常返回函数返回值。
    """
    result = [default]
    exc = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return default, True   # 超时
    if exc[0]:
        return default, False  # 异常
    return result[0], False   # 正常


def _get_monotonicity_tester(store, logger, config):
    global _monotonicity_tester
    if _monotonicity_tester is None:
        try:
            from src.core.monotonicity_tester import MonotonicityTester
            _monotonicity_tester = MonotonicityTester(store, logger, config)
        except ImportError:
            pass
    return _monotonicity_tester


class ICEvaluator:
    """IC评估器"""
    
    @staticmethod
    def calc_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        """计算IC（Information Coefficient）—— Spearman秩相关"""
        valid_mask = factor_values.notna() & future_returns.notna()
        if valid_mask.sum() < 30:
            return 0.0
        return factor_values[valid_mask].corr(future_returns[valid_mask], method='spearman')
    
    @staticmethod
    def calc_pearson_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        """计算Pearson IC"""
        valid_mask = factor_values.notna() & future_returns.notna()
        if valid_mask.sum() < 30:
            return 0.0
        return factor_values[valid_mask].corr(future_returns[valid_mask], method='pearson')
    
    @staticmethod
    def calc_ic_series(
        factor_df: pd.DataFrame,
        return_series: pd.Series,
        periods: list = [5, 10, 20]
    ) -> Dict:
        """计算多期IC"""
        result = {}
        for period in periods:
            # future_ret 是从当期到未来period天的收益
            future_ret = return_series.groupby(level=0).shift(-period)
            # 对齐：factor_df 的索引是 (ts_code, trade_date)，return_series 的索引是 trade_date
            # 重新组织为以 trade_date 为索引的面板数据
            valid_mask = factor_df.notna()
            ic_vals = []
            for date in factor_df.index.get_level_values(0).unique():
                if date not in future_ret.index:
                    continue
                fv = factor_df.loc[date].dropna()
                fr = future_ret.loc[date]
                if len(fv) > 30 and pd.notna(fr):
                    ic = ICEvaluator.calc_ic(fv, pd.Series([fr] * len(fv), index=fv.index))
                    if ic != 0:
                        ic_vals.append(ic)
            if ic_vals:
                result[f'ic_{period}'] = np.mean(ic_vals)
            else:
                result[f'ic_{period}'] = 0.0
        return result
    
    @staticmethod
    def calc_ir(ic_series: pd.Series) -> float:
        """计算IR（Information Ratio）= IC均值 / IC标准差"""
        if len(ic_series) < 10 or ic_series.std() == 0:
            return 0.0
        return ic_series.mean() / ic_series.std()


class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, store, logger, config: Dict = None):
        self.store = store
        self.logger = logger
        self.config = config or {}
        self.ic_evaluator = ICEvaluator()

    def _save_ic_series(self, factor_name: str, ic_results: Dict, periods: List[int]):
        """保存 IC 序列到 factor_ic 表（事务保护，避免删除后插入失败）"""
        try:
            # 只使用 ic_20 的 IC 序列（最常用）
            ic_series_dict = ic_results.get('ic_20_series', {})
            if not ic_series_dict:
                # 退而求其次用 ic_5
                ic_series_dict = ic_results.get('ic_5_series', {})

            if not ic_series_dict:
                self.logger.info(f"[{factor_name}] 无IC序列数据可保存")
                return

            ic_records = []
            for date, ic in ic_series_dict.items():
                # 统一日期格式为 YYYY-MM-DD
                date_str = str(date)
                if 'T' in date_str:
                    date_str = date_str[:10]
                elif ' ' in date_str:
                    date_str = date_str.split(' ')[0]
                ic_records.append({
                    'factor_name': factor_name,
                    'date': date_str,
                    'ic': float(ic),
                    'rank_ic': 0.0
                })

            if ic_records:
                ic_df = pd.DataFrame(ic_records)
                ic_df['date'] = ic_df['date'].astype(str)
                ic_df['ic'] = ic_df['ic'].astype(float)
                ic_df['rank_ic'] = ic_df['rank_ic'].astype(float)

                # 事务保护：删除+插入作为一个事务
                conn = self.store.conn
                with self.store._lock:
                    delete_sql = f"DELETE FROM factor_ic WHERE factor_name = '{factor_name}'"
                    conn.execute(delete_sql)
                    self.store.insert('factor_ic', ic_df)
                    conn.execute("CHECKPOINT")
                self.logger.info(f"[{factor_name}] IC序列已保存: {len(ic_records)} 条")
        except Exception as e:
            self.logger.warning(f"[{factor_name}] 保存IC序列失败: {e}")

    def evaluate_factor(
        self, 
        factor_name: str, 
        start_date: str, 
        end_date: str,
        periods: list = [5, 10, 20]
    ) -> Dict:
        """
        评估单个因子
        从 factors 表读取因子值，计算 IC 序列、IR、分组回测收益
        """
        # 转换日期格式
        start_fmt = self._format_date(start_date)
        end_fmt = self._format_date(end_date)
        
        # 读取因子数据
        factor_df = self.store.df(
            f"""SELECT ts_code, trade_date, value 
                FROM factors 
                WHERE factor_name = '{factor_name}'
                  AND trade_date BETWEEN '{start_fmt}' AND '{end_fmt}'
                ORDER BY trade_date, ts_code"""
        )
        
        if factor_df.empty:
            self.logger.warning(f"因子 {factor_name} 无数据")
            return self._empty_result(factor_name)
        
        # 读取价格数据计算未来收益
        price_df = self.store.df(
            f"""SELECT ts_code, trade_date, close 
                FROM stock_daily 
                WHERE trade_date BETWEEN '{start_fmt}' AND '{end_fmt}'
                ORDER BY ts_code, trade_date"""
        )
        
        if price_df.empty:
            return self._empty_result(factor_name)
        
        # 计算未来收益（未来5日、10日、20日收益）
        for period in periods:
            price_df[f'return_{period}'] = price_df.groupby('ts_code')['close'].pct_change(period).shift(-period)
        
        # 合并因子值和收益
        merged = factor_df.merge(price_df, on=['ts_code', 'trade_date'], how='inner')
        
        if merged.empty or len(merged) < 100:
            return self._empty_result(factor_name)
        
        # ===== 中性化处理（可选，在IC计算前）=====
        try:
            ncfg = getattr(self, 'config', {}).get('neutralization', {})
            if ncfg.get('enabled', True):
                from src.core.neutralizer import Neutralizer
                neutralizer = Neutralizer(self.store, self.logger, self.config)
                # 对每个截面的因子值做行业中性化（超时保护 60s）
                neutralized_values, is_timeout = _timeout_call(
                    60.0,
                    merged['value'].values,  # 超时时回退为原始因子值
                    neutralizer.industry_neutralize,
                    merged['value'].values,
                    merged['ts_code'].values,
                )
                if is_timeout:
                    self.logger.warning(f"[{factor_name}] 中性化超时(60s)，回退原始因子值")
                merged = merged.copy()
                merged['value'] = neutralized_values
                self.logger.info(f"[{factor_name}] 中性化处理完成, IC计算前")
        except Exception as e:
            self.logger.warning(f"中性化失败 [{factor_name}]: {e}")

        # ===== 计算 IC 序列（向量化，按日期分组避免循环内重复SQL）=====
        # merged 已按 trade_date 分组，无需再逐日期循环
        ic_results = {}
        for period in periods:
            ret_col = f'return_{period}'

            # 使用 groupby 一次性计算所有截面的 IC（无重复 SQL）
            ic_data = []
            for date, group in merged.groupby('trade_date'):
                fv = group['value']
                ret = group[ret_col]
                ic = ICEvaluator.calc_ic(fv, ret)
                if ic != 0 and pd.notna(ic):
                    ic_data.append({'date': date, 'ic': ic})

            if ic_data:
                ic_df = pd.DataFrame(ic_data).set_index('date')['ic']
                ic_results[f'ic_{period}'] = float(ic_df.mean())
                ic_results[f'ic_{period}_std'] = float(ic_df.std())
                ic_results[f'ic_{period}_series'] = ic_df.to_dict()
            else:
                ic_results[f'ic_{period}'] = 0.0
                ic_results[f'ic_{period}_std'] = 0.0
                ic_results[f'ic_{period}_series'] = {}
        
        # 计算 IR（使用 ic_20 的 IC 序列）
        ic_series_full = pd.Series(ic_results.get('ic_20_series', {}))
        ir = ICEvaluator.calc_ir(ic_series_full) if len(ic_series_full) > 0 else 0.0
        
        # ===== IC 衰减分析 =====
        ic_decay = {}
        if 'ic_5' in ic_results and ic_results['ic_5'] != 0:
            ic_decay['decay_5_to_10'] = ic_results.get('ic_10', 0) / ic_results['ic_5']
            ic_decay['decay_5_to_20'] = ic_results.get('ic_20', 0) / ic_results['ic_5']
        else:
            ic_decay['decay_5_to_10'] = 0
            ic_decay['decay_5_to_20'] = 0
        
        # ===== 分组回测 =====
        group_returns = self._quantile_backtest(merged, 'value', 'return_20')
        
        # ===== Top vs Bottom 分组收益差 =====
        spread = group_returns.get('top_return', 0) - group_returns.get('bottom_return', 0)

        # ===== 计算换手率 =====
        turnover_rate = self._calc_turnover_rate(factor_df)

        # ===== 胜率（Top组 vs Bottom组）=====
        win_rate = group_returns.get('top_return', 0) > group_returns.get('bottom_return', 0)

        # ===== 分组单调性检验（超时保护 60s）=====
        monotonicity_result_raw, is_timeout = _timeout_call(
            60.0,
            {'monotonicity_score': 0.0, 'passed': False},
            self._eval_monotonicity,
            factor_name, start_date, end_date,
        )
        if is_timeout:
            self.logger.warning(f"[{factor_name}] 单调性检验超时(60s)，跳过")
            monotonicity_result = {'monotonicity_score': 0.0, 'passed': False}
        else:
            monotonicity_result = monotonicity_result_raw

        # 保存 IC 序列到 factor_ic 表
        self._save_ic_series(factor_name, ic_results, periods)

        return {
            'factor_name': factor_name,
            'ic_mean': ic_results.get('ic_20', 0),
            'ic_std': ic_results.get('ic_20_std', 0),
            'ir': float(ir),
            'ic_5': ic_results.get('ic_5', 0),
            'ic_10': ic_results.get('ic_10', 0),
            'ic_20': ic_results.get('ic_20', 0),
            'ic_decay_5': ic_decay.get('decay_5_to_10', 0),
            'ic_decay_10': ic_decay.get('decay_5_to_20', 0),
            'top_group_return': group_returns.get('top_return', 0),
            'bottom_group_return': group_returns.get('bottom_return', 0),
            'group_spread': spread,
            'n_samples': len(merged),
            'group_returns': group_returns,
            'turnover_rate': turnover_rate,
            'win_rate': win_rate,
            # 单调性
            'monotonicity_score': monotonicity_result.get('monotonicity_score', 0.0),
            'monotonicity_passed': monotonicity_result.get('passed', False),
            'monotonicity_details': monotonicity_result,
        }
    
    def _quantile_backtest(
        self, 
        merged: pd.DataFrame, 
        factor_col: str, 
        return_col: str,
        n_groups: int = 5
    ) -> Dict:
        """分组回测：按因子值分 n 组，计算每组未来收益均值"""
        results = {}
        
        for date, group in merged.groupby('trade_date'):
            valid = group[group[factor_col].notna() & group[return_col].notna()]
            if len(valid) < n_groups:
                continue
            
            try:
                valid = valid.copy()
                valid['quantile'] = pd.qcut(valid[factor_col], n_groups, labels=False, duplicates='drop')
                
                for q in range(n_groups):
                    q_returns = valid[valid['quantile'] == q][return_col]
                    key = f'q{q+1}_return'
                    if key not in results:
                        results[key] = []
                    if len(q_returns) > 0:
                        results[key].append(q_returns.mean())
            except Exception:
                continue
        
        if not results:
            return {}
        
        for k, v in results.items():
            results[k] = float(np.mean(v)) if v else 0
        
        # Top 组 = Q5 (最高因子值)，Bottom 组 = Q1 (最低因子值)
        results['top_return'] = results.get('q5_return', 0)
        results['bottom_return'] = results.get('q1_return', 0)
        
        return results
    
    def _calc_turnover_rate(self, factor_df: pd.DataFrame) -> float:
        """计算因子月均换手率（因子值排名变化频率）
        
        计算逻辑：
        - 每月末按因子值排名
        - 换手率 = 排名发生变化的股票占比
        - 年度平均 = 各月换手率均值
        """
        if factor_df.empty or len(factor_df) < 100:
            return 0.0
        
        try:
            # 确保 trade_date 是字符串格式
            df = factor_df.copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['year_month'] = df['trade_date'].dt.to_period('M')
            
            # 获取所有有数据的月份
            months = sorted(df['year_month'].unique())
            if len(months) < 2:
                return 0.0
            
            monthly_turnovers = []
            
            for i in range(1, len(months)):
                prev_month = months[i - 1]
                curr_month = months[i]
                
                # 获取上月末的因子值排名
                prev_data = df[df['year_month'] == prev_month]
                if prev_data.empty:
                    continue
                
                # 获取当月末的因子值排名
                curr_data = df[df['year_month'] == curr_month]
                if curr_data.empty:
                    continue
                
                # 按 ts_code 取每月末最后一个有效因子值
                prev_last = prev_data.sort_values('trade_date').groupby('ts_code')['value'].last().dropna()
                curr_last = curr_data.sort_values('trade_date').groupby('ts_code')['value'].last().dropna()
                
                # 找两期都有数据的股票
                common_codes = prev_last.index.intersection(curr_last.index)
                if len(common_codes) < 50:
                    continue
                
                prev_ranks = prev_last[common_codes].rank(ascending=False)
                curr_ranks = curr_last[common_codes].rank(ascending=False)
                
                # 排名变化比例（排名差 > 0 的占比）
                rank_diff = (curr_ranks - prev_ranks).abs()
                turnover = (rank_diff > 0).sum() / len(common_codes)
                monthly_turnovers.append(turnover)
            
            if monthly_turnovers:
                return float(np.mean(monthly_turnovers))
            return 0.0
        except Exception as e:
            self.logger.warning(f"换手率计算失败: {e}")
            return 0.0

    def _eval_monotonicity(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
    ) -> Dict:
        """执行因子单调性检验（集成 MonotonicityTester）"""
        try:
            tester = _get_monotonicity_tester(self.store, self.logger, self.config)
            if tester is None:
                return {'monotonicity_score': 0.0, 'passed': False}
            return tester.test(factor_name, start_date, end_date)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"单调性检验失败 [{factor_name}]: {e}")
            return {'monotonicity_score': 0.0, 'passed': False}

    def _empty_result(self, factor_name: str) -> Dict:
        """返回空结果"""
        return {
            'factor_name': factor_name,
            'ic_mean': 0.0,
            'ic_std': 0.0,
            'ir': 0.0,
            'ic_5': 0.0,
            'ic_10': 0.0,
            'ic_20': 0.0,
            'ic_decay_5': 0.0,
            'ic_decay_10': 0.0,
            'top_group_return': 0.0,
            'bottom_group_return': 0.0,
            'group_spread': 0.0,
            'n_samples': 0,
            'turnover_rate': 0.0,
            'win_rate': False,
            'monotonicity_score': 0.0,
            'monotonicity_passed': False,
            'monotonicity_details': {},
        }
    
    def _format_date(self, date_str: str) -> str:
        """YYYYMMDD -> YYYY-MM-DD"""
        if date_str is None:
            return ''
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
    
    def rank_factors(self, evaluations: list) -> list:
        """因子排名 - 按 IR 降序"""
        return sorted(evaluations, key=lambda x: x.get('ir', 0), reverse=True)
    
    def _eval_one_factor(self, fname: str, start_date: str, end_date: str) -> Dict:
        """评估单个因子（供并行调用）"""
        result, is_timeout = _timeout_call(
            120.0,
            self._empty_result(fname),
            self.evaluate_factor, fname, start_date, end_date,
        )
        if is_timeout:
            self.logger.warning(f"因子 {fname} 评估超时(120s)，已跳过")
        return result

    def evaluate_multiple(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """批量评估多个因子（每个因子最多 120s 超时，joblib 8进程并行）"""
        n_workers = min(len(factor_names), 8) if HAS_JOBLIB else 1
        if n_workers > 1 and len(factor_names) > 1:
            self.logger.info(f"[并行] 因子评估 {len(factor_names)} 个，workers={n_workers}")
            results = joblib.Parallel(n_jobs=n_workers, prefer="threads", timeout=300)(
                joblib.delayed(self._eval_one_factor)(fname, start_date, end_date)
                for fname in factor_names
            )
        else:
            results = [self._eval_one_factor(fname, start_date, end_date) for fname in factor_names]
        return results
