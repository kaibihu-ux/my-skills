import optuna
from typing import Dict, List

from .genetic_optimizer import GeneticOptimizer
from .rl_optimizer import RLOptimizer


class BayesianOptimizer:
    """贝叶斯优化器"""

    def __init__(self, backtester, logger, start_date: str = None, end_date: str = None):
        self.backtester = backtester
        self.logger = logger
        self.start_date = start_date or '20200101'
        self.end_date = end_date or '20231231'

    def optimize(self, strategy: Dict, n_trials: int = 50) -> Dict:
        """优化策略参数"""
        # 预加载回测数据到内存，避免每次 trial 重复查询 DuckDB
        # 只加载策略使用的因子，减少内存占用
        factor_names = strategy.get('factors', [])
        self.backtester.preload_data(self.start_date, self.end_date, factor_names)

        def objective(trial):
            params = {
                'holding_period': trial.suggest_int('holding_period', 5, 60),
                'stop_loss': trial.suggest_float('stop_loss', 0.03, 0.15),
                'take_profit': trial.suggest_float('take_profit', 0.10, 0.30),
            }
            result = self.backtester.run(strategy, params, self.start_date, self.end_date)
            return result.get('sharpe_ratio', 0)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=4)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value
        }


class GridSearchOptimizer:
    """网格搜索优化器"""

    def __init__(self, backtester, logger, start_date: str = None, end_date: str = None):
        self.backtester = backtester
        self.logger = logger
        self.start_date = start_date or '20200101'
        self.end_date = end_date or '20231231'

    def optimize(self, strategy: Dict, param_grid: Dict) -> Dict:
        """网格搜索最优参数（joblib 4进程并行）"""
        try:
            import joblib
            HAS_JOBLIB = True
        except ImportError:
            HAS_JOBLIB = False

        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = list(itertools.product(*values))

        # 预加载回测数据到内存，避免每个组合重复查询 DuckDB
        # 只加载策略使用的因子，减少内存占用
        factor_names = strategy.get('factors', [])
        self.backtester.preload_data(self.start_date, self.end_date, factor_names)

        def _eval_combo(combo):
            params = dict(zip(keys, combo))
            result = self.backtester.run(strategy, params, self.start_date, self.end_date)
            return combo, params, result.get('sharpe_ratio', -999)

        best_sharpe = -999
        best_params = {}
        best_combo = None

        if HAS_JOBLIB and len(all_combinations) > 4:
            n_workers = min(4, len(all_combinations))
            self.logger.info(f"[GridSearch] {len(all_combinations)} 种组合，workers={n_workers}")
            results = joblib.Parallel(n_jobs=n_workers, prefer="threads", timeout=600)(
                joblib.delayed(_eval_combo)(combo) for combo in all_combinations
            )
            for combo, params, sharpe in results:
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_combo = combo
        else:
            for combo in all_combinations:
                _, params, sharpe = _eval_combo(combo)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    best_combo = combo

        self.logger.info(f"[GridSearch] 最优: Sharpe={best_sharpe:.4f}, 组合={best_combo}")
        return {'best_params': best_params, 'best_value': best_sharpe}
