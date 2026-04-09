import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd

# 导入所有模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.duckdb_store import DuckDBStore
from src.core.data_manager import DataManager
from src.core.factor_miner import FactorMiner, ALL_FACTORS, TREND_FACTORS, TECH_FACTORS, VOL_FACTORS, VOLUME_FACTORS
from src.core.factor_eval import FactorEvaluator
from src.core.strategy_gen import StrategyGenerator
from src.core.optimizer import BayesianOptimizer, GridSearchOptimizer
from src.core.backtester import BacktestExecutor
from src.core.factor_pool import FactorPoolManager
from src.core.performance import PerformanceAnalyzer
from src.utils.logger import Logger
from src.utils.config_loader import ConfigLoader


class SkillAPI:
    def __init__(self):
        self.logger = Logger('astock_factor_forge')
        self.config = ConfigLoader().load()
        self.store = DuckDBStore()
        self.data_mgr = DataManager(self.store, self.logger)
        self.factor_miner = FactorMiner(self.store, self.logger)
        self.factor_eval = FactorEvaluator(self.store, self.logger)
        self.strategy_gen = StrategyGenerator(self.store, self.logger)
        self.factor_pool = FactorPoolManager(self.store, self.logger, self.config)
        self.performance = PerformanceAnalyzer()

        # 初始化数据库表
        self.store.init_tables()

    def execute(self, params: dict) -> dict:
        """唯一执行入口"""
        request_id = params.get('request_id', str(uuid.uuid4()))
        action = params.get('action', 'run_backtest')

        self.logger.info(f"[{request_id}] Action: {action}")

        try:
            if action == 'init':
                return self._action_init(request_id, params)
            elif action == 'update_data':
                return self._action_update_data(request_id, params)
            elif action == 'mine_factors':
                return self._action_mine_factors(request_id, params)
            elif action == 'evaluate_factors':
                return self._action_evaluate_factors(request_id, params)
            elif action == 'generate_strategies':
                return self._action_generate_strategies(request_id, params)
            elif action == 'run_backtest':
                return self._action_run_backtest(request_id, params)
            elif action == 'optimize':
                return self._action_optimize(request_id, params)
            elif action == 'ga_optimize':
                return self._action_ga_optimize(request_id, params)
            elif action == 'rl_optimize':
                return self._action_rl_optimize(request_id, params)
            elif action == 'run_full_pipeline':
                return self._action_run_full_pipeline(request_id, params)
            elif action == 'feature_select':
                return self._action_feature_select(request_id, params)
            else:
                return {'code': 400, 'msg': f'Unknown action: {action}', 'request_id': request_id}

        except Exception as e:
            self.logger.error(f"[{request_id}] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'code': 500, 'msg': str(e), 'request_id': request_id}

    def _action_init(self, request_id: str, params: dict) -> dict:
        """初始化系统"""
        stock_count = self.data_mgr.update_stock_list()
        return {
            'code': 0, 'msg': 'success', 'request_id': request_id,
            'data': {'stock_count': stock_count}
        }

    def _action_update_data(self, request_id: str, params: dict) -> dict:
        """更新数据"""
        start = params.get('start_date', self.config['backtest']['start_date'])
        end = params.get('end_date', self.config['backtest']['end_date'])
        self.data_mgr.update_daily(start, end)
        return {'code': 0, 'msg': 'success', 'request_id': request_id}

    def _action_mine_factors(self, request_id: str, params: dict) -> dict:
        """挖掘因子"""
        n = params.get('n_candidates', 100)
        candidates = self.factor_miner.mine_candidates(n)
        return {'code': 0, 'msg': 'success', 'request_id': request_id,
                'data': {'candidates': len(candidates)}}

    def _action_evaluate_factors(self, request_id: str, params: dict) -> dict:
        """
        评估因子
        - 默认使用 ALL_FACTORS（技术面因子）进行全量评估
        - factor_names 参数可指定，否则默认评估所有可用因子
        - 评估结果存入 factor_pool 表
        """
        start_date = params.get('start_date', self.config['backtest']['start_date'])
        end_date = params.get('end_date', self.config['backtest']['end_date'])

        # 要评估的因子列表：默认 ALL_FACTORS
        factor_names = params.get('factor_names', None)

        if factor_names is None:
            # 默认：只评估技术面因子（基本面和HKT数据暂未入库）
            all_avail = list(set(TREND_FACTORS + TECH_FACTORS + VOL_FACTORS + VOLUME_FACTORS))
            # 过滤已在 factor_miner.py 中实现了计算逻辑的因子
            supported = [
                'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
                'rsi_14', 'rsi_28', 'macd', 'macd_signal',
                'bollinger_position', 'bollinger_bandwidth',
                'volatility_20', 'volatility_60',
                'volume_ratio_20', 'volume_ratio_60',
            ]
            factor_names = [f for f in all_avail if f in supported]

        self.logger.info(f"评估因子数: {len(factor_names)}, 列表: {factor_names}")

        results = self.factor_eval.evaluate_multiple(factor_names, start_date, end_date)

        # 保存到因子池
        for result in results:
            if result.get('n_samples', 0) > 0:
                self.factor_pool.add_factor(result)

        ranked = self.factor_eval.rank_factors(results)

        return {
            'code': 0,
            'msg': 'success',
            'request_id': request_id,
            'data': {
                'evaluations': ranked,
                'top_factor': ranked[0] if ranked else None,
                'count': len(results),
            }
        }

    def _action_generate_strategies(self, request_id: str, params: dict) -> dict:
        """
        生成策略
        - 从因子池获取 Top 10 因子
        - 为每个因子生成多种策略（不同持仓周期、不同止损、不同权重组合）
        - 策略存入 strategy_pool 表
        """
        top_n = params.get('top_n', 10)
        top_factors = self.factor_pool.get_top_factors(n=top_n)

        if not top_factors:
            # 如果没有已评估的因子，使用默认列表
            self.logger.warning("因子池为空，使用默认因子生成策略")
            top_factors = [
                {'factor_name': 'momentum_20', 'type': 'trend', 'ir': 0.5},
                {'factor_name': 'momentum_60', 'type': 'trend', 'ir': 0.4},
                {'factor_name': 'volatility_20', 'type': 'vol', 'ir': 0.3},
                {'factor_name': 'volume_ratio_20', 'type': 'volume', 'ir': 0.3},
                {'factor_name': 'rsi_14', 'type': 'tech', 'ir': 0.2},
            ]

        # 参数空间
        param_space = {
            'holding_periods': self.config['strategy']['holding_periods'],
            'weight_schemes': self.config['strategy']['weight_schemes'],
            'stop_loss': self.config['strategy']['stop_loss'],
        }

        strategies = self.strategy_gen.generate_strategies(top_factors, param_space)

        # 保存策略
        for strat in strategies:
            self.strategy_gen.save_strategy(strat)

        self.logger.info(f"生成策略数: {len(strategies)}")

        return {
            'code': 0,
            'msg': 'success',
            'request_id': request_id,
            'data': {
                'strategies': [
                    {'strategy_id': s['strategy_id'], 'name': s['strategy_name'], 'factors': s.get('factors', [])}
                    for s in strategies
                ],
                'count': len(strategies),
                'top_factors': [f['factor_name'] for f in top_factors],
            }
        }

    def _action_run_backtest(self, request_id: str, params: dict) -> dict:
        """
        运行回测
        - 支持 strategy_id 参数指定策略
        - 如果没有指定，默认使用 strategy_pool 中夏普最高的策略
        - 返回完整绩效指标
        """
        start = params.get('start_date', self.config['backtest']['start_date'])
        end = params.get('end_date', self.config['backtest']['end_date'])
        strategy_id = params.get('strategy_id', None)

        backtester = BacktestExecutor(self.store, self.logger, self.config)

        # 确定要回测的策略
        if strategy_id:
            # 从 strategy_pool 读取指定策略
            strat_df = self.store.df(
                f"SELECT * FROM strategy_pool WHERE strategy_id = '{strategy_id}'"
            )
            if strat_df.empty:
                return {'code': 404, 'msg': f'策略 {strategy_id} 不存在', 'request_id': request_id}
            strat_row = strat_df.iloc[0]
            strategy = {
                'strategy_id': strat_row['strategy_id'],
                'strategy_name': strat_row['strategy_name'],
                'factors': json.loads(strat_row['factors']) if isinstance(strat_row['factors'], str) else strat_row['factors'],
                'parameters': json.loads(strat_row['parameters']) if isinstance(strat_row['parameters'], str) else strat_row.get('parameters', {}),
            }
        else:
            # 默认：取夏普最高的策略
            best_df = self.store.df(
                """SELECT *, 
                      CAST(json_extract_string(metrics, '$.sharpe_ratio') AS DOUBLE) as sharpe
                   FROM strategy_pool 
                   WHERE status IN ('candidate', 'active') 
                   ORDER BY sharpe DESC NULLS LAST LIMIT 10"""
            )
            if best_df.empty:
                # 没有策略，用多因子默认策略
                strategy = {
                    'strategy_id': 'multi_factor_default',
                    'strategy_name': 'multi_factor_default',
                    'factors': ['momentum_20', 'volatility_20', 'volume_ratio_20'],
                    'parameters': {
                        'top_n_stocks': 20,
                        'stop_loss': 0.05,
                        'take_profit': 0.20,
                        'rebalance_frequency': 'weekly',
                        'weight_scheme': 'equal',
                    }
                }
            else:
                # 取夏普比率最高的策略
                strat_row = best_df.iloc[0]
                strategy = {
                    'strategy_id': strat_row['strategy_id'],
                    'strategy_name': strat_row['strategy_name'],
                    'factors': json.loads(strat_row['factors']) if isinstance(strat_row['factors'], str) else strat_row['factors'],
                    'parameters': json.loads(strat_row['parameters']) if isinstance(strat_row['parameters'], str) else strat_row.get('parameters', {}),
                }

        result = backtester.run(strategy, strategy.get('parameters', {}), start, end)

        # 使用 PerformanceAnalyzer 计算完整绩效
        nav_obj = result.get('nav_series')
        trades_list = result.get('trades', [])
        if isinstance(nav_obj, pd.DataFrame) and 'nav' in nav_obj.columns:
            nav_series = nav_obj['nav']
        elif isinstance(nav_obj, pd.Series):
            nav_series = nav_obj
        else:
            nav_series = pd.Series([], dtype=float)
        perf_result = self.performance.analyze(trades_list, nav_series)

        # 更新 strategy_pool 的 metrics
        try:
            metrics = json.dumps({
                'total_return': result.get('total_return', 0),
                'annual_return': result.get('annual_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0),
                'total_trades': result.get('total_trades', 0),
            })
            self.store.execute(
                "UPDATE strategy_pool SET metrics = ?, updated_at = CURRENT_TIMESTAMP WHERE strategy_id = ?",
                [metrics, strategy['strategy_id']]
            )
        except Exception as e:
            self.logger.warning(f"更新策略metrics失败: {e}")

        return {
            'code': 0, 'msg': 'success', 'request_id': request_id,
            'data': {
                'strategy_id': strategy.get('strategy_id'),
                'strategy_name': strategy.get('strategy_name'),
                'total_return': result.get('total_return', 0),
                'annual_return': result.get('annual_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0),
                'total_trades': result.get('total_trades', 0),
                'calmar_ratio': perf_result.get('calmar_ratio', 0),
                'profit_loss_ratio': perf_result.get('profit_loss_ratio', 0),
                'annual_returns': perf_result.get('annual_returns', {}),
            }
        }

    def _action_optimize(self, request_id: str, params: dict) -> dict:
        """
        优化最优策略参数
        1. 从因子池获取 Top3 因子
        2. 对每个因子用 BayesianOptimizer 优化参数
        3. 用 GridSearchOptimizer 做粗粒度搜索
        4. 取最优参数，更新 strategy_pool 中该策略
        5. 返回最优结果
        """
        start_date = params.get('start_date', self.config['backtest']['start_date'])
        end_date = params.get('end_date', self.config['backtest']['end_date'])
        n_trials = params.get('n_trials', 20)

        self.logger.info(f"[{request_id}] 开始参数优化...")

        # 1. 获取 Top3 因子
        top3_factors = self.factor_pool.get_top_factors(n=3)
        if not top3_factors:
            return {'code': 400, 'msg': '因子池为空，请先运行因子评估', 'request_id': request_id}

        backtester = BacktestExecutor(self.store, self.logger, self.config)
        bayes_opt = BayesianOptimizer(backtester, self.logger, start_date=start_date, end_date=end_date)
        grid_opt = GridSearchOptimizer(backtester, self.logger, start_date=start_date, end_date=end_date)

        best_overall = {'sharpe_ratio': -999, 'params': {}, 'factor': None, 'strategy_id': None}

        for factor in top3_factors:
            factor_name = factor.get('factor_name', 'unknown')
            self.logger.info(f"优化因子: {factor_name}")

            base_strategy = {
                'strategy_id': f"opt_{factor_name}",
                'strategy_name': f"opt_{factor_name}",
                'factors': [factor_name],
                'parameters': {'holding_period': 20, 'stop_loss': 0.10, 'take_profit': 0.20}
            }

            # 2. BayesianOptimizer 细粒度优化
            try:
                bayes_result = bayes_opt.optimize(base_strategy, n_trials=n_trials)
                bayes_sharpe = bayes_result.get('best_value', 0)
                self.logger.info(f"  Bayesian最优: Sharpe={bayes_sharpe:.4f}, params={bayes_result.get('best_params')}")
            except Exception as e:
                self.logger.warning(f"  Bayesian优化失败: {e}")
                bayes_result = {'best_params': {}, 'best_value': -999}

            # 3. GridSearchOptimizer 粗粒度搜索
            grid_params = {
                'holding_period': [5, 10, 20, 40, 60],
                'stop_loss': [0.05, 0.08, 0.10, 0.15],
                'take_profit': [0.15, 0.20, 0.25, 0.30],
            }
            try:
                grid_result = grid_opt.optimize(base_strategy, grid_params)
                grid_sharpe = grid_result.get('best_value', 0)
                self.logger.info(f"  Grid最优: Sharpe={grid_sharpe:.4f}, params={grid_result.get('best_params')}")
            except Exception as e:
                self.logger.warning(f"  Grid搜索失败: {e}")
                grid_result = {'best_params': {}, 'best_value': -999}

            # 4. 取两者最优
            if bayes_sharpe >= grid_sharpe:
                best_for_factor = {'params': bayes_result.get('best_params', {}), 'sharpe': bayes_sharpe}
            else:
                best_for_factor = {'params': grid_result.get('best_params', {}), 'sharpe': grid_sharpe}

            # 5. 生成带最优参数的策略并保存
            optimized_strategy = {
                'strategy_id': f"opt_{factor_name}_{uuid.uuid4().hex[:8]}",
                'strategy_name': f"optimized_{factor_name}",
                'factors': [factor_name],
                'parameters': best_for_factor['params'],
                'status': 'candidate'
            }
            self.strategy_gen.save_strategy(optimized_strategy)

            if best_for_factor['sharpe'] > best_overall['sharpe_ratio']:
                best_overall = {
                    'sharpe_ratio': best_for_factor['sharpe'],
                    'params': best_for_factor['params'],
                    'factor': factor_name,
                    'strategy_id': optimized_strategy['strategy_id'],
                }

        # 6. 对最优策略做最终回测验证
        if best_overall['strategy_id']:
            final_bt = backtester.run(
                {'strategy_id': best_overall['strategy_id'], 'factors': [best_overall['factor']]},
                best_overall['params'],
                start_date, end_date
            )
            best_overall['final_backtest'] = {
                'total_return': final_bt.get('total_return', 0),
                'annual_return': final_bt.get('annual_return', 0),
                'sharpe_ratio': final_bt.get('sharpe_ratio', 0),
                'max_drawdown': final_bt.get('max_drawdown', 0),
                'total_trades': final_bt.get('total_trades', 0),
            }

        self.logger.info(f"优化完成，最优因子: {best_overall['factor']}, Sharpe: {best_overall['sharpe_ratio']:.4f}")

        return {
            'code': 0, 'msg': 'success', 'request_id': request_id,
            'data': {
                'best_factor': best_overall['factor'],
                'best_params': best_overall['params'],
                'best_sharpe': best_overall['sharpe_ratio'],
                'best_strategy_id': best_overall['strategy_id'],
                'final_backtest': best_overall.get('final_backtest', {}),
            }
        }

    def _action_ga_optimize(self, request_id: str, params: dict) -> dict:
        """
        遗传算法优化因子组合
        - 使用 GeneticOptimizer 在给定因子空间中搜索最优因子组合和参数
        - 返回最优因子组合、最优参数、历代最优夏普轨迹
        """
        from src.core.genetic_optimizer import GeneticOptimizer

        start_date = params.get('start_date', self.config['backtest']['start_date'])
        end_date = params.get('end_date', self.config['backtest']['end_date'])
        pop_size = params.get('pop_size', 20)
        n_generations = params.get('n_generations', 30)
        mutation_rate = params.get('mutation_rate', 0.15)
        top_n_stocks = params.get('top_n_stocks', 20)

        self.logger.info(f"[{request_id}] GA优化开始 | {start_date}~{end_date}")

        backtester = BacktestExecutor(self.store, self.logger, self.config)

        ga_opt = GeneticOptimizer(
            backtester=backtester,
            logger=self.logger,
            start_date=start_date,
            end_date=end_date,
            pop_size=pop_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
        )

        result = ga_opt.optimize(top_n_stocks=top_n_stocks)

        # 保存最优策略到 strategy_pool
        optimized_strategy = {
            'strategy_id': f"ga_{uuid.uuid4().hex[:8]}",
            'strategy_name': f"ga_optimized",
            'factors': result['best_factors'],
            'parameters': result['best_params'],
            'status': 'candidate',
        }
        self.strategy_gen.save_strategy(optimized_strategy)

        # 精简 history（只保留关键代数）
        gen_history = result.get('generation_history', [])
        slim_history = [
            {'gen': h['gen'], 'best_sharpe': float(h['best_sharpe']), 'avg_sharpe': float(h['avg_sharpe'])}
            for h in gen_history
            if h['gen'] % 5 == 0 or h['gen'] == len(gen_history) - 1
        ]

        self.logger.info(
            f"[{request_id}] GA优化完成 | Best Sharpe={result['best_sharpe']:.4f} | "
            f"Factors={result['best_factors']} | Params={result['best_params']}"
        )

        return {
            'code': 0, 'msg': 'success', 'request_id': request_id,
            'data': {
                'best_sharpe': float(result['best_sharpe']),
                'best_factors': result['best_factors'],
                'best_params': result['best_params'],
                'best_chromosome': result['best_chromosome'],
                'strategy_id': optimized_strategy['strategy_id'],
                'generation_history': slim_history,
                'total_generations': len(gen_history),
            }
        }

    def _action_rl_optimize(self, request_id: str, params: dict) -> dict:
        """
        强化学习优化动态仓位
        - 使用 Q-Learning 学习市场状态到仓位的映射
        - 返回训练好的 Q 表、最优策略、夏普轨迹
        """
        from src.core.rl_optimizer import RLOptimizer

        start_date = params.get('start_date', self.config['backtest']['start_date'])
        end_date = params.get('end_date', self.config['backtest']['end_date'])
        n_episodes = params.get('n_episodes', 50)
        gamma = params.get('gamma', 0.95)
        alpha = params.get('alpha', 0.1)
        epsilon = params.get('epsilon', 0.1)

        self.logger.info(f"[{request_id}] RL优化开始 | Episodes={n_episodes}")

        backtester = BacktestExecutor(self.store, self.logger, self.config)

        # 默认基础策略
        factor_name = params.get('factor_name', 'momentum_20')
        base_strategy = {
            'strategy_id': f"rl_base_{uuid.uuid4().hex[:8]}",
            'strategy_name': 'rl_base',
            'factors': [factor_name],
        }
        base_params = params.get('base_params', {
            'holding_period': 20,
            'stop_loss': 0.10,
            'take_profit': 0.20,
        })

        rl_opt = RLOptimizer(
            backtester=backtester,
            logger=self.logger,
            start_date=start_date,
            end_date=end_date,
            gamma=gamma,
            alpha=alpha,
            epsilon=epsilon,
            n_episodes=n_episodes,
        )

        result = rl_opt.optimize(
            strategy=base_strategy,
            base_params=base_params,
            use_rl_position=True,
        )

        # 精简训练历史
        training_history = result.get('training_history', [])
        slim_history = [
            {'episode': h['episode'], 'total_reward': float(h['total_reward']), 'epsilon': float(h['epsilon'])}
            for h in training_history
            if h['episode'] % 10 == 0 or h['episode'] == len(training_history) - 1
        ]

        self.logger.info(
            f"[{request_id}] RL优化完成 | Final Sharpe={result['final_sharpe']:.4f} | "
            f"States={result['n_states']}"
        )

        return {
            'code': 0, 'msg': 'success', 'request_id': request_id,
            'data': {
                'final_sharpe': float(result['final_sharpe']),
                'n_states': result['n_states'],
                'n_episodes': result['n_episodes'],
                'best_policy': result['best_policy'],
                'training_history': slim_history,
            }
        }

    def _action_run_full_pipeline(self, request_id: str, params: dict) -> dict:
        """运行完整流水线"""
        self.logger.info(f"[{request_id}] 开始全流程...")

        # 1. 初始化
        init_result = self._action_init(request_id, params)

        # 2. 更新数据
        update_result = self._action_update_data(request_id, params)

        # 3. 评估因子
        eval_result = self._action_evaluate_factors(request_id, params)

        # 4. 生成策略
        strat_result = self._action_generate_strategies(request_id, params)

        # 5. 回测
        bt_result = self._action_run_backtest(request_id, params)

        self.logger.info(f"[{request_id}] 全流程完成")

        return {
            'code': 0, 'msg': 'Pipeline completed', 'request_id': request_id,
            'data': {
                'init': init_result.get('data', {}),
                'update': update_result.get('data', {}),
                'evaluation': eval_result.get('data', {}),
                'strategies': strat_result.get('data', {}),
                'backtest': bt_result.get('data', {}),
            }
        }

    def _action_feature_select(self, request_id: str, params: dict) -> dict:
        """使用 LightGBM 筛选最优因子"""
        from src.core.ml_feature_selector import MLFeatureSelector

        factor_names = params.get('factor_names', [])
        if not factor_names:
            # 默认取前 20 个技术面因子
            supported = [
                'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
                'rsi_14', 'rsi_28', 'macd', 'macd_signal',
                'bollinger_position', 'bollinger_bandwidth', 'cci_20',
                'volatility_20', 'volatility_60',
                'volume_ratio_20', 'volume_ratio_60',
                'williams_r', 'adx_14', 'atr_20',
            ]
            factor_names = supported[:20]

        start = params.get('start_date', self.config['backtest']['start_date'])
        end = params.get('end_date', self.config['backtest']['end_date'])

        self.logger.info(
            f"[{request_id}] LightGBM 特征筛选: {len(factor_names)} 个候选因子 "
            f"({start} ~ {end})"
        )

        selector = MLFeatureSelector(self.store, self.logger)
        result = selector.select_features(factor_names, start, end)

        # 如果有数据，同步更新因子池
        if result.get('selected_features'):
            self.logger.info(
                f"LightGBM 筛选结果: {result['n_selected']} 个最优因子, "
                f"验证集 AUC={result.get('model_auc', 0):.4f}"
            )

        return {
            'code': 0,
            'msg': 'success',
            'request_id': request_id,
            'data': result,
        }

    def pause(self):
        """暂停"""
        pass

    def resume(self):
        """恢复"""
        pass

    def destroy(self):
        """销毁"""
        if hasattr(self, 'store'):
            self.store.close()


# 全局实例
_skill_api = None


def get_instance():
    global _skill_api
    if _skill_api is None:
        _skill_api = SkillAPI()
    return _skill_api


def execute(params: dict) -> dict:
    """供 OpenClaw 调用的入口"""
    return get_instance().execute(params)
