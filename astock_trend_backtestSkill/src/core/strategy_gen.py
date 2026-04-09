import json
import uuid
import pandas as pd
from typing import Dict, List


class SignalBuilder:
    """信号构建器"""
    
    @staticmethod
    def build_momentum_signal(factor_value: float, threshold: float = 0.0) -> int:
        """动量信号：1买入, 0持有, -1卖出"""
        if factor_value > threshold:
            return 1
        elif factor_value < -threshold:
            return -1
        return 0
    
    @staticmethod
    def build_ma_cross_signal(ma5: float, ma20: float, ma5_prev: float, ma20_prev: float) -> str:
        """均线交叉信号"""
        if ma5 > ma20 and ma5_prev <= ma20_prev:
            return 'golden_cross'
        elif ma5 < ma20 and ma5_prev >= ma20_prev:
            return 'death_cross'
        return 'hold'
    
    @staticmethod
    def build_rebalance_signal(date: str, frequency: str = 'weekly') -> bool:
        """
        调仓频率信号
        frequency: 'daily' | 'weekly' | 'monthly'
        weekly: 每周一调仓
        monthly: 每月第一个交易日（简化：用日期判断）
        """
        if frequency == 'daily':
            return True
        elif frequency == 'weekly':
            return pd.Timestamp(date).dayofweek == 0  # 周一
        elif frequency == 'monthly':
            return pd.Timestamp(date).day <= 5  # 每月前5个交易日
        return False


class StrategyGenerator:
    """策略生成器"""
    
    def __init__(self, store, logger):
        self.store = store
        self.logger = logger
        self.signal_builder = SignalBuilder()
    
    def generate_strategies(
        self,
        top_factors: List[Dict],
        param_space: Dict,
        ml_predictions: Dict[str, float] = None,
    ) -> List[Dict]:
        """生成策略候选"""
        strategies = []
        base_params = {
            'holding_period': param_space.get('holding_periods', [20])[0],
            'stop_loss': param_space.get('stop_loss', [0.10])[0],
            'take_profit': param_space.get('take_profit', [0.20])[0],
            'rebalance_frequency': param_space.get('rebalance_frequency', ['weekly'])[0],
        }
        for factor in top_factors:
            factor_name = factor.get('name') or factor.get('factor_name', 'unknown')
            for holding_period in param_space.get('holding_periods', [20]):
                for weight_scheme in param_space.get('weight_schemes', ['equal']):
                    for stop_loss in param_space.get('stop_loss', [0.10]):
                        for rebalance_frequency in param_space.get('rebalance_frequency', ['weekly']):
                            take_profit = param_space.get('take_profit', [0.20])[0]
                            strategy = {
                                'strategy_id': str(uuid.uuid4()),
                                'strategy_name': f"{factor_name}_h{holding_period}_{weight_scheme}_rf{rebalance_frequency}_sl{stop_loss}",
                                'factors': [factor_name],
                                'parameters': {
                                    'holding_period': holding_period,
                                    'weight_scheme': weight_scheme,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'rebalance_frequency': rebalance_frequency,
                                },
                                'status': 'candidate'
                            }
                            strategies.append(strategy)

        # ML 增强策略（如果有 ML 预测结果）
        if ml_predictions:
            ml_strats = self.generate_ml_enhanced_strategies(top_factors, ml_predictions, base_params)
            strategies.extend(ml_strats)
            self.logger.info(f"ML增强策略: +{len(ml_strats)} 个")

        return strategies
    
    def generate_ml_enhanced_strategies(
        self,
        top_factors: List[Dict],
        ml_predictions: Dict[str, float],  # XGBoost/LSTM/Transformer 预测概率
        base_params: Dict
    ) -> List[Dict]:
        """
        生成基于 ML 预测增强的策略

        Args:
            top_factors:     顶级因子列表
            ml_predictions:  ML模型预测概率 {stock_code: predicted_prob}
            base_params:     基础参数

        Returns:
            ML增强策略列表
        """
        strategies = []

        for factor_info in top_factors:
            fname = factor_info.get('name') or factor_info.get('factor_name', 'unknown')
            # 用 ML 预测概率替代因子排名
            for weight_scheme in ['ml_probability', 'factor_ml_blend']:
                strat = {
                    'strategy_id': str(uuid.uuid4()),
                    'strategy_name': f"ml_{fname}_{weight_scheme}",
                    'factors': [fname],
                    'parameters': {
                        **base_params,
                        'weight_scheme': weight_scheme,
                        'ml_predictions': ml_predictions,
                    },
                    'status': 'candidate'
                }
                strategies.append(strat)

        self.logger.info(f"ML增强策略: 生成了 {len(strategies)} 个策略")
        return strategies

    def save_strategy(self, strategy: Dict):
        """保存策略到数据库"""
        self.store.execute(
            """INSERT OR REPLACE INTO strategy_pool 
               (strategy_id, strategy_name, factors, parameters, metrics, rank, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, '{}', 0, 'candidate', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            [strategy['strategy_id'], strategy['strategy_name'], 
             json.dumps(strategy['factors']), json.dumps(strategy['parameters'])]
        )
