"""
LightGBM 特征重要性筛选器
使用 LightGBM 计算因子重要性，筛选最优因子
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import List, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.factor_miner import FactorMiner, ALL_FACTORS
from src.core.factor_eval import FactorEvaluator


class MLFeatureSelector:
    """LightGBM 特征重要性筛选器"""

    def __init__(self, store, logger):
        self.store = store
        self.logger = logger
        self.factor_miner = FactorMiner(store, logger)

    def _load_factor_panel(
        self, factor_names: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        加载因子面板数据
        从 stock_daily 和 factors 表构建因子矩阵
        """
        # 转换日期格式：YYYYMMDD -> YYYY-MM-DD
        start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        # 读取日线数据
        daily_df = self.store.df(
            f"SELECT ts_code, trade_date, close FROM stock_daily "
            f"WHERE trade_date BETWEEN '{start_str}' AND '{end_str}' "
            f"ORDER BY ts_code, trade_date"
        )

        if daily_df.empty:
            self.logger.warning("日线数据为空，无法构建因子面板")
            return pd.DataFrame()

        stock_codes = daily_df['ts_code'].unique()
        self.logger.info(f"加载日线数据: {len(daily_df)} 条, {len(stock_codes)} 只股票")

        # 对每只股票计算各因子值，拼成面板
        panel_rows = []

        for code in stock_codes:
            stock_df = daily_df[daily_df['ts_code'] == code].sort_values('trade_date').reset_index(drop=True)
            if len(stock_df) < 30:
                continue

            for factor_name in factor_names:
                try:
                    factor_series = self.factor_miner.calculate_factor(stock_df, factor_name)
                    if len(factor_series) != len(stock_df):
                        continue
                    stock_df = stock_df.copy()
                    stock_df[factor_name] = factor_series.values
                except Exception:
                    continue

            stock_df = stock_df[['ts_code', 'trade_date', 'close'] + [f for f in factor_names if f in stock_df.columns]].copy()
            panel_rows.append(stock_df)

        if not panel_rows:
            return pd.DataFrame()

        panel = pd.concat(panel_rows, ignore_index=True)
        self.logger.info(f"因子面板构建完成: {panel.shape}")
        return panel

    def _build_label(panel: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
        """
        构建标签：未来 forward_days 日收益率分类（涨=1，跌=0）
        """
        panel = panel.sort_values(['ts_code', 'trade_date']).copy()

        def forward_return(group):
            group = group.sort_values('trade_date')
            future = group['close'].shift(-forward_days)
            ret = (future / group['close']) - 1
            return ret

        panel['future_return'] = panel.groupby('ts_code', group_keys=False).apply(forward_return)
        panel['label'] = (panel['future_return'] > 0).astype(int)

        # 过滤无效标签
        panel = panel.dropna(subset=['label'])
        return panel

    def select_features(
        self, factor_names: List[str], start_date: str, end_date: str
    ) -> Dict:
        """
        使用 LightGBM 计算因子重要性，筛选最优因子
        """
        # 1. 构建因子面板
        panel = self._load_factor_panel(factor_names, start_date, end_date)
        if panel.empty:
            return {
                'selected_features': [],
                'all_importance': [],
                'model_auc': 0.0,
                'error': 'No data available',
            }

        # 2. 过滤有效因子列
        available_factors = [f for f in factor_names if f in panel.columns]
        if len(available_factors) < 2:
            self.logger.warning(f"可用因子不足: {available_factors}")
            return {
                'selected_features': available_factors,
                'all_importance': [],
                'model_auc': 0.0,
            }

        # 3. 构建标签
        try:
            panel = self._build_label(panel, forward_days=5)
        except Exception as e:
            self.logger.warning(f"标签构建失败: {e}, 跳过标签构建")
            # fallback: 使用当期收益率方向作为标签
            panel = panel.sort_values(['ts_code', 'trade_date']).copy()
            panel['future_return'] = panel.groupby('ts_code', group_keys=False).apply(
                lambda g: g['close'].pct_change(5).shift(-5)
            )
            panel['label'] = (panel['future_return'] > 0).astype(int)
            panel = panel.dropna(subset=['label'])

        # 4. 准备训练数据
        feature_cols = available_factors
        X = panel[feature_cols].values.astype(np.float64)
        y = panel['label'].values.astype(np.int32)

        # 过滤无穷大和NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(X) < 100:
            self.logger.warning(f"样本量太少: {len(X)}")
            return {
                'selected_features': available_factors,
                'all_importance': [],
                'model_auc': 0.0,
            }

        # 5. 划分训练/验证集（时序划分，后30%为验证集）
        split_idx = int(len(X) * 0.7)
        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]

        # 6. LightGBM 训练
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': 8,
            'seed': 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_data)

        self.logger.info(f"LightGBM 训练中: {len(X_train)} 训练样本, {len(X_valid)} 验证样本")

        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
        )

        # 7. 提取特征重要性
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance('gain'),
        }).sort_values('importance', ascending=False)

        # 8. 筛选：保留前50%
        n_keep = max(1, len(feature_cols) // 2)
        selected = importance_df.head(n_keep)['feature'].tolist()

        # 获取AUC
        best_score = model.best_score.get('valid', {}).get('auc', 0.0)
        if best_score == 0.0:
            best_score = model.best_score.get('valid_0', {}).get('auc', 0.0)

        self.logger.info(
            f"LightGBM 特征筛选完成: {len(feature_cols)} -> {len(selected)} 因子, "
            f"验证集 AUC={best_score:.4f}"
        )

        return {
            'selected_features': selected,
            'all_importance': importance_df.to_dict('records'),
            'model_auc': float(best_score),
            'n_samples': int(len(X)),
            'n_selected': len(selected),
        }
