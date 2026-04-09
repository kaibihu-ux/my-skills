"""
机器学习建模：XGBoost / LSTM / Transformer

支持：
- XGBoost 二分类（涨/跌选股）
- LSTM 时序预测
- Transformer 时序预测

所有阈值/参数从 config/settings.yaml 读取
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# 尝试导入 optional 依赖，缺失时给出友好提示
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


# =============================================================================
# XGBoost 选股模型
# =============================================================================

class XGBoostModel:
    """
    XGBoost 二分类选股模型

    训练数据：
    - X: 因子值（n_stocks × n_factors 面板）
    - y: 未来5日收益是否 > 中位数（1=涨，0=跌）

    所有参数从 config['ml_models']['xgboost'] 读取
    """

    def __init__(self, store, logger, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self._load_config(config)
        self.model: Optional[xgb.Booster] = None
        self.feature_importance_: Optional[Dict] = None

    def _load_config(self, config: Optional[Dict] = None):
        if config is not None:
            mlcfg = config.get('ml_models', {}).get('xgboost', {})
        else:
            mlcfg = {}

        self.enabled = bool(mlcfg.get('enabled', True)) and XGBOOST_AVAILABLE
        self.n_estimators = int(mlcfg.get('n_estimators', 100))
        self.max_depth = int(mlcfg.get('max_depth', 6))
        self.learning_rate = float(mlcfg.get('learning_rate', 0.05))
        self.colsample_bytree = float(mlcfg.get('colsample_bytree', 0.8))
        self.subsample = float(mlcfg.get('subsample', 0.8))
        self.label_horizon = int(mlcfg.get('label_horizon', 5))  # 预测未来N日涨跌
        self.future_period = int(mlcfg.get('future_period', 5))

    def train(
        self,
        factor_names: List[str],
        train_start: str,
        train_end: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        训练 XGBoost 模型

        Args:
            factor_names:  因子列表
            train_start:   训练开始日期 YYYYMMDD
            train_end:     训练结束日期 YYYYMMDD
            params:        可选，覆盖默认参数

        Returns:
            {
                'model': Booster,
                'feature_importance': Dict,
                'auc': float,
                'train_samples': int,
                'params': Dict,
            }
        """
        if not self.enabled:
            self._log("XGBoost 未启用或未安装（pip install xgboost）")
            return self._empty_result()

        start_fmt = self._format_date(train_start)
        end_fmt = self._format_date(train_end)

        # 加载数据
        X, y, dates = self._build_dataset(factor_names, start_fmt, end_fmt)
        if len(X) < 100:
            self._log("训练样本不足")
            return self._empty_result()

        # 合并特征和标签
        train_data = pd.DataFrame(X, columns=factor_names)
        train_data['label'] = y

        # 去除缺失值
        train_data = train_data.dropna()
        if len(train_data) < 100:
            return self._empty_result()

        X_train = train_data[factor_names].values
        y_train = train_data['label'].values

        # 超参数覆盖
        p = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'colsample_bytree': self.colsample_bytree,
            'subsample': self.subsample,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'verbosity': 0,
        }
        if params:
            p.update(params)

        try:
            # 使用原生 XGBoost API（更灵活）
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=factor_names)

            evals = [(dtrain, 'train')]
            self.model = xgb.train(
                p,
                dtrain,
                num_boost_round=p['n_estimators'],
                evals=evals,
                verbose_eval=False,
            )

            # 特征重要性
            importance = self.model.get_score(importance_type='gain')
            self.feature_importance_ = importance

            # 计算训练 AUC
            proba = self.model.predict(dtrain)
            auc = self._calc_auc(proba, y_train)

            self._log(
                f"XGBoost 训练完成: {len(X_train)} 样本, AUC={auc:.4f}, "
                f"因子={factor_names}"
            )

            return {
                'model': self.model,
                'feature_importance': {k: float(v) for k, v in importance.items()},
                'auc': float(auc),
                'train_samples': len(X_train),
                'params': p,
            }

        except Exception as e:
            self._log(f"XGBoost 训练失败: {e}")
            return self._empty_result()

    def predict(self, model, factor_data: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型预测涨跌概率

        Args:
            model:         XGBoost Booster 模型
            factor_data:   DataFrame (n_stocks × n_factors)

        Returns:
            涨跌概率数组 (n_stocks,)
        """
        if model is None or not self.enabled:
            return np.array([])

        try:
            X = factor_data.values
            dtest = xgb.DMatrix(X, feature_names=list(factor_data.columns))
            proba = model.predict(dtest)
            return proba
        except Exception as e:
            self._log(f"XGBoost 预测失败: {e}")
            return np.array([])

    def _build_dataset(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        构建 XGBoost 训练数据集

        Returns:
            X: (n_samples, n_factors)
            y: (n_samples,)  0/1 标签
            dates: 对应日期列表
        """
        if self.store is None:
            return np.array([]), np.array([]), []

        # 读取所有因子数据
        factor_dfs = []
        for fname in factor_names:
            try:
                df = self.store.df(
                    f"""SELECT ts_code, trade_date, value
                        FROM factors
                        WHERE factor_name = '{fname}'
                          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                        ORDER BY trade_date, ts_code"""
                )
                if not df.empty:
                    df = df.rename(columns={'value': fname})
                    factor_dfs.append(df[['ts_code', 'trade_date', fname]])
            except Exception as e:
                self._log(f"加载因子 {fname} 失败: {e}")

        if not factor_dfs:
            return np.array([]), np.array([]), []

        # 按日期合并所有因子
        merged = factor_dfs[0]
        for df in factor_dfs[1:]:
            merged = merged.merge(df, on=['ts_code', 'trade_date'], how='left')

        # 读取价格计算未来收益
        price_df = self.store.df(
            f"""SELECT ts_code, trade_date, close
                FROM stock_daily
                WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY ts_code, trade_date"""
        )
        if price_df.empty:
            return np.array([]), np.array([]), []

        price_df[f'return_{self.future_period}'] = (
            price_df.groupby('ts_code')['close']
            .pct_change(self.future_period)
            .shift(-self.future_period)
        )

        merged = merged.merge(
            price_df[['ts_code', 'trade_date', f'return_{self.future_period}']],
            on=['ts_code', 'trade_date'],
            how='inner'
        )

        # 过滤有效行
        feature_cols = factor_names
        merged = merged.dropna(subset=feature_cols + [f'return_{self.future_period}'])
        if len(merged) < 100:
            return np.array([]), np.array([]), []

        # 构建标签：未来收益 > 中位数为 1
        ret_col = f'return_{self.future_period}'
        median_ret = merged[ret_col].median()
        merged['label'] = (merged[ret_col] > median_ret).astype(int)

        X = merged[factor_names].values.astype(np.float64)
        y = merged['label'].values.astype(np.int32)
        dates = merged['trade_date'].astype(str).tolist()

        return X, y, dates

    @staticmethod
    def _calc_auc(proba: np.ndarray, y_true: np.ndarray) -> float:
        """计算 AUC（纯 numpy 实现）"""
        # 简单的 rank-based AUC
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.5

        pos_proba = proba[y_true == 1]
        neg_proba = proba[y_true == 0]

        # AUC = (正样本排名和 - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
        all_proba = proba
        all_labels = y_true
        n = len(all_labels)
        ranks = np.argsort(np.argsort(-all_proba)) + 1  # 降序排名
        rank_sum = float(np.sum(ranks[all_labels == 1]))
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(np.clip(auc, 0.0, 1.0))

    def _empty_result(self) -> Dict:
        return {
            'model': None,
            'feature_importance': {},
            'auc': 0.0,
            'train_samples': 0,
            'params': {},
        }

    @staticmethod
    def _format_date(date_str: str) -> str:
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[XGBoostModel] {msg}")
        else:
            print(f"[XGBoostModel] {msg}")


# =============================================================================
# LSTM 时序预测模型
# =============================================================================

class LSTMPredictor(nn.Module if TORCH_AVAILABLE else object):
    """
    LSTM 时序预测模型

    输入: (batch, seq_len, input_size)
    输出: (batch, 1) 涨跌概率
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，无法使用 LSTM 模型")

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)


# =============================================================================
# Transformer 时序预测模型
# =============================================================================

class TransformerPredictor(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer Encoder 时序预测模型

    输入: (batch, seq_len, input_size)
    输出: (batch, 1) 涨跌概率
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，无法使用 Transformer 模型")

        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步
        return self.sigmoid(x)


# =============================================================================
# PyTorch 训练工具函数
# =============================================================================

def _train_pytorch_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    device: str = 'cpu',
    logger=None,
) -> Tuple[float, float]:
    """
    通用 PyTorch 模型训练函数

    Returns:
        (train_auc, val_auc)
    """
    if not TORCH_AVAILABLE:
        return 0.0, 0.0

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_proba = model(X_v).cpu().numpy().flatten()
            val_auc = _calc_auc_np(val_proba, y_val)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if logger and (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}, Val AUC={val_auc:.4f}")

    # 最终训练 AUC
    model.eval()
    with torch.no_grad():
        train_proba = model(X_t).cpu().numpy().flatten()
        train_auc = _calc_auc_np(train_proba, y_train)

    return train_auc, best_val_auc


def _build_sequences(
    data: np.ndarray,
    seq_len: int,
    label_col_idx: int,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建时序序列数据

    Args:
        data:         (n_timesteps, n_features)，包含因子值+标签
        seq_len:      序列长度（历史天数）
        label_col_idx: 标签列索引
        horizon:      预测期（未来N天）

    Returns:
        X: (n_samples, seq_len, n_features-1)
        y: (n_samples,)
    """
    X_list, y_list = [], []
    n = len(data)

    for i in range(seq_len, n - horizon):
        seq = data[i - seq_len:i, :]  # seq_len × n_features
        # 排除标签列
        seq_features = np.delete(seq, label_col_idx, axis=1)
        label = data[i + horizon, label_col_idx]
        X_list.append(seq_features)
        y_list.append(label)

    return np.array(X_list), np.array(y_list)


def _calc_auc_np(proba: np.ndarray, y_true: np.ndarray) -> float:
    """纯 numpy AUC 计算"""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_proba = proba[y_true == 1]
    neg_proba = proba[y_true == 0]

    # AUC via Mann-Whitney U statistic
    total = 0.0
    for p in pos_proba:
        total += np.sum(neg_proba < p) + 0.5 * np.sum(neg_proba == p)

    auc = total / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


# =============================================================================
# ML 模型工厂
# =============================================================================

class MLModelFactory:
    """
    统一 ML 模型工厂

    支持 XGBoost / LSTM / Transformer 三种模型
    所有参数从 config/settings.yaml 读取
    """

    def __init__(self, store, logger, config: Optional[Dict] = None):
        self.store = store
        self.logger = logger
        self.config = config or {}

        self._load_config()

        # 初始化各模型
        self.xgb_model = XGBoostModel(store, logger, config)

        # PyTorch 模型缓存（按 stock_code 存储）
        self.lstm_cache: Dict[str, nn.Module] = {}
        self.transformer_cache: Dict[str, nn.Module] = {}

    def _load_config(self):
        mlcfg = self.config.get('ml_models', {})

        self.lstm_cfg = mlcfg.get('lstm', {})
        self.transformer_cfg = mlcfg.get('transformer', {})

        self.lstm_enabled = bool(self.lstm_cfg.get('enabled', False)) and TORCH_AVAILABLE
        self.transformer_enabled = bool(self.transformer_cfg.get('enabled', False)) and TORCH_AVAILABLE

        self.lstm_seq_len = int(self.lstm_cfg.get('seq_len', 20))
        self.lstm_hidden_size = int(self.lstm_cfg.get('hidden_size', 64))
        self.lstm_num_layers = int(self.lstm_cfg.get('num_layers', 2))
        self.lstm_epochs = int(self.lstm_cfg.get('epochs', 50))

        self.tf_d_model = int(self.transformer_cfg.get('d_model', 64))
        self.tf_nhead = int(self.transformer_cfg.get('nhead', 4))
        self.tf_num_layers = int(self.transformer_cfg.get('num_layers', 2))
        self.tf_epochs = int(self.transformer_cfg.get('epochs', 50))

    def train_lstm(
        self,
        stock_code: str,
        factor_names: List[str],
        seq_len: Optional[int] = None,
        train_start: str = '20220101',
        train_end: str = '20240101',
    ) -> float:
        """
        训练 LSTM 预测单只股票未来收益

        Args:
            stock_code:   股票代码
            factor_names: 因子列表
            seq_len:      序列长度（默认从配置读取）
            train_start:  训练开始
            train_end:    训练结束

        Returns:
            验证集 AUC（float）
        """
        if not self.lstm_enabled:
            self._log("LSTM 未启用")
            return 0.0

        seq_len = seq_len or self.lstm_seq_len
        start_fmt = self._format_date(train_start)
        end_fmt = self._format_date(train_end)

        # 加载该股票历史因子数据
        data = self._load_stock_factor_data(stock_code, factor_names, start_fmt, end_fmt)
        if data is None or len(data) < seq_len + 20:
            self._log(f"LSTM [{stock_code}]: 数据不足")
            return 0.0

        # 构建序列
        label_idx = data.shape[1] - 1  # 最后一列是标签
        X_seq, y_seq = _build_sequences(data, seq_len, label_idx, horizon=5)

        if len(X_seq) < 50:
            return 0.0

        # 划分训练/验证集（8:2）
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # 建立模型
        input_size = X_train.shape[2]
        model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
        )

        device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        _, val_auc = _train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            epochs=self.lstm_epochs,
            batch_size=64,
            logger=self.logger,
            device=device,
        )

        # 缓存模型
        self.lstm_cache[stock_code] = model

        self._log(f"LSTM [{stock_code}] 训练完成: Val AUC={val_auc:.4f}")
        return float(val_auc)

    def train_transformer(
        self,
        stock_code: str,
        factor_names: List[str],
        seq_len: Optional[int] = None,
        train_start: str = '20220101',
        train_end: str = '20240101',
    ) -> float:
        """训练 Transformer 预测单只股票未来收益"""
        if not self.transformer_enabled:
            self._log("Transformer 未启用")
            return 0.0

        seq_len = seq_len or self.lstm_seq_len  # 共用 seq_len
        start_fmt = self._format_date(train_start)
        end_fmt = self._format_date(train_end)

        data = self._load_stock_factor_data(stock_code, factor_names, start_fmt, end_fmt)
        if data is None or len(data) < seq_len + 20:
            self._log(f"Transformer [{stock_code}]: 数据不足")
            return 0.0

        label_idx = data.shape[1] - 1
        X_seq, y_seq = _build_sequences(data, seq_len, label_idx, horizon=5)

        if len(X_seq) < 50:
            return 0.0

        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        input_size = X_train.shape[2]
        model = TransformerPredictor(
            input_size=input_size,
            d_model=self.tf_d_model,
            nhead=self.tf_nhead,
            num_layers=self.tf_num_layers,
        )

        device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        _, val_auc = _train_pytorch_model(
            model, X_train, y_train, X_val, y_val,
            epochs=self.tf_epochs,
            batch_size=64,
            logger=self.logger,
            device=device,
        )

        self.transformer_cache[stock_code] = model

        self._log(f"Transformer [{stock_code}] 训练完成: Val AUC={val_auc:.4f}")
        return float(val_auc)

    def _load_stock_factor_data(
        self,
        stock_code: str,
        factor_names: List[str],
        start_date: str,
        end_date: str,
    ) -> Optional[np.ndarray]:
        """
        加载单只股票的因子数据，构建训练数组

        Returns:
            (n_timesteps, n_factors + 1) ndarray
            最后一列是标签（未来收益 > 中位数为 1）
        """
        if self.store is None:
            return None

        # 读取各因子
        dfs = []
        for fname in factor_names:
            try:
                df = self.store.df(
                    f"""SELECT trade_date, value
                        FROM factors
                        WHERE ts_code = '{stock_code}'
                          AND factor_name = '{fname}'
                          AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                        ORDER BY trade_date"""
                )
                if not df.empty:
                    df = df.rename(columns={'value': fname})
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return None

        # 合并
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on='trade_date', how='left')

        # 读取价格
        price_df = self.store.df(
            f"""SELECT trade_date, close
                FROM stock_daily
                WHERE ts_code = '{stock_code}'
                  AND trade_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY trade_date"""
        )
        if price_df.empty:
            return None

        merged = merged.merge(price_df, on='trade_date', how='inner')
        merged = merged.sort_values('trade_date')

        # 计算未来收益并构建标签
        merged['future_ret'] = (
            merged['close'].pct_change(5).shift(-5)
        )
        merged = merged.dropna(subset=['future_ret'])

        if len(merged) < 50:
            return None

        median_ret = merged['future_ret'].median()
        merged['label'] = (merged['future_ret'] > median_ret).astype(int)

        # 构建返回数组
        feature_cols = factor_names + ['label']
        result = merged[feature_cols].values.astype(np.float64)

        return result

    @staticmethod
    def _format_date(date_str: str) -> str:
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(f"[MLModelFactory] {msg}")
        else:
            print(f"[MLModelFactory] {msg}")
