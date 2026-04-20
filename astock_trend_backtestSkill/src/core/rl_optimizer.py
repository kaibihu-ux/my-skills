"""
强化学习优化器 - 动态仓位/择时
使用 Q-Learning 纯 Python 实现，不依赖 tensorflow/pytorch
"""

import random
import json
import threading
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# 并行评估
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# ------------------------------------------------------------------
# 模块级 Worker 函数（用于 ProcessPoolExecutor，必须可 pickle）
# ------------------------------------------------------------------

def _episode_worker(args: Tuple) -> Tuple[float, List[int], Dict]:
    """
    单个 episode 的工作函数，在子进程中执行。
    返回: (episode_reward, actions_taken, q_table_snapshot)
    """
    (
        episode,
        strategy,
        base_params,
        epsilon,
        lookback_days,
        gamma,
        alpha,
        trade_dates,
        store_info,       # (conn_str, db_path) 用于重建 store
        initial_cash,
        commission,
        slippage,
    ) = args

    # 在子进程中重建必要的数据库连接（避免 Store 对象不可 pickle）
    import duckdb
    conn = duckdb.connect(store_info)
    try:
        # 注意：所有子函数定义和主逻辑都在 try 块内，确保 finally 能正确关闭连接

        class MiniStore:
            def __init__(self, conn):
                self.conn = conn

            def df(self, sql):
                return pd.read_sql(sql, self.conn)

        store = MiniStore(conn)

        # 重建因子查询函数（子进程内）
        def get_market_return(date):
            try:
                idx_prices = pd.read_sql(
                    f"""SELECT trade_date, close FROM stock_daily
                        WHERE ts_code = '000001.SH'
                        AND trade_date <= '{date}'
                        ORDER BY trade_date DESC
                        LIMIT {lookback_days}""",
                    conn
                )
                if len(idx_prices) >= 2:
                    ret = (idx_prices.iloc[0]['close'] - idx_prices.iloc[-1]['close']) \
                          / idx_prices.iloc[-1]['close']
                    return float(ret)
            except Exception:
                pass
            return 0.0

        def get_momentum_return(date):
            try:
                dt = pd.to_datetime(date)
                start_dt = dt - pd.Timedelta(days=int(lookback_days * 1.5))
                start_str = start_dt.strftime('%Y-%m-%d')
                mom_df = pd.read_sql(
                    f"""SELECT AVG(value) as avg_mom FROM factors
                        WHERE factor_name = 'momentum_20'
                        AND trade_date BETWEEN '{start_str}' AND '{date}'
                        AND value IS NOT NULL""",
                    conn
                )
                if not mom_df.empty and mom_df.iloc[0]['avg_mom'] is not None:
                    return float(mom_df.iloc[0]['avg_mom'])
            except Exception:
                pass
            return 0.0

        def get_volatility(date):
            try:
                dt = pd.to_datetime(date)
                start_dt = dt - pd.Timedelta(days=int(lookback_days * 1.5))
                start_str = start_dt.strftime('%Y-%m-%d')
                vol_df = pd.read_sql(
                    f"""SELECT AVG(value) as avg_vol FROM factors
                        WHERE factor_name = 'volatility_20'
                        AND trade_date BETWEEN '{start_str}' AND '{date}'
                        AND value IS NOT NULL""",
                    conn
                )
                if not vol_df.empty and vol_df.iloc[0]['avg_vol'] is not None:
                    return float(vol_df.iloc[0]['avg_vol'])
            except Exception:
                pass
            return 0.02

        ACTION_MAP = {0: 0.0, 1: 0.5, 2: 1.0}
        N_ACTIONS = 3
        q_table = defaultdict(lambda: [0.0, 0.0, 0.0])

        def get_state(date, portfolio_value, positions, nav_history):
            market_ret = get_market_return(date)
            market_regime = 1 if market_ret > 0.02 else (-1 if market_ret < -0.02 else 0)
            momentum_ret = get_momentum_return(date)
            momentum_signal = 1 if momentum_ret > 0.01 else (-1 if momentum_ret < -0.01 else 0)
            vol = get_volatility(date)
            vol_signal = 1 if vol > 0.03 else (-1 if vol < 0.015 else 0)
            return (market_regime, momentum_signal, vol_signal)

        def choose_action(state, eps):
            if random.random() < eps:
                return random.randint(0, N_ACTIONS - 1)
            return int(np.argmax(q_table[state]))

        def update_q(state, action, reward, next_state):
            current_q = q_table[state][action]
            max_next_q = max(q_table[next_state]) if next_state in q_table else 0.0
            q_table[state][action] = current_q + alpha * (reward + gamma * max_next_q - current_q)

        def get_positions_value(date, positions):
            total = 0.0
            for ts_code, pos in positions.items():
                try:
                    df = pd.read_sql(
                        f"SELECT close FROM stock_daily WHERE ts_code = '{ts_code}' AND trade_date = '{date}'",
                        conn
                    )
                    if not df.empty:
                        total += pos['shares'] * float(df.iloc[0]['close'])
                except Exception:
                    pass
            return total

        def get_close_price(ts_code, date):
            df = pd.read_sql(
                f"SELECT close FROM stock_daily WHERE ts_code = '{ts_code}' AND trade_date = '{date}'",
                conn
            )
            return float(df.iloc[0]['close']) if not df.empty else 0.0

        def execute_with_position(date, strategy, base_params, positions, cash, position_ratio):
            # 简化版选股信号（基于动量排名前N）
            signals = []
            try:
                factor_df = pd.read_sql(
                    f"""SELECT ts_code, value FROM factors
                        WHERE factor_name = 'momentum_20'
                        AND trade_date = '{date}'
                        AND value IS NOT NULL
                        ORDER BY value DESC LIMIT 20""",
                    conn
                )
                if not factor_df.empty:
                    for _, row in factor_df.head(5).iterrows():
                        price = get_close_price(row['ts_code'], date)
                        if price > 0:
                            signals.append({'ts_code': row['ts_code'], 'direction': 'buy', 'price': price})
            except Exception:
                pass

            for signal in signals:
                ts_code = signal['ts_code']
                direction = signal['direction']
                price = signal['price']
                if direction == 'buy' and cash > 0:
                    invest_amount = cash * position_ratio
                    shares = int(invest_amount * 0.1 / (price * (1 + slippage)))
                    if shares > 0:
                        cost = shares * price * (1 + slippage + commission)
                        cash -= cost
                        positions[ts_code] = {'shares': shares, 'cost': price}
                elif direction == 'sell' and ts_code in positions:
                    pos = positions[ts_code]
                    proceeds = pos['shares'] * price * (1 - slippage - commission)
                    cash += proceeds
                    del positions[ts_code]
            return cash, positions

        # ---- 运行单个 episode ----
        cash = initial_cash
        positions = {}
        nav_history = []
        episode_reward = 0.0
        actions_taken = []

        for i, date in enumerate(trade_dates):
            state = get_state(date, cash, positions, nav_history)
            action_idx = choose_action(state, epsilon)
            actions_taken.append(action_idx)
            position_ratio = ACTION_MAP[action_idx]

            cash, positions = execute_with_position(
                date, strategy, base_params, positions, cash, position_ratio
            )

            portfolio_value = cash + get_positions_value(date, positions)
            next_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else date
            next_state = get_state(next_date, cash, positions, nav_history)

            old_nav = nav_history[-1] if nav_history else initial_cash
            new_nav = portfolio_value
            reward = float(np.log(new_nav / old_nav)) if old_nav > 0 else 0.0

            update_q(state, action_idx, reward, next_state)
            nav_history.append(new_nav)
            episode_reward += reward

    finally:
        # 确保 DuckDB 连接始终被关闭（即使中途异常退出）
        conn.close()

    return episode_reward, actions_taken, dict(q_table)


class RLOptimizer:
    """
    强化学习优化器 - 动态仓位/择时

    状态空间: (market_regime, momentum_signal, volatility_signal) 各3种取值 = 27状态
    行为空间: (满仓/半仓/空仓) = (1.0, 0.5, 0.0)
    算法: Q-Learning
    """

    # 市场状态枚举
    MARKET_REGIME_MAP = {-1: 'bear', 0: 'neutral', 1: 'bull'}

    # 动量信号枚举
    MOMENTUM_MAP = {-1: 'negative', 0: 'neutral', 1: 'positive'}

    # 波动率信号枚举
    VOL_MAP = {-1: 'low_vol', 0: 'medium_vol', 1: 'high_vol'}

    # 动作: 0=空仓(0%), 1=半仓(50%), 2=满仓(100%)
    ACTION_MAP = {0: 0.0, 1: 0.5, 2: 1.0}
    ACTION_NAMES = {0: '空仓', 1: '半仓', 2: '满仓'}
    N_ACTIONS = 3

    def __init__(
        self,
        backtester,
        logger,
        start_date: str = None,
        end_date: str = None,
        gamma: float = 0.95,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.98,
        min_epsilon: float = 0.01,
        n_episodes: int = 5,  # 原50，减少以加快执行
        lookback_days: int = 20,
    ):
        self.backtester = backtester
        self.logger = logger
        self.start_date = self._format_date(start_date) if start_date else '2020-01-01'
        self.end_date = self._format_date(end_date) if end_date else '2026-03-27'

        # Q-Learning 超参数（修复：移入 __init__，原位置是 unreachable bug）
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_episodes = n_episodes
        self.lookback_days = lookback_days

        # Q表: {(regime, momentum, vol): [Q(a0), Q(a1), Q(a2)]}
        self.q_table: Dict[Tuple, List[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

        # 基准净值序列（用于计算状态）
        self._nav_history: List[float] = []
        self._price_history: Dict[str, List[float]] = {}

        # ---- 内存缓存（由 load_all_data_to_memory 填充）----
        self._market_index_cache: Dict[str, float] = {}   # {date: close_price}
        self._momentum_factor_cache: Dict[str, float] = {}  # {date: avg_mom_value}
        self._volatility_factor_cache: Dict[str, float] = {} # {date: avg_vol_value}
        self._data: Optional[pd.DataFrame] = None           # 全量日线数据

        # ---- 检查点 ----
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_lock = threading.Lock()
        self._training_history: List = []
        self._current_episode: int = 0
        self._current_episode_epsilon: float = epsilon
        self.eval_results: List = []

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _format_date(self, date_str: str) -> str:
        """统一日期格式：YYYYMMDD -> YYYY-MM-DD"""
        if date_str is None:
            return ''
        s = str(date_str)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _save_checkpoint(self, force: bool = False):
        """线程安全保存RL训练检查点（每10个episode调用）"""
        if self._checkpoint_path is None:
            return
        with self._checkpoint_lock:
            # JSON不支持tuple key，转为字符串
            q_table_str_keys = {str(k): v for k, v in self.q_table.items()}
            ckpt = {
                'q_table': q_table_str_keys,
                'training_history': self._training_history,
                'current_episode': self._current_episode,
                'current_epsilon': self._current_episode_epsilon,
                'eval_results': self.eval_results,
                'rl_batch_id': getattr(self, '_batch_id', 0),
            }
            tmp = str(self._checkpoint_path) + '.tmp'
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump(ckpt, f, default=str)
            Path(tmp).rename(self._checkpoint_path)
            if force:
                self.logger.info(f"[RL] 检查点已保存: {self._checkpoint_path}")


    def load_checkpoint(self):
        """加载检查点续训"""
        if self._checkpoint_path is None:
            try:
                ckpt_dir = Path(__file__).parent.parent / "checkpoints"
                self._checkpoint_path = ckpt_dir / "rl_checkpoint.json"
            except Exception:
                return
        
        if not self._checkpoint_path.exists():
            self.logger.warning(f"[RL] 检查点不存在：{self._checkpoint_path}")
            return
        
        try:
            with open(self._checkpoint_path, 'r') as f:
                ckpt = json.load(f)
            
            # 恢复 Q 表（字符串 key 转 tuple）
            q_table_str = ckpt.get('q_table', {})
            for k, v in q_table_str.items():
                if isinstance(k, str):
                    key_tuple = tuple(int(x) for x in k.split(','))
                else:
                    key_tuple = k
                self.q_table[key_tuple] = v
            
            # 恢复训练历史
            self._training_history = ckpt.get('training_history', [])
            self._current_episode = ckpt.get('current_episode', 0)
            self._current_episode_epsilon = ckpt.get('current_epsilon', self.epsilon)
            self.eval_results = ckpt.get('eval_results', [])
            
            self.logger.info(f"[RL] 已加载检查点：episode={self._current_episode}")
        except Exception as e:
            self.logger.warning(f"[RL] 加载检查点失败：{e}")

    def run_eval_backtest(self) -> Dict:
        """
        使用 Q-table 执行 Eval 回测
        返回回测结果字典
        """
        self.logger.info("[RL] 开始 Eval 回测...")
        
        # 预加载 539 天数据到内存
        data = self.load_all_data_to_memory()
        
        # 使用 Q-table 回测
        result = self.backtest_with_qtable(data)
        
        self.eval_results.append(result)
        self.logger.info(f"[RL] Eval 回测完成 | Sharpe={result.get('sharpe', 0):.4f}")
        
        return result

    def load_all_data_to_memory(self) -> pd.DataFrame:
        """预加载所有历史数据到内存 DataFrame，并填充缓存"""
        try:
            df = self.backtester.store.df(
                f"""SELECT trade_date, ts_code, close, volume, amount
                    FROM stock_daily
                    WHERE trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ORDER BY trade_date, ts_code"""
            )
            self.logger.info(f"[RL] 已加载 {len(df)} 条数据到内存")

            # ---- 填充市场指数缓存 ----
            index_df = df[df['ts_code'] == '000001.SH'][['trade_date', 'close']].copy()
            index_df = index_df.sort_values('trade_date')
            self._market_index_cache = {
                str(r['trade_date']): float(r['close'])
                for _, r in index_df.iterrows()
            }

            # ---- 填充动量因子缓存 ----
            try:
                mom_df = self.backtester.store.df(
                    f"""SELECT trade_date, AVG(value) as avg_mom
                        FROM factors
                        WHERE factor_name = 'momentum_20'
                        AND trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                        GROUP BY trade_date
                        ORDER BY trade_date"""
                )
                self._momentum_factor_cache = {
                    str(r['trade_date']): float(r['avg_mom'])
                    for _, r in mom_df.iterrows()
                    if r['avg_mom'] is not None
                }
            except Exception as e:
                self.logger.warning(f"[RL] 动量因子缓存加载失败: {e}")

            # ---- 填充波动率因子缓存 ----
            try:
                vol_df = self.backtester.store.df(
                    f"""SELECT trade_date, AVG(value) as avg_vol
                        FROM factors
                        WHERE factor_name = 'volatility_20'
                        AND trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                        GROUP BY trade_date
                        ORDER BY trade_date"""
                )
                self._volatility_factor_cache = {
                    str(r['trade_date']): float(r['avg_vol'])
                    for _, r in vol_df.iterrows()
                    if r['avg_vol'] is not None
                }
            except Exception as e:
                self.logger.warning(f"[RL] 波动率因子缓存加载失败: {e}")

            # 保存 data DataFrame 供 _get_positions_value 使用
            self._data = df
            self.logger.info(
                f"[RL] 缓存填充完成 | market_index={len(self._market_index_cache)}, "
                f"momentum={len(self._momentum_factor_cache)}, "
                f"volatility={len(self._volatility_factor_cache)}"
            )
            return df
        except Exception as e:
            self.logger.warning(f"[RL] 加载数据失败：{e}")
            return pd.DataFrame()

    def backtest_with_qtable(self, data: pd.DataFrame) -> Dict:
        """
        使用训练好的 Q-table 执行回测
        返回回测结果
        """
        if data.empty:
            return {'sharpe': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0}
        
        cash = self.backtester.initial_cash
        positions = {}
        nav_history = []
        
        trade_dates = sorted(data['trade_date'].unique())
        
        for date in trade_dates:
            state = self._get_state(date, cash, positions, nav_history)
            state_key = f"{state[0]},{state[1]},{state[2]}"
            
            # 使用 Q-table 决策
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                action_idx = int(np.argmax(q_values))
                position_ratio = self.ACTION_MAP[action_idx]
            else:
                position_ratio = 1.0  # 默认满仓
            
            # 执行交易
            cash, positions = self._execute_with_position(
                date=date,
                strategy={'factors': []},
                base_params={},
                positions=positions,
                cash=cash,
                position_ratio=position_ratio,
            )
            
            portfolio_value = cash + self._get_positions_value(date, positions)
            nav_history.append(portfolio_value)
        
        # 计算指标
        if len(nav_history) < 2:
            return {'sharpe': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0}
        
        nav_series = pd.Series(nav_history)
        returns = nav_series.pct_change().dropna()
        
        total_return = (nav_history[-1] - nav_history[0]) / nav_history[0]
        
        risk_free = 0.03
        if returns.std() > 0:
            sharpe = (returns.mean() * 252 - risk_free) / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # 计算最大回撤
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = float(drawdown.min())
        
        return {
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': max_drawdown,
            'final_nav': float(nav_history[-1]),
        }


    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy: Dict,
        base_params: Dict = None,
        use_rl_position: bool = True,
        batch_id: int = 0,
        daily_reset: bool = False,
    ) -> Dict:
        """
        训练 Q-Learning 策略，返回 Q 表和最优策略

        Args:
            strategy: 基础选股策略（因子列表等）
            base_params: 基础策略参数
            use_rl_position: 是否用 RL 动态仓位（True=RL仓位, False=固定满仓）

        Returns:
            {
                'q_table': {...},
                'best_policy': {...},
                'training_history': [{'episode': int, 'total_reward': float, 'epsilon': float}, ...],
                'final_sharpe': float,
                'rl_batch_id': int,
                'rl_episodes_done': int,
                'eval_results': [...],
            }
        """
        # batch0 逻辑：清除旧数据或加载 checkpoint 续训
        self._batch_id = batch_id
        if batch_id == 0 or daily_reset:
            self.logger.info(f"[RL] Batch {batch_id}: 清除旧数据，重新开始")
            self.q_table = {}
            self._training_history = []
            self.eval_results = []
        else:
            self.logger.info(f"[RL] Batch {batch_id}: 加载 checkpoint 续训")
            self.load_checkpoint()

        self.logger.info(
            f"[RL] 开始训练 | Batch={batch_id}, Episodes={self.n_episodes}, gamma={self.gamma}, "
            f"alpha={self.alpha}, initial_epsilon={self.epsilon}"
        )

        # 1. 准备历史数据（获取市场状态序列）
        self._prepare_market_data()

        # 自动设置检查点路径
        if self._checkpoint_path is None:
            try:
                ckpt_dir = Path(__file__).parent.parent / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                self._checkpoint_path = ckpt_dir / "rl_checkpoint.json"
            except Exception:
                pass

        training_history = self._training_history
        current_epsilon = self.epsilon

        # ---- 8进程并行执行 episodes ----
        trade_dates = self._get_trade_dates()
        # 获取数据库路径（用于子进程重建连接）
        # DuckDB 文件数据库的 pragma_database_list 返回 NULL，改用 store.db_path
        try:
            db_path = str(self.backtester.store.db_path)
        except Exception:
            db_path = "data/astock_full.duckdb"

        episode_args = [
            (
                episode,
                strategy,
                base_params or {},
                current_epsilon,
                self.lookback_days,
                self.gamma,
                self.alpha,
                trade_dates,
                db_path,
                self.backtester.initial_cash,
                self.backtester.commission,
                self.backtester.slippage,
            )
            for episode in range(self.n_episodes)
        ]

        # 使用 ProcessPoolExecutor(max_workers=8) 并行执行
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_episode_worker, episode_args))

        # 衰减 epsilon（串行，后续每10个episode仍串行checkpoint）
        current_epsilon = max(
            self.min_epsilon,
            self.epsilon * (self.epsilon_decay ** self.n_episodes)
        )

        # 汇总结果
        best_reward = float('-inf')
        best_q_table = None
        for episode, (episode_reward, episode_actions, q_table_snapshot) in enumerate(results):
            training_history.append({
                'episode': episode,
                'total_reward': float(episode_reward),
                'epsilon': float(current_epsilon),
                'actions': episode_actions,
            })

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_q_table = q_table_snapshot

            if episode % 10 == 0 or episode == self.n_episodes - 1:
                avg_reward = episode_reward / max(1, len(episode_actions))
                self.logger.info(
                    f"[RL] Episode {episode:3d}/{self.n_episodes} (并行) | "
                    f"Reward={episode_reward:.4f} | Avg={avg_reward:.4f}"
                )

        # 用最优 episode 的 Q 表
        if best_q_table:
            for k, v in best_q_table.items():
                if isinstance(k, str):
                    key_tuple = tuple(int(x) for x in k.split(','))
                else:
                    key_tuple = k  # 已经是 tuple（从 checkpoint 恢复时）
                self.q_table[key_tuple] = v

        self._training_history = training_history
        self._current_episode = (batch_id + 1) * self.n_episodes
        self._current_episode_epsilon = current_epsilon

        # 最终检查点
        self._save_checkpoint(force=True)

        # 3. 提取最优策略
        best_policy = self._extract_policy()

        # 4. 用最优策略跑最终回测
        final_sharpe = self._eval_policy(strategy, base_params or {}, best_policy, use_rl_position)

        # 5. 转换为可序列化格式
        q_table_serializable = {
            f"{k[0]},{k[1]},{k[2]}": v for k, v in self.q_table.items()
        }

        self.logger.info(f"[RL] 训练完成 | 最终Sharpe={final_sharpe:.4f}")

        return {
            'q_table': q_table_serializable,
            'best_policy': best_policy,
            'training_history': training_history,
            'final_sharpe': float(final_sharpe),
            'n_states': len(self.q_table),
            'n_episodes': self.n_episodes,
            'rl_batch_id': batch_id,
            'rl_episodes_done': (batch_id + 1) * self.n_episodes,
            'eval_results': self.eval_results,
        }

    # ------------------------------------------------------------------
    # Episode 训练
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        strategy: Dict,
        base_params: Dict,
        epsilon: float,
        use_rl_position: bool,
    ) -> Tuple[float, List[int]]:
        """
        运行一个 episode（相当于一次回测 + Q学习更新）
        返回: (总奖励, 动作序列)
        """
        # 重置持仓和现金
        cash = self.backtester.initial_cash
        positions = {}  # {ts_code: {'shares': int, 'cost': float}}
        nav_history = []
        episode_reward = 0.0
        actions_taken = []

        dates = self._get_trade_dates()

        for i, date in enumerate(dates):
            # 获取当前状态
            state = self._get_state(date, cash, positions, nav_history)

            # epsilon-greedy 选择动作
            action_idx = self._choose_action(state, epsilon)
            actions_taken.append(action_idx)
            position_ratio = self.ACTION_MAP[action_idx]

            # 执行交易（带 RL 仓位比例）
            cash, positions = self._execute_with_position(
                date=date,
                strategy=strategy,
                base_params=base_params,
                positions=positions,
                cash=cash,
                position_ratio=position_ratio,
            )

            # 计算持仓市值
            portfolio_value = cash + self._get_positions_value(date, positions)

            # 获取下一个状态（实际上状态转移由市场推动）
            next_date = dates[i + 1] if i + 1 < len(dates) else date
            next_state = self._get_state(next_date, cash, positions, nav_history)

            # 计算奖励 = 当日收益率
            old_nav = nav_history[-1] if nav_history else self.backtester.initial_cash
            new_nav = portfolio_value
            reward = self._get_reward(old_nav, new_nav)

            # Q-Learning 更新
            self._update_q(self.q_table, state, action_idx, reward, next_state)

            nav_history.append(new_nav)
            episode_reward += reward

        return episode_reward, actions_taken

    # ------------------------------------------------------------------
    # 状态 & 奖励
    # ------------------------------------------------------------------

    def _get_state(
        self,
        date: str,
        portfolio_value: float,
        positions: Dict,
        nav_history: List[float],
    ) -> Tuple[int, int, int]:
        """
        离散化状态：
        - market_regime: 基于最近 lookback_days 的指数收益率
        - momentum_signal: 基于持仓股票近期动能
        - volatility_signal: 基于近期波动率
        """
        # 市场状态：对比基准指数收益率
        market_ret = self._get_market_return(date)
        if market_ret > 0.02:
            market_regime = 1   # 牛市
        elif market_ret < -0.02:
            market_regime = -1  # 熊市
        else:
            market_regime = 0   # 震荡

        # 动量信号：基于持仓或市场近期走势
        momentum_ret = self._get_momentum_return(date)
        if momentum_ret > 0.01:
            momentum_signal = 1
        elif momentum_ret < -0.01:
            momentum_signal = -1
        else:
            momentum_signal = 0

        # 波动率信号
        vol = self._get_volatility(date)
        if vol > 0.03:
            vol_signal = 1   # 高波动
        elif vol < 0.015:
            vol_signal = -1  # 低波动
        else:
            vol_signal = 0   # 中等波动

        return (market_regime, momentum_signal, vol_signal)

    def _get_reward(self, old_nav: float, new_nav: float) -> float:
        """奖励 = 对数收益率"""
        if old_nav <= 0:
            return 0.0
        return float(np.log(new_nav / old_nav))

    # ------------------------------------------------------------------
    # Q-Learning 核心
    # ------------------------------------------------------------------

    def _choose_action(self, state: Tuple, epsilon: float) -> int:
        """epsilon-greedy 动作选择"""
        if random.random() < epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        else:
            q_values = self.q_table[state]
            return int(np.argmax(q_values))

    def _update_q(
        self,
        q_table: Dict,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
    ) -> Dict:
        """Q-Learning 更新公式"""
        current_q = q_table[state][action]
        max_next_q = max(q_table[next_state]) if next_state in q_table else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        q_table[state][action] = new_q
        return q_table

    def _extract_policy(self) -> Dict:
        """从 Q 表提取确定性最优策略"""
        policy = {}
        for state, q_values in self.q_table.items():
            best_action = int(np.argmax(q_values))
            policy[f"{state[0]},{state[1]},{state[2]}"] = {
                'action_idx': best_action,
                'action_name': self.ACTION_NAMES[best_action],
                'position_ratio': self.ACTION_MAP[best_action],
                'q_value': float(q_values[best_action]),
                'all_q': [float(q) for q in q_values],
            }
        return policy

    def _eval_policy(
        self,
        strategy: Dict,
        base_params: Dict,
        policy: Dict,
        use_rl_position: bool,
    ) -> float:
        """用最优策略跑回测，返回夏普"""
        cash = self.backtester.initial_cash
        positions = {}
        nav_history = []
        dates = self._get_trade_dates()

        for date in dates:
            state = self._get_state(date, cash, positions, nav_history)
            state_key = f"{state[0]},{state[1]},{state[2]}"

            if use_rl_position and state_key in policy:
                position_ratio = policy[state_key]['position_ratio']
            else:
                position_ratio = 1.0  # 默认满仓

            cash, positions = self._execute_with_position(
                date=date,
                strategy=strategy,
                base_params=base_params,
                positions=positions,
                cash=cash,
                position_ratio=position_ratio,
            )

            portfolio_value = cash + self._get_positions_value(date, positions)
            nav_history.append(portfolio_value)

        # 计算夏普
        if len(nav_history) < 2:
            return 0.0

        nav_series = pd.Series(nav_history)
        returns = nav_series.pct_change().dropna()
        risk_free = 0.03
        if returns.std() > 0:
            sharpe = (returns.mean() * 252 - risk_free) / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        return float(sharpe)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _prepare_market_data(self):
        """预加载市场指数数据（用于状态计算）"""
        self._nav_history = []
        self._price_history = {}

        try:
            index_df = self.backtester.store.df(
                f"""SELECT trade_date, close FROM stock_daily
                    WHERE ts_code = '000001.SH'
                    AND trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ORDER BY trade_date"""
            )
            if not index_df.empty:
                self._nav_history = index_df['close'].tolist()
        except Exception as e:
            self.logger.warning(f"[RL] 加载市场数据失败: {e}")

    def _get_trade_dates(self) -> List[str]:
        """获取回测期交易日期列表"""
        df = self.backtester.store.df(
            f"""SELECT DISTINCT trade_date FROM stock_daily
                WHERE trade_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                ORDER BY trade_date"""
        )
        return df['trade_date'].astype(str).tolist()

    def _get_market_return(self, date: str) -> float:
        """获取截至 date 的市场指数收益率（使用缓存）"""
        if not self._market_index_cache:
            # 缓存为空时，直接查 DuckDB
            try:
                idx_prices = self.backtester.store.df(
                    f"""SELECT trade_date, close FROM stock_daily
                        WHERE ts_code = '000001.SH'
                        AND trade_date <= '{date}'
                        ORDER BY trade_date DESC
                        LIMIT {self.lookback_days}"""
                )
                if len(idx_prices) >= 2:
                    ret = (float(idx_prices.iloc[0]['close']) - float(idx_prices.iloc[-1]['close'])) \
                          / float(idx_prices.iloc[-1]['close'])
                    return float(ret)
            except Exception:
                pass
            return 0.0
        try:
            dates_sorted = sorted(self._market_index_cache.keys())
            d_idx = None
            for i, d in enumerate(dates_sorted):
                if d <= date:
                    d_idx = i
            if d_idx is None or d_idx < self.lookback_days:
                return 0.0
            start_date = dates_sorted[d_idx - self.lookback_days]
            ret = (self._market_index_cache[date] - self._market_index_cache[start_date]) \
                  / self._market_index_cache[start_date]
            return float(ret)
        except Exception:
            return 0.0

    def _get_momentum_return(self, date: str) -> float:
        """获取近期动量收益（使用缓存）"""
        if date in self._momentum_factor_cache:
            return self._momentum_factor_cache[date]
        return 0.0

    def _get_volatility(self, date: str) -> float:
        """获取近期波动率（使用缓存）"""
        if date in self._volatility_factor_cache:
            return self._volatility_factor_cache[date]
        return 0.02  # 默认中等波动

    def _execute_with_position(
        self,
        date: str,
        strategy: Dict,
        base_params: Dict,
        positions: Dict,
        cash: float,
        position_ratio: float = 1.0,
    ) -> Tuple[float, Dict]:
        """
        带仓位的交易执行
        position_ratio: 0.0=空仓, 0.5=半仓, 1.0=满仓
        """
        # 调用底层回测器的选股逻辑，获取当日信号
        # 为了不重复造轮子，这里直接复用 backtester._get_signals
        try:
            signals = self.backtester._get_signals(
                date=date,
                strategy=strategy,
                params=base_params,
                positions=positions,
                current_holdings={},
                rebalance_dates={date},  # 每日都评估
            )
        except Exception:
            signals = []

        slippage = self.backtester.slippage
        commission = self.backtester.commission

        for signal in signals:
            ts_code = signal['ts_code']
            direction = signal['direction']
            price = signal['price']

            if direction == 'buy' and cash > 0:
                # 按 position_ratio 决定实际仓位
                invest_amount = cash * position_ratio
                shares = int(invest_amount * 0.1 / (price * (1 + slippage)))
                if shares > 0:
                    cost = shares * price * (1 + slippage + commission)
                    cash -= cost
                    positions[ts_code] = {
                        'shares': shares,
                        'cost': price,
                    }

            elif direction == 'sell' and ts_code in positions:
                pos = positions[ts_code]
                proceeds = pos['shares'] * price * (1 - slippage - commission)
                cash += proceeds
                del positions[ts_code]

        return cash, positions

    def _get_positions_value(self, date: str, positions: Dict) -> float:
        """计算持仓市值（使用内存 DataFrame）"""
        if self._data is None or self._data.empty:
            return 0.0
        total = 0.0
        date_str = str(date)
        day_data = self._data[self._data['trade_date'].astype(str) == date_str]
        close_map = dict(zip(day_data['ts_code'].astype(str), day_data['close']))
        for ts_code, pos in positions.items():
            if ts_code in close_map:
                total += pos['shares'] * float(close_map[ts_code])
        return total
