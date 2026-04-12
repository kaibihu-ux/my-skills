import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from numba import njit, prange


# ============================================================
# numba 加速的底层计算函数（独立于 pandas，高性能）
# ============================================================

@njit
def _calc_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """numba 加速 RSI 计算"""
    n = len(prices)
    changes = np.zeros(n)
    if n > 1:
        changes[1:] = prices[1:] - prices[:-1]

    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)

    avg_gains = np.zeros(n)
    avg_losses = np.zeros(n)

    # 初始平均
    if n >= period:
        avg_gains[period - 1] = np.mean(gains[:period])
        avg_losses[period - 1] = np.mean(losses[:period])

    for i in range(period, n):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period

    rs = np.zeros(n)
    rs[period - 1:] = avg_gains[period - 1:] / (avg_losses[period - 1:] + 1e-10)
    rsi = np.zeros(n)
    rsi[period - 1:] = 100.0 - (100.0 / (1.0 + rs[period - 1:]))
    for i in range(period - 1):
        rsi[i] = np.nan
    return rsi


@njit
def _calc_macd_numba(prices: np.ndarray, fast: int, slow: int, signal: int) -> tuple:
    """numba 加速 MACD 计算"""
    n = len(prices)
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    dif = np.zeros(n)
    dea = np.zeros(n)
    macd = np.zeros(n)

    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)

    # 初始化 EMA
    if n > 0:
        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]

    for i in range(1, n):
        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i - 1]
        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i - 1]
        dif[i] = ema_fast[i] - ema_slow[i]
        dea[i] = alpha_signal * dif[i] + (1 - alpha_signal) * dea[i - 1]
        macd[i] = (dif[i] - dea[i]) * 2.0

    return dif, dea, macd


@njit(parallel=True)
def _calc_bollinger_numba(prices: np.ndarray, period: int) -> tuple:
    """numba 并行加速布林带"""
    n = len(prices)
    ma = np.zeros(n)
    std = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)

    for i in prange(period - 1, n):
        s = i - period + 1
        ma[i] = 0.0
        for j in range(s, i + 1):
            ma[i] += prices[j]
        ma[i] /= period

        var = 0.0
        for j in range(s, i + 1):
            diff = prices[j] - ma[i]
            var += diff * diff
        std[i] = np.sqrt(var / period)

        upper[i] = ma[i] + 2.0 * std[i]
        lower[i] = ma[i] - 2.0 * std[i]

    position = np.zeros(n)
    bandwidth = np.zeros(n)
    for i in range(period - 1, n):
        bw = upper[i] - lower[i]
        if bw > 1e-10:
            position[i] = (prices[i] - lower[i]) / bw
            bandwidth[i] = bw / (ma[i] + 1e-10)
        else:
            position[i] = 0.5
            bandwidth[i] = 0.0

    return position, bandwidth


@njit
def _calc_cci_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """numba 加速 CCI"""
    n = len(close)
    tp = (high + low + close) / 3.0
    cci = np.zeros(n)

    for i in range(period - 1, n):
        typical = tp[i]
        s = i - period + 1
        sma = 0.0
        for j in range(s, i + 1):
            sma += tp[j]
        sma /= period

        mad = 0.0
        for j in range(s, i + 1):
            mad += abs(tp[j] - sma)
        mad /= period

        if mad > 1e-10:
            cci[i] = (typical - sma) / (0.015 * mad)
        else:
            cci[i] = 0.0

    for i in range(period - 1):
        cci[i] = np.nan
    return cci


@njit
def _calc_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """numba 加速 ATR"""
    n = len(close)
    tr = np.zeros(n)
    if n > 0:
        tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    for i in range(period - 1):
        atr[i] = np.nan
    return atr


@njit
def _calc_adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """numba 加速 ADX"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_val = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr_val[i] = max(hl, max(hpc, lpc))

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    adx = np.zeros(n)
    # Smooth with Wilder's method
    period_di = period
    alpha_di = 1.0 / period_di

    smooth_plus = 0.0
    smooth_minus = 0.0
    smooth_tr = 0.0

    for i in range(1, period + 1):
        smooth_tr += tr_val[i]
        smooth_plus += plus_dm[i]
        smooth_minus += minus_dm[i]

    if smooth_tr > 1e-10:
        di_plus = (smooth_plus / smooth_tr) * 100.0
        di_minus = (smooth_minus / smooth_tr) * 100.0
        dx = (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)) * 100.0
        adx[period] = dx

    for i in range(period + 1, n):
        smooth_tr = smooth_tr - smooth_tr / period_di + tr_val[i]
        smooth_plus = smooth_plus - smooth_plus / period_di + plus_dm[i]
        smooth_minus = smooth_minus - smooth_minus / period_di + minus_dm[i]

        if smooth_tr > 1e-10:
            di_plus = (smooth_plus / smooth_tr) * 100.0
            di_minus = (smooth_minus / smooth_tr) * 100.0
            dx = (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)) * 100.0
            adx[i] = adx[i - 1] * (period_di - 1) / period_di + dx

    for i in range(period):
        adx[i] = np.nan
    return adx


@njit
def _calc_williams_r_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """numba 加速 Williams %R"""
    n = len(close)
    wr = np.zeros(n)
    for i in range(period - 1, n):
        s = i - period + 1
        highest = high[s]
        lowest = low[s]
        for j in range(s + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        denom = highest - lowest
        if denom > 1e-10:
            wr[i] = (highest - close[i]) / denom * -100.0
        else:
            wr[i] = -50.0
    for i in range(period - 1):
        wr[i] = np.nan
    return wr


# ============================================================
# 因子名称常量
# ============================================================
TREND_FACTORS = ['momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
                 'acceleration_20', 'momentum_volume_corr_20', 'price_relative_20', 'high_low_ratio_20']

TECH_FACTORS = ['ma5_ma20_cross', 'trend_strength_rsq', 'trend_slope', 'supertrend',
                'ichimoku_a', 'ichimoku_b', 'ichimoku_cloud',
                'rsi_14', 'rsi_28', 'macd', 'macd_signal', 'kdj_k', 'kdj_j',
                'bollinger_position', 'bollinger_bandwidth', 'cci_20', 'williams_r', 'adx_14', 'atr_20']

VOL_FACTORS = ['volatility_20', 'volatility_60', 'volatility_ratio', 
               'max_drawdown_20', 'max_drawdown_60', 'downside_volatility', 'volatility_skew']

VOLUME_FACTORS = ['volume_ratio_20', 'volume_ratio_60', 'volume_ma5_crossover',
                  'amount_ma20', 'vol_price_divergence', 'turnover_rate', 'turnover_rate_ma5']

FUNDAMENTAL_FACTORS = ['pe', 'pb', 'ps', 'pcf', 'roe', 'roa', 'gross_margin', 'net_margin',
                        'revenue_growth', 'profit_growth', 'operating_cash_flow', 'debt_to_equity',
                        'earnings_yield', 'book_to_market', 'f_score']

HKT_FACTORS = ['hkt_hold_ratio', 'hkt_hold_ratio_change_5', 'hkt_hold_ratio_change_20', 'hkt_net_flow_20']

COMPOSITE_FACTORS = ['composite_momentum', 'quality_momentum', 'value_momentum',
                     'liquidity_adjusted_momentum', 'alpha_factor_pca']

ALL_FACTORS = (TREND_FACTORS + TECH_FACTORS + VOL_FACTORS + VOLUME_FACTORS + 
               FUNDAMENTAL_FACTORS + HKT_FACTORS + COMPOSITE_FACTORS)


class FactorCalculator:
    """因子计算器"""
    
    @staticmethod
    def calc_momentum(df: pd.DataFrame, period: int) -> pd.Series:
        """计算动量因子"""
        return df['close'].pct_change(period)
    
    @staticmethod
    def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal).mean()
        macd = (dif - dea) * 2
        return dif, dea, macd
    
    @staticmethod
    def calc_bollinger(df: pd.DataFrame, period: int = 20) -> tuple:
        """计算布林带"""
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        bandwidth_val = upper - lower
        # 防止除零
        position = (df['close'] - lower) / bandwidth_val.replace(0, np.nan)
        bandwidth = bandwidth_val / ma.replace(0, np.nan)
        return position, bandwidth
    
    @staticmethod
    def calc_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算波动率"""
        return df['close'].pct_change().rolling(period).std()
    
    @staticmethod
    def calc_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算量比"""
        return df['vol'] / df['vol'].rolling(period).mean()


class FactorMiner:
    """因子挖掘机"""
    
    def __init__(self, store, logger):
        self.store = store
        self.logger = logger
        self.calculator = FactorCalculator()
    
    def mine_candidates(self, n: int = 100) -> List[Dict]:
        """生成因子候选"""
        candidates = []
        # 技术因子候选
        for name in ALL_FACTORS[:n]:
            candidates.append({'name': name, 'type': 'technical'})
        return candidates
    
    def calculate_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单个因子（优先使用 numba 加速版本）"""
        if factor_name in ('rsi_14', 'rsi_28'):
            period = int(factor_name.split('_')[1])
            prices = df['close'].values.astype(np.float64)
            result = _calc_rsi_numba(prices, period)
            return pd.Series(result, index=df.index)

        elif factor_name in ('macd', 'macd_signal'):
            prices = df['close'].values.astype(np.float64)
            dif, dea, macd = _calc_macd_numba(prices, 12, 26, 9)
            if factor_name == 'macd':
                return pd.Series(macd, index=df.index)
            else:
                return pd.Series(dif, index=df.index)

        elif factor_name in ('bollinger_position', 'bollinger_bandwidth'):
            prices = df['close'].values.astype(np.float64)
            pos, bw = _calc_bollinger_numba(prices, 20)
            if factor_name == 'bollinger_position':
                return pd.Series(pos, index=df.index)
            else:
                return pd.Series(bw, index=df.index)

        elif factor_name in ('cci_20',):
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            result = _calc_cci_numba(high, low, close, 20)
            return pd.Series(result, index=df.index)

        elif factor_name in ('atr_20',):
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            result = _calc_atr_numba(high, low, close, 20)
            return pd.Series(result, index=df.index)

        elif factor_name in ('adx_14',):
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            result = _calc_adx_numba(high, low, close, 14)
            return pd.Series(result, index=df.index)

        elif factor_name in ('williams_r',):
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            result = _calc_williams_r_numba(high, low, close, 14)
            return pd.Series(result, index=df.index)

        # 以下使用原始 pandas 实现
        elif factor_name == 'momentum_20':
            return self.calculator.calc_momentum(df, 20)
        elif factor_name == 'momentum_5':
            return self.calculator.calc_momentum(df, 5)
        elif factor_name == 'momentum_10':
            return self.calculator.calc_momentum(df, 10)
        elif factor_name == 'momentum_60':
            return self.calculator.calc_momentum(df, 60)
        elif factor_name == 'momentum_120':
            return self.calculator.calc_momentum(df, 120)
        elif factor_name == 'momentum_250':
            return self.calculator.calc_momentum(df, 250)
        elif factor_name == 'volatility_120':
            return self.calculator.calc_volatility(df, 120)
        elif factor_name == 'volatility_20':
            return self.calculator.calc_volatility(df, 20)
        elif factor_name == 'volatility_60':
            return self.calculator.calc_volatility(df, 60)
        elif factor_name == 'volume_ratio_20':
            return self.calculator.calc_volume_ratio(df, 20)
        elif factor_name == 'volume_ratio_60':
            return self.calculator.calc_volume_ratio(df, 60)

        # ... 更多因子
        return pd.Series(0, index=df.index)
    
    def batch_calculate(self, df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """批量计算因子"""
        result = pd.DataFrame(index=df.index)
        for name in factor_names:
            result[name] = self.calculate_factor(df, name)
        return result
