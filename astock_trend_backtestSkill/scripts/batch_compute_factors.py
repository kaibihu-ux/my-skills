#!/usr/bin/env python3
"""
批量预计算60个因子（跳过已存在的6个）
写入 DuckDB factors 表
每处理完一只股票直接写入，避免内存堆积
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 已存在的因子（不重算）
SKIP_FACTORS = {
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
    'volatility_20', 'volume_ratio_20'
}

# ============================================================
# 因子计算函数
# ============================================================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def calc_momentum_120(close):
    return close.pct_change(120)


def calc_acceleration_20(close):
    mom = close.pct_change(20)
    return mom.diff(20)


def calc_momentum_volume_corr_20(close, vol):
    ret = close.pct_change(20)
    return ret.rolling(20).corr(vol)


def calc_price_relative_20(close):
    return close / close.rolling(20).mean()


def calc_high_low_ratio_20(close, high, low):
    hl = high.rolling(20).max() - low.rolling(20).min()
    return (close - low.rolling(20).min()) / hl.replace(0, np.nan)


def calc_ma5_ma20_cross(close):
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    return np.sign(ma5 - ma20)


def calc_trend_strength_rsq(close):
    """rolling 20 线性回归 R²"""
    n = 20
    rsq = pd.Series(np.nan, index=close.index)
    for i in range(n - 1, len(close)):
        win = close.iloc[i - n + 1:i + 1].values
        if len(win) == n and not np.any(np.isnan(win)):
            idx = np.arange(n, dtype=float)
            cov = np.cov(idx, win)[0, 1]
            var_idx = np.var(idx)
            var_win = np.var(win)
            if var_idx > 0 and var_win > 0:
                ss_res = np.sum((win - (np.mean(win) + cov / var_idx * (idx - np.mean(idx)))) ** 2)
                ss_tot = np.var(win) * n
                rsq.iloc[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rsq


def calc_trend_slope(close):
    n = 20
    idx = np.arange(n, dtype=float)
    rslt = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(n - 1, len(close)):
        win = close.iloc[i - n + 1:i + 1].values
        if len(win) == n and not np.any(np.isnan(win)):
            cov = np.cov(idx, win)[0, 1]
            var_idx = np.var(idx)
            if var_idx > 0:
                rslt.iloc[i] = cov / var_idx / close.iloc[i]
    return rslt


def calc_supertrend(close, high, low, period=20, mult=3.0):
    atr = calc_atr_20(high, low, close)
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    trend = pd.Series(1.0, index=close.index)
    for i in range(1, len(close)):
        if pd.isna(upper.iloc[i]):
            continue
        if close.iloc[i] > (upper.iloc[i - 1] if not pd.isna(upper.iloc[i - 1]) else np.nan):
            trend.iloc[i] = 1.0
        elif close.iloc[i] < (lower.iloc[i - 1] if not pd.isna(lower.iloc[i - 1]) else np.nan):
            trend.iloc[i] = -1.0
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


def _ichimoku_base(high, low, period):
    return (high.rolling(period).max() + low.rolling(period).min()) / 2


def calc_ichimoku_a(high, low, close):
    tenkan = _ichimoku_base(high, low, 9)
    kijun = _ichimoku_base(high, low, 26)
    return ((tenkan + kijun) / 2).shift(26)


def calc_ichimoku_b(high, low, close):
    return ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)


def calc_ichimoku_cloud(high, low, close):
    a = calc_ichimoku_a(high, low, close)
    b = calc_ichimoku_b(high, low, close)
    return (a - b).abs()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close):
    dif = _ema(close, 12) - _ema(close, 26)
    dea = _ema(dif, 9)
    return dif, dea


def calc_macd_signal(close):
    dif, dea = calc_macd(close)
    return dif - dea


def _kdj(high, low, close, n=9, m1=3, m2=3):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    rsv = (close - ll) / (hh - ll + 1e-9) * 100
    k = pd.Series(50.0, index=close.index)
    d = pd.Series(50.0, index=close.index)
    for i in range(n, len(close)):
        if pd.isna(rsv.iloc[i]):
            continue
        k.iloc[i] = 2/3 * k.iloc[i-1] + 1/3 * rsv.iloc[i]
        d.iloc[i] = 2/3 * d.iloc[i-1] + 1/3 * k.iloc[i]
    j = m1 * k - m2 * d
    return k, d, j


def calc_kdj_k(high, low, close):
    k, _, _ = _kdj(high, low, close)
    return k


def calc_kdj_j(high, low, close):
    _, _, j = _kdj(high, low, close)
    return j


def calc_bollinger_position(close):
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bw = upper - lower
    return (close - lower) / bw.replace(0, np.nan)


def calc_bollinger_bandwidth(close):
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / ma.replace(0, np.nan)


def calc_cci_20(high, low, close):
    tp = (high + low + close) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    return (tp - sma) / (0.015 * mad + 1e-9)


def calc_williams_r(high, low, close, period=14):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return (highest - close) / (highest - lowest + 1e-9) * -100


def calc_adx_14(high, low, close, n=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(n).mean()


def calc_atr_20(high, low, close):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(20).mean()


def calc_volatility_60(close):
    return close.pct_change().rolling(60).std()


def calc_volatility_ratio(close):
    v20 = close.pct_change().rolling(20).std()
    v60 = close.pct_change().rolling(60).std()
    return v20 / v60.replace(0, np.nan)


def calc_max_drawdown_20(close):
    roll_max = close.rolling(20, min_periods=1).max()
    dd = close / roll_max - 1
    return dd.rolling(20).min()


def calc_max_drawdown_60(close):
    roll_max = close.rolling(60, min_periods=1).max()
    dd = close / roll_max - 1
    return dd.rolling(60).min()


def calc_downside_volatility(close):
    returns = close.pct_change()
    neg = returns.where(returns < 0, 0)
    return neg.rolling(20).std()


def calc_volatility_skew(close):
    returns = close.pct_change()
    up = returns.where(returns > 0, 0).rolling(20).std()
    down = returns.where(returns < 0, 0).rolling(20).std()
    return up / down.replace(0, np.nan)


def calc_volume_ratio_60(vol):
    return vol / vol.rolling(60).mean()


def calc_volume_ma5_crossover(vol):
    ma5 = vol.rolling(5).mean()
    above = vol > ma5
    prev_above = above.shift(1).fillna(False)
    return (above & ~prev_above & (vol > vol.shift(1))).astype(float)


def calc_amount_ma20(amount):
    ma = amount.rolling(20).mean()
    return amount / ma.replace(0, np.nan) - 1


def calc_vol_price_divergence(close, vol):
    return close.pct_change(20) - vol.pct_change(20)


def calc_turnover_rate(vol):
    return vol / vol.rolling(20).mean()


def calc_turnover_rate_ma5(vol):
    tr = vol / vol.rolling(20).mean()
    return tr.rolling(5).mean()


# ============================================================
# 向量化批量因子计算
# ============================================================

def compute_factors_vectorized(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    vol: pd.Series,
    amount: pd.Series,
    dates: pd.Series
) -> pd.DataFrame:
    """
    对单只股票计算所有36个新因子，返回长表 DataFrame
    列: factor_name, trade_date, value
    所有 Series 必须按相同顺序排列（已排序）
    """
    rows = []

    # --- TREND (5) ---
    # momentum_120
    v = calc_momentum_120(close)
    for dt, val in zip(dates, v):
        rows.append(('momentum_120', dt, val))

    # acceleration_20
    v = calc_acceleration_20(close)
    for dt, val in zip(dates, v):
        rows.append(('acceleration_20', dt, val))

    # momentum_volume_corr_20
    v = calc_momentum_volume_corr_20(close, vol)
    for dt, val in zip(dates, v):
        rows.append(('momentum_volume_corr_20', dt, val))

    # price_relative_20
    v = calc_price_relative_20(close)
    for dt, val in zip(dates, v):
        rows.append(('price_relative_20', dt, val))

    # high_low_ratio_20
    v = calc_high_low_ratio_20(close, high, low)
    for dt, val in zip(dates, v):
        rows.append(('high_low_ratio_20', dt, val))

    # --- TECH (19) ---
    v = calc_ma5_ma20_cross(close)
    for dt, val in zip(dates, v):
        rows.append(('ma5_ma20_cross', dt, val))

    v = calc_trend_strength_rsq(close)
    for dt, val in zip(dates, v):
        rows.append(('trend_strength_rsq', dt, val))

    v = calc_trend_slope(close)
    for dt, val in zip(dates, v):
        rows.append(('trend_slope', dt, val))

    v = calc_supertrend(close, high, low)
    for dt, val in zip(dates, v):
        rows.append(('supertrend', dt, val))

    v = calc_ichimoku_a(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('ichimoku_a', dt, val))

    v = calc_ichimoku_b(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('ichimoku_b', dt, val))

    v = calc_ichimoku_cloud(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('ichimoku_cloud', dt, val))

    v = calc_rsi(close, 14)
    for dt, val in zip(dates, v):
        rows.append(('rsi_14', dt, val))

    v = calc_rsi(close, 28)
    for dt, val in zip(dates, v):
        rows.append(('rsi_28', dt, val))

    dif, _ = calc_macd(close)
    for dt, val in zip(dates, dif):
        rows.append(('macd', dt, val))

    v = calc_macd_signal(close)
    for dt, val in zip(dates, v):
        rows.append(('macd_signal', dt, val))

    v = calc_kdj_k(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('kdj_k', dt, val))

    v = calc_kdj_j(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('kdj_j', dt, val))

    v = calc_bollinger_position(close)
    for dt, val in zip(dates, v):
        rows.append(('bollinger_position', dt, val))

    v = calc_bollinger_bandwidth(close)
    for dt, val in zip(dates, v):
        rows.append(('bollinger_bandwidth', dt, val))

    v = calc_cci_20(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('cci_20', dt, val))

    v = calc_williams_r(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('williams_r', dt, val))

    v = calc_adx_14(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('adx_14', dt, val))

    v = calc_atr_20(high, low, close)
    for dt, val in zip(dates, v):
        rows.append(('atr_20', dt, val))

    # --- VOL (6) ---
    v = calc_volatility_60(close)
    for dt, val in zip(dates, v):
        rows.append(('volatility_60', dt, val))

    v = calc_volatility_ratio(close)
    for dt, val in zip(dates, v):
        rows.append(('volatility_ratio', dt, val))

    v = calc_max_drawdown_20(close)
    for dt, val in zip(dates, v):
        rows.append(('max_drawdown_20', dt, val))

    v = calc_max_drawdown_60(close)
    for dt, val in zip(dates, v):
        rows.append(('max_drawdown_60', dt, val))

    v = calc_downside_volatility(close)
    for dt, val in zip(dates, v):
        rows.append(('downside_volatility', dt, val))

    v = calc_volatility_skew(close)
    for dt, val in zip(dates, v):
        rows.append(('volatility_skew', dt, val))

    # --- VOLUME (6) ---
    v = calc_volume_ratio_60(vol)
    for dt, val in zip(dates, v):
        rows.append(('volume_ratio_60', dt, val))

    v = calc_volume_ma5_crossover(vol)
    for dt, val in zip(dates, v):
        rows.append(('volume_ma5_crossover', dt, val))

    v = calc_amount_ma20(amount)
    for dt, val in zip(dates, v):
        rows.append(('amount_ma20', dt, val))

    v = calc_vol_price_divergence(close, vol)
    for dt, val in zip(dates, v):
        rows.append(('vol_price_divergence', dt, val))

    v = calc_turnover_rate(vol)
    for dt, val in zip(dates, v):
        rows.append(('turnover_rate', dt, val))

    v = calc_turnover_rate_ma5(vol)
    for dt, val in zip(dates, v):
        rows.append(('turnover_rate_ma5', dt, val))

    df = pd.DataFrame(rows, columns=['factor_name', 'trade_date', 'value'])
    # 清理
    df['value'] = df['value'].replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================
# 主程序
# ============================================================

def main():
    sys.path.insert(0, '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill')
    from src.core.duckdb_store import DuckDBStore

    DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
    store = DuckDBStore(DB_PATH)

    # 获取所有股票列表
    ts_codes = store.df(
        "SELECT DISTINCT ts_code FROM stock_daily ORDER BY ts_code"
    )['ts_code'].tolist()
    total = len(ts_codes)
    print(f"[INFO] 共 {total} 只股票待处理", flush=True)

    t0 = time.time()
    total_rows = 0
    written = 0

    INSERT_SQL = (
        "INSERT INTO factors (factor_name, ts_code, trade_date, value, zscore) "
        "VALUES (?, ?, CAST(? AS DATE), ?, NULL)"
    )

    for i, ts_code in enumerate(ts_codes):
        try:
            # 读取单只股票数据
            df_s = store.df(
                f"SELECT trade_date, open, high, low, close, vol, amount "
                f"FROM stock_daily WHERE ts_code = '{ts_code}' ORDER BY trade_date"
            )
            if df_s.empty:
                continue

            # 确保列类型正确
            df_s['close'] = df_s['close'].astype(float)
            df_s['high'] = df_s['high'].astype(float)
            df_s['low'] = df_s['low'].astype(float)
            df_s['vol'] = df_s['vol'].astype(float)
            if 'amount' not in df_s.columns or df_s['amount'].isna().all():
                df_s['amount'] = df_s['vol'] * df_s['close'] / 100  # proxy
            else:
                df_s['amount'] = df_s['amount'].astype(float)

            dates = df_s['trade_date']
            close = df_s['close']
            high = df_s['high']
            low = df_s['low']
            vol = df_s['vol']
            amount = df_s['amount']

            # 计算因子
            factor_df = compute_factors_vectorized(close, high, low, vol, amount, dates)

            if factor_df.empty:
                continue

            # 过滤无效值
            factor_df = factor_df.dropna(subset=['value'])

            # 转为长表并清理
            factor_df['ts_code'] = ts_code
            factor_df = factor_df.dropna(subset=['value'])

            if factor_df.empty:
                continue

            # trade_date 转为 YYYY-MM-DD 字符串
            factor_df['trade_date'] = factor_df['trade_date'].apply(lambda x: str(x)[:10])

            # 构造最终 DataFrame（列顺序与 factors 表一致）
            # zscore 不放在 DataFrame 里，用 SQL NULL 替代
            out = factor_df[['factor_name', 'ts_code', 'trade_date', 'value']].copy()
            out['value'] = out['value'].astype(float)

            if out.empty:
                continue

            # DuckDB bulk insert：注册临时表 + INSERT ... SELECT
            tbl = f'fb_{i}'
            try:
                store.conn.register(tbl, out)
                store.conn.execute(
                    f"INSERT INTO factors (factor_name, ts_code, trade_date, value, zscore) "
                    f"SELECT factor_name, ts_code, CAST(trade_date AS DATE), value, NULL::DOUBLE "
                    f"FROM {tbl} "
                    f"ON CONFLICT (factor_name, ts_code, trade_date) DO NOTHING"
                )
                written += len(out)
            except Exception as e1:
                # 兜底：逐行 INSERT
                for _, r in out.iterrows():
                    try:
                        store.conn.execute(
                            "INSERT INTO factors (factor_name, ts_code, trade_date, value, zscore) "
                            "VALUES (?, ?, CAST(? AS DATE), ?, NULL) "
                            "ON CONFLICT (factor_name, ts_code, trade_date) DO NOTHING",
                            [r['factor_name'], r['ts_code'], r['trade_date'], float(r['value'])]
                        )
                        written += 1
                    except Exception:
                        pass
            finally:
                # 必须清理临时表，避免残留影响后续注册
                try:
                    store.conn.unregister(tbl)
                except Exception:
                    pass

        except Exception as e:
            print(f"[WARN] ts_code={ts_code} error: {e}", flush=True)

        # 每100只股票打印进度
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"[PROGRESS] {i+1}/{total} stocks, "
                f"written={written:,}, "
                f"elapsed={elapsed:.0f}s, "
                f"rate={rate:.1f}/s, "
                f"ETA={eta:.0f}s",
                flush=True
            )

    elapsed = time.time() - t0
    print(f"\n[DONE] 总耗时: {elapsed:.1f}s, 共写入 {written:,} 行", flush=True)

    # 验证
    print("\n=== 因子行数验证 ===", flush=True)
    result = store.df(
        "SELECT factor_name, COUNT(*) as cnt FROM factors GROUP BY factor_name ORDER BY cnt DESC"
    )
    print(result.to_string(index=False), flush=True)

    expected_factors = {
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
        'momentum_120', 'acceleration_20', 'momentum_volume_corr_20',
        'price_relative_20', 'high_low_ratio_20',
        'ma5_ma20_cross', 'trend_strength_rsq', 'trend_slope', 'supertrend',
        'ichimoku_a', 'ichimoku_b', 'ichimoku_cloud',
        'rsi_14', 'rsi_28', 'macd', 'macd_signal',
        'kdj_k', 'kdj_j', 'bollinger_position', 'bollinger_bandwidth',
        'cci_20', 'williams_r', 'adx_14', 'atr_20',
        'volatility_20', 'volatility_60', 'volatility_ratio',
        'max_drawdown_20', 'max_drawdown_60', 'downside_volatility', 'volatility_skew',
        'volume_ratio_20', 'volume_ratio_60', 'volume_ma5_crossover',
        'amount_ma20', 'vol_price_divergence', 'turnover_rate', 'turnover_rate_ma5'
    }
    stored_factors = set(result['factor_name'].tolist())
    missing = expected_factors - stored_factors
    zero_count = set(result[result['cnt'] == 0]['factor_name'].tolist())

    print(f"\n总因子数: {len(stored_factors)} (期望 {len(expected_factors)})", flush=True)
    if missing:
        print(f"[WARN] 缺失因子: {missing}", flush=True)
    if zero_count:
        print(f"[WARN] 无数据因子: {zero_count}", flush=True)
    if not missing and not zero_count:
        print(f"[PASS] 所有 {len(expected_factors)} 个因子均有数据!", flush=True)

    store.close()


if __name__ == '__main__':
    main()
