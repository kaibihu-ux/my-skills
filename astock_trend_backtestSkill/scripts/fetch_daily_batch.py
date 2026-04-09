#!/usr/bin/env python3
"""独立抓取脚本：处理指定股票列表，结果写入自己的 JSON（不碰 DuckDB）

多数据源优先级（逐级降级）：
    1. curl (Eastmoney API)  — 最快，免费
    2. curl (Tencent ifzq)    — Eastmoney备选
    3. curl (Sina getKLineData) — 腾讯备选
    4. baostock             — 稳，登录制
    5. akshare              — Python库，备选

某来源超时/返回空/报错 → 自动尝试下一级

Usage:
    python3 scripts/fetch_daily_batch.py <group_id> <target_date>
    Example: python3 scripts/fetch_daily_batch.py 0 2026-04-07
"""
import os, sys, time, json, subprocess, pandas as pd

# 清除代理（baostock 不走代理）
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

SKILL_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
CKPT_DIR = f'{SKILL_DIR}/checkpoints'
LOG_FILE = None  # 初始化为 None，main() 中设置

GROUP_ID = sys.argv[1] if len(sys.argv) > 1 else '0'
TARGET_DATE = sys.argv[2] if len(sys.argv) > 2 else None

LIST_FILE = None  # 惰性初始化
RESULT_FILE = None

# 交易所前缀映射
PREFIX_MAP = {
    '600': 'sh', '601': 'sh', '603': 'sh', '605': 'sh',
    '688': 'sh', '689': 'sh',
    '000': 'sz', '001': 'sz', '002': 'sz', '003': 'sz',
    '300': 'sz', '301': 'sz',
}

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

# =========================================================================
# 数据源 1: curl + Eastmoney API（日K线，最优先）
# =========================================================================
def fetch_curl_eastmoney(code, trade_date):
    """通过 curl 请求 Eastmoney 日K线接口"""
    # secid: 1=上交所, 0=深交所
    market = '1' if code.startswith(('600','601','603','605','688','689')) else '0'
    secid = f"{market}.{code}"

    url = (
        f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        f"&fields1=f1,f2,f3,f4,f5"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        f"&klt=101"       # 日K
        f"&fqt=1"        # 前复权
        f"&beg={trade_date.replace('-','')}"
        f"&end={trade_date.replace('-','')}"
    )

    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '8', url],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        import json as _json
        data = _json.loads(result.stdout)
        if not data.get('data') or not data['data'].get('klines'):
            return None

        kline_str = data['data']['klines'][0]  # e.g. "2026-04-07,10.12,9.98,10.17,9.95,378826,380103392.00,2.17"
        fields = kline_str.split(',')
        if len(fields) < 7:
            return None

        # fields: 日期,开盘,收盘,最高,最低,成交量,成交额[,涨跌幅,...]
        row = {
            'date':       fields[0],
            'open':       float(fields[1]),
            'close':      float(fields[2]),
            'high':       float(fields[3]),
            'low':        float(fields[4]),
            'vol':        float(fields[5]),
            'amount':     float(fields[6]),
            'pct_chg':    float(fields[7]) if len(fields) > 7 else 0.0,
        }
        row['ts_code'] = code
        row['trade_date'] = row['date']
        row['limit_up'] = row['pct_chg'] >= 9.9
        row['limit_down'] = row['pct_chg'] <= -9.9
        row['is_st'] = False
        row['suspended'] = row['vol'] == 0

        df = pd.DataFrame([row])
        return df

    except Exception:
        return None

# =========================================================================
# 数据源 1b: curl + Tencent ifzq API（日K线，Eastmoney备选）
# =========================================================================
def fetch_curl_tencent(code, trade_date):
    """通过 curl 请求腾讯 ifzq 日K线接口（东方财富系）"""
    # 沪市 sh, 深市 sz
    prefix = 'sh' if code.startswith(('600','601','603','605','688','689')) else 'sz'
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?_var=kline_dayqfq"
        f"&param={prefix}{code},day,{trade_date},{trade_date},10,qfq"
    )
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '8', '-A', 'Mozilla/5.0', url],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        raw = result.stdout.strip()
        json_start = raw.find('{')
        if json_start < 0:
            return None
        import json as _json
        data = _json.loads(raw[json_start:])
        key = f"{prefix}{code}"
        if key not in data.get('data', {}):
            return None
        klines = data['data'][key].get('qfqday', [])
        if not klines or len(klines[0]) < 6:
            return None

        # kline格式: [date, open, close, high, low, vol]
        row = klines[0]
        df = pd.DataFrame([{
            'date':       row[0],
            'open':       float(row[1]),
            'close':      float(row[2]),
            'high':       float(row[3]),
            'low':        float(row[4]),
            'vol':        float(row[5]),
            'amount':     0.0,
            'pct_chg':    0.0,
        }])
        df['ts_code'] = code
        df['trade_date'] = df['date']
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        return df
    except Exception:
        return None

# =========================================================================
# 数据源 1c: curl + Sina 历史K线 API（日K线，腾讯备选）
# =========================================================================
def fetch_curl_sina(code, trade_date):
    """通过 curl 请求新浪财经历史K线接口"""
    # 沪市 sh, 深市 sz
    prefix = 'sh' if code.startswith(('600','601','603','605','688','689')) else 'sz'
    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
        f"/CN_MarketData.getKLineData"
        f"?symbol={prefix}{code}"
        f"&scale=240"     # 240分钟=日K
        f"&ma=no"
        f"&datalen=1"
    )
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '8',
             '-H', 'Referer: http://finance.sina.com.cn',
             '-A', 'Mozilla/5.0', url],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        import json as _json
        data = _json.loads(result.stdout)
        if not data or len(data) == 0:
            return None

        row = data[0]  # 最新一天
        if row.get('day') != trade_date:
            return None

        df = pd.DataFrame([{
            'date':       row['day'],
            'open':       float(row['open']),
            'close':      float(row['close']),
            'high':       float(row['high']),
            'low':        float(row['low']),
            'vol':        float(row['volume']),
            'amount':     0.0,
            'pct_chg':    0.0,
        }])
        df['ts_code'] = code
        df['trade_date'] = df['date']
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        return df
    except Exception:
        return None

# =========================================================================
# 数据源 2: baostock
# =========================================================================
def fetch_baostock(code, trade_date):
    """通过 baostock 获取单只股票单日数据"""
    import baostock as bs

    bs_code = f"{PREFIX_MAP.get(code[:3], 'sz')}.{code}"
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=trade_date, end_date=trade_date,
            frequency='d', adjustflag='3'  # 后复权
        )
        bs.logout()

        if rs.error_code != '0':
            return None

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return None

        df = pd.DataFrame(data, columns=rs.fields)
        df['ts_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
        df['trade_date'] = df['date']
        df.rename(columns={'volume': 'vol', 'pctChg': 'pct_chg'}, inplace=True)

        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0

        return df

    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return None

# =========================================================================
# 数据源 3: akshare（最后备选）
# =========================================================================
def fetch_akshare(code, trade_date):
    """通过 akshare 获取单只股票单日数据"""
    try:
        import akshare as ak

        symbol_map = {
            '600': 'sh600', '601': 'sh601', '603': 'sh603', '605': 'sh605',
            '688': 'sh688', '689': 'sh689',
            '000': 'sz000', '001': 'sz001', '002': 'sz002', '003': 'sz003',
            '300': 'sz300', '301': 'sz301',
        }
        symbol = symbol_map.get(code[:3], f'sz{code}')
        start = trade_date.replace('-', '')
        end = start

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period='daily',
            start_date=start,
            end_date=end,
            adjust='qfq'
        )

        if df is None or df.empty:
            return None

        # 列名标准化
        rename_map = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'vol',
            '成交额': 'amount',
            '涨跌幅': 'pct_chg',
            '股票代码': 'ts_code',
        }
        df = df.rename(columns=rename_map)
        df = df.rename(columns={'trade_date': 'date'})

        df['ts_code'] = df['ts_code'].astype(str).str.zfill(6)
        # 移除前缀（sz/sh）
        df['ts_code'] = df['ts_code'].str.replace('sz', '').str.replace('sh', '')

        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['trade_date'] = df['date']
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0

        return df

    except Exception:
        return None

# =========================================================================
# 主函数：逐级降级获取
# =========================================================================
def fetch_one(code, trade_date):
    """尝试多个数据源，按优先级：
    curl(eastmoney) → curl(tencent) → curl(sina) → baostock → akshare"""

    # 1. curl (Eastmoney)
    df = fetch_curl_eastmoney(code, trade_date)
    if df is not None and not df.empty:
        return df, 'em'

    # 2. curl (Tencent)
    df = fetch_curl_tencent(code, trade_date)
    if df is not None and not df.empty:
        return df, 'tx'

    # 3. curl (Sina)
    df = fetch_curl_sina(code, trade_date)
    if df is not None and not df.empty:
        return df, 'sina'

    # 4. baostock
    df = fetch_baostock(code, trade_date)
    if df is not None and not df.empty:
        return df, 'bs'

    # 5. akshare
    df = fetch_akshare(code, trade_date)
    if df is not None and not df.empty:
        return df, 'ak'

    return None, 'none'

def main():
    global TARGET_DATE, LOG_FILE, LIST_FILE, RESULT_FILE

    LOG_FILE = f'/tmp/update_history_fetch_{GROUP_ID}.log'
    LIST_FILE = f'{CKPT_DIR}/batch_{GROUP_ID}.json'
    RESULT_FILE = f'{CKPT_DIR}/batch_{GROUP_ID}_results.json'

    if TARGET_DATE is None:
        import datetime
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        TARGET_DATE = yesterday.strftime('%Y-%m-%d')

    if not os.path.exists(LIST_FILE):
        log(f"[ERROR] 股票列表不存在: {LIST_FILE}")
        return

    with open(LIST_FILE) as f:
        codes = json.load(f)

    log(f"=== 组 {GROUP_ID} | 日期 {TARGET_DATE} | {len(codes)} 只 | {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    results = []
    source_stats = {'em': 0, 'tx': 0, 'sina': 0, 'bs': 0, 'ak': 0, 'none': 0}

    for i, code in enumerate(codes):
        df, source = fetch_one(code, TARGET_DATE)
        if df is not None and not df.empty:
            results.append({'code': code, 'data': df.to_dict('records')[0]})
            source_stats[source] += 1

        if (i + 1) % 50 == 0:
            log(f"  进度 {i+1}/{len(codes)} | 已有数据 {len(results)} | em:{source_stats['em']} tx:{source_stats['tx']} sina:{source_stats['sina']} bs:{source_stats['bs']} ak:{source_stats['ak']} none:{source_stats['none']}")

        # 间隔0.2秒，避免被限流
        time.sleep(0.2)

    # 写入结果 JSON
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f)

    total_success = sum(v for k, v in source_stats.items() if k != 'none')
    log(f"组 {GROUP_ID} 完成: {len(codes)} 只抓取, {total_success} 只有数据 -> {RESULT_FILE}")
    log(f"  来源: em={source_stats['em']} tx={source_stats['tx']} sina={source_stats['sina']} bs={source_stats['bs']} ak={source_stats['ak']} 失败={source_stats['none']}")

if __name__ == '__main__':
    main()
