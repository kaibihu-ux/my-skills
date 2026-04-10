"""
数据获取管理类
- 股票来源：baostock.query_all_stock()（主板过滤：600/601/603 + 000/001）
- 数据源：5级降级（Eastmoney → 腾讯 → 新浪 → baostock → akshare）
- 抓取间隔：0.2秒/只（防限流）
- 并行：多进程（max_workers）
"""
import os
import subprocess
import json
import time
import baostock as bs
import pandas as pd
import numpy as np
from typing import Optional, List
import multiprocessing as mp

from .stock_pool import AShareMainBoardFilter

# ========================================================================
# 股票代码前缀映射
# ========================================================================
_PREFIX_MAP = {
    '600': 'sh', '601': 'sh', '603': 'sh',  # 上证
    '000': 'sz', '001': 'sz',                # 深证主板
}

def _get_bs_all_stock_codes(trade_date: str) -> List[str]:
    """用 baostock.query_all_stock 获取指定日期的主板股票代码列表

    返回格式：['600000', '600001', ..., '000001', ...]（去掉 sh./sz. 前缀）
    仅保留主板：沪（600/601/603）+ 深（000/001）
    """
    bs.login()
    rs = bs.query_all_stock(day=trade_date)
    bs.logout()
    if rs.error_code != '0':
        return []
    codes = []
    while rs.next():
        row = rs.get_row_data()
        code = row[0]  # e.g. "sh.600000"
        # 去掉前缀
        code = code.replace('sh.', '').replace('sz.', '')
        codes.append(code)
    # 过滤：仅主板
    main_board = [c for c in codes if c[:3] in {'600', '601', '603', '000', '001'}]
    return sorted(set(main_board))


def _get_bs_code(ts_code: str) -> str:
    """ts_code -> baostock格式 (sh.600000 / sz.000001)"""
    prefix = _PREFIX_MAP.get(ts_code[:3], 'sz')
    return f"{prefix}.{ts_code}"

# ========================================================================
# Curl 方式获取（优先）：腾讯历史K线 -> 新浪日K
# ========================================================================
def _curl_eastmoney(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """Eastmoney 日K线，返回 df 或 None"""
    market = '1' if ts_code.startswith(('600','601','603','605','688','689')) else '0'
    secid = f"{market}.{ts_code}"
    url = (
        f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        f"&fields1=f1,f2,f3,f4,f5"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
        f"&klt=101"
        f"&fqt=1"
        f"&beg={trade_date.replace('-','')}"
        f"&end={trade_date.replace('-','')}"
    )
    try:
        r = subprocess.run(
            ['curl', '-s', '--max-time', '8', url],
            capture_output=True, text=True, timeout=12
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        data = json.loads(r.stdout)
        if not data.get('data') or not data['data'].get('klines'):
            return None
        kline_str = data['data']['klines'][0]
        fields = kline_str.split(',')
        if len(fields) < 7:
            return None
        row = {
            'ts_code': ts_code,
            'trade_date': fields[0],
            'open':       float(fields[1]),
            'close':      float(fields[2]),
            'high':       float(fields[3]),
            'low':        float(fields[4]),
            'vol':        float(fields[5]),
            'amount':     float(fields[6]),
            'pct_chg':    float(fields[7]) if len(fields) > 7 else 0.0,
            'limit_up': False,
            'limit_down': False,
            'is_st': False,
            'suspended': False,
        }
        row['limit_up'] = row['pct_chg'] >= 9.9
        row['limit_down'] = row['pct_chg'] <= -9.9
        row['suspended'] = row['vol'] == 0
        return pd.DataFrame([row])
    except Exception:
        return None

def _get_tx_prefix(ts_code: str) -> str:
    """ts_code -> 腾讯API前缀 (sh / sz)"""
    return 'sh' if ts_code.startswith(('600','601','603','605','688','689')) else 'sz'


def _curl_tencent(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """腾讯历史K线（前复权），返回 df 或 None"""
    tx_prefix = _get_tx_prefix(ts_code)  # sh 或 sz（无点！）
    try:
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?_var=kline_dayqfq&param={tx_prefix}{ts_code},day,{trade_date},{trade_date},10,qfq")
        r = subprocess.run(
            ['curl', '-s', '--max-time', '8', url, '-H', 'Referer: https://gu.qq.com/'],
            capture_output=True, text=False, timeout=12
        )
        raw = r.stdout.decode('utf-8', errors='ignore')
        if 'kline_dayqfq=' not in raw:
            return None
        json_str = raw.split('kline_dayqfq=', 1)[1]
        data = json.loads(json_str)
        key = f"{tx_prefix}{ts_code}"
        klines = data.get('data', {}).get(key, {}).get('qfqday', [])
        for kline in klines:
            if kline[0] == trade_date:
                vol = kline[5]
                if isinstance(vol, str):
                    vol = float(vol.replace(',', '')) * 100
                else:
                    vol = float(vol) * 100
                df = pd.DataFrame([{
                    'ts_code': ts_code,
                    'trade_date': kline[0],
                    'open': float(kline[1]),
                    'close': float(kline[2]),
                    'high': float(kline[3]),
                    'low': float(kline[4]),
                    'vol': vol,
                    'amount': 0.0,
                    'pct_chg': 0.0,
                    'limit_up': False,
                    'limit_down': False,
                    'is_st': False,
                    'suspended': False,
                }])
                return df
    except Exception:
        pass
    return None

def _curl_sina(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """新浪日K线，返回 df 或 None"""
    tx_prefix = _get_tx_prefix(ts_code)  # sh 或 sz（无点！）
    try:
        url = (f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
               f"/CN_MarketData.getKLineData?symbol={tx_prefix}{ts_code}&scale=240&ma=no&datalen=5")
        r = subprocess.run(
            ['curl', '-s', '--max-time', '8', url,
             '-H', 'Referer: http://finance.sina.com.cn',
             '-A', 'Mozilla/5.0'],
            capture_output=True, text=False, timeout=12
        )
        raw = r.stdout.decode('utf-8', errors='ignore')
        data = json.loads(raw)
        for kline in data:
            if kline['day'] == trade_date:
                df = pd.DataFrame([{
                    'ts_code': ts_code,
                    'trade_date': kline['day'],
                    'open': float(kline['open']),
                    'close': float(kline['close']),
                    'high': float(kline['high']),
                    'low': float(kline['low']),
                    'vol': float(kline['volume']),
                    'amount': 0.0,
                    'pct_chg': 0.0,
                    'limit_up': False,
                    'limit_down': False,
                    'is_st': False,
                    'suspended': False,
                }])
                return df
    except Exception:
        pass
    return None

def _fetch_akshare(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """akshare 日K线，返回 df 或 None"""
    try:
        import akshare as ak
        symbol_map = {
            '600': 'sh600', '601': 'sh601', '603': 'sh603', '605': 'sh605',
            '688': 'sh688', '689': 'sh689',
            '000': 'sz000', '001': 'sz001', '002': 'sz002', '003': 'sz003',
        }
        symbol = symbol_map.get(ts_code[:3], f'sz{ts_code}')
        start = trade_date.replace('-', '')
        end = start
        df = ak.stock_zh_a_hist(symbol=symbol, period='daily',
                                  start_date=start, end_date=end, adjust='qfq')
        if df is None or df.empty:
            return None
        rename_map = {
            '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'vol',
            '成交额': 'amount', '涨跌幅': 'pct_chg', '股票代码': 'ts_code',
        }
        df = df.rename(columns=rename_map)
        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['ts_code'] = df['ts_code'].astype(str).str.zfill(6)
        df['ts_code'] = df['ts_code'].str.replace('sz', '').str.replace('sh', '')
        df['trade_date'] = df['trade_date'].astype(str)
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                'vol', 'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
        return df[cols]
    except Exception:
        return None

def _fetch_with_fallback(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """五链路备援获取：Eastmoney -> 腾讯 -> 新浪 -> baostock -> akshare"""
    # 链路1: Eastmoney
    df = _curl_eastmoney(ts_code, trade_date)
    if df is not None and not df.empty:
        return df
    # 链路2: 腾讯
    df = _curl_tencent(ts_code, trade_date)
    if df is not None and not df.empty:
        return df
    # 链路3: 新浪
    df = _curl_sina(ts_code, trade_date)
    if df is not None and not df.empty:
        return df
    # 链路4: baostock
    df = _baostock_fetch_one(ts_code, trade_date)
    if df is not None and not df.empty:
        return df
    # 链路5: akshare
    df = _fetch_akshare(ts_code, trade_date)
    if df is not None and not df.empty:
        return df
    return None

def _baostock_fetch_one(ts_code: str, trade_date: str) -> Optional[pd.DataFrame]:
    """baostock 获取单只股票单日数据（兜底用）"""
    try:
        bs_code = _get_bs_code(ts_code)
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=trade_date, end_date=trade_date,
            frequency='d', adjustflag='3'
        )
        bs.logout()
        if rs.error_code != '0':
            return None
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['ts_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
        df['trade_date'] = df['date']
        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                'vol', 'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
        return df[cols]
    except Exception:
        try:
            bs.logout()
        except:
            pass
        return None

def _parallel_fetch_batch(codes: List[str], trade_date: str, interval: float = 0.05) -> List[pd.DataFrame]:
    """独立进程：批量获取股票日线（curl优先+间隔控速）"""
    results = []
    for code in codes:
        time.sleep(interval)  # 间隔控速
        df = _fetch_with_fallback(code, trade_date)
        if df is not None and not df.empty:
            results.append(df)
    return results

# ========================================================================
# 原有 baostock 多股批量接口（保留，向后兼容）
# ========================================================================
def _baostock_fetch_batch(codes: List[str], start_fmt: str, end_fmt: str) -> List[pd.DataFrame]:
    """独立进程：批量获取股票日线数据（仅baostock，用于区间获取）"""
    results = []
    bs.login()
    for code in codes:
        try:
            bs_code = _get_bs_code(code)
            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,code,open,high,low,close,volume,amount,pctChg',
                start_date=start_fmt, end_date=end_fmt,
                frequency='d', adjustflag='3'
            )
            if rs.error_code != '0':
                continue
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                continue
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['ts_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
            df['trade_date'] = df['date']
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['limit_up'] = df['pct_chg'] >= 9.9
            df['limit_down'] = df['pct_chg'] <= -9.9
            df['is_st'] = False
            df['suspended'] = df['vol'] == 0
            cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'vol', 'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
            results.append(df[cols])
        except Exception:
            pass
    bs.logout()
    return results

# ========================================================================
# DataManager 主类
# ========================================================================
class DataManager:
    """数据获取管理类（curl优先 + 10进程并行）"""

    def __init__(self, store, logger):
        self.store = store
        self.logger = logger
        self.filter = AShareMainBoardFilter()
        self._session_codes: List[str] = []
        self._bs_logged_in = False

    def _bs_login(self):
        if not self._bs_logged_in:
            bs.login()
            self._bs_logged_in = True

    def _bs_logout(self):
        if self._bs_logged_in:
            bs.logout()
            self._bs_logged_in = False

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        if not date_str:
            return date_str
        d = date_str.strip()
        if len(d) == 8 and d.isdigit():
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return d

    def _to_baostock_code(self, ts_code: str) -> str:
        prefix = ts_code[:3] if ts_code.startswith(('600', '00')) else ts_code[:3]
        if prefix in _PREFIX_MAP:
            return f"{_PREFIX_MAP[prefix]}.{ts_code}"
        return f"sz.{ts_code}"

    def update_stock_list(self) -> int:
        """更新股票列表"""
        self.logger.info("更新股票列表...")
        self._bs_login()

        rs = bs.query_all_stock(day='2020-01-02')
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
        raw = pd.DataFrame(data_list, columns=rs.fields)
        raw.columns = ['code', 'status', 'name']
        raw = raw[raw['status'] == '1']

        raw['exchange'] = raw['code'].str[:3]
        raw['ts_code'] = raw['code'].str.replace('sh.', '').str.replace('sz.', '')

        def is_main_board_stock(row):
            code = row['ts_code']
            ex = row['exchange']
            if ex == 'sh.' and code.startswith(('600', '601', '603')):
                return True
            if ex == 'sz.' and code.startswith(('000', '001')):
                return True
            return False

        df = raw[raw.apply(is_main_board_stock, axis=1)].copy()
        df = df[df['ts_code'].str.match(r'^\d{6}$')].copy()
        df = df[['ts_code', 'name']].copy()
        df['symbol'] = df['ts_code']
        df['list_date'] = pd.NaT
        df['delist_date'] = pd.NaT
        df['industry'] = None
        df['market_cap'] = None
        df = df[['ts_code', 'symbol', 'name', 'list_date', 'delist_date', 'industry', 'market_cap']]

        self.store.execute("DELETE FROM stock_list")
        self.store.conn.execute("INSERT INTO stock_list SELECT * FROM df")
        self._session_codes = df['ts_code'].tolist()
        self.logger.info(f"股票列表更新完成: {len(df)} 只主板股票")
        return len(df)

    def _fetch_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取单只股票日线区间（baostock，用于多日查询）"""
        try:
            self._bs_login()
            bs_code = self._to_baostock_code(ts_code)
            start_fmt = self._normalize_date(start_date)
            end_fmt = self._normalize_date(end_date)
            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,code,open,high,low,close,volume,amount,pctChg',
                start_date=start_fmt, end_date=end_fmt,
                frequency='d', adjustflag='3'
            )
            if rs.error_code != '0':
                return None
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                return None
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['ts_code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
            df['trade_date'] = df['date']
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['limit_up'] = df['pct_chg'] >= 9.9
            df['limit_down'] = df['pct_chg'] <= -9.9
            df['is_st'] = False
            df['suspended'] = df['vol'] == 0
            cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'vol', 'amount', 'pct_chg', 'limit_up', 'limit_down', 'is_st', 'suspended']
            return df[cols]
        except Exception:
            return None

    def update_daily(self, start_date: str, end_date: str, max_workers: int = 8, sync_stock_list: bool = True, force: bool = False) -> int:
        """更新日线数据（多进程并行获取，每个进程独立连接）

        注意：区间更新（start_date != end_date）使用 baostock 多日模式；
        单日增量更新请使用 update_daily_incremental()。

        force=True 时强制全量重新拉取（跳过"已有数据"检查），用于盘后确保收盘价。

        股票来源：baostock.query_all_stock()（主板，过滤600/601/603+000/001）
        """
        # 区分单日 vs 多日
        start_fmt = self._normalize_date(start_date)
        end_fmt = self._normalize_date(end_date)

        if start_fmt == end_fmt and not force:
            # 单日：使用 curl 并行（更快）
            return self._update_daily_curl(start_fmt, max_workers=max_workers)

        # 多日：使用 baostock 并行
        self.logger.info(f"更新日线数据 {start_fmt} ~ {end_fmt}... (并行 workers={max_workers})")

        # 用 baostock.query_all_stock 获取基准股票
        self._session_codes = _get_bs_all_stock_codes(start_fmt)
        total = len(self._session_codes)
        self.logger.info(f"baostock基准: {total} 只")

        if total == 0:
            self.logger.warning("baostock.query_all_stock 返回空，使用 stock_list 兜底")
            stocks_df = self.store.df("SELECT ts_code FROM stock_list")
            self._session_codes = stocks_df['ts_code'].tolist()
            total = len(self._session_codes)

        chunk_size = max(1, (total + max_workers - 1) // max_workers)
        chunks = [self._session_codes[i:i+chunk_size] for i in range(0, total, chunk_size)]
        self.logger.info(f"分 {len(chunks)} 批，每批约 {chunk_size} 只")

        buffer_dfs = []
        with mp.Pool(max_workers) as pool:
            results_list = pool.starmap(
                _baostock_fetch_batch,
                [(chunk, start_fmt, end_fmt) for chunk in chunks]
            )
            for batch_results in results_list:
                buffer_dfs.extend(batch_results)

        if buffer_dfs:
            self._flush_daily(buffer_dfs)

        success_count = len(buffer_dfs)
        self.logger.info(f"日线更新完成，成功: {success_count}/{total} 只股票")
        return success_count

    def _update_daily_curl(self, trade_date: str, max_workers: int = 8, force: bool = False) -> int:
        """单日增量更新：bs.query_all_stock基准 + 五链路curl备援 + 进程并行"""
        self.logger.info(f"单日增量更新 {trade_date} (bs基准, 五链路备援, workers={max_workers})")

        # 清除代理环境变量（curl直接连）
        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
            os.environ.pop(k, None)

        # force=True 时：先探查一只股票，确认数据已发布再删旧数据
        if force:
            probe_code = self.store.df(
                f"SELECT ts_code FROM stock_daily WHERE trade_date = '{trade_date}' LIMIT 1"
            )
            if not probe_code.empty:
                test_df = _fetch_with_fallback(probe_code.iloc[0]['ts_code'], trade_date)
                if test_df is None or test_df.empty:
                    self.logger.warning(f"今日数据未发布（{trade_date}），跳过强制刷新，保留现有数据")
                    return 0

        # 用 baostock.query_all_stock 获取该日期主板股票
        self._session_codes = _get_bs_all_stock_codes(trade_date)
        self.logger.info(f"baostock基准: {len(self._session_codes)} 只")

        if not self._session_codes:
            self.logger.warning("baostock.query_all_stock 返回空，使用 stock_list 兜底")
            stocks_df = self.store.df("SELECT ts_code FROM stock_list")
            self._session_codes = stocks_df['ts_code'].tolist()

        # 排除已有数据（force=True 时跳过，强制全量拉取确保收盘价）
        if force:
            to_fetch = list(self._session_codes)
        else:
            existing = set(self.store.df(
                f"SELECT ts_code FROM stock_daily WHERE trade_date = '{trade_date}'"
            )['ts_code'].tolist())
            to_fetch = [c for c in self._session_codes if c not in existing]
        total = len(to_fetch)
        self.logger.info(f"baostock基准: {len(self._session_codes)}, 已有: {len(existing)}, 待取: {total}")

        if total == 0:
            self.logger.info("无待取数据")
            return 0

        # 分块并行
        chunk_size = max(1, (total + max_workers * 4 - 1) // (max_workers * 4))
        chunks = [to_fetch[i:i+chunk_size] for i in range(0, total, chunk_size)]
        self.logger.info(f"分 {len(chunks)} 批，每批 ~{chunk_size} 只, workers={max_workers}")

        buffer_dfs = []
        interval = 0.2  # 200ms/只，防限流

        with mp.Pool(max_workers) as pool:
            results_list = pool.starmap(
                _parallel_fetch_batch,
                [(chunk, trade_date, interval) for chunk in chunks]
            )
            for batch_results in results_list:
                buffer_dfs.extend(batch_results)
                self.logger.info(f"累积: {len(buffer_dfs)} 只")

        if buffer_dfs:
            self._flush_daily(buffer_dfs)

        success_count = len(buffer_dfs)
        self.logger.info(f"单日更新完成，成功: {success_count}/{total} 只 ({trade_date})")
        return success_count

    def _flush_daily(self, dfs: List[pd.DataFrame]):
        """批量写入日线数据"""
        if not dfs:
            return
        combined = pd.concat(dfs, ignore_index=True)
        if combined.empty:
            return
        try:
            for ts_code in combined['ts_code'].unique():
                mask = combined['ts_code'] == ts_code
                dates = combined.loc[mask, 'trade_date'].unique()
                for td in dates:
                    self.store.conn.execute(
                        f"DELETE FROM stock_daily WHERE ts_code = '{ts_code}' AND trade_date = '{td}'"
                    )
            self.store.conn.execute("INSERT INTO stock_daily BY NAME SELECT * FROM combined")
        except Exception as e:
            self.logger.error(f"写入日线数据失败: {e}")

    def get_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股日线"""
        start_fmt = self._normalize_date(start_date)
        end_fmt = self._normalize_date(end_date)
        return self.store.df(
            "SELECT * FROM stock_daily WHERE ts_code = ? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            [ts_code, start_fmt, end_fmt]
        )

    def get_all_daily(self, trade_date: str) -> pd.DataFrame:
        """获取某日所有股票日线"""
        trade_date_fmt = self._normalize_date(trade_date)
        return self.store.df(
            "SELECT * FROM stock_daily WHERE trade_date = ?", [trade_date_fmt]
        )

    def get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        start_fmt = self._normalize_date(start_date)
        end_fmt = self._normalize_date(end_date)
        df = self.store.df(
            "SELECT DISTINCT trade_date FROM stock_daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
            [start_fmt, end_fmt]
        )
        return df['trade_date'].astype(str).tolist()
