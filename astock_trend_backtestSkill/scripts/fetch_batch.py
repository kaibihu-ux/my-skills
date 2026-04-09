#!/usr/bin/env python3
"""独立抓取脚本：处理指定股票列表，结果写入自己的 JSON（不碰 DuckDB）"""
import os, sys, time, json, pandas as pd

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

CKPT_DIR = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/checkpoints'
LOG_FILE = f'/tmp/update_20260403_fetch_{sys.argv[1]}.log'
TARGET_DATE = '2026-04-03'
PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}
LIST_FILE = f'{CKPT_DIR}/batch_{sys.argv[1]}.json'
RESULT_FILE = f'{CKPT_DIR}/batch_{sys.argv[1]}_results.json'

def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')
            f.flush()
    except Exception:
        pass

def fetch_one(code):
    import baostock as bs
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,pctChg',
            start_date=TARGET_DATE, end_date=TARGET_DATE,
            frequency='d', adjustflag='3'
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
        df['ts_code'] = df['code'].str.replace('sh.','').str.replace('sz.','')
        df['trade_date'] = df['date']
        df.rename(columns={'volume':'vol','amount':'amount','pctChg':'pct_chg'}, inplace=True)
        for col in ['open','high','low','close','vol','amount','pct_chg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['limit_up'] = df['pct_chg'] >= 9.9
        df['limit_down'] = df['pct_chg'] <= -9.9
        df['is_st'] = False
        df['suspended'] = df['vol'] == 0
        return df
    except:
        try:
            bs.logout()
        except:
            pass
        return None

def main():
    if not os.path.exists(LIST_FILE):
        log(f"股票列表不存在: {LIST_FILE}")
        return

    with open(LIST_FILE) as f:
        codes = json.load(f)

    log(f"=== 组 {sys.argv[1]} 开始 {len(codes)} 只 {time.strftime('%H:%M:%S')} ===")

    results = []
    for i, code in enumerate(codes):
        df = fetch_one(code)
        if df is not None and not df.empty:
            results.append({'code': code, 'data': df.to_dict('records')[0]})

        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(codes)} | 已有数据 {len(results)}")

        # 每50只刷新baostock会话
        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    # 保存结果到JSON
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f)

    log(f"组 {sys.argv[1]} 完成: {len(codes)} 只抓取, {len(results)} 只有数据 -> {RESULT_FILE}")

if __name__ == '__main__':
    main()
