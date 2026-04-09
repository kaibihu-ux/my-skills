import os, sys, time, duckdb, pandas as pd, baostock as bs

for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
con = duckdb.connect(DB_PATH, read_only=False)
target_date = '2026-04-03'

# 清空测试
con.execute(f"DELETE FROM stock_daily WHERE trade_date = '{target_date}'")
cnt = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{target_date}'").fetchone()[0]
print(f"清空后剩余: {cnt}")

PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}
test_codes = ['600000', '000001', '600036']

bs.login()
for code in test_codes:
    bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
    rs = bs.query_history_k_data_plus(bs_code,
        'date,code,open,high,low,close,volume,amount,pctChg',
        start_date=target_date, end_date=target_date,
        frequency='d', adjustflag='3')
    print(f"{code}: error={rs.error_code}", end='')
    data = []
    while rs.next(): data.append(rs.get_row_data())
    print(f", rows={len(data)}", end='')
    if data:
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
        cols = ['ts_code','trade_date','open','high','low','close','vol','amount',
                'pct_chg','limit_up','limit_down','is_st','suspended']
        df = df[cols]
        try:
            con.register('dft', df)
            con.execute("INSERT INTO stock_daily BY NAME SELECT * FROM dft")
            con.unregister('dft')
            print(f" -> 写入成功! pct_chg={df['pct_chg'].iloc[0]:.2f}%")
        except Exception as e:
            print(f" -> 写入失败: {e}")
    else:
        print()

bs.logout()
cnt = con.execute(f"SELECT COUNT(*) FROM stock_daily WHERE trade_date = '{target_date}'").fetchone()[0]
print(f"\n最终 {target_date}: {cnt} 条")
con.close()
