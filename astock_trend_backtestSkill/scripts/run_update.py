#!/usr/bin/env python3
"""原子化数据更新：杀掉锁持有者 → 运行更新 → 重启 ga_optimization"""
import os, sys, time, signal, subprocess, duckdb

# 取消代理
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
    os.environ.pop(k, None)

DB_PATH = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill/data/astock_full.duckdb'
LOG_FILE = '/tmp/update_20260403.log'

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def kill_dblock_holders():
    """杀掉所有持有 DuckDB 锁的进程"""
    result = subprocess.run(['lsof', '-t', DB_PATH], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')
    killed = []
    for pid in pids:
        pid = pid.strip()
        if pid and pid.isdigit():
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed.append(pid)
            except:
                pass
    return killed

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False

def wait_until_unlocked(max_wait=30):
    """等待 DuckDB 锁释放，最多 max_wait 秒"""
    for _ in range(max_wait):
        result = subprocess.run(['lsof', '-t', DB_PATH], capture_output=True, text=True)
        if not result.stdout.strip():
            return True
        time.sleep(1)
    return False

def main():
    log("=" * 50)
    log(f"开始数据更新 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Kill lock holders
    killed = kill_dblock_holders()
    if killed:
        log(f"已杀掉锁持有进程: {killed}")
        time.sleep(2)
    
    # Step 2: Verify unlocked
    if not wait_until_unlocked(10):
        log("警告: DuckDB 仍然被锁定，继续尝试...")
    
    # Step 3: Run data update
    skill_root = '/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill'
    sys.path.insert(0, skill_root)
    
    from src.api.skill_api import get_instance
    import baostock as bs
    import pandas as pd
    
    PREFIX_MAP = {'600':'sh','601':'sh','603':'sh','000':'sz','001':'sz'}
    
    def fetch_one(code, start_fmt, end_fmt):
        bs_code = f"{PREFIX_MAP.get(code[:3],'sz')}.{code}"
        try:
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
            return df[cols]
        except:
            return None
    
    api = get_instance()
    store = api.store
    target_date = '2026-04-03'
    
    stocks = store.df("SELECT ts_code FROM stock_list ORDER BY ts_code")
    all_codes = stocks['ts_code'].tolist()
    
    existing = store.df(f"SELECT ts_code FROM stock_daily WHERE trade_date = '{target_date}'")
    existing_codes = set(existing['ts_code'].tolist())
    to_fetch = [c for c in all_codes if c not in existing_codes]
    
    log(f"总数: {len(all_codes)}, 已有: {len(existing_codes)}, 待取: {len(to_fetch)}")
    
    bs.login()
    success = 0
    fail = 0
    fail_codes = []
    
    for i, code in enumerate(to_fetch):
        # 每次循环前检查是否有新的锁持有者（ga_optimization 被重启）
        result = subprocess.run(['lsof', '-t', DB_PATH], capture_output=True, text=True)
        lock_pids = [p for p in result.stdout.strip().split('\n') if p.strip().isdigit()]
        if lock_pids:
            log(f"警告: 检测到新进程 {lock_pids} 获取了 DuckDB 锁，中断更新")
            break
        
        df = fetch_one(code, target_date, target_date)
        if df is not None and not df.empty:
            try:
                store.conn.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{target_date}'")
                store.conn.execute(f"INSERT INTO stock_daily BY NAME SELECT * FROM df")
                success += 1
            except Exception:
                fail += 1
                fail_codes.append(code)
        else:
            fail += 1
            fail_codes.append(code)
        
        if (i+1) % 100 == 0:
            log(f"  {i+1}/{len(to_fetch)} | 成功 {success} | 失败 {fail}")
        
        time.sleep(0.5)
    
    bs.logout()
    log(f"首轮: 成功 {success} | 失败 {fail}")
    
    # 重试
    if fail_codes:
        log(f"重试 {len(fail_codes)} 只...")
        time.sleep(5)
        retry_ok = 0
        for code in fail_codes[:]:
            df = fetch_one(code, target_date, target_date)
            if df is not None and not df.empty:
                try:
                    store.conn.execute(f"DELETE FROM stock_daily WHERE ts_code = '{code}' AND trade_date = '{target_date}'")
                    store.conn.execute(f"INSERT INTO stock_daily BY NAME SELECT * FROM df")
                    retry_ok += 1
                    fail_codes.remove(code)
                except:
                    pass
            time.sleep(2.0)
        log(f"重试成功: {retry_ok}")
    
    final = store.df(f"SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date = '{target_date}'")
    log(f"最终: {final['cnt'].iloc[0]} 条记录")
    
    # Step 4: 重启 ga_optimization
    log("重启 ga_optimization...")
    subprocess.Popen(
        ['python3', 'scheduler/tasks.py', '--task', 'ga_optimization'],
        cwd='/home/hyh/.openclaw/my-skills/astock_trend_backtestSkill',
        stdout=open('/dev/null','w'), stderr=open('/dev/null','w')
    )
    log("完成!")

if __name__ == '__main__':
    main()
