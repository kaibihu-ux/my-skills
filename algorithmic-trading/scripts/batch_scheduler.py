#!/usr/bin/env python3
"""
打板策略分批优化调度器
- 交易日（周一~周五，非节假日）：23:30 ~ 次日 06:30
- 非交易日（周末、节假日）：21:00 ~ 次日 06:30
- 复用量化回测系统的交易日判断逻辑

进程管理（方案三）：
1. 进程组管理：subprocess.Popen用os.setsid()让子进程成组，SIGTERM时os.killpg()一锅端
2. 每个子进程独立lock文件（含PPID），daemon启动前清理stale lock
3. daemon定期扫描并wait已死子进程
"""

import sys
import os
import time
import signal
import json
import subprocess
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# =============================================================================
# 交易日判断逻辑（从量化回测系统复制）
# =============================================================================
NON_TRADING_DAYS_2026 = {
    # 元旦（1月1日）
    '20260101',
    # 春节（2月17日-23日）
    '20260217', '20260218', '20260219', '20260220', '20260221', '20260222', '20260223',
    # 清明节（4月4日-6日）
    '20260404', '20260405', '20260406',
    # 劳动节（5月1日-5日）
    '20260501', '20260502', '20260503', '20260504', '20260505',
    # 端午节（6月19日-21日）
    '20260619', '20260620', '20260621',
    # 中秋节（9月25日-27日）
    '20260925', '20260926', '20260927',
    # 国庆节（10月1日-7日）
    '20261001', '20261002', '20261003', '20261004', '20261005', '20261006', '20261007',
}


def is_trading_day(d: Optional[date] = None) -> bool:
    """判断是否为A股交易日（排除周末和所有非交易日）"""
    if d is None:
        d = date.today()
    date_str = d.strftime('%Y%m%d')
    if date_str in NON_TRADING_DAYS_2026:
        return False
    # 排除周六、周日
    if d.weekday() >= 5:
        return False
    return True


def is_in_trading_window() -> bool:
    """判断当前是否在策略运行时间窗口内"""
    now = datetime.now()
    hour = now.hour

    if is_trading_day():
        # 交易日：23:30 ~ 次日 06:30
        if hour >= 23 or hour < 6:
            return True
    else:
        # 非交易日：21:00 ~ 次日 06:30
        if hour >= 21 or hour < 6:
            return True
    return False


def get_next_window_start() -> float:
    """计算距离下一个窗口开始的秒数"""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()

    if is_trading_day():
        if hour >= 23:
            tomorrow = date.today() + timedelta(days=1)
            while not is_trading_day(tomorrow):
                tomorrow += timedelta(days=1)
            next_trading = tomorrow
            next_dt = datetime.combine(next_trading, datetime.strptime("23:30", "%H:%M").time())
        elif hour < 6:
            next_dt = now.replace(hour=6, minute=30, second=0, microsecond=0)
        else:
            next_dt = now.replace(hour=23, minute=30, second=0, microsecond=0)
            if next_dt <= now:
                next_dt += timedelta(days=1)
    else:
        if hour >= 21:
            next_dt = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if next_dt <= now:
                next_dt += timedelta(days=1)
        else:
            next_dt = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if next_dt <= now:
                next_dt += timedelta(days=1)

    return (next_dt - now).total_seconds()


def get_window_end_seconds() -> float:
    """计算距离当前窗口结束的秒数（负数表示已超时）"""
    now = datetime.now()
    hour = now.hour

    if is_trading_day():
        if hour >= 23:
            end_dt = now.replace(hour=6, minute=30, second=0, microsecond=0)
            if end_dt <= now:
                end_dt += timedelta(days=1)
        else:
            return -1
    else:
        if hour >= 21:
            end_dt = now.replace(hour=6, minute=30, second=0, microsecond=0)
            if end_dt <= now:
                end_dt += timedelta(days=1)
        else:
            end_dt = now.replace(hour=6, minute=30, second=0, microsecond=0)

    return max((end_dt - now).total_seconds(), 0)


# =============================================================================
# 调度器主逻辑
# =============================================================================
SKILL_DIR = Path(__file__).parent.parent
CHECKPOINT_FILE = SKILL_DIR / 'checkpoint_state.json'
LOG_DIR = SKILL_DIR / 'output'
LOG_DIR.mkdir(exist_ok=True)

# 新增：locks目录用于进程锁管理
LOCKS_DIR = SKILL_DIR / 'locks'
LOCKS_DIR.mkdir(exist_ok=True)


def get_param_grid(quick: bool = False) -> Dict:
    if quick:
        return {
            'rsi_buy': [25, 30, 35],
            'rsi_sell': [65, 70, 75],
            'macd_fast': [10, 12],
            'macd_slow': [24, 26],
            'macd_signal': [8, 9],
            'bb_period': [18, 20, 22],
            'bb_std': [1.5, 2.0, 2.5],
            'vol_period': [15, 20],
            'vol_multiplier': [1.2, 1.5],
            'breakout_period': [18, 20],
            'stop_loss_pct': [2, 3, 5],
            'take_profit_pct': [8, 10],
            'min_conditions': [2, 3, 4],
        }
    else:
        return {
            'rsi_buy': [20, 25, 30, 35, 40],
            'rsi_sell': [60, 65, 70, 75, 80],
            'macd_fast': [8, 10, 12, 15],
            'macd_slow': [20, 24, 26, 30],
            'macd_signal': [7, 9, 11],
            'bb_period': [15, 18, 20, 22, 25],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'vol_period': [10, 15, 20, 25],
            'vol_multiplier': [1.0, 1.5, 2.0, 2.5],
            'breakout_period': [15, 18, 20, 25],
            'stop_loss_pct': [2, 3, 4, 5, 7],
            'take_profit_pct': [5, 8, 10, 15, 20],
            'min_conditions': [2, 3, 4, 5],
        }


def get_all_symbols() -> List[str]:
    """从数据库获取所有股票代码"""
    try:
        import duckdb
        db_path = SKILL_DIR / 'data' / 'astock_full.duckdb'
        conn = duckdb.connect(str(db_path), read_only=True)
        stocks = conn.execute('SELECT ts_code FROM stock_list').fetchall()
        conn.close()
        return [s[0] for s in stocks]
    except Exception as e:
        print(f"⚠️  获取股票列表失败: {e}，使用默认标的")
        return ['000001.SZ', '000002.SZ', '600000.SH']


def write_log(msg: str) -> None:
    """写日志"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    log_file = LOG_DIR / f'scheduler_{date.today().strftime("%Y%m%d")}.log'
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_batch(symbols: List[str], start_date: str, end_date: str, quick: bool = False) -> None:
    """运行一批优化"""
    import importlib.util
    sys.path.insert(0, str(SKILL_DIR / 'scripts'))

    # 动态导入 optimizer
    spec = importlib.util.spec_from_file_location('optimizer', SKILL_DIR / 'scripts' / 'optimizer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ParameterOptimizer = mod.ParameterOptimizer

    optimizer = ParameterOptimizer(symbols, start_date, end_date)
    param_grid = get_param_grid(quick)
    optimizer.batch_grid_search(
        param_grid,
        checkpoint_path=str(CHECKPOINT_FILE),
        batch_size=2,
        top_n=20,
        quick=quick,
    )


def _snapshot_checkpoint() -> Tuple[int, List]:
    """返回当前已完成的组合数和结果"""
    if not CHECKPOINT_FILE.exists():
        return 0, []
    with open(CHECKPOINT_FILE, 'r') as f:
        ck = json.load(f)
    return len(ck.get('completed_idx', [])), ck.get('results', [])


# =============================================================================
# 进程锁管理（方案三）
# =============================================================================

def _get_daemon_lock_file() -> Path:
    """获取daemon主进程锁文件路径"""
    return LOCKS_DIR / 'daemon.lock'


def _read_lock_info(lock_file: Path) -> Optional[Dict]:
    """读取锁文件信息，返回None表示无效"""
    try:
        if not lock_file.exists():
            return None
        with open(lock_file, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_lock_info(lock_file: Path, data: Dict) -> None:
    """写入锁文件信息"""
    with open(lock_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False)


def _is_lock_stale(lock_file: Path) -> bool:
    """检查锁文件是否过期（进程已死）"""
    lock_info = _read_lock_info(lock_file)
    if lock_info is None:
        return True
    
    pid = lock_info.get('pid')
    if pid is None:
        return True
    
    try:
        # 检查进程是否还活着
        os.kill(pid, 0)
        return False  # 进程还活着，锁有效
    except (ProcessLookupError, PermissionError, OSError):
        return True  # 进程已死，锁过期


def _acquire_daemon_lock() -> bool:
    """尝试获取daemon锁，返回是否成功"""
    lock_file = _get_daemon_lock_file()
    
    if not _is_lock_stale(lock_file):
        return False
    
    try:
        data = {
            'pid': os.getpid(),
            'ppid': os.getppid(),
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
        }
        _write_lock_info(lock_file, data)
        return True
    except OSError:
        return False


def _release_daemon_lock() -> None:
    """释放daemon锁"""
    lock_file = _get_daemon_lock_file()
    try:
        lock_file.unlink()
    except OSError:
        pass


def _cleanup_stale_locks() -> int:
    """清理所有stale lock文件，返回清理的数量"""
    cleaned = 0
    if not LOCKS_DIR.exists():
        return 0
    
    for lock_file in LOCKS_DIR.glob('*.lock'):
        if lock_file.name == 'daemon.lock':
            # daemon.lock由主进程自己管理
            if _is_lock_stale(lock_file):
                try:
                    lock_file.unlink()
                    cleaned += 1
                except OSError:
                    pass
        else:
            # 子进程锁文件
            lock_info = _read_lock_info(lock_file)
            if lock_info is not None:
                pid = lock_info.get('pid')
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                        # 进程还活着，检查是否是当前进程的子进程
                        parent_pid = lock_info.get('parent_pid')
                        if parent_pid != os.getpid():
                            # 不是当前进程的子进程，可能是孤立的
                            try:
                                os.kill(pid, 0)
                            except ProcessLookupError:
                                # 进程已死，清理
                                lock_file.unlink()
                                cleaned += 1
                    except (ProcessLookupError, PermissionError, OSError):
                        # 进程已死，清理
                        try:
                            lock_file.unlink()
                            cleaned += 1
                        except OSError:
                            pass
    return cleaned


def _get_child_lock_file(child_pid: int) -> Path:
    """获取指定子进程的锁文件路径"""
    return LOCKS_DIR / f'child_{child_pid}.lock'


def _acquire_child_lock(child_pid: int) -> Path:
    """为子进程创建锁文件"""
    lock_file = _get_child_lock_file(child_pid)
    try:
        data = {
            'pid': child_pid,
            'parent_pid': os.getpid(),
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        _write_lock_info(lock_file, data)
    except OSError:
        pass
    return lock_file


def _release_child_lock(child_pid: int) -> None:
    """释放子进程锁"""
    lock_file = _get_child_lock_file(child_pid)
    try:
        lock_file.unlink()
    except OSError:
        pass


def _scan_and_wait_dead_children() -> int:
    """
    扫描locks目录，wait已死但未被wait的子进程
    返回wait的子进程数量
    """
    waited = 0
    if not LOCKS_DIR.exists():
        return 0
    
    for lock_file in LOCKS_DIR.glob('child_*.lock'):
        lock_info = _read_lock_info(lock_file)
        if lock_info is None:
            continue
        
        pid = lock_info.get('pid')
        parent_pid = lock_info.get('parent_pid')
        
        if parent_pid != os.getpid():
            # 不是当前进程的子进程锁，跳过
            continue
        
        if pid is None:
            continue
        
        try:
            # 检查进程是否已死
            pid_int = int(pid) if isinstance(pid, str) else pid
            # 使用os.WNOHANG方式检测
            result = os.waitpid(pid_int, os.WNOHANG)
            if result[0] != 0:
                # 子进程已死，清理锁文件
                try:
                    lock_file.unlink()
                    waited += 1
                except OSError:
                    pass
        except (ProcessLookupError, ChildProcessError, ValueError, OSError):
            # 进程已死或不存在，清理锁文件
            try:
                lock_file.unlink()
                waited += 1
            except OSError:
                pass
    
    return waited


def _terminate_process_group(proc: subprocess.Popen, timeout: int = 5) -> None:
    """
    终止整个进程组（SIGTERM后SIGKILL）
    """
    if proc is None or proc.poll() is not None:
        return
    
    try:
        # 先尝试SIGTERM
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # 进程已死或没有权限
        return
    
    # 等待进程结束
    start = time.time()
    while proc.poll() is None and (time.time() - start) < timeout:
        time.sleep(0.5)
    
    # 如果还没结束，SIGKILL
    if proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            time.sleep(0.5)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    
    # 确保清理
    if proc.poll() is None:
        proc.kill()


def run_once(
    symbols: List[str],
    start_date: str,
    end_date: str,
    quick: bool,
    check_interval: int = 300,
    max_run_seconds: int = 50 * 60,
    child_lock_file: Optional[Path] = None,
) -> Dict:
    """
    执行一次优化任务，运行到窗口结束或任务完成。
    返回 dict: {completed_before, completed_after, batch_time, results, batch_num}
    """
    remaining = get_window_end_seconds()
    write_log(f"⏰ 在运行窗口内，剩余 {remaining/3600:.1f} 小时")

    # 记录开始前状态
    completed_before, _ = _snapshot_checkpoint()
    write_log(f"   断点记录: 已完成 {completed_before} 组合")

    write_log(f"🚀 启动优化进程（快速模式，区间 {start_date}~{end_date}）...")

    log_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'batch_{log_ts}.log'
    symbols_file = LOG_DIR / f'symbols_{os.getpid()}.json'
    with open(symbols_file, 'w', encoding='utf-8') as f:
        json.dump(symbols, f)

    # 使用 os.setsid() 创建新进程组
    proc = subprocess.Popen(
        [
            sys.executable, '-c',
            f"""
import sys, json
sys.path.insert(0, '{SKILL_DIR}/scripts')
with open('{symbols_file}', 'r') as sf:
    symbols = json.load(sf)
import importlib.util
spec = importlib.util.spec_from_file_location('optimizer', '{SKILL_DIR}/scripts/optimizer.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ParameterOptimizer = mod.ParameterOptimizer
optimizer = ParameterOptimizer(symbols, '{start_date}', '{end_date}')
param_grid = {get_param_grid(quick)}
optimizer.batch_grid_search(param_grid, '{CHECKPOINT_FILE}', batch_size=2, top_n=20, quick={quick})
"""
        ],
        stdout=open(log_file, 'w', encoding='utf-8'),
        stderr=subprocess.STDOUT,
        cwd=str(SKILL_DIR),
        preexec_fn=os.setsid,  # 创建新进程组
    )
    
    # 为子进程创建锁文件
    _acquire_child_lock(proc.pid)
    
    write_log(f"   子进程 PID: {proc.pid}，日志: {log_file.name}")
    write_log(f"   子进程进程组PGID: {os.getpgid(proc.pid)}")

    start_time = time.time()
    while True:
        time.sleep(check_interval)
        retcode = proc.poll()
        elapsed = time.time() - start_time
        
        # 定期扫描并wait已死子进程
        waited = _scan_and_wait_dead_children()
        if waited > 0:
            write_log(f"   清理了 {waited} 个已死子进程锁文件")
        
        if retcode is not None:
            write_log(f"⚠️  子进程已结束 (exit={retcode})，日志: {log_file.name}")
            _release_child_lock(proc.pid)
            break
        
            # 超时终止逻辑已移除，改为 cron 模式自然退出
        # if elapsed >= max_run_seconds:
        #     write_log(f"⏰ 超时 ({elapsed/3600:.1f}h)，发送 SIGTERM 优雅停止...")
        #     _terminate_process_group(proc)
        #     _release_child_lock(proc.pid)
        #     break
        
        if not is_in_trading_window():
            write_log("⏸️  运行窗口结束，发送 SIGTERM 优雅停止...")
            _terminate_process_group(proc)
            _release_child_lock(proc.pid)
            break
        
        remaining = get_window_end_seconds()
        if int(remaining) % 600 < check_interval:
            write_log(f"   仍在运行窗口内，已运行 {elapsed/3600:.1f}h，剩余 {remaining/3600:.1f} 小时")

    batch_time = time.time() - start_time
    completed_after, results = _snapshot_checkpoint()
    batch_num = completed_after // 50 if completed_after > 0 else 1

    return {
        'completed_before': completed_before,
        'completed_after': completed_after,
        'batch_time': batch_time,
        'results': results,
        'batch_num': batch_num,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='打板策略分批优化调度器')
    parser.add_argument('--daemon', action='store_true',
                        help='常驻模式：不在窗口内时睡眠等待，窗口内运行（默认模式）')
    parser.add_argument('--cron', action='store_true',
                        help='Cron模式：不在窗口内时立即退出，不等待')
    parser.add_argument('--hours', type=float, default=None,
                        help='运行小时数（用于cron模式，默认1小时）')
    args = parser.parse_args()

    # 清理stale locks
    cleaned = _cleanup_stale_locks()
    if cleaned > 0:
        print(f"🧹 清理了 {cleaned} 个过期锁文件")

    # 进程锁：防止多实例并行
    if not _acquire_daemon_lock():
        sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 另一实例正在运行，退出\n")
        sys.exit(0)

    write_log("=" * 50)
    write_log("🟢 打板策略调度器启动")
    write_log(f"   模式: {'Cron分批' if (args.cron or args.hours is not None) else '常驻'}")
    write_log(f"   交易日: {'是' if is_trading_day() else '否'}")
    write_log(f"   当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(f"   Locks目录: {LOCKS_DIR}")
    write_log("=" * 50)

    # 加载报告模块
    try:
        sys.path.insert(0, str(SKILL_DIR / 'scripts'))
        import _reporter
        report_init_fn = _reporter.report_init
        report_batch_fn = _reporter.report_batch_completion
        report_milestone_fn = _reporter.report_milestone
        write_log("📣 报告模块: 已加载")
    except Exception as e:
        write_log(f"⚠️  报告模块加载失败: {e}，报告功能禁用")
        report_init_fn = report_batch_fn = report_milestone_fn = None

    symbols = get_all_symbols()
    write_log(f"📊 股票总数: {len(symbols)}")

    start_date = '2024-01-01'
    end_date = '2026-03-31'
    quick = True

    if args.cron or args.hours is not None:
        # Cron模式：运行指定小时数后自然退出，不等待窗口
        run_hours = args.hours if args.hours else 1
        max_run_seconds = int(run_hours * 3600)

        if not is_in_trading_window():
            next_start = get_next_window_start()
            write_log(f"😴 不在运行窗口内，Cron模式直接退出（下次窗口: {next_start/3600:.1f}h后）")
            _release_daemon_lock()
            sys.exit(0)

        # 发送每日汇总报告
        if report_init_fn:
            try:
                report_init_fn()
            except Exception as e:
                write_log(f"⚠️  每日汇总推送失败: {e}")

        write_log(f"⏱️  Cron模式：运行 {run_hours}h（{max_run_seconds}s）后自然退出")
        stats = run_once(symbols, start_date, end_date, quick, max_run_seconds=max_run_seconds)
        completed_before = stats['completed_before']
        completed_after = stats['completed_after']
        batch_time = stats['batch_time']
        results = stats['results']
        batch_num = stats['batch_num']

        # 推送批次完成报告
        if report_batch_fn and completed_after > completed_before:
            try:
                # 推算总组合数
                param_keys = []
                if CHECKPOINT_FILE.exists():
                    with open(CHECKPOINT_FILE, 'r') as f:
                        ck = json.load(f)
                    param_keys = ck.get('param_keys', [])
                total = 1
                for k in param_keys:
                    pg = get_param_grid(quick)
                    if k in pg:
                        total *= len(pg[k])

                top3 = _reporter._get_top_params(results, top_n=3) if results else []
                report_batch_fn(
                    batch_size=completed_after - completed_before,
                    batch_num=batch_num,
                    completed_total=completed_after,
                    total=total,
                    batch_time_seconds=batch_time,
                    top_results=top3,
                )

                # 推送里程碑报告（每100组）
                if report_milestone_fn:
                    milestones = [100, 200, 300, 500, 1000, 2000, 5000]
                    for m in milestones:
                        if completed_before < m <= completed_after:
                            report_milestone_fn(
                                completed=completed_after,
                                total=total,
                                top_results=top3,
                            )
            except Exception as e:
                write_log(f"⚠️  报告推送失败: {e}")

    else:
        # 常驻模式：睡眠等待窗口
        while not is_in_trading_window():
            sleep_secs = get_next_window_start()
            write_log(f"😴 不在运行窗口内，等待 {sleep_secs/3600:.1f}h...")
            time.sleep(min(sleep_secs, 3600))
            # 定期清理stale locks
            cleaned = _cleanup_stale_locks()
            if cleaned > 0:
                write_log(f"🧹 清理了 {cleaned} 个过期锁文件")
            # 定期扫描并wait已死子进程
            waited = _scan_and_wait_dead_children()
            if waited > 0:
                write_log(f"   等待了 {waited} 个已死子进程")
        run_once(symbols, start_date, end_date, quick)

    _release_daemon_lock()


if __name__ == '__main__':
    main()
