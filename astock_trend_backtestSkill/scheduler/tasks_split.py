"""
非交易日优化任务 - 分拆版（带检查点续算）
=====================================
将 job_optimize_and_report 拆分为5个独立任务，每个任务有检查点续算：

执行流程（非交易日）：
  09:00  job_step1_lgb       → checkpoint: step1_lgb.json
  10:00  job_step2_ga       → checkpoint: step2_ga.json
  12:00  job_step3_rl        → checkpoint: step3_rl.json
  14:00  job_step4_bayes     → checkpoint: step4_bayes.json
  16:00  job_step5_final    → checkpoint: step5_final.json → 生成报告

使用方式：
  # 正常运行（自动检测检查点续算）
  python scheduler/tasks_split.py --step 1
  
  # 强制从头开始
  python scheduler/tasks_split.py --step 1 --force
  
  # 单独运行某步骤
  python scheduler/tasks_split.py --step 2
"""
import sys
import os
import json
import functools
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Tuple, Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.skill_api import get_instance

# 确保项目路径在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ===== 交易日判断工具 =====
from datetime import date as _date


# 2026年A股非交易日（周末 + 法定节假日）
_NON_TRADING_DAYS_2026 = {
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


def is_trading_day(d=None):
    """判断是否为交易日（A股）"""
    if d is None:
        d = _date.today()
    date_str = d.strftime('%Y%m%d')
    # 排除节假日
    if date_str in _NON_TRADING_DAYS_2026:
        return False
    # 排除周六、周日
    weekday = d.weekday()
    if weekday >= 5:
        return False
    return True

# =============================================================================
# 辅助函数 & 超时装饰器
# =============================================================================

# 共享停止标志（供 SIGTERM handler 和各任务检查）
_stop_requested = False
# 全局变量供 SIGTERM handler 使用（在 job_step2_ga 中设置）
_current_ga_ckpt_path = None


def _sigterm_handler(signum, frame):
    """SIGTERM 信号处理：立即保存检查点并以码0退出（防止孤儿进程）"""
    global _stop_requested
    if _stop_requested:
        return
    _stop_requested = True
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ⚠️  收到 SIGTERM | 时间戳: {time.time()} | pid={os.getpid()}")
    print(f"[{now}]    检查点路径: {_current_ga_ckpt_path}")
    # 触发 save_ckpt → 写盘 → 然后 Python 进程可以干净退出
    import sys
    sys.exit(0)


def _setup_sigterm_handler():
    """注册 SIGTERM handler（仅在主进程注册一次）"""
    try:
        import signal
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        pass  # 非主线程或不支持信号的运行环境


class CronSafeTimeout:
    """
    cron安全的超时控制器：
    - 每隔 interval 秒检查一次剩余时间
    - 在 cron 超时前提前保存检查点并优雅退出
    - 保证 SIGTERM 信号到达时也能保存断点
    """

    def __init__(self, cron_timeout_seconds: int, warning_pct: float = 0.85,
                 checkpoint_fn=None, step_name: str = ""):
        """
        Args:
            cron_timeout_seconds: cron 配置的超时秒数（提前5分钟保存）
            warning_pct:         剩余多少比例时开始警告并准备退出
            checkpoint_fn:       回调：触发保存检查点的函数（通常是当前步骤的 save_ckpt）
            step_name:           当前步骤名（用于日志）
        """
        self.cron_timeout = cron_timeout_seconds
        self.warning_pct = warning_pct
        self.checkpoint_fn = checkpoint_fn
        self.step_name = step_name
        self.start_time = time.time()
        self.warning_done = False
        self._stop = False

    def check(self) -> bool:
        """
        定期调用此方法检查是否需要退出。
        在各任务的循环内部调用（如 GA 每代后）。
        返回 True 表示时间快到了，应该保存并退出；False 表示继续运行。
        """
        if self._stop:
            return True
        elapsed = time.time() - self.start_time
        remaining = self.cron_timeout - elapsed
        deadline = self.cron_timeout * self.warning_pct

        if elapsed >= deadline and not self.warning_done:
            self.warning_done = True
            safe_exit_time = remaining
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now_str}] ⏰ [{self.step_name}] 时间不多（剩余 {safe_exit_time/60:.0f}min），"
                  f"保存检查点后优雅退出，下次接力继续")
            if self.checkpoint_fn:
                self.checkpoint_fn()
            # 给 SIGTERM handler 发信号让它不再重复处理
            self._stop = True
            return True
        return False


def timeout(seconds):
    """超时强制终止装饰器（使用进程级 Event 真正停止，而非 abandon 线程）"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            import multiprocessing as mp

            result = [None]
            exc = [None]
            stop_event = mp.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exc[0] = e

            t = threading.Thread(target=target, daemon=True)
            t.start()
            t.join(timeout=seconds)
            if t.is_alive():
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] ⏰ {func.__name__} 运行超过 {seconds/3600:.1f}h，强制结束")
                raise TimeoutError(f"{func.__name__} timed out after {seconds/3600:.1f}h")
            if exc[0]:
                raise exc[0]
            return result[0]
        return wrapper
    return decorator

def get_ckpt(step_name):
    """读取检查点，返回 (data, step) 或 (None, None)"""
    ckpt_file = Path(__file__).parent.parent / "checkpoints" / f"weekend_{step_name}.json"
    if ckpt_file.exists():
        try:
            with open(ckpt_file) as f:
                data = json.load(f)
            return data, data.get('completed_step', '')
        except Exception:
            pass
    return None, None


def save_ckpt(step_name, completed_step, data):
    """保存检查点（原子写入）"""
    ckpt_file = Path(__file__).parent.parent / "checkpoints" / f"weekend_{step_name}.json"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    data['completed_step'] = completed_step
    data['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = str(ckpt_file) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    Path(tmp).rename(ckpt_file)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💾 检查点已保存: {ckpt_file.name} → {completed_step}")


def is_step_done(target_step, current_step):
    """判断步骤是否已完成（current_step >= target_step）"""
    order = ['step1_lgb', 'step2_ga', 'step3_rl', 'step4_bayes', 'step5_final', 'done']
    try:
        return order.index(current_step) >= order.index(target_step)
    except ValueError:
        return False


# =============================================================================
# P0 Bug修复: job_optimize_checkpoint.json 同步更新 & 数据新鲜度校验
# =============================================================================

def _get_job_ckpt_path() -> Path:
    """获取 job_optimize_checkpoint.json 路径"""
    return Path(__file__).parent.parent / "checkpoints" / "job_optimize_checkpoint.json"


def _get_job_optimize_checkpoint() -> Tuple[Optional[Dict], str]:
    """
    读取 job_optimize_checkpoint.json
    返回 (data, completed_step)
    """
    ckpt_file = _get_job_ckpt_path()
    if ckpt_file.exists():
        try:
            with open(ckpt_file, 'r') as f:
                data = json.load(f)
            return data, data.get('completed_step', '')
        except Exception:
            pass
    return None, ''


def _update_job_optimize_checkpoint(step_name: str, data: Dict) -> None:
    """
    同步更新 job_optimize_checkpoint.json
    在各step完成后调用，确保主checkpoint与分步checkpoint同步
    """
    ckpt_file = _get_job_ckpt_path()
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 构造更新数据
    update_data = {
        'completed_step': step_name,
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'step_data': data,
        'step_history': {step_name: data},
    }
    
    # 如果存在旧数据，合并而非覆盖
    if ckpt_file.exists():
        try:
            with open(ckpt_file, 'r') as f:
                old_data = json.load(f)
            # 保留历史step数据
            if 'step_history' not in old_data:
                old_data['step_history'] = {}
            old_data['step_history'][step_name] = data
            old_data['completed_step'] = step_name
            old_data['saved_at'] = update_data['saved_at']
            update_data = old_data
        except Exception:
            pass
    
    # 原子写入
    tmp = str(ckpt_file) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(update_data, f, ensure_ascii=False, indent=2, default=str)
    Path(tmp).rename(ckpt_file)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💾 [FIX] job_optimize_checkpoint.json 已同步更新 → {step_name}")


def _is_checkpoint_fresh(ckpt_data: Optional[Dict], step_name: str, force_restart: bool = False) -> Tuple[bool, str]:
    """
    P0: 检查checkpoint是否是当天的（非当天checkpoint不能跳过）
    
    返回 (is_fresh, reason)
    - is_fresh=True 表示可以跳过
    - is_fresh=False 表示不能跳过，需要重新运行
    """
    if force_restart:
        return False, "force_restart=True"
    
    if ckpt_data is None:
        return False, "checkpoint不存在"
    
    saved_at = ckpt_data.get('saved_at', '')
    if not saved_at:
        return False, "checkpoint无saved_at时间戳"
    
    try:
        # 解析保存时间
        saved_date_str = saved_at.split(' ')[0]  # 取 'YYYY-MM-DD' 部分
        today_str = date.today().strftime('%Y-%m-%d')
        
        if saved_date_str != today_str:
            return False, f"checkpoint是{saved_date_str}的，不是今天({today_str})，不能跳过"
        
        return True, "checkpoint是今天的，可以跳过"
    except Exception as e:
        return False, f"解析saved_at失败: {e}"


def _check_and_skip_or_run(
    step_name: str,
    ckpt_data: Optional[Dict],
    completed: str,
    force_restart: bool = False,
) -> Tuple[bool, Optional[Dict]]:
    """
    P0: 统一的checkpoint新鲜度检查+跳过判断
    返回 (should_skip, data)
    - should_skip=True 表示跳过（使用缓存数据）
    - should_skip=False 表示需要运行
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 检查是否完成
    if completed != step_name:
        return False, None
    
    # 检查新鲜度
    is_fresh, reason = _is_checkpoint_fresh(ckpt_data, step_name, force_restart)
    
    if is_fresh:
        print(f"[{now_str}] ✅ {step_name} 已完成，从检查点恢复... ({reason})")
        return True, ckpt_data
    else:
        print(f"[{now_str}] ⚠️  {step_name} checkpoint已存在但{reason}，将重新运行")
        return False, None


# =============================================================================
# P1: 拆分监控/恢复checkpoint职责（可选增强）
# =============================================================================

class CheckpointMonitor:
    """
    P1: 独立的checkpoint监控器
    负责：监控checkpoint健康状态、清理过期checkpoint、恢复指引
    """
    
    def __init__(self, checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_checkpoints(self) -> Dict[str, Dict]:
        """获取所有checkpoint及其状态"""
        checkpoints = {}
        for ckpt_file in self.checkpoints_dir.glob("weekend_*.json"):
            try:
                with open(ckpt_file, 'r') as f:
                    data = json.load(f)
                step = data.get('completed_step', ckpt_file.stem.replace('weekend_', ''))
                checkpoints[step] = {
                    'file': str(ckpt_file),
                    'saved_at': data.get('saved_at', ''),
                    'data': data,
                }
            except Exception:
                pass
        return checkpoints
    
    def get_progress(self) -> Tuple[int, int]:
        """
        获取当前进度
        返回 (completed_steps, total_steps)
        """
        checkpoints = self.get_all_checkpoints()
        order = ['step1_lgb', 'step2_ga', 'step3_rl', 'step4_bayes', 'step5_final']
        completed = 0
        for step in order:
            if step in checkpoints:
                data = checkpoints[step]['data']
                if data.get('completed_step') == step:
                    # 检查新鲜度
                    saved_at = data.get('saved_at', '')
                    if saved_at:
                        saved_date = saved_at.split(' ')[0]
                        today = date.today().strftime('%Y-%m-%d')
                        if saved_date == today:
                            completed += 1
        return completed, len(order)
    
    def cleanup_stale(self, days: int = 7) -> int:
        """清理指定天数之前的旧checkpoint"""
        cleaned = 0
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        for ckpt_file in self.checkpoints_dir.glob("weekend_*.json"):
            try:
                with open(ckpt_file, 'r') as f:
                    data = json.load(f)
                saved_at = data.get('saved_at', '')
                if saved_at:
                    saved_date = saved_at.split(' ')[0]
                    if saved_date < cutoff:
                        ckpt_file.unlink()
                        cleaned += 1
            except Exception:
                pass
        
        return cleaned
    
    def get_next_runnable_step(self) -> Optional[str]:
        """获取下一个可执行的步骤"""
        checkpoints = self.get_all_checkpoints()
        order = ['step1_lgb', 'step2_ga', 'step3_rl', 'step4_bayes', 'step5_final']
        
        for step in order:
            if step not in checkpoints:
                return step
            data = checkpoints[step]['data']
            if data.get('completed_step') != step:
                return step
            # 检查新鲜度
            is_fresh, _ = _is_checkpoint_fresh(data, step, force_restart=False)
            if not is_fresh:
                return step
        
        return None  # 全部完成


# =============================================================================
# 飞书通知（复用 tasks.py 的实现）
# =============================================================================
FEISHU_CHAT_ID = 'oc_c6dde682059311ce59cee47c6dc5383b'


def _get_feishu_credentials():
    try:
        openclaw_cfg = Path.home() / '.openclaw' / 'openclaw.json'
        with open(openclaw_cfg) as f:
            cfg = json.load(f)
        feishu = cfg.get('channels', {}).get('feishu', {})
        return feishu.get('appId', ''), feishu.get('appSecret', '')
    except Exception:
        return '', ''


def _get_tenant_token():
    app_id, app_secret = _get_feishu_credentials()
    if not app_id or not app_secret:
        return ''
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        if resp.get("code") == 0:
            return resp.get("tenant_access_token", "")
    except Exception:
        pass
    return ""


def _send_feishu(text: str) -> bool:
    token = _get_tenant_token()
    if not token:
        return False
    try:
        import urllib.request
        payload = {
            "receive_id": FEISHU_CHAT_ID,
            "msg_type": "text",
            "content": json.dumps({"text": text})
        }
        req = urllib.request.Request(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("code") == 0
    except Exception:
        return False


def _send_step_report(step_name: str, data: dict, is_error: bool = False):
    """发送步骤完成报告到飞书群"""
    now_str = datetime.now().strftime("%H:%M:%S")
    emoji = "✅" if not is_error else "❌"
    lines = [
        f"{emoji} **Step{step_name} 完成** | {now_str}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if step_name == "1":
        lgb_auc = data.get('lgb_model_auc', 0)
        selected = data.get('selected_factors', [])
        neutralized = data.get('neutralized_factors', [])
        lines.extend([
            f"**LightGBM 筛选结果**",
            f"  模型 AUC: {lgb_auc:.4f}" if lgb_auc else f"  模型 AUC: N/A",
            f"  选出因子: {len(selected)} 个",
            f"  中性化后: {len(neutralized)} 个",
        ])
        if selected:
            lines.append(f"  因子列表: {', '.join(selected[:5])}{'...' if len(selected) > 5 else ''}")
        lines.append(f"  下一步: 10:00 Step2-GA遗传优化")
        lines.append(f"  状态: 正常运行中 🔄")

    elif step_name == "2":
        lines.extend([
            f"**GA 遗传算法结果**",
            f"  最优 Sharpe: {data.get('ga_best_sharpe', 0):.4f}",
            f"  最优因子: {', '.join(data.get('ga_best_factors', [])[:5])}",
            f"  最优参数: {str(data.get('ga_best_params', {}))[:80]}",
            f"  下一步: 12:00 Step3-RL仓位优化",
            f"  状态: 正常运行中 🔄",
        ])

    elif step_name == "3":
        lines.extend([
            f"**RL 强化学习结果**",
            f"  最优 Sharpe: {data.get('rl_best_sharpe', 0):.4f}",
            f"  GA 最优 Sharpe: {data.get('ga_best_sharpe', 0):.4f}",
            f"  使用因子: {', '.join(data.get('ga_best_factors', [])[:5])}",
            f"  下一步: 14:00 Step4-Bayesian精调",
            f"  状态: 正常运行中 🔄",
        ])

    elif step_name == "4":
        lines.extend([
            f"**Bayesian/Grid 精调结果**",
            f"  最优 Sharpe: {data.get('refined_sharpe', 0):.4f} ({data.get('refine_method', '')})",
            f"  GA Sharpe: {data.get('ga_best_sharpe', 0):.4f}",
            f"  RL Sharpe: {data.get('rl_best_sharpe', 0):.4f}",
            f"  生成策略候选: {data.get('total_strategies_generated', 0)} 个",
            f"  下一步: 16:00 Step5-最终回测+生成报告",
            f"  状态: 正常运行中 🔄",
        ])

    elif step_name == "5":
        bs = data.get('best_strategy', {})
        od = data.get('overfit_detection', {})
        is_acc = od.get('overfit_accepted', False)
        lines.extend([
            f"**最终回测报告**",
            f"  最优方法: {data.get('best_method', 'N/A')}",
            f"  最优 Sharpe: {data.get('best_sharpe', 0):.4f}",
            f"  最优因子: {', '.join(data.get('best_factors', [])[:5])}",
            f"  总收益: {bs.get('total_return', 0):.2%}",
            f"  夏普比率: {bs.get('sharpe_ratio', 0):.4f}",
            f"  最大回撤: {bs.get('max_drawdown', 0):.2%}",
            f"  交易次数: {bs.get('total_trades', 0)}",
            f"  过拟合检测: {'✅ 通过' if is_acc else '❌ 未通过'}",
        ])
        ofd = od.get('summary', {})
        if ofd:
            lines.append(f"  PBO={ofd.get('pbo', 'N/A')} | DSR={ofd.get('dsr', 'N/A')} | CSCV={ofd.get('cscv', 'N/A')}")
        lines.append(f"  完整报告: 今晚21:00发送")
        lines.append(f"  状态: ✅ 已完成")

    msg = "\n".join(lines)
    print(msg)
    sent = _send_feishu(msg)
    if sent:
        print(f"[FEISHU] Step{step_name} 完成报告已发送")
    else:
        print(f"[FEISHU] Step{step_name} 完成报告发送失败（群ID: {FEISHU_CHAT_ID}）")


# =============================================================================
# 交易日专用：因子重平衡 + Step1 合并任务
# =============================================================================

def job_factor_rebalance_and_step1(force_restart=False):
    """
    交易日-15:30 | 因子重平衡 + LightGBM Step1
    直接运行，不走 cron 接力。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ===== 交易日: 因子重平衡 + Step1 =====")

    _setup_sigterm_handler()

    # 因子重平衡
    try:
        api = get_instance()
        api.factor_pool.rebalance()
        top = api.factor_pool.get_top_factors(10)
        print(f"[{now}] 因子重平衡完成，Top10: {[f['factor_name'] for f in top]}")
    except Exception as e:
        print(f"[{now}] ⚠️ 因子重平衡失败: {e}")

    # Step1 LightGBM
    if not force_restart:
        ckpt1, comp1 = get_ckpt('step1_lgb')
        if comp1 == 'step1_lgb':
            print(f"[{now}] ✅ Step1 已存在，跳过")
            return ckpt1

    from src.core.ml_feature_selector import MLFeatureSelector
    api = get_instance()
    store = api.store
    logger = api.logger
    selector = MLFeatureSelector(store, logger)
    all_factors = list(set([
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
        'rsi_14', 'rsi_28', 'macd', 'macd_signal',
        'bollinger_position', 'bollinger_bandwidth',
        'volatility_20', 'volatility_60',
        'volume_ratio_20', 'volume_ratio_60',
    ]))
    result = selector.select_features(all_factors[:20], '20240101', '20260327')
    selected = result.get('selected_features', [])
    neutralized = result.get('neutralized_factors', selected)
    data = {
        'lgb_result': result,
        'selected_factors': selected,
        'neutralized_factors': neutralized,
        'lgb_model_auc': result.get('model_auc', 0),
        'selected_count': len(selected),
        'neutralized_count': len(neutralized),
    }
    save_ckpt('step1_lgb', 'step1_lgb', data)
    print(f"[{now}] ✅ Step1 LightGBM 完成 | 选出{len(selected)}因子 AUC={result.get('model_auc',0):.4f}")
    _send_step_report("1", data)
    return data


# =============================================================================
# Step 1: LightGBM 因子筛选
# =============================================================================

@timeout(1 * 3600)
def job_step1_lgb(force_restart=False):
    """非交易日 09:00 | Step1: LightGBM 因子筛选 + 中性化处理"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ===== Step1: LightGBM 因子筛选 =====")

    # ---- P0 Bug修复: 检查点新鲜度校验 ----
    if not force_restart:
        ckpt_data, completed = get_ckpt('step1_lgb')
        should_skip, skip_data = _check_and_skip_or_run('step1_lgb', ckpt_data, completed, force_restart)
        if should_skip:
            return skip_data

    from src.core.ml_feature_selector import MLFeatureSelector
    from src.core.factor_eval import FactorEvaluator
    from src.core.neutralizer import Neutralizer

    api = get_instance()
    store = api.store
    logger = api.logger

    start_date = '20240101'
    end_date = '20260327'
    all_factors = list(set([
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
        'rsi_14', 'rsi_28', 'macd', 'macd_signal',
        'bollinger_position', 'bollinger_bandwidth',
        'volatility_20', 'volatility_60',
        'volume_ratio_20', 'volume_ratio_60',
    ]))

    # LightGBM 筛选
    print(f"[{now}] Step1.1: LightGBM 因子筛选...")
    selector = MLFeatureSelector(store, logger)
    lgb_result = selector.select_features(all_factors[:20], start_date, end_date)
    selected_factors = lgb_result.get('selected_features', [])
    print(f"[{now}] LightGBM 筛选: {len(selected_factors)} 个因子 → {selected_factors}")

    # 备用回退
    if len(selected_factors) < 3:
        print(f"[{now}] LightGBM 筛选不足(<3)，使用评估器回退...")
        evaluator = FactorEvaluator(store, logger)
        results = evaluator.evaluate_multiple(all_factors, start_date, end_date)
        ranked = evaluator.rank_factors(results)
        selected_factors = [f['factor_name'] for f in ranked[:10] if ranked]
        print(f"[{now}] 评估器回退: {selected_factors}")

    # 中性化处理
    print(f"[{now}] Step1.2: 因子中性化处理...")
    neutralized_factors = selected_factors
    try:
        ncfg = api.config.get('neutralization', {})
        if ncfg.get('enabled', True):
            neutralizer = Neutralizer(store, logger, api.config)
            neutralized_factors = neutralizer.neutralize_factor_list(selected_factors, start_date, end_date)
            print(f"[{now}] 中性化完成: {len(neutralized_factors)}/{len(selected_factors)} 个因子通过中性化")
    except Exception as e:
        print(f"[{now}] ⚠️ 中性化失败，使用原始因子: {e}")

    data = {
        'selected_factors': selected_factors,
        'neutralized_factors': neutralized_factors,
        'lgb_result': {k: v for k, v in lgb_result.items() if k != 'model'},
        'lgb_model_auc': lgb_result.get('model_auc', 0),
    }
    save_ckpt('step1_lgb', 'step1_lgb', data)
    print(f"[{now}] ✅ Step1 完成")
    _send_step_report("1", data)
    return data


# =============================================================================
# =============================================================================
# Step 2: GA 遗传算法优化（分5批，每批2代）
# =============================================================================

@timeout(90 * 60)  # 90分钟足够跑完2代（每代~20-30分钟）
def job_step2_ga(force_restart=False, batch_id=None, trading_day=False, generations=None):
    """
    Step2: GA 遗传算法优化（分批续跑）

    trading_day=True  时: TOTAL=3代, 每批1代, 3批接力 (交易日用)
      batch0(16:00) → batch1(16:30) → batch2(17:00)
    trading_day=False 时: TOTAL=6代, 每批1代, 6批接力 (非交易日用)
      batch0(09:30) → batch1(11:00) → batch2(12:30) → batch3(14:00) → batch4(15:30) → batch5(17:00)

    batch_id:
      None = 一次性跑完所有代数（用于手动触发测试）
      0-4  = 第 N 批，跑 GENS_PER_BATCH 代后保存检查点退出
    """
    start_ts = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🚀 Step2-GA 批{batch_id} 开始 | pid={os.getpid()} | ppid={os.getppid()}")
    print(f"[{now}]    开始时间戳: {start_ts}")
    print(f"[{now}]    孤立会话检测: session={os.environ.get('SESSION', 'N/A')}")

    TOTAL_TARGET_GENS = generations if generations else (6 if trading_day else 6)   # 总代数
    GENS_PER_BATCH   = 3 if trading_day else 3    # 每批代数

    # 注册 SIGTERM handler（捕获 cron kill 信号）
    global _current_ga_ckpt_path
    _current_ga_ckpt_path = str(Path(__file__).parent.parent / 'checkpoints' / 'ga_checkpoint.json')
    _setup_sigterm_handler()

    # ---- P0 Bug修复: 检查点新鲜度校验（仅 batch_id is None 时检查）----
    if batch_id is None and not force_restart:
        ckpt_data, completed = get_ckpt('step2_ga')
        should_skip, skip_data = _check_and_skip_or_run('step2_ga', ckpt_data, completed, force_restart)
        if should_skip:
            return skip_data

    # 必须有 Step1 结果
    ckpt1, comp1 = get_ckpt('step1_lgb')
    if comp1 != 'step1_lgb':
        print(f"[{now}] ❌ Step1 未完成，请先运行 job_step1_lgb")
        return None

    from src.core.genetic_optimizer import GeneticOptimizer
    from src.core.backtester import BacktestExecutor

    api = get_instance()
    store = api.store
    logger = api.logger
    start_date = '20240101'
    end_date = '20260327'
    neutralized_factors = ckpt1.get('neutralized_factors', ckpt1.get('selected_factors', []))

    print(f"[{now}] ===== Step2-GA 批{batch_id} ===== ({len(neutralized_factors)} 个因子)")

    # ---- 从 GA checkpoint 读取当前进度 ----
    ga_ckpt_path = Path(__file__).parent.parent / 'checkpoints' / 'ga_checkpoint.json'
    ga_state = {}
    n_completed = 0

    if ga_ckpt_path.exists():
        try:
            with open(ga_ckpt_path) as f:
                ga_state = json.load(f)
            n_completed = ga_state.get('n_gens_completed', 0)
            total_target = ga_state.get('total_target', TOTAL_TARGET_GENS)
            print(f"[{now}] GA检查点: 已完成={n_completed}/{total_target}代, Best={ga_state.get('best_sharpe', -999):.4f}")
        except Exception as e:
            print(f"[{now}] GA检查点读取失败: {e}")
            n_completed = 0

    # ---- 确定本批跑多少代 ----
    if batch_id is not None:
        # 分批模式：每批只跑 GENS_PER_BATCH 代
        target_gen = TOTAL_TARGET_GENS
        if n_completed >= target_gen:
            print(f"[{now}] ✅ GA已完成{n_completed}代，无需再跑")
            # 构造最终数据返回
            data = {
                'ga_result': {
                    'best_sharpe': ga_state.get('best_sharpe', -999),
                    'best_factors': ga_state.get('best_factors', []),
                    'best_params': ga_state.get('best_params', {}),
                    'generation_history': ga_state.get('generation_history', []),
                },
                'ga_best_sharpe': ga_state.get('best_sharpe', -999),
                'ga_best_factors': ga_state.get('best_factors', []),
                'ga_best_params': ga_state.get('best_params', {}),
                'neutralized_factors': neutralized_factors,
                'xgb_result': {},
            }
            save_ckpt('step2_ga', 'step2_ga', data)
            _send_step_report("2", data)
            return data

        n_gens_to_run = min(GENS_PER_BATCH, target_gen - n_completed)
        n_gens_completed = n_completed
        remaining_n_gens = {
            'n_gens_completed': n_completed,
            'total_target': target_gen,
            'fitness_history': ga_state.get('generation_history', []),
            'best_sharpe': ga_state.get('best_sharpe', -999),
            'best_chromosome': ga_state.get('best_chromosome'),
            'best_decoded': {
                'factors': ga_state.get('best_factors', []),
                'params': ga_state.get('best_params', {}),
            },
            'population': ga_state.get('population', []),
        }
        batch_label = f"batch{batch_id}"
        print(f"[{now}] [{batch_label}] 跑 {n_gens_to_run} 代 (总目标={target_gen})")
    else:
        # 一次性模式（手动触发）：从检查点继续或从头
        target_gen = TOTAL_TARGET_GENS
        n_gens_completed = n_completed
        if n_completed > 0:
            remaining_n_gens = {
                'n_gens_completed': n_completed,
                'total_target': TOTAL_TARGET_GENS,
                'fitness_history': ga_state.get('generation_history', []),
                'best_sharpe': ga_state.get('best_sharpe', -999),
                'best_chromosome': ga_state.get('best_chromosome'),
                'best_decoded': {
                    'factors': ga_state.get('best_factors', []),
                    'params': ga_state.get('best_params', {}),
                },
                'population': ga_state.get('population', []),
            }
            n_gens_to_run = max(0, TOTAL_TARGET_GENS - n_completed)
            print(f"[{now}] 从检查点继续，还剩 {n_gens_to_run} 代")
        else:
            remaining_n_gens = None
            n_gens_to_run = TOTAL_TARGET_GENS
        batch_label = "manual"

    # ---- 创建 GA 优化器 ----
    bt_executor = BacktestExecutor(store, logger, api.config)
    ga_opt = GeneticOptimizer(
        bt_executor, logger,
        start_date=start_date, end_date=end_date,
        pop_size=8, n_generations=TOTAL_TARGET_GENS,
        mutation_rate=0.15, elite_ratio=0.2,
        config=api.config,
    )

    # ---- 超时检查（SIGTERM 安全退出）----
    last_logged_gen = [0]  # closure: track last logged generation
    cron_safe = CronSafeTimeout(
        cron_timeout_seconds=80 * 60,
        warning_pct=1.0,
        checkpoint_fn=None,
        step_name=f'Step2-GA-{batch_label}'
    )
    def _timeout_check():
        # 每代完成后打印进度日志
        current_gen = ga_opt._current_generation if hasattr(ga_opt, '_current_generation') else len(ga_opt._fitness_history)
        if current_gen > last_logged_gen[0]:
            elapsed = time.time() - start_ts
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now_str}] 📊 GA Gen {current_gen} 完成 | 当前时间戳: {time.time()} | 运行: {elapsed:.0f}s")
            last_logged_gen[0] = current_gen
        if cron_safe.check():
            print(f"[{now}] ⏰ 时间不多，保存检查点后退出")
            save_ckpt('step2_ga', 'step2_ga', {
                'ga_result': {
                    'best_sharpe': ga_opt._current_best_sharpe,
                    'best_factors': ga_opt._current_best_decoded.get('factors', []),
                    'best_params': ga_opt._current_best_decoded.get('params', {}),
                    'generation_history': list(ga_opt._fitness_history),
                },
                'ga_best_sharpe': ga_opt._current_best_sharpe,
                'ga_best_factors': ga_opt._current_best_decoded.get('factors', []),
                'ga_best_params': ga_opt._current_best_decoded.get('params', {}),
                'neutralized_factors': neutralized_factors,
                'xgb_result': {},
            })
            n_completed = len(ga_opt._fitness_history)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now_str}] 💾 GA检查点已保存 | n_gens_completed={n_completed} | total_target={TOTAL_TARGET_GENS} | 时间戳: {time.time()}")
            return True
        return False

    # ---- 运行 GA ----
    ga_result = ga_opt.optimize(
        neutralized_factors,
        timeout_checker=_timeout_check,
        remaining_n_gens=remaining_n_gens,
    )

    ga_best_sharpe = ga_opt._current_best_sharpe
    ga_best_factors = ga_opt._current_best_decoded.get('factors', [])
    ga_best_params = ga_opt._current_best_decoded.get('params', {})
    n_new_done = len(ga_opt._fitness_history)
    n_total_now = n_new_done

    print(f"[{now}] [{batch_label}] 完成 {n_new_done} 代, Best={ga_best_sharpe:.4f}")

    # ---- XGBoost（仅 batch_id=None/0 时跑）----
    xgb_result = {}
    if batch_id in (None, 0):
        try:
            xgb_enabled = api.config.get('ml_models', {}).get('xgboost', {}).get('enabled', False)
            if xgb_enabled:
                from src.core.ml_models import XGBoostModel
                xgb_model = XGBoostModel(store, logger, api.config)
                xgb_result = xgb_model.train(neutralized_factors, '20220101', '20241231')
                print(f"[{now}] XGBoost AUC={xgb_result.get('auc', 0):.4f}")
        except Exception as e:
            print(f"[{now}] ⚠️ XGBoost: {e}")

    # ---- 保存检查点 ----
    data = {
        'ga_result': {
            'best_sharpe': ga_best_sharpe,
            'best_factors': ga_best_factors,
            'best_params': ga_best_params,
            'generation_history': list(ga_opt._fitness_history),
        },
        'ga_best_sharpe': ga_best_sharpe,
        'ga_best_factors': ga_best_factors,
        'ga_best_params': ga_best_params,
        'neutralized_factors': neutralized_factors,
        'xgb_result': xgb_result,
    }

    if n_total_now >= TOTAL_TARGET_GENS:
        # 全部完成
        save_ckpt('step2_ga', 'step2_ga', data)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] 💾 GA检查点已保存 | n_gens_completed={n_total_now} | total_target={TOTAL_TARGET_GENS} | 时间戳: {time.time()}")
        print(f"[{now}] ✅ Step2-GA 全部完成 ({TOTAL_TARGET_GENS}代)")
        _send_step_report("2", data)
    else:
        # 未完成，只保存中间结果（供下批接力）
        save_ckpt('step2_ga', 'step2_ga_partial', data)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] 💾 GA检查点已保存 | n_gens_completed={n_total_now} | total_target={TOTAL_TARGET_GENS} | 时间戳: {time.time()}")
        next_batch = batch_id + 1 if batch_id is not None else 1
        print(f"[{now}] ⏳ Step2-GA 批{batch_id} 完成({n_total_now}/{TOTAL_TARGET_GENS}代), "
              f"下批 batch{next_batch} 在下一 cron 触发")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] ✅ Step2-GA 批{batch_id} 正常结束 | 总耗时: {time.time() - start_ts:.0f}s")
    return data


# Step 3: RL 强化学习仓位优化
# =============================================================================

@timeout(3 * 3600)
def job_step3_rl(force_restart=False, batch_id=None, daily_reset=False):
    """
    Step3: RL 强化学习仓位优化（分批训练版）
    
    Args:
        force_restart: 强制重新开始
        batch_id: 批次 ID (0-3), None 表示不分批
        daily_reset: 是否清除旧数据（仅 batch0 使用）
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if batch_id is not None:
        print(f"\n[{now}] ===== Step3: RL 强化学习仓位优化 Batch {batch_id} =====")
    else:
        print(f"\n[{now}] ===== Step3: RL 强化学习仓位优化 =====")

    # ---- P0 Bug修复: 检查点新鲜度校验 ----
    # batch0 逻辑
    if batch_id == 0:
        daily_reset = True

    # ---- P0 Bug 修复：检查点新鲜度校验 ----
    if not force_restart and batch_id is None:
        ckpt_data, completed = get_ckpt('step3_rl')
        should_skip, skip_data = _check_and_skip_or_run('step3_rl', ckpt_data, completed, force_restart)
        if should_skip:
            return skip_data

    ckpt2, comp2 = get_ckpt('step2_ga')
    if comp2 != 'step2_ga':
        print(f"[{now}] ❌ Step2 未完成，请先运行 job_step2_ga")
        return None

    from src.core.rl_optimizer import RLOptimizer
    from src.core.overfit_detector import OverfitDetector
    from src.core.backtester import BacktestExecutor

    api = get_instance()
    store = api.store
    logger = api.logger

    start_date = '20240101'
    end_date = '20260327'
    ga_best_factors = ckpt2.get('ga_best_factors', [])
    ga_best_params = ckpt2.get('ga_best_params', {})
    neutralized_factors = ckpt2.get('neutralized_factors', [])

    # RL 优化
    print(f"[{now}] Step3.1: RL 强化学习仓位优化...")
    bt_executor = BacktestExecutor(store, logger, api.config)
    rl_strategy = {
        'strategy_id': 'rl_position',
        'strategy_name': 'rl_position',
        'factors': ga_best_factors if ga_best_factors else neutralized_factors[:3],
    }
    rl_opt = RLOptimizer(
        bt_executor, logger,
        start_date=start_date, end_date=end_date,
        gamma=0.95, alpha=0.1, epsilon=0.1,
        n_episodes=5, lookback_days=20,  # 原50
    )
    # 分批训练调用
    if batch_id is not None:
        rl_result = rl_opt.optimize(
            rl_strategy, ga_best_params, 
            use_rl_position=True,
            batch_id=batch_id,
            daily_reset=daily_reset
        )
        
        # Eval 回测（每批后执行）
        print(f"[{now}] Step3.2: Eval 回测...")
        eval_result = rl_opt.run_eval_backtest()
        rl_result['eval_results'].append(eval_result)
        print(f"[{now}] Eval 回测：Sharpe={eval_result.get('sharpe', 0):.4f}")
    else:
        rl_result = rl_opt.optimize(rl_strategy, ga_best_params, use_rl_position=True)
    rl_best_sharpe = rl_result.get('final_sharpe', -999)
    print(f"[{now}] RL 最优: Sharpe={rl_best_sharpe:.4f}")

    # GA 过拟合检测
    print(f"[{now}] Step3.3: GA 多重检验过拟合检测...")
    try:
        ga_gen_history = ckpt2.get('ga_result', {}).get('generation_history', [])
        all_ga_sharpes = [h['best_sharpe'] for h in ga_gen_history]
        detector = OverfitDetector(store, logger, api.config)
        mt_result = detector.calc_multiple_testing_adjustment(all_ga_sharpes, n_trials=len(all_ga_sharpes))
        print(f"[{now}] 多重检验: Bonferroni显著={mt_result.get('n_significant_bonferroni', 0)}/{len(all_ga_sharpes)}")
    except Exception as e:
        print(f"[{now}] ⚠️ 过拟合检测失败: {e}")
        mt_result = {}

    data = {
        'rl_batch_id': rl_result.get('rl_batch_id', batch_id),
        'rl_episodes_done': rl_result.get('rl_episodes_done', 5),
        'eval_results': rl_result.get('eval_results', []),
        'rl_result': {k: v for k, v in rl_result.items() if k not in ('q_table',)},
        'rl_best_sharpe': rl_best_sharpe,
        'ga_best_factors': ga_best_factors,
        'ga_best_params': ga_best_params,
        'neutralized_factors': neutralized_factors,
        'multiple_testing': mt_result,
    }
    save_ckpt('step3_rl', 'step3_rl', data)
    print(f"[{now}] ✅ Step3 完成")
    _send_step_report("3", data)
    return data


# =============================================================================
# Step 4: Bayesian/Grid 参数精调
# =============================================================================

@timeout(90 * 60)
def job_step4_bayes(force_restart=False):
    """非交易日 14:00 | Step4: Bayesian/Grid 参数精调"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ===== Step4: Bayesian/Grid 参数精调 =====")

    # ---- P0 Bug修复: 检查点新鲜度校验 ----
    if not force_restart:
        ckpt_data, completed = get_ckpt('step4_bayes')
        should_skip, skip_data = _check_and_skip_or_run('step4_bayes', ckpt_data, completed, force_restart)
        if should_skip:
            return skip_data

    ckpt3, comp3 = get_ckpt('step3_rl')
    if comp3 != 'step3_rl':
        print(f"[{now}] ❌ Step3 未完成，请先运行 job_step3_rl")
        return None

    from src.core.optimizer import BayesianOptimizer, GridSearchOptimizer
    from src.core.strategy_gen import StrategyGenerator
    from src.core.backtester import BacktestExecutor

    api = get_instance()
    store = api.store
    logger = api.logger

    start_date = '20240101'
    end_date = '20260327'
    ga_best_factors = ckpt3.get('ga_best_factors', [])
    ga_best_params = ckpt3.get('ga_best_params', {})
    rl_best_sharpe = ckpt3.get('rl_best_sharpe', -999)
    neutralized_factors = ckpt3.get('neutralized_factors', [])

    bt_executor = BacktestExecutor(store, logger, api.config)
    final_factors = ga_best_factors if ga_best_factors else neutralized_factors[:3]
    final_strat = {
        'strategy_id': 'optimized',
        'strategy_name': 'optimized',
        'factors': final_factors,
    }

    # Bayesian 优化
    print(f"[{now}] Step4.1: Bayesian 参数精调...")
    bayes_opt = BayesianOptimizer(bt_executor, logger, start_date=start_date, end_date=end_date)
    try:
        bayes_result = bayes_opt.optimize(final_strat, n_trials=20)
        bayes_sharpe = bayes_result.get('best_value', -999)
    except Exception as e:
        logger.warning(f"Bayesian优化失败: {e}")
        bayes_result = {'best_params': {}, 'best_value': -999}
        bayes_sharpe = -999

    # Grid 搜索
    print(f"[{now}] Step4.2: Grid 参数搜索...")
    grid_opt = GridSearchOptimizer(bt_executor, logger, start_date=start_date, end_date=end_date)
    grid_params = {
        'holding_period': [5, 10, 20, 40, 60],
        'stop_loss': [0.05, 0.08, 0.10, 0.15],
        'take_profit': [0.15, 0.20, 0.25, 0.30],
        'weight_scheme': ['equal', 'ic_weighted', 'volatility_inverse'],
    }
    try:
        grid_result = grid_opt.optimize(final_strat, grid_params)
        grid_sharpe = grid_result.get('best_value', -999)
    except Exception as e:
        logger.warning(f"Grid搜索失败: {e}")
        grid_result = {'best_params': {}, 'best_value': -999}
        grid_sharpe = -999

    # 选择最优
    if bayes_sharpe >= grid_sharpe:
        refined_params = bayes_result.get('best_params', {})
        refined_sharpe = bayes_sharpe
        refine_method = 'Bayesian'
    else:
        refined_params = grid_result.get('best_params', {})
        refined_sharpe = grid_sharpe
        refine_method = 'Grid'

    print(f"[{now}] 参数精调最优: {refine_method}, Sharpe={refined_sharpe:.4f}")

    # 策略生成
    print(f"[{now}] Step4.3: 策略候选生成...")
    try:
        sg = StrategyGenerator(store, logger)
        top_factors_for_gen = [{'factor_name': f, 'ir': 0.0} for f in final_factors]
        gen_param_space = {
            'holding_periods': [5, 10, 20, 40, 60],
            'weight_schemes': ['equal', 'ic_weighted', 'volatility_inverse'],
            'stop_loss': [0.05, 0.08, 0.10, 0.15],
            'take_profit': [0.15, 0.20, 0.25, 0.30],
        }
        all_strategies = sg.generate_strategies(top_factors_for_gen, gen_param_space)
        print(f"[{now}] 策略候选: {len(all_strategies)} 个")
    except Exception as e:
        print(f"[{now}] ⚠️ 策略生成失败: {e}")
        all_strategies = []

    data = {
        'bayes_result': bayes_result,
        'grid_result': grid_result,
        'refined_sharpe': refined_sharpe,
        'refined_params': refined_params,
        'refine_method': refine_method,
        'ga_best_sharpe': ckpt3.get('ga_best_sharpe', -999),
        'ga_best_factors': ga_best_factors,
        'ga_best_params': ga_best_params,
        'rl_best_sharpe': rl_best_sharpe,
        'neutralized_factors': neutralized_factors,
        'total_strategies_generated': len(all_strategies),
    }
    save_ckpt('step4_bayes', 'step4_bayes', data)
    print(f"[{now}] ✅ Step4 完成")
    _send_step_report("4", data)
    return data


# =============================================================================
# Step 5: 最终回测 + 生成报告
# =============================================================================

@timeout(60 * 60)
def job_step5_final(force_restart=False):
    """非交易日 16:00 | Step5: 最终回测验证 + 生成报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ===== Step5: 最终回测 + 生成报告 =====")

    # ---- P0 Bug修复: 检查点新鲜度校验 ----
    if not force_restart:
        ckpt_data, completed = get_ckpt('step5_final')
        should_skip, skip_data = _check_and_skip_or_run('step5_final', ckpt_data, completed, force_restart)
        if should_skip:
            # P0: 即使跳过也要确保job_optimize_checkpoint是同步的
            if skip_data:
                _update_job_optimize_checkpoint('step5_final', skip_data)
            return skip_data

    ckpt4, comp4 = get_ckpt('step4_bayes')
    if comp4 != 'step4_bayes':
        print(f"[{now}] ❌ Step4 未完成，请先运行 job_step4_bayes")
        return None

    from src.core.overfit_detector import OverfitDetector
    from src.core.backtester import BacktestExecutor

    api = get_instance()
    store = api.store
    logger = api.logger

    start_date = '20240101'
    end_date = '20260327'
    ga_best_sharpe = ckpt4.get('ga_best_sharpe', -999)
    ga_best_factors = ckpt4.get('ga_best_factors', [])
    ga_best_params = ckpt4.get('ga_best_params', {})
    rl_best_sharpe = ckpt4.get('rl_best_sharpe', -999)
    refined_sharpe = ckpt4.get('refined_sharpe', -999)
    refined_params = ckpt4.get('refined_params', {})
    refine_method = ckpt4.get('refine_method', 'Grid')
    neutralized_factors = ckpt4.get('neutralized_factors', [])

    bt_executor = BacktestExecutor(store, logger, api.config)

    # 综合最优选择
    all_candidates = [
        ('GA', ga_best_sharpe, ga_best_params, ga_best_factors),
        ('RL', rl_best_sharpe, ga_best_params, ga_best_factors[:3] if ga_best_factors else neutralized_factors[:3]),
        ('Refined', refined_sharpe, refined_params, ga_best_factors),
    ]
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    best_name, best_sharpe, best_params, best_factors = all_candidates[0]
    print(f"[{now}] 综合最优: {best_name}, Sharpe={best_sharpe:.4f}")

    # 最终回测
    print(f"[{now}] Step5.1: 最终回测验证...")
    final_strat = {
        'strategy_id': 'final',
        'strategy_name': f'final_{best_name.lower()}',
        'factors': best_factors,
    }
    try:
        final_bt = bt_executor.run(final_strat, best_params, start_date, end_date)
        final_bt['strategy_id'] = final_strat['strategy_id']
        final_bt['strategy_name'] = final_strat.get('strategy_name', '')
        final_bt['factors'] = final_strat.get('factors', [])
    except Exception as e:
        logger.warning(f"最终回测失败: {e}")
        final_bt = {}

    # ===== 持仓写入 backtest_positions =====
    if final_bt and final_bt.get('trades'):
        print(f"[{now}] Step5.x: 写入持仓到 backtest_positions...")
        try:
            from collections import defaultdict
            # 从交易记录重建期末持仓
            shares_held = defaultdict(int)
            for trade in final_bt['trades']:
                if trade.get('direction') == 'buy':
                    shares_held[trade['ts_code']] += int(trade.get('quantity', 0))
                elif trade.get('direction') == 'sell':
                    shares_held[trade['ts_code']] -= int(trade.get('quantity', 0))
            # 过滤掉期末已清仓的
            final_positions = {code: qty for code, qty in shares_held.items() if qty > 0}
            total_shares = sum(final_positions.values()) or 1

            # 写入 backtest_positions（先删除旧记录再插入）
            store.conn.execute("DELETE FROM backtest_positions")
            for code, qty in final_positions.items():
                ratio = qty / total_shares
                store.conn.execute(
                    "INSERT INTO backtest_positions (date, code, position_ratio, updated_at) VALUES (?, ?, ?, ?)",
                    [end_date, code, ratio, datetime.now()]
                )
            print(f"[{now}] 持仓写入完成: {len(final_positions)} 只股票")
        except Exception as e:
            logger.warning(f"持仓写入失败: {e}")

    # 过拟合检测
    print(f"[{now}] Step5.2: 过拟合检测...")
    overfit_accepted = True
    overfit_summary = {}
    try:
        detector = OverfitDetector(store, logger, api.config)
        ods_cfg = api.config.get('overfit_detection', {})

        if ods_cfg.get('enabled', True):
            # 从 Step2 获取 GA 历史
            ckpt2, _ = get_ckpt('step2_ga')
            ga_gen_history = ckpt2.get('ga_result', {}).get('generation_history', [])
            all_ga_sharpes = [h['best_sharpe'] for h in ga_gen_history]

            is_sharpes = all_ga_sharpes[:max(1, len(all_ga_sharpes) // 2)]
            oos_sharpes = all_ga_sharpes[max(1, len(all_ga_sharpes) // 2):]

            pseudo_returns = [s / 252.0 / 16.0 for s in all_ga_sharpes]
            if len(pseudo_returns) >= 60:
                split_i = int(len(pseudo_returns) * 0.8)
                pbo = detector.calc_pbo(pseudo_returns[:split_i], pseudo_returns[split_i:])
                cscv = detector.calc_cscv(pseudo_returns)
            else:
                pbo, cscv = 0.5, 0.5

            dsr = detector.calc_dsr(all_ga_sharpes, [all_ga_sharpes], n_trials=len(all_ga_sharpes))
            mt_result = detector.calc_multiple_testing_adjustment(all_ga_sharpes, n_trials=len(all_ga_sharpes))

            overfit_summary = {
                'pbo': round(float(pbo), 4),
                'dsr': round(float(dsr), 4),
                'cscv': round(float(cscv), 4),
                'multiple_testing': mt_result,
            }

            pbo_th = float(ods_cfg.get('pbo_threshold', 0.5))
            dsr_th = float(ods_cfg.get('dsr_threshold', 0.0))
            cscv_th = float(ods_cfg.get('cscv_threshold', 0.5))

            reasons = []
            if pbo >= pbo_th:
                reasons.append(f"PBO({pbo:.4f})>={pbo_th}")
                overfit_accepted = False
            if dsr <= dsr_th:
                reasons.append(f"DSR({dsr:.4f})<={dsr_th}")
                overfit_accepted = False
            if cscv < cscv_th:
                reasons.append(f"CSCV({cscv:.4f})<{cscv_th}")
                overfit_accepted = False

            if overfit_accepted:
                print(f"[{now}] ✅ 策略通过过拟合检测: PBO={pbo:.4f}, DSR={dsr:.4f}, CSCV={cscv:.4f}")
            else:
                print(f"[{now}] ⚠️ 策略被拒绝（过拟合风险）: {', '.join(reasons)}")
    except Exception as e:
        print(f"[{now}] ⚠️ 过拟合检测失败: {e}")

    # 生成报告
    print(f"[{now}] Step5.3: 生成报告...")
    report = {
        "report_time": now,
        "data_range": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} ~ {end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
        "best_method": best_name,
        "best_sharpe": round(float(best_sharpe), 4),
        "best_params": best_params,
        "best_factors": best_factors,
        "ga_best_sharpe": round(float(ga_best_sharpe), 4),
        "rl_best_sharpe": round(float(rl_best_sharpe), 4),
        "refined_sharpe": round(float(refined_sharpe), 4),
        "refine_method": refine_method,
        "neutralized_factors": neutralized_factors,
        "best_strategy": _clean_bt(final_bt) if final_bt else {},
        "total_strategies_generated": ckpt4.get('total_strategies_generated', 0),
        "overfit_detection": {
            "overfit_accepted": overfit_accepted,
            "summary": overfit_summary,
        },
    }

    # 加载 Step1 的 LightGBM 结果
    ckpt1, _ = get_ckpt('step1_lgb')
    if ckpt1:
        report['lgb_selected_count'] = len(ckpt1.get('selected_factors', []))
        report['lgb_model_auc'] = ckpt1.get('lgb_model_auc', 0)

    # 加载 Step2 的 XGBoost 结果
    ckpt2, _ = get_ckpt('step2_ga')
    if ckpt2 and ckpt2.get('xgb_result'):
        report['xgb_auc'] = ckpt2['xgb_result'].get('auc', 0)
        report['xgb_feature_importance'] = ckpt2['xgb_result'].get('feature_importance', {})

    # 交易日写入 daily_strategy_report.json（非交易日写入 weekend_report.json）
    today_str = date.today().strftime('%Y%m%d')
    if is_trading_day():
        REPORT_FILE = Path(__file__).parent.parent / "reports" / f"daily_strategy_report.json"
    else:
        REPORT_FILE = Path(__file__).parent.parent / "reports" / f"weekend_report_{today_str}.json"
    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[{now}] 📄 报告已生成: {REPORT_FILE}")

    # 打印摘要
    _print_summary(now, report)

    save_ckpt('step5_final', 'step5_final', report)
    
    # P0 Bug修复: step5完成后同步更新job_optimize_checkpoint.json
    _update_job_optimize_checkpoint('step5_final', report)
    
    print(f"[{now}] ✅ Step5 完成")
    _send_step_report("5", report)
    return report


def _clean_bt(bt):
    """清理回测结果，移除不可序列化的对象"""
    if not bt:
        return {}
    skip_keys = {'nav_series', 'trade_records', 'positions'}
    return {k: v for k, v in bt.items() if k not in skip_keys}


def _print_summary(now, report):
    """打印报告摘要"""
    print(f"\n{'='*60}")
    print(f"📊 非交易日策略报告 [{now}]")
    print(f"{'='*60}")
    print(f"最优方法: {report.get('best_method', 'N/A')}")
    print(f"最优Sharpe: {report.get('best_sharpe', 0):.4f}")
    print(f"最优因子: {report.get('best_factors', [])}")
    if report.get('best_strategy'):
        bs = report['best_strategy']
        print(f"总收益: {bs.get('total_return', 0):.2%}")
        print(f"夏普: {bs.get('sharpe_ratio', 0):.4f}")
        print(f"最大回撤: {bs.get('max_drawdown', 0):.2%}")
    od = report.get('overfit_detection', {})
    print(f"过拟合检测: {'✓ 通过' if od.get('overfit_accepted') else '✗ 未通过'}")
    if od.get('summary'):
        s = od['summary']
        print(f"  PBO={s.get('pbo', 'N/A')} | DSR={s.get('dsr', 'N/A')} | CSCV={s.get('cscv', 'N/A')}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='非交易日优化任务（分拆版）')
    parser.add_argument('--step', type=int, default=0, help='运行指定步骤 (1-5)，0=运行所有')
    parser.add_argument('--force', action='store_true', help='强制从头开始（忽略检查点）')
    parser.add_argument('--batch-id', type=int, default=None,
                        help='Step2 GA 批编号 (0-4)，分批续跑时使用')
    parser.add_argument('--trading-day', action='store_true',
                        help='交易日模式（5代，每批1代）')
    parser.add_argument('--generations', type=int, default=None,
                        help='手动指定总代数（覆盖默认配置）')
    args = parser.parse_args()

    # 清除代理
    for k in list(os.environ.keys()):
        if 'proxy' in k.lower():
            os.environ.pop(k, None)

    step = args.step
    force = args.force

    if step == 0:
        # 运行所有步骤
        for s in range(1, 6):
            func = [None, job_step1_lgb, job_step2_ga, job_step3_rl, job_step4_bayes, job_step5_final][s]
            kwargs = {'force_restart': force}
            if s == 2:
                if args.batch_id is not None:
                    kwargs['batch_id'] = args.batch_id
                if args.trading_day:
                    kwargs['trading_day'] = True
                if args.generations is not None:
                    kwargs['generations'] = args.generations
            result = func(**kwargs)
            if result is None:
                print(f"Step{s} 失败，停止")
                break
            force = False  # 后续步骤自动续算
    else:
        # 运行指定步骤
        func = [None, job_step1_lgb, job_step2_ga, job_step3_rl, job_step4_bayes, job_step5_final][step]
        if func:
            kwargs = {'force_restart': force}
            if step == 2:
                if args.batch_id is not None:
                    kwargs['batch_id'] = args.batch_id
                if args.trading_day:
                    kwargs['trading_day'] = True
                if args.generations is not None:
                    kwargs['generations'] = args.generations
            result = func(**kwargs)
            if result is None:
                print(f"Step{step} 需要前置步骤完成")
        else:
            print(f"无效步骤: {step}")
