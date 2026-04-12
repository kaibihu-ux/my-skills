"""
打板策略进度报告模块
- 本地存档（JSON）
- 飞书推送（使用 OpenClaw 配置的飞书 App 凭证）
- 三类报告：批次完成 / 百组里程碑 / 每日汇总
"""

import os
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional


# =============================================================================
# 配置
# =============================================================================
SKILL_DIR = Path(__file__).parent.parent
OPENCLAW_CONFIG = Path.home() / '.openclaw' / 'openclaw.json'
REPORT_DIR = SKILL_DIR / 'output' / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = SKILL_DIR / 'checkpoint_state.json'

# 固定推送到当前飞书群
FEISHU_CHAT_ID = 'oc_76322b48fca60ff9f2837b78443ee1da'


# =============================================================================
# 飞书 API（使用 OpenClaw 配置的 App 凭证）
# =============================================================================
def _get_feishu_credentials() -> tuple:
    """从 openclaw.json 读取飞书 appId 和 appSecret"""
    try:
        with open(OPENCLAW_CONFIG) as f:
            cfg = json.load(f)
        feishu = cfg.get('channels', {}).get('feishu', {})
        return feishu.get('appId', ''), feishu.get('appSecret', '')
    except Exception:
        return '', ''


def _get_tenant_token() -> str:
    """获取飞书 tenant_access_token"""
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


def _send_feishu_message(text: str) -> bool:
    """发送文本消息到飞书群，返回是否成功"""
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
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("code") == 0
    except Exception as e:
        print(f"[REPORTER] 飞书推送失败: {e}")
        return False


# =============================================================================
# 工具函数
# =============================================================================
def _load_checkpoint():
    """加载断点"""
    if not CHECKPOINT_FILE.exists():
        return None
    with open(CHECKPOINT_FILE, 'r') as f:
        return json.load(f)


def _load_previous_reports() -> dict:
    """加载今日已有报告（避免重复推送）"""
    today = date.today().strftime('%Y%m%d')
    report_file = REPORT_DIR / f'{today}.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            return json.load(f)
    return {'milestones_sent': [], 'batches_sent': [], 'daily_summary_sent': False}


def _save_report(data: dict):
    """保存报告到本地"""
    today = date.today().strftime('%Y%m%d')
    report_file = REPORT_DIR / f'{today}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_top_params(results: list, top_n=3) -> list:
    """获取最优参数组合"""
    valid = [r for r in results if r and r.get('total_return') is not None]
    if not valid:
        return []
    sorted_results = sorted(valid, key=lambda x: x.get('total_return', 0), reverse=True)
    return sorted_results[:top_n]


def _format_params(p: dict) -> str:
    """格式化参数为单行文本"""
    return (f"RSI({p.get('rsi_buy',0)},{p.get('rsi_sell',0)}) "
            f"MACD({p.get('macd_fast',0)},{p.get('macd_slow',0)},{p.get('macd_signal',0)}) "
            f"BB({p.get('bb_period',0)},{p.get('bb_std',0)}) "
            f"VOL({p.get('vol_period',0)},{p.get('vol_multiplier',0)}) "
            f"Breakout:{p.get('breakout_period',0)} "
            f"SL:{p.get('stop_loss_pct',0)}% TP:{p.get('take_profit_pct',0)}% "
            f"min_cond:{p.get('min_conditions',0)}")


def _send_feishu(message: str) -> bool:
    """发送飞书文本消息到群"""
    return _send_feishu_message(message)


# =============================================================================
# 三类报告
# =============================================================================

def report_batch_completion(batch_size: int, batch_num: int, completed_total: int,
                            total: int, batch_time_seconds: float,
                            top_results: list) -> bool:
    """
    报告每批次完成

    Args:
        batch_size: 本批次完成的组合数
        batch_num: 批次序号
        completed_total: 累计已完成
        total: 总组合数
        batch_time_seconds: 本批次耗时（秒）
        top_results: 目前最优 top3 结果
        webhook: 飞书 Webhook（可选）

    Returns:
        是否推送成功
    """
    reports = _load_previous_reports()

    # 避免重复推送（同一 batch 只推送一次）
    batch_key = f"batch_{batch_num}"
    if batch_key in reports.get('batches_sent', []):
        return False
    reports.setdefault('batches_sent', []).append(batch_key)
    _save_report(reports)

    # 构建消息
    top_lines = []
    for i, r in enumerate(top_results, 1):
        top_lines.append(
            f"#{i} 收益: {r.get('total_return', 0):+.2f}% | "
            f"Sharpe: {r.get('sharpe', 0):.4f} | "
            f"回撤: {r.get('max_drawdown', 0):.2f}% | "
            f"交易: {r.get('trade_count', 0)}次"
        )
        top_lines.append(f"   {_format_params(r.get('params', {}))}")

    progress = f"{completed_total}/{total} ({100*completed_total/total:.1f}%)"
    eta = f"约 {(total-completed_total)/batch_size * batch_time_seconds / 60:.0f} 分钟" if batch_size > 0 else "—"

    msg = (
        f"📊 打板策略优化报告\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ 批次 #{batch_num} 完成\n\n"
        f"• 完成: +{batch_size} 组合（累计 {progress}）\n"
        f"• 本批耗时: {batch_time_seconds/60:.1f} 分钟\n"
        f"• 预计剩余: {eta}\n"
    )
    if top_lines:
        msg += "\n🏆 当前 Top3:\n" + "\n".join(f"• {line}" for line in top_lines)

    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    print(f"[REPORTER] 批次报告: batch#{batch_num} +{batch_size} 累计{completed_total}/{total}")

    sent = _send_feishu(msg)
    if not sent:
        print(f"[REPORTER] 飞书推送失败，仅存档")
    return sent


def report_milestone(completed: int, total: int, top_results: list) -> bool:
    """
    报告每完成 100 组里程碑
    """
    reports = _load_previous_reports()

    # 避免重复推送
    milestone_key = f"milestone_{completed}"
    if milestone_key in reports.get('milestones_sent', []):
        return False
    reports.setdefault('milestones_sent', []).append(milestone_key)
    _save_report(reports)

    top_lines = []
    for i, r in enumerate(top_results, 1):
        top_lines.append(
            f"#{i} {r.get('total_return', 0):+.2f}% | "
            f"Sharpe: {r.get('sharpe', 0):.4f} | "
            f"回撤: {r.get('max_drawdown', 0):.2f}%"
        )
        top_lines.append(f"   {_format_params(r.get('params', {}))}")

    msg = (
        f"📊 打板策略优化报告\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 里程碑: 已完成 {completed}/{total} 组\n"
    )
    if top_lines:
        msg += "\n🏆 Top3:\n" + "\n".join(f"• {line}" for line in top_lines)

    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    print(f"[REPORTER] 里程碑报告: {completed}/{total}")
    return _send_feishu(msg)


def report_daily_summary() -> bool:
    """
    每日汇总报告（当日首次 cron 触发时调用）
    """
    ck = _load_checkpoint()
    if not ck:
        return False

    results = ck.get('results', [])
    completed = len(ck.get('completed_idx', []))
    total_param_keys = ck.get('param_keys', [])
    saved_at = ck.get('saved_at', '未知')

    # 推算总组合数
    param_grid = _get_param_grid_template()
    total = 1
    for k in total_param_keys:
        if k in param_grid:
            total *= len(param_grid[k])

    top_results = _get_top_params(results, top_n=5)

    top_lines = []
    for i, r in enumerate(top_results, 1):
        top_lines.append(
            f"#{i} 收益: {r.get('total_return', 0):+.2f}% | "
            f"Sharpe: {r.get('sharpe', 0):.4f} | "
            f"回撤: {r.get('max_drawdown', 0):.2f}% | "
            f"胜率: {r.get('win_rate', 0):.1f}% | "
            f"交易: {r.get('trade_count', 0)}次"
        )
        top_lines.append(f"   {_format_params(r.get('params', {}))}")

    msg = (
        f"📊 打板策略每日汇总\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"
        f"• 总体进度: {completed}/{total} ({100*completed/total:.1f}%)\n"
        f"• 有效结果: {len(results)} 组\n"
        f"• 最后保存: {saved_at}\n"
    )
    if top_lines:
        msg += "\n🏆 历史最优 Top5:\n" + "\n".join(f"• {line}" for line in top_lines)
    else:
        msg += "\n（暂无有效结果）"

    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    print(f"[REPORTER] 每日汇总: {completed}/{total}")
    return _send_feishu(msg)


def _get_param_grid_template() -> dict:
    """获取快速模式参数网格（用于估算总数）"""
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


def report_init():
    """scheduler 启动时调用，发送每日首次汇总"""
    return report_daily_summary()
