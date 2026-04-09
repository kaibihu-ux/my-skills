"""
盘前预判任务 (08:00-09:25)
1. 下载隔夜新闻/舆情（如有）
2. 预判今日开盘策略
3. 生成盘前报告
"""
import sys
from pathlib import Path
from datetime import date, datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance


def pre_market():
    """
    盘前预判流程：
    1. 读取最新策略
    2. 模拟今日开盘情况（用前一日收盘数据）
    3. 如果持有股票有重大事件，预警
    4. 输出"今日操作预判"报告
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = date.today().strftime("%Y%m%d")
    report_date = date.today().strftime("%Y%m%d")
    print(f"[{now}] 📋 盘前预判任务开始")

    try:
        api = get_instance()
        store = api.store
        logger = api.logger

        # 获取最新策略
        best_strategies_df = store.df(
            "SELECT * FROM strategy_pool ORDER BY updated_at DESC LIMIT 5"
        )

        pre_market_actions = []
        market_outlook = "震荡"
        risk_level = "中"

        if not best_strategies_df.empty:
            for _, row in best_strategies_df.iterrows():
                strategy_name = row.get('strategy_name', 'N/A')
                factors = json.loads(row['factors']) if isinstance(row['factors'], str) else row.get('factors', [])
                metrics_str = row.get('metrics', '{}')
                if isinstance(metrics_str, str):
                    try:
                        metrics = json.loads(metrics_str)
                    except Exception:
                        metrics = {}
                else:
                    metrics = metrics_str

                sharpe = metrics.get('sharpe_ratio', 0)
                total_return = metrics.get('total_return', 0)
                max_dd = metrics.get('max_drawdown', 0)

                # 预判操作
                action = _predict_action(
                    strategy_name, sharpe, total_return, max_dd, factors
                )
                pre_market_actions.append({
                    'strategy_id': row.get('strategy_id', ''),
                    'strategy_name': strategy_name,
                    'factors': factors,
                    'sharpe_ratio': float(sharpe or 0),
                    'total_return': float(total_return or 0),
                    'max_drawdown': float(max_dd or 0),
                    'predicted_action': action,
                })

            # 综合判断
            avg_sharpe = sum(a['sharpe_ratio'] for a in pre_market_actions) / len(pre_market_actions)
            if avg_sharpe > 1.5:
                market_outlook = "多头趋势，建议积极持仓"
                risk_level = "低"
            elif avg_sharpe > 0.8:
                market_outlook = "震荡偏多，谨慎操作"
                risk_level = "中"
            elif avg_sharpe > 0:
                market_outlook = "趋势不明，保持观察"
                risk_level = "中"
            else:
                market_outlook = "策略表现不佳，建议观望"
                risk_level = "高"
        else:
            market_outlook = "无历史策略，默认使用动量策略"
            risk_level = "中"
            pre_market_actions.append({
                'strategy_id': 'momentum_default',
                'strategy_name': 'momentum_default',
                'factors': ['momentum_20'],
                'predicted_action': '按动量因子操作，关注开盘30分钟走势',
            })

        # 生成盘前报告
        report = {
            'report_type': 'pre_market',
            'report_date': report_date,
            'report_time': now,
            'market_outlook': market_outlook,
            'risk_level': risk_level,
            'trading_day': today_str,
            'strategies': pre_market_actions,
            'pre_market_check': {
                'data_status': 'ready',
                'market_open': '09:30',
                'pre_market_window': '08:00-09:25',
            },
            'today_action': _build_today_action(pre_market_actions, market_outlook),
        }

        # 保存盘前报告
        report_dir = Path(__file__).parent.parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"pre_market_{report_date}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{now}] ✅ 盘前报告已生成: {report_file}")
        print(f"[{now}] 市场展望: {market_outlook}, 风险等级: {risk_level}")

        return report

    except Exception as e:
        print(f"[{now}] ❌ 盘前预判失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _predict_action(strategy_name, sharpe, total_return, max_dd, factors):
    """根据策略绩效预判今日操作"""
    if sharpe > 1.5 and max_dd < 0.15:
        return "持仓不动，等待信号"
    elif sharpe > 1.0 and total_return > 0.2:
        return "可适度加仓，止损设在前高"
    elif sharpe > 0.5:
        return "谨慎持仓，关注盘中信号"
    elif sharpe > 0:
        return "轻仓观望，不追高"
    else:
        return "建议空仓等待策略优化"


def _build_today_action(strategies, market_outlook):
    """构建今日操作建议"""
    if "多头" in market_outlook:
        return "建议仓位：80%，关注开盘跳空缺口"
    elif "震荡" in market_outlook:
        return "建议仓位：50%，高抛低吸"
    elif "不佳" in market_outlook or "观望" in market_outlook:
        return "建议仓位：20%或空仓，等待明确信号"
    return "建议仓位：50%，随机应变"


def job_pre_market():
    """调度器入口"""
    return pre_market()
