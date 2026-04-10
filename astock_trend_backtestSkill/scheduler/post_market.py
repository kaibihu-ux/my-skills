"""
盘后复盘任务 (18:00-19:00)
1. 下载今日完整日线数据（Baostock 在收盘后 2-5 小时发布当日数据）
2. 计算今日持仓绩效
3. 更新因子数据库
4. 生成盘后报告
"""
import sys
from pathlib import Path
from datetime import date, datetime
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance


def _normalize_date(date_str: str) -> str:
    """支持 YYYYMMDD 和 YYYY-MM-DD，输出 YYYY-MM-DD（DuckDB 标准）。"""
    d = date_str.strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def post_market():
    """
    盘后复盘流程：
    1. 下载今日数据（18:00后baostock有完整数据）
    2. 今日绩效计算（读取今日净值，对比昨日）
    3. 预警记录（记录今日所有异常事件）
    4. 生成盘后报告
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = date.today().strftime("%Y%m%d")
    today_fmt = date.today().strftime("%Y-%m-%d")  # DuckDB 标准格式
    report_date = date.today().strftime("%Y%m%d")
    print(f"[{now}] 📊 盘后复盘任务开始")

    try:
        api = get_instance()
        store = api.store
        logger = api.logger

        # ===== 1. 下载今日完整数据（强制全量更新，确保收盘价） =====
        print(f"[{now}] 下载今日数据（强制全量，确保收盘价）...")
        try:
            # force=True：跳过"已有数据"检查，强制全量拉取，确保收盘价而非盘中实时价
            api.data_mgr.update_daily(today_str, today_str, force=True)
            print(f"[{now}] 今日数据下载完成")
        except Exception as e:
            print(f"[{now}] 数据下载失败（可能非交易日或数据未更新）: {e}")

        # ===== 1.5 数据新鲜度检查 =====
        try:
            latest_factor_date = store.conn.execute(
                "SELECT MAX(trade_date) FROM factors"
            ).fetchone()[0]
            if latest_factor_date:
                # 计算距离今天有多少个交易日
                td_count = store.conn.execute(
                    f"SELECT COUNT(*) FROM stock_daily WHERE trade_date > '{latest_factor_date}' AND trade_date <= '{today_fmt}'"
                ).fetchone()[0]
                if td_count >= 2:
                    alert_msg = f"⚠️ 因子数据陈旧！最新因子日期: {latest_factor_date}，已间隔 {td_count} 个交易日未更新"
                    print(f"[{now}] {alert_msg}")
                    # 写入 alerts 表供盘后报告记录
                    store.conn.execute(
                        "INSERT INTO alerts (alert_type, message, created_at) VALUES (?, ?, ?)",
                        ['DATA_FRESHNESS', alert_msg, datetime.now()]
                    )
                else:
                    print(f"[{now}] 因子数据新鲜度正常（最新: {latest_factor_date}，间隔{td_count}个交易日）")
        except Exception as e:
            print(f"[{now}] 数据新鲜度检查失败: {e}")

        # ===== 2. 今日行情快照 =====
        market_summary = _get_market_snapshot(store, today_fmt)

        # ===== 3. 计算今日绩效 =====
        print(f"[{now}] 计算今日绩效...")
        perf_today = _calc_today_performance(store, today_fmt)

        # ===== 4. 读取今日报告 =====
        alert_events = _collect_today_alerts(store, today_fmt)

        # ===== 5. 更新因子绩效 =====
        print(f"[{now}] 更新因子绩效...")
        try:
            api.execute({'action': 'evaluate_factors'})
        except Exception as e:
            print(f"[{now}] 因子评估失败: {e}")

        # ===== 6. 生成盘后报告 =====
        report = {
            'report_type': 'post_market',
            'report_date': report_date,
            'report_time': now,
            'trading_day': today_str,
            'market_summary': market_summary,
            'today_performance': perf_today,
            'alert_events': alert_events,
            'next_day_outlook': _build_next_day_outlook(perf_today, market_summary),
        }

        # 保存盘后报告
        report_dir = Path(__file__).parent.parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"post_market_{report_date}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{now}] ✅ 盘后报告已生成: {report_file}")
        if perf_today:
            print(f"[{now}] 今日收益: {_format_pct(perf_today.get('daily_return', 0))}")
            print(f"[{now}] 预警事件: {len(alert_events)} 条")

        return report

    except Exception as e:
        print(f"[{now}] ❌ 盘后复盘失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _get_market_snapshot(store, today_fmt):
    """获取今日大盘行情快照"""
    try:
        df = store.df(
            f"""
            SELECT trade_date, ts_code, close, pct_chg, vol
            FROM stock_daily
            WHERE trade_date = '{today_fmt}'
            ORDER BY pct_chg DESC
            LIMIT 20
            """
        )
        if df.empty:
            return {'top_gainers': [], 'top_losers': [], 'total_stocks': 0, 'avg_change': 0}

        df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
        gainers = df.nlargest(5, 'pct_chg')[['ts_code', 'close', 'pct_chg']].to_dict('records')
        losers = df.nsmallest(5, 'pct_chg')[['ts_code', 'close', 'pct_chg']].to_dict('records')
        avg_change = float(df['pct_chg'].mean())

        return {
            'top_gainers': [
                {'code': str(g['ts_code']), 'close': float(g['close']), 'change_pct': float(g['pct_chg'])}
                for g in gainers
            ],
            'top_losers': [
                {'code': str(l['ts_code']), 'close': float(l['close']), 'change_pct': float(l['pct_chg'])}
                for l in losers
            ],
            'total_stocks': len(df),
            'avg_change': round(avg_change, 4),
        }
    except Exception as e:
        return {'error': str(e)}


def _calc_today_performance(store, today_fmt):
    """计算今日绩效"""
    try:
        # 读取今日净值（如有模拟持仓记录）
        nav_df = store.df(
            f"SELECT * FROM backtest_nav WHERE date = '{today_fmt}' ORDER BY updated_at DESC LIMIT 1"
        )
        yesterday_fmt = _normalize_date(_get_yesterday(date.today().strftime("%Y%m%d")))
        nav_yesterday_df = store.df(
            f"SELECT * FROM backtest_nav WHERE date = '{yesterday_fmt}' ORDER BY updated_at DESC LIMIT 1"
        )

        if not nav_df.empty:
            today_nav = float(nav_df.iloc[0].get('nav', 1.0))
            yesterday_nav = 1.0
            if not nav_yesterday_df.empty:
                yesterday_nav = float(nav_yesterday_df.iloc[0].get('nav', 1.0))
            daily_return = (today_nav - yesterday_nav) / yesterday_nav if yesterday_nav else 0

            return {
                'nav': round(today_nav, 4),
                'daily_return': round(daily_return, 4),
                'cumulative_return': round(today_nav - 1.0, 4),
            }
    except Exception:
        pass
    return {'nav': 1.0, 'daily_return': 0.0, 'cumulative_return': 0.0}


def _collect_today_alerts(store, today_fmt):
    """收集今日预警事件"""
    alerts = []
    try:
        alert_df = store.df(
            f"SELECT * FROM alerts WHERE created_at LIKE '{today_fmt}%' ORDER BY created_at DESC LIMIT 20"
        )
        if not alert_df.empty:
            for _, row in alert_df.iterrows():
                alerts.append({
                    'type': str(row.get('alert_type', '')),
                    'message': str(row.get('message', '')),
                    'created_at': str(row.get('created_at', '')),
                })
    except Exception:
        pass
    return alerts


def _get_yesterday(today_str):
    """获取昨日日期字符串"""
    from datetime import datetime, timedelta
    try:
        d = datetime.strptime(today_str, '%Y%m%d') - timedelta(days=1)
        return d.strftime('%Y%m%d')
    except Exception:
        return today_str


def _format_pct(v):
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def _build_next_day_outlook(perf_today, market_summary):
    """构建下一日展望"""
    daily_return = perf_today.get('daily_return', 0)
    avg_change = market_summary.get('avg_change', 0)

    if daily_return > 0.03 or avg_change > 2:
        return "强势上涨后，注意明日可能回调，建议逢高减仓"
    elif daily_return < -0.03 or avg_change < -2:
        return "大幅下跌后超卖，明日可能反弹修复，可关注优质标的"
    elif daily_return > 0:
        return "小幅上涨，趋势延续，持有或适度加仓"
    elif daily_return < 0:
        return "小幅回调，趋势未破，逢低布局"
    return "市场平稳，明日延续震荡格局"


def job_post_market():
    """调度器入口"""
    return post_market()
