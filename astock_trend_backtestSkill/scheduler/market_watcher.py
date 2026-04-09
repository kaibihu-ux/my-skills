"""
盘中监控：每30分钟检查持仓股票
触发条件：
- 单只股票涨跌超5%：预警
- 持仓整体回撤超3%：触发止损检查
- 持仓整体收益超5%：触发止盈检查
"""
import sys
from pathlib import Path
from datetime import datetime, date
import time
import threading
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance


def is_trading_day(d=None):
    """判断是否为交易日（排除周末和A股重要节假日）"""
    if d is None:
        d = date.today()
    # 周末
    if d.weekday() >= 5:
        return False
    # 重要节假日（简化判断，可扩展为从网上抓取）
    year = d.year
    # 元旦
    holidays = [
        f"{year}0101", f"{year}0102", f"{year}0103",
        # 春节（简化：每年正月初一前后各3天，动态计算）
        # 清明
        # 劳动节
        # 端午
        # 中秋
        # 国庆
    ]
    if d.strftime('%Y%m%d') in holidays:
        return False
    return True


def is_market_open(d=None):
    """判断当前是否在交易时间内"""
    now = datetime.now()
    if not is_trading_day(d):
        return False
    h, m = now.hour, now.minute
    return (h == 9 and m >= 30) or (h == 10) or (h == 11 and m < 30) or (h == 13) or (h == 14) or (h == 15 and m < 5)


class MarketWatcher:
    """
    盘中监控：每30分钟检查持仓股票行情
    运行在后台线程，不阻塞调度器主循环
    """

    def __init__(self, check_interval=1800):  # 30分钟
        self.check_interval = check_interval
        self.api = None
        self.store = None
        self.logger = None
        self._running = False
        self._thread = None
        # 阈值配置
        self.single_stock_threshold = 0.05    # 单只股票涨跌超5%
        self.portfolio_drawdown_threshold = 0.03  # 持仓整体回撤超3%
        self.portfolio_gain_threshold = 0.05   # 持仓整体收益超5%

    def _ensure_api(self):
        """懒加载API实例"""
        if self.api is None:
            self.api = get_instance()
            self.store = self.api.store
            self.logger = self.api.logger

    def start(self):
        """启动后台监控线程"""
        if self._running:
            print(f"[{datetime.now()}] MarketWatcher 已运行，跳过启动")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MarketWatcher")
        self._thread.start()
        print(f"[{datetime.now()}] MarketWatcher 后台监控已启动（每{self.check_interval // 60}分钟检查一次）")

    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print(f"[{datetime.now()}] MarketWatcher 已停止")

    def _run_loop(self):
        """后台主循环"""
        while self._running and is_market_open():
            self._ensure_api()
            try:
                self.check_positions()
                self.check_market_alerts()
            except Exception as e:
                print(f"[{datetime.now()}] MarketWatcher 检查异常: {e}")
                import traceback
                traceback.print_exc()
            # 分段sleep，避免长时间阻塞
            for _ in range(6):  # 6 * 5min = 30min
                if not self._running or not is_market_open():
                    break
                time.sleep(300)

    def check_positions(self):
        """检查持仓股票行情"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today_str = date.today().strftime("%Y%m%d")

        try:
            # 从回测引擎获取当前模拟持仓
            # 读取最新保存的持仓信号
            positions = self._get_current_positions(today_str)
            if not positions:
                print(f"[{now_str}] MarketWatcher: 无持仓记录，跳过持仓检查")
                return

            # 检查每只持仓股票的涨跌情况
            alerts = []
            for pos in positions:
                code = pos.get('code', '')
                try:
                    quote_df = self.store.df(
                        f"""
                        SELECT trade_date, ts_code, close, (close - pre_close) / pre_close as change_pct
                        FROM stock_daily
                        WHERE ts_code = '{code}' AND trade_date = '{today_str}'
                        ORDER BY trade_date DESC LIMIT 1
                        """
                    )
                    if quote_df.empty:
                        continue

                    close = float(quote_df.iloc[0]['close'])
                    pre_close = float(quote_df.iloc[0]['close']) / (1 + float(quote_df.iloc[0]['change_pct'])) if float(quote_df.iloc[0]['change_pct']) != 0 else float(quote_df.iloc[0]['close'])
                    change_pct = float(quote_df.iloc[0]['change_pct'])

                    if abs(change_pct) >= self.single_stock_threshold:
                        alert_type = "SINGLE_STOCK_ALERT"
                        direction = "大涨" if change_pct > 0 else "大跌"
                        alerts.append({
                            'type': alert_type,
                            'code': code,
                            'change_pct': round(change_pct, 4),
                            'close': close,
                            'message': f"持仓股票 {code} {direction} {abs(change_pct)*100:.2f}%，请关注",
                        })
                        print(f"[{now_str}] 🚨 预警: {code} {direction} {abs(change_pct)*100:.2f}%")
                except Exception as e:
                    print(f"[{now_str}] 检查持仓 {code} 失败: {e}")

            # 检查持仓整体绩效
            self._check_portfolio_alerts(positions, today_str)

            # 保存预警记录
            if alerts:
                self._save_alerts(alerts)

        except Exception as e:
            print(f"[{now_str}] 检查持仓失败: {e}")

    def _get_current_positions(self, today_str):
        """获取当前模拟持仓"""
        try:
            # 从最新回测结果中读取持仓
            pos_df = self.store.df(
                f"""
                SELECT * FROM backtest_positions
                WHERE date <= '{today_str}'
                ORDER BY date DESC, updated_at DESC
                LIMIT 20
                """
            )
            if pos_df.empty:
                return []
            # 按code去重，取最新
            seen = set()
            positions = []
            for _, row in pos_df.iterrows():
                code = str(row.get('code', ''))
                if code and code not in seen:
                    seen.add(code)
                    positions.append({
                        'code': code,
                        'date': str(row.get('date', '')),
                    })
            return positions
        except Exception:
            return []

    def _check_portfolio_alerts(self, positions, today_str):
        """检查持仓整体预警"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 读取今日模拟净值
            nav_df = self.store.df(
                f"SELECT nav FROM backtest_nav WHERE date = '{today_str}' ORDER BY updated_at DESC LIMIT 1"
            )
            yesterday_str = _get_yesterday_str(today_str)
            nav_yesterday_df = self.store.df(
                f"SELECT nav FROM backtest_nav WHERE date = '{yesterday_str}' ORDER BY updated_at DESC LIMIT 1"
            )

            if nav_df.empty:
                return

            today_nav = float(nav_df.iloc[0]['nav'])
            yesterday_nav = float(nav_yesterday_df.iloc[0]['nav']) if not nav_yesterday_df.empty else 1.0
            cumulative_return = today_nav - 1.0
            daily_return = (today_nav - yesterday_nav) / yesterday_nav if yesterday_nav else 0

            # 回撤检查（从峰值回撤）
            peak_nav = self.store.df(
                f"SELECT MAX(nav) as peak FROM backtest_nav WHERE date <= '{today_str}'"
            )
            peak = float(peak_nav.iloc[0]['peak']) if not peak_nav.empty else 1.0
            drawdown = (peak - today_nav) / peak if peak else 0

            alerts = []

            if drawdown >= self.portfolio_drawdown_threshold:
                msg = f"持仓整体回撤已达 {drawdown*100:.2f}%（峰值={peak:.4f}，当前={today_nav:.4f}），建议检查止损"
                alerts.append({'type': 'DRAWDOWN_ALERT', 'message': msg, 'drawdown': round(drawdown, 4)})
                print(f"[{now_str}] ⚠️ 回撤预警: {msg}")

            if cumulative_return >= self.portfolio_gain_threshold:
                msg = f"持仓整体收益已达 {cumulative_return*100:.2f}%，注意止盈"
                alerts.append({'type': 'TAKE_PROFIT_ALERT', 'message': msg, 'gain': round(cumulative_return, 4)})
                print(f"[{now_str}] 💰 止盈预警: {msg}")

            if alerts:
                self._save_alerts(alerts)

        except Exception as e:
            print(f"[{now_str}] 组合预警检查失败: {e}")

    def check_market_alerts(self):
        """检查大盘异常"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today_str = date.today().strftime("%Y%m%d")

        try:
            # 用沪深300指数代表大盘（简化：实际可用000300.SH）
            index_code = '000300.SH'
            index_df = self.store.df(
                f"""
                SELECT trade_date, ts_code, close,
                       (close - pre_close) / pre_close as change_pct
                FROM stock_daily
                WHERE ts_code = '{index_code}' AND trade_date = '{today_str}'
                ORDER BY trade_date DESC LIMIT 1
                """
            )
            if index_df.empty:
                return

            change_pct = float(index_df.iloc[0]['change_pct']) / 100.0

            if abs(change_pct) >= 0.02:
                direction = "大涨" if change_pct > 0 else "大跌"
                msg = f"大盘指数（沪深300）{direction} {abs(change_pct)*100:.2f}%，市场异常波动，关注风险"
                print(f"[{now_str}] 📈 大盘预警: {msg}")
                self._save_alerts([{'type': 'MARKET_ALERT', 'message': msg, 'change_pct': round(change_pct, 4)}])
        except Exception as e:
            print(f"[{now_str}] 大盘检查失败: {e}")

    def _save_alerts(self, alerts):
        """保存预警记录"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            for alert in alerts:
                try:
                    self.store.execute(
                        """
                        INSERT INTO alerts (alert_type, message, created_at)
                        VALUES (?, ?, ?)
                        """,
                        [alert.get('type', ''), alert.get('message', ''), now_str]
                    )
                except Exception:
                    # 表可能不存在，跳过
                    pass
        except Exception as e:
            print(f"[{now_str}] 保存预警失败: {e}")


def _get_yesterday_str(today_str):
    """获取昨日字符串"""
    from datetime import datetime, timedelta
    try:
        d = datetime.strptime(today_str, '%Y%m%d') - timedelta(days=1)
        return d.strftime('%Y%m%d')
    except Exception:
        return today_str


def job_market_watcher():
    """调度器入口：启动盘中监控（后台线程）"""
    watcher = MarketWatcher(check_interval=1800)  # 30分钟
    watcher.start()
    return watcher
