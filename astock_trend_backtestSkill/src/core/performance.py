import pandas as pd
import numpy as np
from typing import Dict, List


class PerformanceAnalyzer:
    """绩效分析器"""
    
    @staticmethod
    def calc_sharpe(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calc_max_drawdown(nav: pd.Series) -> float:
        """计算最大回撤"""
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        return drawdown.min()
    
    @staticmethod
    def calc_calmar(nav: pd.Series, periods_per_year: int = 252) -> float:
        """计算卡玛比率"""
        annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1 if len(nav) > 0 else 0
        max_dd = abs(PerformanceAnalyzer.calc_max_drawdown(nav))
        return annual_return / max_dd if max_dd > 0 else 0
    
    @staticmethod
    def calc_win_rate(trades: List[Dict]) -> float:
        """计算胜率"""
        if not trades:
            return 0
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trades)
    
    @staticmethod
    def calc_profit_loss_ratio(trades: List[Dict]) -> float:
        """计算盈亏比"""
        wins = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losses = [abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0]
        if not wins or not losses:
            return 0
        return np.mean(wins) / np.mean(losses)
    
    def analyze(self, trades: List[Dict], nav_series: pd.Series) -> Dict:
        """完整绩效分析"""
        returns = nav_series.pct_change().dropna()
        
        # 年度收益
        nav_df = pd.DataFrame({'nav': nav_series})
        nav_df['year'] = pd.to_datetime(nav_df.index).year
        annual_returns = nav_df.groupby('year')['nav'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else 0
        )
        
        return {
            'total_return': (nav_series.iloc[-1] / nav_series.iloc[0]) - 1 if len(nav_series) > 0 else 0,
            'annual_return': (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252 / len(nav_series)) - 1 if len(nav_series) > 100 else 0,
            'sharpe_ratio': self.calc_sharpe(returns),
            'max_drawdown': self.calc_max_drawdown(nav_series),
            'calmar_ratio': self.calc_calmar(nav_series),
            'win_rate': self.calc_win_rate(trades),
            'profit_loss_ratio': self.calc_profit_loss_ratio(trades),
            'total_trades': len(trades),
            'annual_returns': annual_returns.to_dict(),
        }
