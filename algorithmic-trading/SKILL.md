---
name: algorithmic-trading
description: "基于 PyBroker 的算法交易回测框架，支持技术指标策略和机器学习策略集成。用于回测、策略验证时使用。"
---

# PyBroker Backtest - 算法交易回测框架

## 描述

基于 PyBroker 的简化版算法交易回测工具。支持技术指标策略、机器学习策略集成和快速回测验证。

## 功能

- 📊 技术指标策略（RSI、MACD、布林带等）
- 🤖 机器学习模型集成（scikit-learn）
- 🔄 Walkforward 前向验证
- 📉 风险管理（止损、止盈）
- 💾 智能缓存加速
- 📈 回测报告生成

## 使用方法

### 基础策略

```bash
# 运行示例策略
python scripts/backtest.py --strategy basic --symbols AAPL MSFT --start 2022-01-01 --end 2023-12-31

# 机器学习策略
python scripts/backtest.py --strategy ml --symbols AAPL --start 2022-01-01 --end 2023-12-31

# 生成报告
python scripts/backtest.py --strategy basic --symbols AAPL --report output/report.html
```

### Python API

```python
from scripts.backtest import BacktestRunner

runner = BacktestRunner()

# 运行综合策略A（5指标）
result = runner.run_comprehensive_strategy(
    symbols=['AAPL', 'MSFT'],
    start_date='2022-01-01',
    end_date='2023-12-31',
    rsi_buy=30, rsi_sell=70,
    vol_multiplier=1.5,
    stop_loss_pct=3
)

# 运行基础策略
result = runner.run_basic_strategy(
    symbols=['AAPL', 'MSFT'],
    start_date='2022-01-01',
    end_date='2023-12-31'
)

print(result.metrics_df)
```

## 安装

```bash
pip install -r requirements.txt
```

## 依赖

- lib-pybroker>=1.2.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- yfinance>=0.2.0

## 策略示例

### 突破策略
- 20 日最高价突破买入
- 持有 5 天
- 2% 止损

### RSI 策略
- RSI < 30 超卖买入
- RSI > 70 超买卖出
- 动态止损

### 综合策略A（推荐）
**结合全部5个核心指标，严格筛选：**
- RSI < 30（超卖）
- MACD 金叉（MACD线上穿信号线）
- 价格触及布林带下轨
- 成交量 > 20日均量 × 1.5（放量确认）
- 20日最高价突破（趋势确认）

**买入：5个条件同时满足**
**卖出：RSI > 70（超买）或触发止损

### 参数穷举优化

```bash
# 快速模式（减少参数组合）
python scripts/optimizer.py --symbols AAPL --start 2022-01-01 --end 2023-12-31 --quick

# 完整模式（所有参数组合）
python scripts/optimizer.py --symbols AAPL TSLA --start 2022-01-01 --end 2023-12-31

# 输出Top 30
python scripts/optimizer.py --symbols AAPL --start 2022-01-01 --end 2023-12-31 --top 30
```

**优化参数范围：**
- RSI: 买入阈值(20-40) / 卖出阈值(60-80)
- MACD: 快线(8-15) / 慢线(20-30) / 信号线(7-11)
- 布林带: 周期(15-25) / 标准差(1.5-3.0)
- 成交量: 周期(10-25) / 倍数(1.0-2.5)
- 突破: 周期(15-25)
- 止损(2-7%) / 止盈(5-20%)

### 机器学习策略
- 随机森林分类器
- 特征：RSI、MACD、成交量
- Walkforward 验证

## 输出格式

### 回测指标
```
Total Return: 25.3%
Sharpe Ratio: 1.45
Max Drawdown: -8.2%
Win Rate: 62%
```

## 注意事项

- 首次运行会下载历史数据
- 建议使用 3 年以上数据
- 仅供学习研究，不构成投资建议

## 作者

派蒙 (Paimon) - 基于 PyBroker 二次开发

## 许可证

Apache 2.0 License
