# A股趋势投资量化因子工厂

7×24小时全自动因子挖掘与策略优化系统。

## 功能

- 50+ 因子自动挖掘与评估
- IC/IR 分析、分组回测、衰减验证
- 策略自动生成与参数优化
- 7×24 小时无人值守运行
- 因子池/策略池优胜劣汰

## 数据源

- **baostock**（免费，无需 API Token）
- 覆盖 A股全量主板股票（600/601/603/000/001）
- 日线数据：开高低收成交量

## 项目结构

```
astock_trend_backtestSkill/
├── config/settings.yaml       # 完整配置（回测参数/因子参数/调度时间）
├── requirements.txt            # Python 依赖
├── data/                       # DuckDB 数据库（自动创建）
│   └── astock_full.duckdb     # 股票日线 + 因子数据
├── src/
│   ├── constant.py             # 50+ 因子常量定义
│   ├── api/skill_api.py        # Skill 统一入口
│   └── core/
│       ├── duckdb_store.py     # DuckDB 存储引擎
│       ├── data_manager.py     # baostock 数据获取
│       ├── backtester.py       # 回测引擎 + 因子批量计算
│       ├── factor_miner.py     # 因子挖掘（50+ 因子计算）
│       ├── factor_eval.py      # IC/IR 评估、分组回测
│       ├── strategy_gen.py      # 策略生成器
│       ├── optimizer.py        # 贝叶斯 + 网格搜索
│       ├── factor_pool.py      # 因子池管理（优胜劣汰）
│       └── performance.py      # 绩效分析
├── scheduler/tasks.py          # APScheduler 调度器
└── scripts/
    └── fetch_all_mainboard.py  # 全量股票数据下载脚本
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from src.api.skill_api import execute

# 1. 初始化（更新股票列表）
execute({'action': 'init'})

# 2. 完整流水线（初始化 + 数据更新 + 因子挖掘 + 评估 + 策略生成 + 回测）
execute({'action': 'run_full_pipeline'})

# 3. 单次回测
execute({'action': 'run_backtest', 'start_date': '20200101', 'end_date': '20231231'})

# 4. 评估因子
execute({'action': 'evaluate_factors', 'factor_names': ['momentum_20', 'volatility_20']})

# 5. 生成策略
execute({'action': 'generate_strategies'})
```

## 调度运行

```bash
python scheduler/tasks.py
```

调度时间表（可在 `config/settings.yaml` 中修改）：
- `00:00` — 数据更新
- `01:00` — 因子挖掘
- `03:00` — 因子评估
- `05:00` — 策略生成
- `07:00` — 回测
- `18:00` — 因子池重平衡

## 数据库

- 路径：`data/astock_full.duckdb`
- 主要表：`stock_daily`（316万条日线）、`factors`（因子值）
- 可用 `duckdb-cli data/astock_full.duckdb` 直接查询

## 策略说明

- **默认策略**：动量 Top20，每年1月首个交易日再平衡
- **持仓逻辑**：动量转负或排名下降超过阈值则卖出
- **选股范围**：A股主板（600/601/603/000/001），排除ST

## 作者

拉姆（Lam）@ 2026-03-29
