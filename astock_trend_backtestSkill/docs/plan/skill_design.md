# A股趋势投资量化因子工厂 — 设计规划文档

**Skill 名称：** `astock_factor_forge_skill`  
**版本：** v1.0.0  
**日期：** 2026-03-29  
**设计者：** 拉姆（Lam）

---

## 1. 背景与目标

### 1.1 业务背景

- **市场**：中国 A 股主板（沪深主板，排除科创板、创业板、北交所）
- **股票池**：~3200 只（600/601/603 + 000/001 开头）
- **频率**：日频选股
- **风格**：趋势投资（Trend Following）
- **目标**：7×24 小时全自动因子挖掘、回测验证、策略优化

### 1.2 核心目标

1. **7×24 小时全自动运转** — 无需人工干预，持续运行
2. **因子自动挖掘** — 从价量数据、财务数据、北向资金中自动发现有效因子
3. **因子自动评估** — IC/IR 分析、分组回测、衰减验证
4. **策略自动生成** — 参数空间自动探索、最优参数组合输出
5. **优胜劣汰** — 因子池/策略池动态排名，差的自动淘汰

---

## 2. 系统架构

### 2.1 模块层次

```
┌──────────────────────────────────────────────────────────────────┐
│                     FactorForge · 量化因子工厂                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ FactorMiner  │───▶│  FactorEval  │───▶│ StrategyGen  │        │
│  │  因子挖掘机   │    │   因子评估器  │    │   策略生成器  │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                   │                   │                  │
│         └───────────────────┴───────────────────┘                  │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────┐         │
│  │               FactorPool · 因子池（DuckDB）           │         │
│  └──────────────────────────────────────────────────────┘         │
│                             │                                      │
│         ┌───────────────────┼───────────────────┐                  │
│         ▼                   ▼                   ▼                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐            │
│  │ Backtester  │     │ Optimizer  │     │ ReportGen  │            │
│  │   回测引擎  │     │   优化器   │     │   报告生成  │            │
│  └────────────┘     └────────────┘     └────────────┘            │
│         │                   │                   │                  │
│         └───────────────────┴───────────────────┘                  │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │               ResultsDB · 结果数据库（DuckDB）         │          │
│  └──────────────────────────────────────────────────────┘          │
│                             │                                      │
│         ┌───────────────────┼───────────────────┐                  │
│         ▼                   ▼                   ▼                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐             │
│  │ Scheduler   │     │  Monitor   │     │  Alerter   │             │
│  │   任务调度器  │     │   监控面板  │     │   告警通知  │             │
│  └────────────┘     └────────────┘     └────────────┘             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 职责 | 核心类/函数 |
|------|------|-------------|
| **StockPoolFilter** | A股主板股票池过滤 | `AShareMainBoardFilter` |
| **DataMgr** | 数据获取、清洗 | `TushareFetcher`, `AKShareFetcher`, `DataCleaner` |
| **DuckDBStore** | 数据持久化、SQL查询 | `DuckDBStore`, `TableManager` |
| **FactorMiner** | 自动因子挖掘 | `TechnicalFactorMiner`, `FundamentalFactorMiner`, `CrossSectionFactorMiner` |
| **FactorEval** | 因子评估 | `ICEvaluator`, `IRAnalyzer`, `GroupBacktester`, `DecayAnalyzer` |
| **StrategyGen** | 策略自动生成 | `SignalBuilder`, `ParameterSpace`, `ConditionGenerator` |
| **Optimizer** | 参数优化 | `GridSearchOptimizer`, `BayesianOptimizer`, `GAOptimizer` |
| **Backtester** | 回测执行 | `EventBacktester`, `PortfolioSimulator` |
| **FactorPool** | 因子池管理 | `PoolManager`, `RankingEngine`, `淘汰机制` |
| **Scheduler** | 7×24任务调度 | `APSchedulerRunner`, `TaskQueue` |
| **Monitor** | 监控面板 | `MetricsDashboard`, `ProgressTracker` |
| **Alerter** | 告警通知 | `FeishuBot`, `TelegramBot` |
| **BacktestOrchestrator** | 流程编排 | `PipelineController`, `StateManager` |

---

## 3. 功能流程

### 3.1 每日流水线（Daily Pipeline）

```
00:00 ──▶ 数据更新（下载最新行情 + 财务数据）
01:00 ──▶ 因子挖掘（50-200个新因子候选）
03:00 ──▶ 因子评估（IC/IR分析）
05:00 ──▶ 策略生成 + 参数优化
07:00 ──▶ 全市场回测（Top因子）
09:00 ──▶ 新策略入场（小资金模拟）
11:00 ──▶ 上午盘数据分析
13:00 ──▶ 下午盘跟踪
15:30 ──▶ 日终清算 + 报告生成
16:00 ──▶ 北向资金数据更新
18:00 ──▶ 因子池重排（淘汰差的）
21:00 ──▶ 夜盘期货数据接入（如有）
23:00 ──▶ 日报生成
循环  ──▶ 回到 00:00
```

### 3.2 因子挖掘流程

```
原始数据
    │
    ├── 技术因子 ──▶ 随机组合 ──▶ 50-100个候选
    ├── 基本面因子 ──▶ 两两/三三组合 ──▶ 30-50个候选
    ├── 量价因子 ──▶ 统计关系 ──▶ 20-30个候选
    └── 复合因子 ──▶ PCA/IC提取 ──▶ 10-20个候选
    │
    ▼
因子评估（IC < 0.02 淘汰）
    │
    ▼
进入因子池（排名）
```

---

## 4. 因子体系（50+ 因子）

### 4.1 价格动量类

| 因子名称 | 描述 | 计算方法 |
|----------|------|----------|
| `momentum_5/10/20/60/120/250` | 多周期动量 | 收益率 |
| `acceleration_20` | 动量加速度 | 动量变化率 |
| `momentum_volume_corr_20` | 量价相关性 | Corr(return, volume) |
| `price_relative_20` | 相对强弱 | 个股收益/市场收益 |
| `high_low_ratio_20` | 高低价比 | high/low |

### 4.2 趋势识别类

| 因子名称 | 描述 |
|----------|------|
| `ma5_ma20_cross` | 均线金叉 |
| `trend_strength_rsq` | R²趋势强度 |
| `trend_slope` | 趋势斜率 |
| `supertrend` | SuperTrend |
| `ichimoku_a/b/cloud` | 一目均衡表 |

### 4.3 技术指标类

| 因子名称 | 描述 |
|----------|------|
| `rsi_14/28` | RSI |
| `macd/macd_signal` | MACD |
| `kdj_k/kdj_j` | KDJ |
| `bollinger_position/bandwidth` | 布林带 |
| `cci_20` | CCI |
| `williams_r` | 威廉指标 |
| `adx_14` | ADX |
| `atr_20` | ATR |

### 4.4 波动率类

| 因子名称 | 描述 |
|----------|------|
| `volatility_20/60` | 波动率 |
| `downside_volatility` | 下行波动率 |
| `max_drawdown_20/60` | 最大回撤 |

### 4.5 成交量类

| 因子名称 | 描述 |
|----------|------|
| `volume_ratio_20/60` | 量比 |
| `turnover_rate` | 换手率 |
| `obv` | 能量潮 |

### 4.6 估值基本面类

| 因子名称 | 描述 |
|----------|------|
| `pe/pb/ps/pcf` | 估值指标 |
| `roe/roa` | 盈利能力 |
| `gross/net_margin` | 利润率 |
| `revenue/profit_growth` | 成长能力 |

### 4.7 北向资金类

| 因子名称 | 描述 |
|----------|------|
| `hkt_hold_ratio` | 北向持股占比 |
| `hkt_hold_change_5/20` | 北向持股变化 |

### 4.8 复合/机器学习因子

| 因子名称 | 描述 |
|----------|------|
| `composite_momentum` | 复合动量 |
| `quality_momentum` | 质量动量 |
| `liquidity_adjusted_momentum` | 流动性调整动量 |
| `alpha_factor_pca` | PCA提取Alpha |

---

## 5. 股票池

### 5.1 A股主板范围

| 板块 | 代码前缀 | 数量（约） |
|------|----------|-----------|
| 上证主板 | 600, 601, 603 | ~1700 |
| 深证主板 | 000, 001 | ~1500 |
| **合计** | | **~3200** |

### 5.2 排除规则

- ❌ 科创板（688xxx）
- ❌ 创业板（300xxx）
- ❌ 北交所（8xxxxx）
- ❌ B股（200xxx）
- ❌ ST/\*ST股
- ❌ 上市不足60日新股
- ❌ 日均成交额<1000万

---

## 6. 配置参数

```yaml
stock_pool:
  include_codes: ['600', '601', '603', '000', '001']
  exclude_codes: ['688', '300', '8', '200']
  exclude_st: true
  exclude_new_stock_days: 60
  min_daily_amount: 10000000  # 1000万

backtest:
  start_date: "20200101"
  end_date: "20231231"
  initial_cash: 10000000
  commission: 0.0003
  slippage: 0.001
  benchmark: "000300.SH"

factor_mining:
  candidates_per_round: 100
  min_ic_threshold: 0.02
  min_ir_threshold: 0.3

factor_pool:
  max_size: 100
  promotion_threshold_ir: 1.0
  demotion_threshold_ir: 0.3
  eviction_threshold_ir: 0.3

strategy:
  holding_periods: [5, 10, 20, 60]
  weight_schemes: ['equal', 'ic_weighted', 'volatility_inverse']
  stop_loss: [0.05, 0.10, 0.15]
  take_profit: [0.10, 0.20, 0.30]

scheduler:
  data_update: "00:00"
  factor_mining: "01:00"
  factor_eval: "03:00"
  strategy_gen: "05:00"
  backtest: "07:00"
  report: "15:30"
  pool_rebalance: "18:00"
```

---

## 7. DuckDB 表结构

```sql
-- 主板股票列表
CREATE TABLE stock_list (
    ts_code VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    industry VARCHAR,
    market_cap BIGINT
);

-- 日线行情
CREATE TABLE stock_daily (
    ts_code VARCHAR,
    trade_date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    vol DOUBLE, amount DOUBLE, pct_chg DOUBLE,
    limit_up BOOLEAN, limit_down BOOLEAN,
    is_st BOOLEAN, suspended BOOLEAN,
    PRIMARY KEY (ts_code, trade_date)
);

-- 财务数据
CREATE TABLE fundamentals (
    ts_code VARCHAR, ann_date DATE, end_date DATE,
    pe DOUBLE, pb DOUBLE, roe DOUBLE,
    revenue_growth DOUBLE, profit_growth DOUBLE,
    PRIMARY KEY (ts_code, end_date, ann_date)
);

-- 北向资金
CREATE TABLE hkt_data (
    ts_code VARCHAR, trade_date DATE,
    hold_ratio DOUBLE, net_flow_20d DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 因子池
CREATE TABLE factor_pool (
    factor_name VARCHAR, ts_code VARCHAR, trade_date DATE,
    value DOUBLE, zscore DOUBLE, ic DOUBLE, ir DOUBLE,
    rank INT,
    PRIMARY KEY (factor_name, ts_code, trade_date)
);

-- 策略池
CREATE TABLE strategy_pool (
    strategy_id VARCHAR,
    strategy_name VARCHAR,
    factors JSON,
    parameters JSON,
    metrics JSON,  -- sharpe/max_dd/win_rate/return
    rank INT,
    status VARCHAR,  -- active/suspended/evicted
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (strategy_id)
);

-- 交易记录
CREATE TABLE trades (
    trade_id VARCHAR,
    strategy_id VARCHAR,
    ts_code VARCHAR,
    trade_date DATE,
    direction VARCHAR,  -- buy/sell
    price DOUBLE,
    quantity INT,
    amount DOUBLE,
    signal_reason VARCHAR,
    PRIMARY KEY (trade_id)
);
```

---

## 8. 技术栈

| 组件 | 技术 |
|------|------|
| **语言** | Python 3.10+ |
| **数据库** | DuckDB |
| **数据处理** | pandas, numpy |
| **因子计算** | talib, statsmodels, scipy |
| **机器学习** | LightGBM, scikit-learn |
| **优化** | optuna（贝叶斯优化） |
| **调度** | APScheduler |
| **加速** | numba |
| **告警** | 飞书机器人 |
| **报告** | Jinja2 + HTML |

---

## 9. 验收标准

| 指标 | 目标 |
|------|------|
| 全天候运转 | 7×24 小时无人值守 |
| 因子挖掘速度 | 每轮 50-100 个候选因子 |
| 回测速度 | 全市场 3200 只，3 年，< 10 分钟 |
| 因子存活率 | IC_IR > 0.3 才进入因子池 |
| 策略淘汰 | 夏普 < 0.5 自动淘汰 |
| 报告输出 | 每日 HTML 报告自动生成 |

---

## 10. 目录结构

```
astock_factor_forge/
├── config/
│   └── settings.yaml
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── stock_pool.py        # 股票池过滤
│   │   ├── data_manager.py      # 数据获取
│   │   ├── duckdb_store.py      # DuckDB存储
│   │   ├── factor_miner.py     # 因子挖掘
│   │   ├── factor_eval.py      # 因子评估
│   │   ├── strategy_gen.py     # 策略生成
│   │   ├── optimizer.py         # 参数优化
│   │   ├── backtester.py       # 回测引擎
│   │   ├── factor_pool.py      # 因子池管理
│   │   ├── portfolio.py        # 组合管理
│   │   ├── risk_manager.py     # 风险管理
│   │   └── performance.py      # 绩效分析
│   ├── api/
│   │   ├── __init__.py
│   │   └── skill_api.py        # Skill执行接口
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   └── date_utils.py
│   └── constant.py
├── scheduler/
│   └── tasks.py                # APScheduler任务
├── reports/
│   └── templates/              # HTML报告模板
├── data/                       # DuckDB数据库文件
├── logs/                       # 日志文件
├── test/
│   ├── unit/
│   └── integration/
├── docs/plan/
│   ├── skill_design.md
│   └── skill_process.mmd
├── scripts/
│   └── init_db.sql
├── openclaw.json
├── requirements.txt
└── README.md
```
