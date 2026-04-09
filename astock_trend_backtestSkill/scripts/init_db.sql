-- A股因子工厂数据库初始化脚本

-- 股票列表
CREATE TABLE IF NOT EXISTS stock_list (
    ts_code VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    name VARCHAR,
    list_date DATE,
    delist_date DATE,
    industry VARCHAR,
    market_cap BIGINT
);

-- 日线行情
CREATE TABLE IF NOT EXISTS stock_daily (
    ts_code VARCHAR,
    trade_date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    vol DOUBLE, amount DOUBLE, pct_chg DOUBLE,
    limit_up BOOLEAN, limit_down BOOLEAN,
    is_st BOOLEAN, suspended BOOLEAN,
    PRIMARY KEY (ts_code, trade_date)
);

-- 财务数据
CREATE TABLE IF NOT EXISTS fundamentals (
    ts_code VARCHAR,
    ann_date DATE,
    end_date DATE,
    pe DOUBLE, pb DOUBLE, roe DOUBLE,
    revenue_growth DOUBLE, profit_growth DOUBLE,
    PRIMARY KEY (ts_code, end_date, ann_date)
);

-- 北向资金
CREATE TABLE IF NOT EXISTS hkt_data (
    ts_code VARCHAR,
    trade_date DATE,
    hold_ratio DOUBLE,
    net_flow_20d DOUBLE,
    PRIMARY KEY (ts_code, trade_date)
);

-- 因子表
CREATE TABLE IF NOT EXISTS factors (
    factor_name VARCHAR,
    ts_code VARCHAR,
    trade_date DATE,
    value DOUBLE,
    zscore DOUBLE,
    PRIMARY KEY (factor_name, ts_code, trade_date)
);

-- 因子池
CREATE TABLE IF NOT EXISTS factor_pool (
    factor_name VARCHAR PRIMARY KEY,
    avg_ic DOUBLE,
    avg_ir DOUBLE,
    ic_series JSON,
    rank INT,
    status VARCHAR,
    updated_at TIMESTAMP
);

-- 策略池
CREATE TABLE IF NOT EXISTS strategy_pool (
    strategy_id VARCHAR PRIMARY KEY,
    strategy_name VARCHAR,
    factors JSON,
    parameters JSON,
    metrics JSON,
    rank INT,
    status VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- 交易记录
CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR PRIMARY KEY,
    strategy_id VARCHAR,
    ts_code VARCHAR,
    trade_date DATE,
    direction VARCHAR,
    price DOUBLE,
    quantity INT,
    amount DOUBLE,
    signal_reason VARCHAR
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_daily_code_date ON stock_daily(ts_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_factors_code_date ON factors(ts_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_factors_name ON factors(factor_name);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
