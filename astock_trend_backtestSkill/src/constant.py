"""
常量定义模块
包含所有因子名称、市场代码等常量
"""

# ==================== 因子名称常量 ====================

# 趋势动量因子 (9个)
TREND_FACTORS = [
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
    'acceleration_20', 'momentum_volume_corr_20', 'price_relative_20', 'high_low_ratio_20'
]

# 技术指标因子 (19个)
TECH_FACTORS = [
    'ma5_ma20_cross', 'trend_strength_rsq', 'trend_slope', 'supertrend',
    'ichimoku_a', 'ichimoku_b', 'ichimoku_cloud',
    'rsi_14', 'rsi_28', 'macd', 'macd_signal', 'kdj_k', 'kdj_j',
    'bollinger_position', 'bollinger_bandwidth', 'cci_20', 'williams_r', 'adx_14', 'atr_20'
]

# 波动率因子 (7个)
VOL_FACTORS = [
    'volatility_20', 'volatility_60', 'volatility_ratio',
    'max_drawdown_20', 'max_drawdown_60', 'downside_volatility', 'volatility_skew'
]

# 成交量因子 (7个)
VOLUME_FACTORS = [
    'volume_ratio_20', 'volume_ratio_60', 'volume_ma5_crossover',
    'amount_ma20', 'vol_price_divergence', 'turnover_rate', 'turnover_rate_ma5'
]

# 基本面因子 (15个)
FUNDAMENTAL_FACTORS = [
    'pe', 'pb', 'ps', 'pcf', 'roe', 'roa', 'gross_margin', 'net_margin',
    'revenue_growth', 'profit_growth', 'operating_cash_flow', 'debt_to_equity',
    'earnings_yield', 'book_to_market', 'f_score'
]

# 北向资金因子 (4个)
HKT_FACTORS = [
    'hkt_hold_ratio', 'hkt_hold_ratio_change_5', 'hkt_hold_ratio_change_20', 'hkt_net_flow_20'
]

# 复合因子 (5个)
COMPOSITE_FACTORS = [
    'composite_momentum', 'quality_momentum', 'value_momentum',
    'liquidity_adjusted_momentum', 'alpha_factor_pca'
]

# 所有因子汇总 (9+19+7+7+15+4+5 = 66个)
ALL_FACTORS = (
    TREND_FACTORS + TECH_FACTORS + VOL_FACTORS + VOLUME_FACTORS +
    FUNDAMENTAL_FACTORS + HKT_FACTORS + COMPOSITE_FACTORS
)

# ==================== 市场代码常量 ====================

# 主板包含代码
MAIN_BOARD_INCLUDE_CODES = ('600', '601', '603', '000', '001')

# 排除代码（科创板、创业板、北交所等）
MAIN_BOARD_EXCLUDE_CODES = ('688', '300', '8', '200')

# 指数代码
BENCHMARK_CODES = {
    '沪深300': '000300.SH',
    '中证500': '000905.SH',
    '中证1000': '000852.SH',
    '上证指数': '000001.SH',
    '深证成指': '399001.SZ',
}

# ==================== 交易相关常量 ====================

# 交易日历路径
TRADE_CALENDAR_PATH = 'data/trade_calendar.csv'

# 涨停跌停判断阈值
LIMIT_UP_THRESHOLD = 0.0995  # 接近10%涨幅
LIMIT_DOWN_THRESHOLD = -0.0995  # 接近10%跌幅

# 停牌判断
SUSPENDED_THRESHOLD = 0  # 成交量为0视为停牌

# ==================== 数据库常量 ====================

# 数据库路径
DEFAULT_DB_PATH = 'data/astock.duckdb'

# 表名
TABLE_STOCK_LIST = 'stock_list'
TABLE_STOCK_DAILY = 'stock_daily'
TABLE_FUNDAMENTALS = 'fundamentals'
TABLE_HKT_DATA = 'hkt_data'
TABLE_FACTORS = 'factors'
TABLE_FACTOR_POOL = 'factor_pool'
TABLE_STRATEGY_POOL = 'strategy_pool'
TABLE_TRADES = 'trades'

# ==================== 绩效评估常量 ====================

# 默认无风险利率
DEFAULT_RISK_FREE_RATE = 0.03

# 年化交易日数
TRADING_DAYS_PER_YEAR = 252

# ==================== 因子池管理常量 ====================

# 因子池最大容量
DEFAULT_FACTOR_POOL_SIZE = 100

# IR阈值
IR_PROMOTION_THRESHOLD = 1.0
IR_DEMOTION_THRESHOLD = 0.3
IR_EVICTION_THRESHOLD = 0.3

# IC阈值
IC_MIN_THRESHOLD = 0.02

# ==================== 策略参数常量 ====================

# 默认持仓周期（天）
DEFAULT_HOLDING_PERIODS = [5, 10, 20, 60]

# 默认权重方案
DEFAULT_WEIGHT_SCHEMES = ['equal', 'ic_weighted', 'volatility_inverse']

# 默认止损止盈
DEFAULT_STOP_LOSS = [0.05, 0.10, 0.15]
DEFAULT_TAKE_PROFIT = [0.10, 0.20, 0.30]

# ==================== 回测常量 ====================

# 默认初始资金
DEFAULT_INITIAL_CASH = 10000000

# 默认手续费率
DEFAULT_COMMISSION = 0.0003

# 默认滑点
DEFAULT_SLIPPAGE = 0.001
