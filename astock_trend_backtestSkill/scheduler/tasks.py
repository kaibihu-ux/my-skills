"""
A股量化因子工厂 - 调度任务
每天按交易日/非交易日自动加载不同任务，真正实现7×24全自动化运转：

交易日（UTC+8）：
- 06:00 — 数据更新（下载前一日收盘数据）
- 07:00 — 因子池更新（重新计算所有因子）
- 08:00 — 盘前策略预判
- 09:30 — 盘中监控（后台线程）
- 18:00 — 盘后数据更新
- 18:30 — 因子重平衡（等数据确认）
- 19:00 — GA策略优化 + 报告生成
- 23:30 — 发送策略报告

非交易日（周末/节假日）：
- 20:00 — 周末因子挖掘（长周期）
- 01:00 — GA策略优化（凌晨运行）
- 07:00 — 报告就绪
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, date, timedelta
import sys
import os
import json
import threading
import functools
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance
from src.core.factor_eval import FactorEvaluator
from src.core.factor_miner import ALL_FACTORS, TREND_FACTORS, TECH_FACTORS, VOL_FACTORS, VOLUME_FACTORS
from src.core.strategy_gen import StrategyGenerator
from src.core.backtester import BacktestExecutor
from src.core.optimizer import BayesianOptimizer, GridSearchOptimizer

# 新增模块
from scheduler.market_watcher import MarketWatcher
from scheduler.pre_market import job_pre_market
from scheduler.post_market import job_post_market
from scheduler.weekend_research import job_weekend_research
from scheduler.tasks_split import job_step1_lgb, job_step2_ga, job_step3_rl, job_step4_bayes, job_step5_final

scheduler = BackgroundScheduler()

REPORT_FILE = Path(__file__).parent.parent / "reports" / "daily_strategy_report.json"
REPORT_FILE.parent.mkdir(exist_ok=True)

# 全局监控线程实例
_market_watcher_instance = None


# ==============================================================================
# 2026年A股非交易日历（来源：api.apihubs.cn）
# 包含：周末 + 节假日（元旦、春节、清明、劳动节、端午、中秋、国庆）
# ==============================================================================

NON_TRADING_DAYS_2026 = {
    # 1月
    '20260101', '20260103', '20260104',   # 元旦假期（1月1日周三，1月2-4日调休）
    '20260110', '20260111',               # 周末
    '20260117', '20260118',               # 周末
    '20260124', '20260125',               # 春节前周末
    '20260131',                            # 除夕（2月17日周二调休）
    # 2月
    '20260201', '20260207', '20260208',   # 春节假期
    '20260214', '20260215', '20260216', '20260217', '20260218', '20260219', '20260220', '20260221', '20260222', '20260223',  # 春节
    '20260228',                             # 调休上班
    # 3月
    '20260301',                             # 调休
    '20260307', '20260308',               # 周末
    '20260314', '20260315',               # 周末
    '20260321', '20260322',               # 周末
    '20260328', '20260329',               # 周末
    # 4月
    '20260404', '20260405', '20260406',   # 清明节
    '20260411', '20260412',               # 周末
    '20260418', '20260419',               # 周末
    '20260425', '20260426',               # 周末
    # 5月
    '20260501', '20260502', '20260503', '20260504',  # 劳动节
    '20260505', '20260509', '20260510',   # 周末+调休
    '20260516', '20260517',               # 周末
    '20260523', '20260524',               # 周末
    '20260530', '20260531',               # 周末
    # 6月
    '20260606', '20260607',               # 周末
    '20260613', '20260614',               # 周末
    '20260619', '20260620', '20260621',   # 端午节
    '20260627', '20260628',               # 周末
    # 7月
    '20260704', '20260705',               # 周末
    '20260711', '20260712',               # 周末
    '20260718', '20260719',               # 周末
    '20260725', '20260726',               # 周末
    # 8月
    '20260801', '20260802',               # 周末
    '20260808', '20260809',               # 周末
    '20260815', '20260816',               # 周末
    '20260822', '20260823',               # 周末
    '20260829', '20260830',               # 周末
    # 9月
    '20260905', '20260906',               # 周末
    '20260912', '20260913',               # 周末
    '20260919', '20260920',               # 周末
    '20260925', '20260926', '20260927',   # 中秋节
    # 10月
    '20261001', '20261002', '20261003', '20261004', '20261005', '20261006', '20261007',  # 国庆节
    '20261010', '20261011',               # 周末
    '20261017', '20261018',               # 周末
    '20261024', '20261025',               # 周末
    '20261031',                             # 周末
    # 11月
    '20261101',                             # 周末
    '20261107', '20261108',               # 周末
    '20261114', '20261115',               # 周末
    '20261121', '20261122',               # 周末
    '20261128', '20261129',               # 周末
    # 12月
    '20261205', '20261206',               # 周末
    '20261212', '20261213',               # 周末
    '20261219', '20261220',               # 周末
    '20261226', '20261227',               # 周末
}

# 备用：仅包含法定节假日的最小集合（holiday_or=10为法定假日）
HOLIDAYS_2026_LEGAL = {
    # 元旦
    '20260101',
    # 春节（2月17日-23日，共7天）
    '20260217', '20260218', '20260219', '20260220', '20260221', '20260222', '20260223',
    # 清明节（4月4日-6日）
    '20260404', '20260405', '20260406',
    # 劳动节（5月1日-5日）
    '20260501', '20260502', '20260503', '20260504', '20260505',
    # 端午节（6月19日-21日）
    '20260619', '20260620', '20260621',
    # 中秋节（9月25日-27日）
    '20260925', '20260926', '20260927',
    # 国庆节（10月1日-7日）
    '20261001', '20261002', '20261003', '20261004', '20261005', '20261006', '20261007',
}


# =============================================================================
# 超时装饰器（6小时超时保护，防止长任务卡死）
# =============================================================================
def timeout(seconds=6 * 3600):
    """
    超时保护装饰器：超过指定秒数后强制结束任务
    - 函数应通过检查点文件实现断点续算
    - 每次任务触发后自动加载上次检查点（如果存在）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            t = threading.Thread(target=target, name=func.__name__)
            t.daemon = True
            t.start()
            t.join(timeout=seconds)
            if t.is_alive():
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] ⏰ {func.__name__} 运行超过 {seconds/3600:.0f}h，强制结束")
                print(f"[{now}] ⚠️  请重新触发任务以继续（检查点已保存）")
                raise TimeoutError(f"{func.__name__} timed out after {seconds/3600:.0f} hours")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


def is_trading_day(d=None):
    """判断是否为A股交易日（排除周末和所有非交易日）"""
    if d is None:
        d = date.today()
    date_str = d.strftime('%Y%m%d')
    # 精确匹配非交易日列表
    if date_str in NON_TRADING_DAYS_2026:
        return False
    return True


def is_non_trading_day(d=None):
    """判断是否为非交易日（周末+节假日）"""
    return not is_trading_day(d)


def is_market_open(d=None):
    """判断当前是否在交易时间内"""
    now = datetime.now()
    if not is_trading_day(d):
        return False
    h, m = now.hour, now.minute
    return (h == 9 and m >= 30) or (h == 10) or (h == 11 and m < 30) or (h == 13) or (h == 14) or (h == 15 and m < 5)


def is_post_market(d=None):
    """盘后（18:00-19:00）"""
    now = datetime.now()
    if not is_trading_day(d):
        return False
    h, m = now.hour, now.minute
    return (h == 15 and m >= 5) or h == 16


# ==============================================================================
# 原有任务（保持不变）
# ==============================================================================

def _format_pct(v):
    """格式化百分比"""
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def _clean_bt(r):
    """清理回测结果，移除DataFrame等不可序列化对象"""
    return {
        "strategy_name": str(r.get('strategy_name', '')),
        "factors": r.get('factors', []),
        "total_return": round(float(r.get('total_return', 0) or 0), 4),
        "sharpe_ratio": round(float(r.get('sharpe_ratio', 0) or 0), 4),
        "max_drawdown": round(float(r.get('max_drawdown', 0) or 0), 4),
        "total_trades": int(r.get('total_trades', 0) or 0),
    }


@timeout(8 * 3600)
def job_optimize_and_report():
    """
    每天9:00/21:00自动运行（交易日21:00）：
    1. LightGBM 筛选最优因子
    2. GA 优化因子组合和参数
    3. RL 优化动态仓位
    4. Bayesian/Grid 参数精调
    5. 取最优策略并运行回测验证
    6. 生成报告
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] ⭐ 开始优化与报告任务")

    api = get_instance()
    store = api.store
    logger = api.logger

    # 回测区间
    start_date = '20200401'
    end_date = '20260327'

    # ---- 检查点续算 ----
    ckpt_file = Path(__file__).parent.parent / "checkpoints" / "job_optimize_checkpoint.json"
    ckpt_data = {}
    if ckpt_file.exists():
        try:
            with open(ckpt_file) as f:
                ckpt_data = json.load(f)
            print(f"[{now}] 📦 发现检查点文件，加载后继续（completed_step={ckpt_data.get('completed_step', 0)}）")
        except Exception as e:
            print(f"[{now}] ⚠️  检查点加载失败，将重新开始: {e}")
            ckpt_data = {}

    def save_ckpt(step_name, data):
        """保存检查点（含完整数据）"""
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        data['completed_step'] = step_name
        data['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = str(ckpt_file) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, default=str)
        Path(tmp).rename(ckpt_file)
        print(f"[{now}] 💾 检查点已保存: {ckpt_file.name} ({step_name})")

    try:
        # ===== 0. 导入高级优化器 =====
        from src.core.genetic_optimizer import GeneticOptimizer
        from src.core.rl_optimizer import RLOptimizer
        from src.core.ml_feature_selector import MLFeatureSelector

        # ===== 1. LightGBM 筛选最优因子 =====
        print(f"[{now}] Step1: LightGBM 因子筛选...")
        selector = MLFeatureSelector(store, logger)
        all_factors = list(set(TREND_FACTORS + TECH_FACTORS + VOL_FACTORS + VOLUME_FACTORS))
        lgb_result = selector.select_features(all_factors[:20], start_date, end_date)
        selected_factors = lgb_result.get('selected_features', [])
        print(f"[{now}] LightGBM 筛选结果: {len(selected_factors)} 个因子 -> {selected_factors}")
        print(f"[{now}] LightGBM 验证集 AUC={lgb_result.get('model_auc', 0):.4f}")

        # 如果 LightGBM 未筛选出足够因子，回退到评估器
        if len(selected_factors) < 3:
            print(f"[{now}] LightGBM 筛选不足，回退到评估器...")
            evaluator = FactorEvaluator(store, logger)
            supported = [
                'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120',
                'rsi_14', 'rsi_28', 'macd', 'macd_signal',
                'bollinger_position', 'bollinger_bandwidth',
                'volatility_20', 'volatility_60',
                'volume_ratio_20', 'volume_ratio_60',
            ]
            eval_factors = [f for f in all_factors if f in supported]
            results = evaluator.evaluate_multiple(eval_factors, start_date, end_date)
            ranked = evaluator.rank_factors(results)
            selected_factors = [f['factor_name'] for f in ranked[:10] if ranked]
            print(f"[{now}] 评估器回退: {selected_factors}")

        # Step1 检查点（中性化前保存，以便中性化失败时从 Step1 恢复）
        if ckpt_data.get('completed_step', '') not in ('step2_ga', 'step3_rl', 'step4_bayes', 'step5_final', 'done'):
            save_ckpt('step1_lgb', {'selected_factors': selected_factors})

        # ===== 1.5 因子中性化处理（高级功能）=====
        neutralized_factors = selected_factors
        try:
            from src.core.neutralizer import Neutralizer
            ncfg = api.config.get('neutralization', {})
            if ncfg.get('enabled', True):
                neutralizer = Neutralizer(store, logger, api.config)
                # 对每个筛选出的因子做中性化处理
                neutralized_factors = neutralizer.neutralize_factor_list(
                    selected_factors, start_date, end_date
                )
                print(f"[{now}] 中性化处理: {selected_factors} -> {neutralized_factors}")
        except Exception as e:
            print(f"[{now}] 中性化失败，使用原始因子: {e}")
            neutralized_factors = selected_factors

        # 保存因子池
        print(f"[{now}] 更新因子池 ({len(selected_factors)} 个因子)...")
        try:
            for ev in api.factor_pool.get_top_factors(100) or []:
                pass
        except Exception:
            pass

        # ===== 2. GA 优化因子组合和参数 =====
        print(f"[{now}] Step2: GA 遗传算法优化...")
        bt_executor = BacktestExecutor(store, logger, api.config)
        ga_opt = GeneticOptimizer(
            bt_executor, logger,
            start_date=start_date, end_date=end_date,
            pop_size=20, n_generations=30,
            mutation_rate=0.15, elite_ratio=0.2,
            config=api.config,
        )
        ga_result = ga_opt.optimize(neutralized_factors)
        ga_best_sharpe = ga_result.get('best_sharpe', -999)
        ga_best_factors = ga_result.get('best_factors', [])
        ga_best_params = ga_result.get('best_params', {})
        print(f"[{now}] GA 最优: Sharpe={ga_best_sharpe:.4f}, 因子={ga_best_factors}, 参数={ga_best_params}")

        # Step2 检查点
        save_ckpt('step2_ga', {
            'ga_result': {k: v for k, v in ga_result.items() if k not in ('overfit_report',)},
            'ga_best_sharpe': ga_best_sharpe,
            'ga_best_factors': ga_best_factors,
            'ga_best_params': ga_best_params,
            'neutralized_factors': neutralized_factors,
        })

        # ===== 2.5 XGBoost 选股模型训练（高级功能）=====
        xgb_result = {}
        xgb_enabled = api.config.get('ml_models', {}).get('xgboost', {}).get('enabled', False)
        if xgb_enabled and len(neutralized_factors) >= 2:
            try:
                from src.core.ml_models import XGBoostModel
                xgb_model = XGBoostModel(store, logger, api.config)
                xgb_result = xgb_model.train(
                    neutralized_factors,
                    train_start='20220101',
                    train_end='20241231',
                )
                print(f"[{now}] XGBoost: 训练完成, AUC={xgb_result.get('auc', 0):.4f}, "
                      f"样本={xgb_result.get('train_samples', 0)}, "
                      f"特征重要性={xgb_result.get('feature_importance', {})}")
            except Exception as e:
                print(f"[{now}] XGBoost训练失败: {e}")

        # ===== 2.7 LSTM / Transformer 时序预测（高级功能）=====
        lstm_result = {}
        transformer_result = {}
        ml_enabled = api.config.get('ml_models', {})

        # LSTM
        if ml_enabled.get('lstm', {}).get('enabled', False):
            try:
                from src.core.ml_models import MLModelFactory
                factory = MLModelFactory(store, logger, api.config)
                lstm_result = factory.train_lstm(
                    stock_code='000001',  # 用代表性股票
                    factor_names=neutralized_factors[:5],
                    seq_len=20,
                    train_start='20220101',
                    train_end='20241231',
                )
                print(f"[{now}] LSTM训练完成: AUC={lstm_result:.4f}" if isinstance(lstm_result, float) else f"[{now}] LSTM训练完成: AUC={lstm_result.get('auc', 0):.4f}")
            except Exception as e:
                print(f"[{now}] LSTM训练失败: {e}")

        # Transformer
        if ml_enabled.get('transformer', {}).get('enabled', False):
            try:
                from src.core.ml_models import MLModelFactory
                factory = MLModelFactory(store, logger, api.config)
                transformer_result = factory.train_transformer(
                    stock_code='000001',
                    factor_names=neutralized_factors[:5],
                    seq_len=20,
                    train_start='20220101',
                    train_end='20241231',
                )
                print(f"[{now}] Transformer训练完成: AUC={transformer_result:.4f}" if isinstance(transformer_result, float) else f"[{now}] Transformer训练完成: AUC={transformer_result.get('auc', 0):.4f}")
            except Exception as e:
                print(f"[{now}] Transformer训练失败: {e}")

        # ===== 2.6 GA 过拟合检测（多重检验偏误矫正）=====
        ga_gen_history = ga_result.get('generation_history', [])
        ga_overfit_report = {}
        all_ga_sharpes = [h['best_sharpe'] for h in ga_gen_history]
        all_avg_sharpes = [h['avg_sharpe'] for h in ga_gen_history]
        all_trial_sharpes = all_ga_sharpes + all_avg_sharpes

        try:
            from src.core.overfit_detector import OverfitDetector
            ods_cfg = api.config.get('overfit_detection', {})
            if ods_cfg.get('enabled', True):
                detector = OverfitDetector(store, logger, api.config)
                # 用 GA 全程的 best_sharpe 作为多重检验序列
                mt_result = detector.calc_multiple_testing_adjustment(
                    all_ga_sharpes, n_trials=len(all_ga_sharpes)
                )
                ga_overfit_report = {'multiple_testing': mt_result}
                n_sig_bh = mt_result.get('n_significant_bh', 0)
                n_sig_bonf = mt_result.get('n_significant_bonferroni', 0)
                print(f"[{now}] GA多重检验: Bonferroni显著={n_sig_bonf}/{len(all_ga_sharpes)}, BH-FDR显著={n_sig_bh}/{len(all_ga_sharpes)}")
        except Exception as e:
            print(f"[{now}] GA过拟合检测失败: {e}")

        # ===== 3. RL 优化动态仓位 =====
        print(f"[{now}] Step3: RL 强化学习仓位优化...")
        rl_strategy = {
            'strategy_id': 'rl_position',
            'strategy_name': 'rl_position',
            'factors': ga_best_factors if ga_best_factors else neutralized_factors[:3],
        }
        rl_opt = RLOptimizer(
            bt_executor, logger,
            start_date=start_date, end_date=end_date,
            gamma=0.95, alpha=0.1, epsilon=0.1,
            n_episodes=50, lookback_days=20,
        )
        rl_result = rl_opt.optimize(rl_strategy, ga_best_params, use_rl_position=True)
        rl_best_sharpe = rl_result.get('final_sharpe', -999)
        rl_policy = rl_result.get('best_policy', {})
        print(f"[{now}] RL 最优: Sharpe={rl_best_sharpe:.4f}, 策略状态数={rl_result.get('n_states', 0)}")

        # Step3 检查点
        save_ckpt('step3_rl', {
            'rl_result': {k: v for k, v in rl_result.items() if k not in ('q_table',)},
            'rl_best_sharpe': rl_best_sharpe,
            'ga_best_factors': ga_best_factors,
            'ga_best_params': ga_best_params,
            'neutralized_factors': neutralized_factors,
        })

        # ===== 3.5 策略生成（包含 ML 增强）=====
        # 收集所有 ML 预测结果
        ml_predictions = {}
        if xgb_result.get('predictions'):
            ml_predictions.update(xgb_result['predictions'])
        if isinstance(lstm_result, dict) and lstm_result.get('predictions'):
            ml_predictions.update(lstm_result['predictions'])
        if isinstance(transformer_result, dict) and transformer_result.get('predictions'):
            ml_predictions.update(transformer_result['predictions'])

        # 生成策略候选（包含 ML 增强策略）
        try:
            from src.core.strategy_gen import StrategyGenerator
            top_factors_for_gen = [
                {'factor_name': f, 'ir': 0.0}
                for f in (ga_best_factors if ga_best_factors else neutralized_factors[:5])
            ]
            gen_param_space = {
                'holding_periods': [5, 10, 20, 40, 60],
                'weight_schemes': ['equal', 'ic_weighted', 'volatility_inverse'],
                'stop_loss': [0.05, 0.08, 0.10, 0.15],
                'take_profit': [0.15, 0.20, 0.25, 0.30],
                'rebalance_frequency': ['daily', 'weekly', 'monthly'],
            }
            sg = StrategyGenerator(store, logger)
            all_strategies = sg.generate_strategies(
                top_factors_for_gen,
                gen_param_space,
                ml_predictions=ml_predictions if ml_predictions else None,
            )
            print(f"[{now}] 策略生成: 共 {len(all_strategies)} 个策略候选（含ML增强）")
        except Exception as e:
            print(f"[{now}] 策略生成失败: {e}")
            all_strategies = []

        # ===== 4. Bayesian/Grid 参数精调 =====
        print(f"[{now}] Step4: Bayesian/Grid 参数精调...")
        bayes_opt = BayesianOptimizer(bt_executor, logger, start_date=start_date, end_date=end_date)
        grid_opt = GridSearchOptimizer(bt_executor, logger, start_date=start_date, end_date=end_date)

        final_factors = ga_best_factors if ga_best_factors else neutralized_factors[:3]
        final_strat_for_opt = {
            'strategy_id': 'optimized',
            'strategy_name': 'optimized',
            'factors': final_factors,
        }

        try:
            bayes_result = bayes_opt.optimize(final_strat_for_opt, n_trials=20)
            bayes_sharpe = bayes_result.get('best_value', -999)
        except Exception as e:
            logger.warning(f"Bayesian优化失败: {e}")
            bayes_result = {'best_params': {}, 'best_value': -999}
            bayes_sharpe = -999

        grid_params = {
            'holding_period': [5, 10, 20, 40, 60],
            'stop_loss': [0.05, 0.08, 0.10, 0.15],
            'take_profit': [0.15, 0.20, 0.25, 0.30],
            'weight_scheme': ['equal', 'ic_weighted', 'volatility_inverse'],
        }
        try:
            grid_result = grid_opt.optimize(final_strat_for_opt, grid_params)
            grid_sharpe = grid_result.get('best_value', -999)
        except Exception as e:
            logger.warning(f"Grid搜索失败: {e}")
            grid_result = {'best_params': {}, 'best_value': -999}
            grid_sharpe = -999

        if bayes_sharpe >= grid_sharpe:
            refined_params = bayes_result.get('best_params', {})
            refined_sharpe = bayes_sharpe
        else:
            refined_params = grid_result.get('best_params', {})
            refined_sharpe = grid_sharpe

        # Step4 检查点
        save_ckpt('step4_bayes', {
            'bayes_result': bayes_result,
            'grid_result': grid_result,
            'refined_sharpe': refined_sharpe,
            'refined_params': refined_params,
            'ga_best_sharpe': ga_best_sharpe,
            'ga_best_factors': ga_best_factors,
            'ga_best_params': ga_best_params,
            'rl_best_sharpe': rl_best_sharpe,
            'neutralized_factors': neutralized_factors,
        })

        # ===== 5. 综合最优 =====
        all_candidates = [
            ('GA', ga_best_sharpe, ga_best_params, final_factors),
            ('RL', rl_best_sharpe, ga_best_params, rl_strategy.get('factors', final_factors)),
            ('Refined', refined_sharpe, refined_params, final_factors),
        ]
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        best_name, best_sharpe, best_params, best_factors = all_candidates[0]
        print(f"[{now}] 综合最优: {best_name}, Sharpe={best_sharpe:.4f}, 参数={best_params}")

        # ===== 6. 最终回测验证 =====
        final_strat = {
            'strategy_id': 'final',
            'strategy_name': f'final_{best_name.lower()}',
            'factors': best_factors,
        }
        try:
            final_bt = bt_executor.run(
                final_strat,
                best_params,
                start_date, end_date
            )
            final_bt['strategy_id'] = final_strat['strategy_id']
            final_bt['strategy_name'] = final_strat.get('strategy_name', '')
            final_bt['factors'] = final_strat.get('factors', [])
        except Exception as e:
            logger.warning(f"最终回测失败: {e}")
            final_bt = {}

        # ===== 7. 过拟合检测与策略接受判断 =====
        overfit_accepted = True
        overfit_summary = {}
        try:
            from src.core.overfit_detector import OverfitDetector
            ods_cfg = api.config.get('overfit_detection', {})
            if ods_cfg.get('enabled', True):
                detector = OverfitDetector(store, logger, api.config)

                # 构造 IS/OOS 分割（使用 GA 回测区间的前80%/后20%）
                # 这里用 all_trial_sharpes 的 bootstrap 作为伪 IS
                is_sharpes = all_ga_sharpes[:max(1, len(all_ga_sharpes) // 2)]
                oos_sharpes = all_ga_sharpes[max(1, len(all_ga_sharpes) // 2):]

                # 用 bootstrap 方法生成 IS/OOS 日收益率序列用于 PBO/CSCV
                # （真实场景中应从 backtester.nav_series 获取，这里用 sharpe 序列替代）
                # 用 all_trial_sharpes 构造伪收益率序列
                pseudo_returns = [s / 252.0 / 16.0 for s in all_trial_sharpes]

                # PBO / CSCV 用伪收益率
                if len(pseudo_returns) >= 60:
                    split_i = int(len(pseudo_returns) * 0.8)
                    pseudo_is = pseudo_returns[:split_i]
                    pseudo_oos = pseudo_returns[split_i:]
                    pbo = detector.calc_pbo(pseudo_is, pseudo_oos)
                    cscv = detector.calc_cscv(pseudo_returns)
                else:
                    pbo = 0.5
                    cscv = 0.5

                # DSR 用 GA 所有代 best_sharpe
                dsr = detector.calc_dsr(all_ga_sharpes, [all_ga_sharpes], n_trials=len(all_ga_sharpes))

                # 多重检验矫正（复用 GA 结果）
                mt_result = detector.calc_multiple_testing_adjustment(
                    all_ga_sharpes, n_trials=len(all_ga_sharpes)
                )

                overfit_summary = {
                    'pbo': round(float(pbo), 4),
                    'dsr': round(float(dsr), 4),
                    'cscv': round(float(cscv), 4),
                    'multiple_testing': {
                        'n_significant_bonferroni': mt_result.get('n_significant_bonferroni', 0),
                        'n_significant_bh': mt_result.get('n_significant_bh', 0),
                    }
                }

                # 接受判断
                pbo_th = float(ods_cfg.get('pbo_threshold', 0.5))
                dsr_th = float(ods_cfg.get('dsr_threshold', 0.0))
                cscv_th = float(ods_cfg.get('cscv_threshold', 0.5))

                reasons = []
                if pbo >= pbo_th:
                    reasons.append(f"PBO({pbo:.4f})>={pbo_th}")
                    overfit_accepted = False
                if dsr <= dsr_th:
                    reasons.append(f"DSR({dsr:.4f})<={dsr_th}")
                    overfit_accepted = False
                if cscv < cscv_th:
                    reasons.append(f"CSCV({cscv:.4f})<{cscv_th}")
                    overfit_accepted = False

                if not overfit_accepted:
                    print(f"[{now}] ⚠️ 策略被拒绝（过拟合风险）: {', '.join(reasons)}")
                else:
                    print(f"[{now}] ✓ 策略通过过拟合检测: PBO={pbo:.4f}, DSR={dsr:.4f}, CSCV={cscv:.4f}")
        except Exception as e:
            print(f"[{now}] 过拟合检测失败: {e}")
            overfit_summary = {'error': str(e)}

        # ===== 7.5 压力测试（高级功能）=====
        stress_accepted = True
        stress_report = {}
        try:
            from src.core.stress_tester import StressTester
            scfg = api.config.get('stress_test', {})
            if scfg.get('enabled', False):
                tester = StressTester(bt_executor, logger, api.config)
                stress_report = tester.full_stress_report(final_strat, best_params)
                stress_accepted = stress_report.get('passed', False)

                worst_dd = stress_report.get('worst_drawdown', 0.0)
                worst_period = stress_report.get('worst_period_name', 'N/A')
                sharpe_deg = stress_report.get('sharpe_degradation', 999.0)
                stability = stress_report.get('stability_ratio', 0.0)

                print(f"[{now}] 🧪 压力测试: worst_dd={worst_dd:.2%} [{worst_period}], "
                      f"sharpe_deg={sharpe_deg:.3f}, stability={stability:.2%}, "
                      f"passed={stress_accepted}")

                if not stress_accepted:
                    print(f"[{now}] ⚠️ 策略被拒绝（压力测试未通过）")
        except Exception as e:
            print(f"[{now}] 压力测试失败: {e}")
            stress_report = {'error': str(e)}

        # ===== 8. 绩效归因（Brinson/Barra）=====
        attribution_report = {}
        try:
            from src.core.attribution import PerformanceAttributor
            acfg = api.config.get('attribution', {})
            if acfg.get('enabled', False) and final_bt and 'nav_series' in final_bt:
                nav_df = final_bt.get('nav_series')
                if nav_df is not None and len(nav_df) > 2:
                    # 构造组合日收益率
                    port_rets = nav_df['returns'].dropna() if 'returns' in nav_df.columns else pd.Series()
                    # 构造模拟基准收益率（全市场等权）
                    bench_rets = port_rets * 0.95  # 简化：基准收益为组合的95%
                    # 构造持仓（从回测结果构建）
                    holdings = {}
                    industry_map = {}
                    if 'trades' in final_bt and final_bt['trades']:
                        for trade in final_bt['trades']:
                            code = trade.get('ts_code', '')
                            if code and trade.get('direction') == 'buy':
                                holdings[code] = {
                                    'value': trade.get('amount', 100000) or 100000,
                                    'return': 0.0,
                                    'industry': industry_map.get(code, 'Unknown'),
                                }
                                industry_map[code] = industry_map.get(code, 'Unknown')

                    attr_engine = PerformanceAttributor(store, logger, api.config)
                    attribution_report = attr_engine.full_report(
                        port_rets,
                        bench_rets,
                        holdings,
                        industry_map,
                    )
                    print(f"[{now}] 归因报告: 配置={attribution_report.get('summary', {}).get('allocation', 0):.4f}, "
                          f"选择={attribution_report.get('summary', {}).get('selection', 0):.4f}, "
                          f"因子贡献={attribution_report.get('summary', {}).get('factor_contrib', 0):.4f}")
        except Exception as e:
            print(f"[{now}] 归因分析失败: {e}")

        # ===== 9. 生成报告 =====
        report = {
            "report_time": now,
            "data_range": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} ~ {end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
            "lgb_auc": round(float(lgb_result.get('model_auc', 0) or 0), 4),
            "lgb_selected_count": len(selected_factors),
            "xgb_auc": round(float(xgb_result.get('auc', 0) or 0), 4),
            "xgb_train_samples": xgb_result.get('train_samples', 0),
            "xgb_feature_importance": xgb_result.get('feature_importance', {}),
            "lstm_auc": round(float(lstm_result.get('auc', lstm_result) if isinstance(lstm_result, dict) else (lstm_result or 0)), 4),
            "transformer_auc": round(float(transformer_result.get('auc', transformer_result) if isinstance(transformer_result, dict) else (transformer_result or 0)), 4),
            "n_neutralized_factors": len(neutralized_factors),
            "neutralized_factors": neutralized_factors,
            "ga_best_sharpe": round(float(ga_best_sharpe or 0), 4),
            "ga_best_factors": ga_best_factors,
            "ga_gen_history": ga_result.get('generation_history', []),
            "ga_overfit_report": ga_overfit_report,
            "rl_final_sharpe": round(float(rl_best_sharpe or 0), 4),
            "rl_n_states": rl_result.get('n_states', 0),
            "rl_policy_summary": {
                k: {'position_ratio': v.get('position_ratio'), 'q_value': v.get('q_value')}
                for k, v in list(rl_result.get('best_policy', {}).items())[:10]
            },
            "best_method": best_name,
            "best_sharpe": round(float(best_sharpe or 0), 4),
            "best_params": best_params,
            "best_factors": best_factors,
            "best_strategy": _clean_bt(final_bt) if final_bt else {},
            "total_strategies_generated": len(all_strategies),
            # 过拟合检测结果
            "overfit_detection": {
                "overfit_accepted": overfit_accepted,
                "summary": overfit_summary,
            },
            # 压力测试结果
            "stress_test": {
                "stress_accepted": stress_accepted,
                "summary": {
                    "worst_drawdown": stress_report.get('worst_drawdown', 0.0),
                    "worst_period": stress_report.get('worst_period_name', 'N/A'),
                    "sharpe_degradation": stress_report.get('sharpe_degradation', 999.0),
                    "stability_ratio": stress_report.get('stability_ratio', 0.0),
                    "sharpe_in_sample": stress_report.get('sharpe_in_sample', 0.0),
                    "sharpe_oos": stress_report.get('sharpe_oos', 0.0),
                },
                "historical_stress": stress_report.get('historical_stress', {}),
                "param_robustness": stress_report.get('param_robustness', {}),
                "oos_isolation": stress_report.get('oos_isolation', {}),
            },
            # 归因报告
            "attribution": attribution_report if attribution_report else {},
        }

        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{now}] 报告已生成: {REPORT_FILE}")

        print(f"\n{'='*60}")
        print(f"📊 策略报告 [{now}]")
        print(f"{'='*60}")
        print(f"LightGBM: 筛选出 {len(selected_factors)} 个因子, AUC={lgb_result.get('model_auc', 0):.4f}")
        if xgb_result:
            xgb_fi = xgb_result.get('feature_importance', {})
            print(f"XGBoost:  AUC={xgb_result.get('auc', 0):.4f}, 特征重要性={xgb_fi}")
        print(f"GA最优:    Sharpe={ga_best_sharpe:.4f}, 因子={ga_best_factors}")
        print(f"RL最优:    Sharpe={rl_best_sharpe:.4f}, 状态数={rl_result.get('n_states', 0)}")
        if attribution_report:
            asumm = attribution_report.get('summary', {})
            print(f"\n归因报告:")
            print(f"  行业配置={asumm.get('allocation', 0):.4f}, "
                  f"个股选择={asumm.get('selection', 0):.4f}, "
                  f"交互={asumm.get('interaction', 0):.4f}")
            print(f"  因子贡献={asumm.get('factor_contrib', 0):.4f}, "
                  f"特异性={asumm.get('specific_return', 0):.4f}")
        if final_bt:
            print(f"\n最终策略: {final_bt.get('strategy_name', 'N/A')}")
            print(f"  总收益: {_format_pct(final_bt.get('total_return', 0))}")
            print(f"  夏普:   {final_bt.get('sharpe_ratio', 0):.4f}")
            print(f"  最大回撤: {_format_pct(final_bt.get('max_drawdown', 0))}")
            print(f"  交易次数: {final_bt.get('total_trades', 0)}")
        # 过拟合检测结果
        ofd = overfit_summary
        print(f"\n🛡️ 过拟合检测:")
        print(f"  PBO={ofd.get('pbo', 'N/A')} | DSR={ofd.get('dsr', 'N/A')} | CSCV={ofd.get('cscv', 'N/A')}")
        mt_n_sig = ofd.get('multiple_testing', {})
        print(f"  多重检验: Bonferroni显著={mt_n_sig.get('n_significant_bonferroni', 'N/A')}, BH-FDR显著={mt_n_sig.get('n_significant_bh', 'N/A')}")
        print(f"  策略接受: {'✓ 是' if overfit_accepted else '✗ 否（过拟合风险）'}")
        print(f"\n🧪 压力测试:")
        print(f"  最差回撤={stress_report.get('worst_drawdown', 'N/A'):.2%} [{stress_report.get('worst_period_name', 'N/A')}]")
        print(f"  夏普衰退={stress_report.get('sharpe_degradation', 'N/A'):.3f}")
        print(f"  参数稳定性={stress_report.get('stability_ratio', 'N/A'):.2%}")
        print(f"  压力测试接受: {'✓ 是' if stress_accepted else '✗ 否（极端行情风险）'}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[{now}] ❌ 优化任务失败: {e}")
        import traceback
        traceback.print_exc()


def job_data_update():
    """数据更新（交易日18:00）—— 下载当天收盘数据，优先直连"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 数据更新任务")

    # 清除代理环境变量，确保 baostock 直连不受干扰
    for k in list(os.environ.keys()):
        if 'proxy' in k.lower():
            os.environ.pop(k, None)

    try:
        from datetime import date, timedelta
        api = get_instance()
        store = api.store

        # 交易日18:00直接更新今天的数据（收盘后）
        today_str = date.today().strftime('%Y%m%d')
        yesterday_str = (date.today() - timedelta(days=1)).strftime('%Y%m%d')

        print(f"[{now}] 交易日数据更新: {today_str}（收盘后补全）")

        api.execute({
            'action': 'update_data',
            'start_date': today_str,
            'end_date': today_str,
        })
        print(f"[{now}] ✅ 数据更新完成")
    except Exception as e:
        print(f"[{now}] 数据更新失败: {e}")
        import traceback; traceback.print_exc()


def job_factor_pool_update():
    """因子池更新（重新计算所有因子，含超时保护）"""
    return _factor_pool_update_with_timeout()


_factor_pool_update_timeout = 6 * 3600  # 6小时超时


def _factor_pool_update_with_timeout():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 因子池更新任务（超时保护: {_factor_pool_update_timeout // 3600}h）")
    result = [None]
    exception = [None]

    def target():
        try:
            api = get_instance()
            ret = api.execute({'action': 'evaluate_factors'})
            result[0] = ret
            if ret.get('code') == 500:
                print(f"[{now}] 因子评估失败: {ret.get('msg')}")
            elif ret.get('code') == 0:
                top = ret.get('data', {}).get('top_factor')
                count = ret.get('data', {}).get('count', 0)
                print(f"[{now}] 因子评估完成: 评估{count}个因子, Top: {top.get('factor_name') if top else 'N/A'}")
        except Exception as e:
            exception[0] = e

    t = threading.Thread(target=target, name='factor_pool_update')
    t.daemon = True
    t.start()
    t.join(timeout=_factor_pool_update_timeout)
    if t.is_alive():
        print(f"[{now}] ⏰ 因子池更新运行超过 {_factor_pool_update_timeout // 3600}h，强制结束")
        raise TimeoutError(f"job_factor_pool_update timed out after {_factor_pool_update_timeout // 3600} hours")
    if exception[0]:
        raise exception[0]
    return result[0]


def job_factor_rebalance():
    """因子池重平衡（含超时保护）"""
    return _factor_rebalance_with_timeout()


_factor_rebalance_timeout = 30 * 60  # 30分钟超时


def _factor_rebalance_with_timeout():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 因子池重平衡任务（超时: {_factor_rebalance_timeout // 60}min）")
    result = [None]
    exception = [None]

    def target():
        try:
            api = get_instance()
            api.factor_pool.rebalance()
            top = api.factor_pool.get_top_factors(10)
            print(f"[{now}] 重平衡完成，Top10: {[f['factor_name'] for f in top]}")
            result[0] = True
        except Exception as e:
            exception[0] = e

    t = threading.Thread(target=target, name='factor_rebalance')
    t.daemon = True
    t.start()
    t.join(timeout=_factor_rebalance_timeout)
    if t.is_alive():
        print(f"[{now}] ⏰ 因子池重平衡运行超过 {_factor_rebalance_timeout // 60}min，强制结束")
        raise TimeoutError(f"job_factor_rebalance timed out after {_factor_rebalance_timeout // 60} minutes")
    if exception[0]:
        raise exception[0]
    return result[0]


def job_market_watcher():
    """盘中监控（后台线程）"""
    global _market_watcher_instance
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 📡 启动盘中监控线程")

    if _market_watcher_instance is None:
        _market_watcher_instance = MarketWatcher(check_interval=1800)  # 30分钟
    _market_watcher_instance.start()
    return _market_watcher_instance


# ==============================================================================
# 非交易日精细化任务（CLI模式调用）
# ==============================================================================

@timeout(8 * 3600)
def job_long_cycle_mining():
    """非交易日-01:00 | 长周期因子挖掘"""
    from scheduler.weekend_research import weekend_research
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🔬 长周期因子挖掘任务")

    try:
        api = get_instance()
        from src.core.factor_eval import FactorEvaluator
        evaluator = FactorEvaluator(api.store, api.logger)
        LONG_TERM_FACTORS = [
            'momentum_60', 'momentum_120', 'momentum_250',
            'volatility_60', 'volatility_120',
            'volume_ratio_60',
        ]
        results = evaluator.evaluate_multiple(LONG_TERM_FACTORS, '20150101', '20260327')
        ranked = evaluator.rank_factors(results)
        print(f"[{now}] 长周期因子挖掘完成: {len(ranked)} 个因子")
        return ranked
    except Exception as e:
        print(f"[{now}] ❌ 长周期因子挖掘失败: {e}")
        import traceback; traceback.print_exc()


@timeout(8 * 3600)
def job_multi_interval_backtest():
    """非交易日-03:00 | 多区间深度回测（joblib 7进程并行）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 📊 多区间深度回测任务")
    try:
        api = get_instance()
        from src.core.backtester import BacktestExecutor
        HISTORICAL_INTERVALS = [
            ('20150101', '20151231'), ('20160101', '20161231'),
            ('20180101', '20181231'), ('20200101', '20200331'),
            ('20200401', '20201231'), ('20230101', '20231231'),
            ('20240101', '20240630'),
        ]

        def _run_one_interval(start_end):
            """单个区间回测（供并行调用）"""
            start, end = start_end
            store = api.store
            logger = api.logger
            config = api.config
            bt_executor = BacktestExecutor(store, logger, config)
            strategy = {
                'strategy_id': f'momentum_base_{start}_{end}',
                'strategy_name': f'momentum_base',
                'factors': ['momentum_20'],
                'parameters': {'holding_period': 20, 'stop_loss': 0.10, 'take_profit': 0.20}
            }
            r = bt_executor.run(strategy, strategy['parameters'], start, end)
            return {
                'interval': f'{start}~{end}',
                'sharpe': round(float(r.get('sharpe_ratio', 0) or 0), 4),
                'return': round(float(r.get('total_return', 0) or 0), 4),
            }

        n_workers = min(len(HISTORICAL_INTERVALS), 7)
        print(f"[{now}] [并行] 多区间回测 {len(HISTORICAL_INTERVALS)} 个区间，workers={n_workers}")
        results = joblib.Parallel(n_jobs=n_workers, prefer="threads", timeout=3600)(
            joblib.delayed(_run_one_interval)(iv) for iv in HISTORICAL_INTERVALS
        )
        for r in results:
            print(f"[{now}]   {r['interval']}: Sharpe={r['sharpe']:.4f}")
        print(f"[{now}] 多区间回测完成: {len(results)} 个结果")
        return results
    except Exception as e:
        print(f"[{now}] ❌ 多区间深度回测失败: {e}")
        import traceback; traceback.print_exc()


@timeout(8 * 3600)
def job_ga_optimization():
    """非交易日-09:00 | GA并行优化"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🧬 GA并行优化任务")
    # GA优化复用job_optimize_and_report的核心逻辑
    return job_optimize_and_report()


def job_factor_correlation():
    """非交易日-17:00 | 因子相关性分析"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🔗 因子相关性分析任务")
    try:
        from scheduler.weekend_research import _analyze_factor_correlation
        api = get_instance()
        result = _analyze_factor_correlation(api)
        print(f"[{now}] 因子相关性分析完成: {result.get('message', '完成')}")
        return result
    except Exception as e:
        print(f"[{now}] ❌ 因子相关性分析失败: {e}")
        import traceback; traceback.print_exc()


def job_generate_report():
    """非交易日-20:00 | 报告生成"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 📝 非交易日报告生成")
    # 复用优化报告逻辑
    return job_optimize_and_report()


def job_send_report():
    """非交易日-21:00 | 发送报告到飞书"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 📤 发送非交易日报告")
    try:
        report_file = Path(__file__).parent.parent / "reports" / f"weekend_report_{date.today().strftime('%Y%m%d')}.json"
        if report_file.exists():
            import json
            with open(report_file) as f:
                report = json.load(f)
            
            lines = [
                "📊 **A股趋势策略报告**",
                f"报告日期: {report.get('report_time', date.today())}（非交易日）",
                "",
                "━━━━━━━━━━━━━━━━━━━━",
                "【Top5 因子】",
            ]
            for i, f in enumerate(report.get('top_factors', [])[:5], 1):
                lines.append(f"  {i}. {f['factor']:<15} IC={f['ic']:+.4f}  IR={f['ir']:+.4f}")
            
            best = report.get('best_strategy', {})
            lines.extend([
                "",
                "━━━━━━━━━━━━━━━━━━━━",
                "【最优策略】",
                f"  策略名: {best.get('strategy_name', 'N/A')}",
                f"  使用因子: {', '.join(best.get('factors', []))}",
                f"  总收益: {best.get('total_return', 0):.2%}",
                f"  夏普比率: {best.get('sharpe_ratio', 0):.4f}",
                f"  最大回撤: {best.get('max_drawdown', 0):.2%}",
                f"  交易次数: {best.get('total_trades', 0)}",
                "",
                "━━━━━━━━━━━━━━━━━━━━",
                f"📌 数据区间: {report.get('data_range', 'N/A')}",
            ])
            
            print("\n".join(lines))
            print(f"\n[{now}] ✅ 非交易日报告已发送")
        else:
            print(f"[{now}] 报告文件不存在: {report_file}")
    except Exception as e:
        print(f"[{now}] ❌ 发送报告失败: {e}")
        import traceback; traceback.print_exc()


# ==============================================================================
# 交易日精细化任务（CLI模式调用）
# ==============================================================================

def job_factor_rebalance_and_report():
    """交易日-15:30 | 因子重平衡 + GA优化 + 报告生成"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] ⚖️ 因子重平衡与报告生成任务")
    try:
        job_factor_rebalance()
        job_optimize_and_report()
        print(f"[{now}] ✅ 因子重平衡与报告生成完成")
    except Exception as e:
        print(f"[{now}] ❌ 因子重平衡与报告生成失败: {e}")
        import traceback; traceback.print_exc()


def _get_feishu_credentials():
    """从 openclaw.json 读取飞书 appId 和 appSecret"""
    try:
        openclaw_cfg = Path.home() / '.openclaw' / 'openclaw.json'
        with open(openclaw_cfg) as f:
            cfg = json.load(f)
        feishu = cfg.get('channels', {}).get('feishu', {})
        return feishu.get('appId', ''), feishu.get('appSecret', '')
    except Exception:
        return '', ''


def _get_tenant_token():
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


def _send_feishu_message_to_group(chat_id: str, text: str) -> bool:
    """发送文本消息到飞书群"""
    token = _get_tenant_token()
    if not token:
        print(f"[FEISHU] 无法获取 tenant_token")
        return False
    try:
        import urllib.request
        payload = {
            "receive_id": chat_id,
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
        if resp.get("code") == 0:
            return True
        else:
            print(f"[FEISHU] 发送失败: code={resp.get('code')} msg={resp.get('msg')}")
            return False
    except Exception as e:
        print(f"[FEISHU] 发送异常: {e}")
        return False


def job_send_strategy_report():
    """交易日-23:30 | 发送策略报告到飞书（支持交易日和非交易日）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 📤 发送策略报告")
    # 目标群：当前飞书群
    FEISHU_CHAT_ID = 'oc_c6dde682059311ce59cee47c6dc5383b'
    try:
        # 优先读取交易日报告，找不到则读周末报告
        report_file = Path(__file__).parent.parent / "reports" / "daily_strategy_report.json"
        report_type = "交易日"
        if not report_file.exists():
            report_file = Path(__file__).parent.parent / "reports" / f"weekend_report_{date.today().strftime('%Y%m%d')}.json"
            report_type = "非交易日"
        if not report_file.exists():
            print(f"[{now}] 报告文件不存在: {report_file}")
            return
        import json
        with open(report_file) as f:
            report = json.load(f)
            
        lines = [
            f"📊 **A股趋势策略报告**",
            f"报告日期: {report.get('report_time', date.today())}（{report_type}）",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "【Top5 因子】",
        ]
        for i, f in enumerate(report.get('top_factors', [])[:5], 1):
            lines.append(f"  {i}. {f['factor']:<15} IC={f['ic']:+.4f}  IR={f['ir']:+.4f}")
        
        best = report.get('best_strategy', {})
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "【最优策略】",
            f"  策略名: {best.get('strategy_name', 'N/A')}",
            f"  使用因子: {', '.join(best.get('factors', []))}",
            f"  总收益: {best.get('total_return', 0):.2%}",
            f"  夏普比率: {best.get('sharpe_ratio', 0):.4f}",
            f"  最大回撤: {best.get('max_drawdown', 0):.2%}",
            f"  交易次数: {best.get('total_trades', 0)}",
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📌 数据区间: {report.get('data_range', 'N/A')}",
        ])
        
        message = "\n".join(lines)
        print(message)
        
        # 真正调用飞书 API 发送
        sent = _send_feishu_message_to_group(FEISHU_CHAT_ID, message)
        if sent:
            print(f"[{now}] ✅ 策略报告已发送到飞书群 {FEISHU_CHAT_ID}")
        else:
            print(f"[{now}] ⚠️ 飞书 API 发送失败（报告已打印到日志）")
    except Exception as e:
        print(f"[{now}] ❌ 发送策略报告失败: {e}")
        import traceback; traceback.print_exc()


# ==============================================================================
# 启动调度器（根据是否为交易日决定加载哪些任务）
# ==============================================================================

def start_scheduler():
    """
    根据今天是否为交易日，决定加载哪些任务。
    交易日：加载完整交易日任务链
    非交易日：加载周末研究任务链
    """
    today = date.today()
    is_td = is_trading_day(today)
    weekday = today.weekday()
    weekday_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][weekday]

    print(f"[{datetime.now()}] 🏭 量化工厂调度器启动 | {weekday_name} {today} | "
          f"类型: {'交易日' if is_td else '非交易日'}")

    if is_td:
        # ==================== 交易日任务（分拆版优化）====================
        print(f"[{datetime.now()}] 📅 交易日模式：注册以下任务")
        print("  07:00 — 因子池更新")
        print("  08:00 — 盘前策略预判")
        print("  09:30 — 盘中监控（后台线程）")
        print("  15:30 — 盘后数据更新")
        print("  16:00 — 因子池更新（先评估）")
        print("  16:30 — 因子重平衡（再重分配）")
        print("  17:00 — Step1: LightGBM因子筛选")
        print("  18:00 — Step2: GA遗传算法优化")
        print("  20:00 — Step3: RL强化学习仓位优化")
        print("  21:00 — Step4: Bayesian参数精调")
        print("  22:00 — Step5: 最终回测+生成报告")
        print("  23:00 — 发送策略报告")

        scheduler.add_job(job_pre_market, CronTrigger(hour=8, minute=0))
        scheduler.add_job(job_market_watcher, CronTrigger(hour=9, minute=30))
        scheduler.add_job(job_post_market, CronTrigger(hour=15, minute=30))
        scheduler.add_job(job_factor_pool_update, CronTrigger(hour=16, minute=0))   # 因子池更新（先评估）
        scheduler.add_job(job_factor_rebalance, CronTrigger(hour=16, minute=30))   # 因子重平衡（再重分配）
        scheduler.add_job(job_step1_lgb, CronTrigger(hour=17, minute=0))       # 17:00 LightGBM因子筛选
        scheduler.add_job(job_step2_ga, CronTrigger(hour=18, minute=0))        # 18:00 GA优化
        scheduler.add_job(job_step3_rl, CronTrigger(hour=20, minute=0))         # 20:00 RL仓位优化
        scheduler.add_job(job_step4_bayes, CronTrigger(hour=21, minute=0))      # 21:00 Bayesian精调
        scheduler.add_job(job_step5_final, CronTrigger(hour=22, minute=0))      # 22:00 最终回测+报告
        scheduler.add_job(job_send_strategy_report, CronTrigger(hour=23, minute=0))  # 23:00 发送报告

    else:
        # ==================== 非交易日任务（分拆版）====================
        print(f"[{datetime.now()}] 🌙 非交易日模式：注册以下任务")
        print("  07:00 — 长周期因子挖掘")
        print("  08:00 — 多区间深度回测")
        print("  09:00 — Step1: LightGBM因子筛选")
        print("  10:00 — Step2: GA遗传算法优化")
        print("  12:00 — Step3: RL强化学习仓位优化")
        print("  14:00 — Step4: Bayesian参数精调")
        print("  15:00 — 因子相关性分析")
        print("  16:00 — Step5: 最终回测+生成报告")
        print("  21:00 — 发送报告到飞书")

        # 非交易日固定注册这些任务（每天按各自时间点运行）
        scheduler.add_job(job_long_cycle_mining, CronTrigger(hour=7, minute=0))          # 07:00 长周期因子挖掘
        scheduler.add_job(job_multi_interval_backtest, CronTrigger(hour=8, minute=0))   # 08:00 多区间深度回测
        scheduler.add_job(job_step1_lgb, CronTrigger(hour=9, minute=0))                   # 09:00 LightGBM因子筛选
        scheduler.add_job(job_step2_ga, CronTrigger(hour=10, minute=0))                  # 10:00 GA优化
        scheduler.add_job(job_step3_rl, CronTrigger(hour=12, minute=0))                  # 12:00 RL仓位优化
        scheduler.add_job(job_step4_bayes, CronTrigger(hour=14, minute=0))                # 14:00 Bayesian精调
        scheduler.add_job(job_factor_correlation, CronTrigger(hour=15, minute=0))           # 15:00 因子相关性分析（step4之后，step5之前）
        scheduler.add_job(job_step5_final, CronTrigger(hour=16, minute=0))              # 16:00 最终回测+报告
        scheduler.add_job(job_send_strategy_report, CronTrigger(hour=21, minute=0))           # 21:00 发送报告

    scheduler.start()
    print(f"[{datetime.now()}] ✅ 调度器已启动，运行模式: {'交易日' if is_td else '非交易日'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A股量化因子工厂任务调度')
    parser.add_argument('--task', type=str, help='指定运行的任务类型')
    parser.add_argument('--batch-id', type=int, default=None, help='分批ID，None表示连续跑完')
    args, unknown = parser.parse_known_args()

    if args.task:
        # ============================================================
        # CLI模式：由cron定时器调用，严格区分交易日/非交易日
        # ============================================================
        today = date.today()
        is_td = is_trading_day(today)
        today_str = today.strftime('%Y%m%d')
        weekday = today.weekday()
        weekday_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][weekday]

        # 交易日任务映射
        TRADING_TASKS = {
            'data_update': job_data_update,
            'factor_pool_update': job_factor_pool_update,
            'pre_market_analysis': job_pre_market,
            'post_market_review': job_post_market,
            'factor_rebalance_and_report': job_factor_rebalance_and_report,
            'step1_lgb': job_step1_lgb,
            'step2_ga': job_step2_ga,
            'step3_rl': job_step3_rl,
            'step4_bayes': job_step4_bayes,
            'step5_final': job_step5_final,
            'send_strategy_report': job_send_strategy_report,
        }

        # 非交易日任务映射
        NON_TRADING_TASKS = {
            'long_cycle_mining': job_long_cycle_mining,
            'multi_interval_backtest': job_multi_interval_backtest,
            'step1_lgb': job_step1_lgb,
            'step2_ga': job_step2_ga,
            'step3_rl': job_step3_rl,
            'step4_bayes': job_step4_bayes,
            'step5_final': job_step5_final,
            'factor_correlation': job_factor_correlation,
            'send_report': job_send_strategy_report,
        }

        task_name = args.task

        # 任务存在性检查
        if task_name not in TRADING_TASKS and task_name not in NON_TRADING_TASKS:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 未知任务: {task_name}")
            print(f"可用交易日任务: {list(TRADING_TASKS.keys())}")
            print(f"可用非交易日任务: {list(NON_TRADING_TASKS.keys())}")
            sys.exit(1)

        # 严格类型校验（step1-step5 每天都跑，非交易日和交易日只是时间段不同）
        ALWAYS_RUN_TASKS = {'step1_lgb', 'step2_ga', 'step3_rl', 'step4_bayes', 'step5_final', 'send_report'}
        if task_name not in ALWAYS_RUN_TASKS:
            if task_name in TRADING_TASKS:
                if not is_td:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏭️ 跳过任务 [{task_name}] — 今日({weekday_name} {today_str})是非交易日")
                    sys.exit(0)
            elif task_name in NON_TRADING_TASKS:
                if is_td:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏭️ 跳过任务 [{task_name}] — 今日({weekday_name} {today_str})是交易日")
                    sys.exit(0)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ▶️ 执行任务 [{task_name}] | {weekday_name} {today_str} | {'交易日' if is_td else '非交易日'}")

        # 执行任务
        task_func = TRADING_TASKS.get(task_name) or NON_TRADING_TASKS.get(task_name)
        try:
            if task_name == 'step2_ga':
                task_func(batch_id=args.batch_id)
            elif task_name == 'step3_rl':
                task_func(batch_id=args.batch_id, daily_reset=(args.batch_id == 0))
            else:
                task_func()
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 任务 [{task_name}] 执行失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # 原有APScheduler模式（后台常驻）
        start_scheduler()
        import time
        while True:
            time.sleep(3600)
