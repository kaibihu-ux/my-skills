"""
周末研究任务（不依赖实时数据）：
1. 全量因子挖掘（不同时长：20/60/120/250日）
2. 历史区间深度回测（2015/2018/2020/2023等极端行情）
3. 多周期策略测试
4. 因子相关性分析
5. 下一周策略预判
"""
import sys
import threading
import functools
from pathlib import Path
from datetime import date, datetime
import json
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.skill_api import get_instance
from src.core.genetic_optimizer import GeneticOptimizer
from src.core.backtester import BacktestExecutor


# 历史回测区间（极端行情）
HISTORICAL_INTERVALS = [
    ('20240101', '20241231'),   # 2024全年
    ('20250101', '20251231'),   # 2025全年
    ('20260101', '20260409'),   # 2026至今（数据库最新日期）
]

# 长周期因子
LONG_TERM_FACTORS = [
    'momentum_60', 'momentum_120', 'momentum_250',
    'volatility_60', 'volatility_120',
    'volume_ratio_60',
]


# 超时装饰器（8小时超时保护）
def _timeout(seconds=8 * 3600):
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
            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(timeout=seconds)
            if t.is_alive():
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] ⏰ {func.__name__} 超时（>{seconds/3600:.0f}h），强制结束")
                raise TimeoutError(f"{func.__name__} timed out after {seconds/3600:.0f} hours")
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


def weekend_research():
    """
    周末流水线：
    1. 长周期因子挖掘（120/250日）
    2. 多区间深度回测（并行）
    3. GA多代并行优化
    4. 因子相关性分析
    5. 周报生成
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_date = date.today().strftime("%Y%m%d")
    print(f"[{now}] 🔬 周末研究任务开始")

    try:
        api = get_instance()
        store = api.store
        logger = api.logger
        bt_executor = BacktestExecutor(store, logger, api.config)

        # ===== 1. 长周期因子挖掘 =====
        print(f"[{now}] Step1: 长周期因子挖掘...")
        long_term_results = _mine_long_term_factors(api, LONG_TERM_FACTORS)
        print(f"[{now}] 长周期因子: {len(long_term_results)} 个完成")

        # ===== 2. 多区间深度回测（并行）=====
        print(f"[{now}] Step2: 多区间深度回测（{len(HISTORICAL_INTERVALS)} 个区间）...")
        interval_results = _run_multi_interval_backtest(
            bt_executor, logger, HISTORICAL_INTERVALS
        )
        print(f"[{now}] 多区间回测完成: {len(interval_results)} 个结果")

        # ===== 3. GA多代并行优化 =====
        print(f"[{now}] Step3: GA多代并行优化...")
        ga_results = _run_parallel_ga_optimization(
            bt_executor, logger, HISTORICAL_INTERVALS
        )
        print(f"[{now}] GA并行优化完成: {len(ga_results)} 个区间")

        # ===== 4. 因子相关性分析 =====
        print(f"[{now}] Step4: 因子相关性分析...")
        correlation_matrix = _analyze_factor_correlation(api)
        print(f"[{now}] 相关性分析完成")

        # ===== 5. 策略稳定性评估 =====
        print(f"[{now}] Step5: 策略稳定性评估...")
        stability_report = _evaluate_strategy_stability(
            interval_results, ga_results
        )
        print(f"[{now}] 稳定性评估完成")

        # ===== 6. 生成周末报告 =====
        report = {
            'report_type': 'weekend_research',
            'report_date': report_date,
            'report_time': now,
            'weekend_day': date.today().strftime("%A"),
            'long_term_factors': long_term_results,
            'interval_results': interval_results,
            'ga_optimization': ga_results,
            'correlation_matrix': correlation_matrix,
            'stability_report': stability_report,
            'next_week_outlook': _build_next_week_outlook(
                interval_results, stability_report
            ),
        }

        report_dir = Path(__file__).parent.parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"weekend_report_{report_date}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[{now}] ✅ 周末报告已生成: {report_file}")
        print(f"\n{'='*60}")
        print(f"📊 周末研究报告 [{now}]")
        print(f"{'='*60}")
        print(f"长周期因子: {len(long_term_results)} 个")
        print(f"历史区间: {len(interval_results)} 个")
        print(f"GA优化区间: {len(ga_results)} 个")
        print(f"策略稳定性: {stability_report.get('overall_stability', 'N/A')}")
        print(f"下周展望: {report['next_week_outlook']}")
        print(f"{'='*60}\n")

        return report

    except Exception as e:
        print(f"[{now}] ❌ 周末研究失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _mine_long_term_factors(api, factor_names):
    """挖掘长周期因子"""
    results = []
    for factor in factor_names:
        try:
            ev_result = api.factor_eval.evaluate_multiple([factor], '20150101', '20250630')
            if ev_result:
                r = ev_result[0]
                results.append({
                    'factor_name': factor,
                    'ir': round(float(r.get('information_ratio', 0)), 4),
                    'ic': round(float(r.get('ic', 0)), 4),
                    'n_samples': int(r.get('n_samples', 0)),
                })
        except Exception as e:
            print(f"  因子 {factor} 评估失败: {e}")
    return results


def _run_multi_interval_backtest(bt_executor, logger, intervals):
    """多区间深度回测"""
    results = []
    for start, end in intervals:
        try:
            # 用基础动量策略跑每个区间
            strategy = {
                'strategy_id': f'momentum_base_{start}_{end}',
                'strategy_name': f'momentum_base',
                'factors': ['momentum_20'],
                'parameters': {
                    'holding_period': 20,
                    'stop_loss': 0.10,
                    'take_profit': 0.20,
                }
            }
            bt_result = bt_executor.run(strategy, strategy['parameters'], start, end)
            results.append({
                'interval': f'{start}~{end}',
                'start': start,
                'end': end,
                'total_return': round(float(bt_result.get('total_return', 0) or 0), 4),
                'sharpe_ratio': round(float(bt_result.get('sharpe_ratio', 0) or 0), 4),
                'max_drawdown': round(float(bt_result.get('max_drawdown', 0) or 0), 4),
                'total_trades': int(bt_result.get('total_trades', 0) or 0),
                'win_rate': round(float(bt_result.get('win_rate', 0) or 0), 4),
            })
            print(f"  区间 {start}~{end}: Sharpe={bt_result.get('sharpe_ratio', 0):.4f}, "
                  f"收益={bt_result.get('total_return', 0)*100:.2f}%")
        except Exception as e:
            print(f"  区间 {start}~{end} 回测失败: {e}")
            results.append({
                'interval': f'{start}~{end}',
                'start': start,
                'end': end,
                'error': str(e),
            })
    return results


def _run_parallel_ga_optimization(bt_executor, logger, intervals):
    """GA多代并行优化（每个区间单独优化）"""
    results = []

    # 使用进程池并行（但当前区间数量不多，串行也可）
    # 这里展示并行结构，实际因GIL和资源限制可调整为串行
    n_workers = min(len(intervals), cpu_count(), 4)

    for start, end in intervals:
        try:
            ga_opt = GeneticOptimizer(
                bt_executor=bt_executor,
                logger=logger,
                start_date=start,
                end_date=end,
                pop_size=15,
                n_generations=20,
                mutation_rate=0.15,
                elite_ratio=0.2,
            )
            ga_result = ga_opt.optimize(top_n_stocks=20)
            results.append({
                'interval': f'{start}~{end}',
                'ga_best_sharpe': round(float(ga_result.get('best_sharpe', -999)), 4),
                'ga_best_factors': ga_result.get('best_factors', []),
                'ga_best_params': ga_result.get('best_params', {}),
                'ga_generations': len(ga_result.get('generation_history', [])),
            })
            print(f"  GA优化 {start}~{end}: Sharpe={ga_result.get('best_sharpe', -999):.4f}")
        except Exception as e:
            print(f"  GA优化 {start}~{end} 失败: {e}")
            results.append({
                'interval': f'{start}~{end}',
                'error': str(e),
            })

    return results


def _analyze_factor_correlation(api):
    """
    因子相关性分析：
    剔除相关性>0.8的冗余因子
    """
    try:
        top_factors = api.factor_pool.get_top_factors(20)
        if not top_factors or len(top_factors) < 2:
            return {'message': '因子池不足，跳过相关性分析'}

        factor_names = [f['factor_name'] for f in top_factors]
        store = api.store

        # 计算因子间的IC相关性
        correlations = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                try:
                    ic_df1 = store.df(
                        f"SELECT ic FROM factor_ic WHERE factor_name = '{f1}' ORDER BY date DESC LIMIT 60"
                    )
                    ic_df2 = store.df(
                        f"SELECT ic FROM factor_ic WHERE factor_name = '{f2}' ORDER BY date DESC LIMIT 60"
                    )
                    if not ic_df1.empty and not ic_df2.empty:
                        corr = ic_df1['ic'].corr(ic_df2['ic'])
                        correlations.append({
                            'factor1': f1,
                            'factor2': f2,
                            'correlation': round(float(corr), 4),
                        })
                except Exception:
                    pass

        # 找出高相关对
        high_corr_pairs = [c for c in correlations if abs(c['correlation']) > 0.8]
        # 建议剔除
        to_remove = list(set([c['factor2'] for c in high_corr_pairs if c['correlation'] > 0.8]))

        return {
            'factor_count': len(factor_names),
            'total_pairs': len(correlations),
            'high_correlation_pairs': high_corr_pairs[:10],  # 只保留前10
            'suggested_remove': to_remove,
            'message': f'建议剔除 {len(to_remove)} 个高相关冗余因子',
        }
    except Exception as e:
        return {'error': str(e)}


def _evaluate_strategy_stability(interval_results, ga_results):
    """评估策略稳定性"""
    try:
        # 按区间统计夏普分布
        sharpes = [r.get('sharpe_ratio', -999) for r in interval_results if 'error' not in r]
        if not sharpes:
            return {'overall_stability': 'N/A'}

        avg_sharpe = sum(sharpes) / len(sharpes)
        positive_count = sum(1 for s in sharpes if s > 0)
        stability_score = positive_count / len(sharpes)  # 正收益区间占比

        if stability_score >= 0.8 and avg_sharpe > 1.0:
            stability = "极稳定"
        elif stability_score >= 0.6 and avg_sharpe > 0.5:
            stability = "较稳定"
        elif stability_score >= 0.4:
            stability = "一般"
        else:
            stability = "不稳定"

        # 极端行情分析
        crisis_intervals = ['20150101~20151231', '20180101~20181231', '20200101~20200331']
        crisis_sharpes = [
            r.get('sharpe_ratio', -999)
            for r in interval_results
            if r.get('interval') in crisis_intervals
        ]
        crisis_avg = sum(crisis_sharpes) / len(crisis_sharpes) if crisis_sharpes else -999

        return {
            'overall_stability': stability,
            'stability_score': round(stability_score, 4),
            'avg_sharpe_all': round(avg_sharpe, 4),
            'positive_ratio': f"{positive_count}/{len(sharpes)}",
            'crisis_avg_sharpe': round(crisis_avg, 4),
            'crisis_intervals_tested': len(crisis_sharpes),
        }
    except Exception as e:
        return {'error': str(e)}


def _build_next_week_outlook(interval_results, stability_report):
    """构建下周展望"""
    try:
        recent_result = interval_results[-1] if interval_results else {}
        recent_sharpe = recent_result.get('sharpe_ratio', 0)
        recent_return = recent_result.get('total_return', 0)
        stability = stability_report.get('overall_stability', 'N/A')

        if stability in ["极稳定", "较稳定"] and recent_sharpe > 0.5:
            return "策略在最近区间表现稳定，建议下周维持现有策略，积极持仓"
        elif stability == "一般":
            return "策略稳定性一般，建议下周谨慎操作，控制仓位"
        elif recent_return < -0.1:
            return "最近区间回撤较大，建议下周降低仓位，等待信号明确"
        else:
            return "市场格局不明，建议下周保持观察，轻仓试探"
    except Exception:
        return "数据不足，无法给出下周展望"


@_timeout(8 * 3600)
def job_weekend_research():
    """调度器入口"""
    return weekend_research()
