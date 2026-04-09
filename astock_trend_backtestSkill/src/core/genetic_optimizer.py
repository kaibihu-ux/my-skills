"""
遗传算法优化器 - 用于因子组合和参数协同优化
不依赖 tensorflow/pytorch，纯 Python + numpy 实现
"""

import numpy as np
import random
import json
import threading
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 并行评估
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class GeneticOptimizer:
    """遗传算法优化器 - 用于因子组合和参数协同优化"""

    # 染色体编码映射表
    FACTOR_MAP = {
        0: None,                          # 无因子
        1: 'momentum_20',
        2: 'momentum_60',
        3: 'volatility_20',
        4: 'volume_ratio_20',
        5: 'rsi_14',
        6: 'macd',
        7: 'bollinger_position',
    }

    HOLDING_PERIOD_MAP = {0: 5, 1: 10, 2: 20, 3: 60}
    STOP_LOSS_MAP = {0: 0.05, 1: 0.10, 2: 0.15}
    N_STOCKS_MAP = {0: 10, 1: 20, 2: 30}

    # 染色体长度: 5因子位 + 1持仓期 + 1止损 + 1持仓数 = 8
    CHROMOSOME_LEN = 8

    def __init__(
        self,
        backtester,
        logger,
        start_date: str = None,
        end_date: str = None,
        pop_size: int = 20,
        n_generations: int = 30,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        tournament_size: int = 3,
        config: Dict = None,
    ):
        self.backtester = backtester
        self.logger = logger
        self.config = config
        self.start_date = start_date or '20200101'
        self.end_date = end_date or '20231231'

        # GA 超参数
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size

        # ---- 并行 + 检查点 ----
        self._n_jobs = 1  # 顺序执行（DuckDB不可pickle）
        # 检查点路径（由外部设置，或默认）
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_lock = threading.Lock()
        # 当前最优（用于检查点保存）
        self._current_best_sharpe = -999.0
        self._current_best_chrom = None
        self._current_best_decoded = {}
        self._fitness_history = []

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def optimize(
        self,
        available_factors: List[str] = None,
        initial_pop: List[Dict] = None,
        top_n_stocks: int = 20,
        timeout_checker=None,
        remaining_n_gens: int = None,
    ) -> Dict:
        """
        运行遗传算法优化因子组合和策略参数

        Args:
            available_factors: 可用因子列表，默认使用内置 FACTOR_MAP
            initial_pop: 初始种群（可选）
            top_n_stocks: 选股数量
            timeout_checker: 超时检查回调，接收 timeout_checker() -> bool，
                             返回 True 表示时间不多应保存并退出
            remaining_n_gens: 续跑专用。传入 dict：
                             {n_gens_completed, total_target, fitness_history,
                              best_sharpe, best_chromosome, best_decoded, population}
                             用于 cron 多批接力场景，告知 GA 已完成多少代，
                             GA 自动续跑剩余代数。

        Returns:
            {
                'best_chromosome': [...],
                'best_factors': [...],
                'best_params': {...},
                'best_sharpe': float,
                'generation_history': [{'gen': int, 'best': float, 'avg': float}, ...]
            }
        """
        # 自动设置检查点路径
        if self._checkpoint_path is None:
            try:
                ckpt_dir = Path(__file__).parent.parent / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                self._checkpoint_path = ckpt_dir / "ga_checkpoint.json"
            except Exception:
                pass

        # ---- 确定总代数 & 起始代数 ----
        _total_target = self.n_generations
        _resume_from = 0
        # 使用实例变量（供 _save_checkpoint 读取）
        self._fitness_history = []
        self._current_population = []

        if remaining_n_gens is not None:
            # 外部传入续跑状态（cron 多批接力）
            n_done = remaining_n_gens.get('n_gens_completed', 0)
            _total_target = remaining_n_gens.get('total_target', self.n_generations)
            _resume_from = n_done
            self._fitness_history = list(remaining_n_gens.get('fitness_history', []))
            self._current_best_sharpe = remaining_n_gens.get('best_sharpe', -999)
            self._current_best_chrom = remaining_n_gens.get('best_chromosome')
            self._current_best_decoded = remaining_n_gens.get('best_decoded', {})
            self._current_population = list(remaining_n_gens.get('population', []))
            n_gens_to_run = max(0, _total_target - _resume_from)
            self.logger.info(
                f"[GA] 续跑 | 已完成={n_done}/{_total_target}代 | "
                f"本次跑={n_gens_to_run}代 | Best={self._current_best_sharpe:.4f}"
            )
        else:
            # 普通模式：从文件检查点恢复
            loaded = self._load_checkpoint()
            if loaded:
                _resume_from = loaded.get('n_gens_completed', 0)
                _total_target = loaded.get('total_target', self.n_generations)
                self._fitness_history = list(loaded.get('generation_history', []))
                self._current_population = list(loaded.get('population', []))
                n_gens_to_run = max(0, _total_target - _resume_from)
                self.logger.info(
                    f"[GA] 检查点恢复 | 已完成={_resume_from}/{_total_target}代 | "
                    f"本次跑={n_gens_to_run}代"
                )
            else:
                n_gens_to_run = _total_target

        self.logger.info(
            f"[GA] 开始优化 | pop={self.pop_size}, target={_total_target}, "
            f"resume={_resume_from}, 本次={n_gens_to_run}代, "
            f"jobs={self._n_jobs}"
        )

        # ---- 初始化种群 ----
        if self._current_population:
            population = self._current_population
        else:
            population = self._init_population(initial_pop)

        # ---- 主循环 ----
        for gen in range(_resume_from, _resume_from + n_gens_to_run):
            if HAS_JOBLIB and self._n_jobs > 1:
                # 顺序执行（避免DuckDB无法pickle）
                fitness_scores = [self._evaluate(chrom) for chrom in population]
            else:
                fitness_scores = [self._evaluate(chrom) for chrom in population]

            best_idx = int(np.argmax(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            avg_fitness = float(np.mean(fitness_scores))

            self._fitness_history.append({
                'gen': gen,
                'best_sharpe': best_fitness,
                'avg_sharpe': avg_fitness,
            })

            decoded = self._decode(population[best_idx])

            if best_fitness > self._current_best_sharpe:
                self._current_best_sharpe = best_fitness
                self._current_best_chrom = population[best_idx]
                self._current_best_decoded = decoded

            self._save_checkpoint(force=True)

            if timeout_checker and timeout_checker():
                self.logger.info(f"[GA] 超时退出 Gen={gen}")
                break

            self.logger.info(
                f"[GA] Gen {gen:3d} | Best={best_fitness:.4f} | "
                f"Avg={avg_fitness:.4f} | Factors={decoded['factors']}"
            )

            # 提前终止
            if len(self._fitness_history) > 10:
                recent = [h['best_sharpe'] for h in self._fitness_history[-10:]]
                if max(recent) - min(recent) < 0.001:
                    self.logger.info(f"[GA] 提前终止 Gen={gen}")
                    break

            # 选择
            selected = [self._select(population, fitness_scores) for _ in range(self.pop_size)]
            # 交叉
            offsprings = []
            for i in range(0, len(selected) - 1, 2):
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(selected[i], selected[i + 1])
                    offsprings.extend([c1, c2])
                else:
                    offsprings.extend([selected[i][:], selected[i + 1][:]])
            # 突变
            offsprings = [self._mutate(c) for c in offsprings]
            # 精英保留
            population = self._elitism(population, fitness_scores, offsprings, self.elite_ratio)
            self._current_population = population

        # ---- 最终评估 ----
        if HAS_JOBLIB and self._n_jobs > 1:
            final_fitness = [self._evaluate(chrom) for chrom in population]
        else:
            final_fitness = [self._evaluate(chrom) for chrom in population]

        best_idx = int(np.argmax(final_fitness))
        best_chrom = population[best_idx]
        best_decoded = self._decode(best_chrom)

        if final_fitness[best_idx] > self._current_best_sharpe:
            self._current_best_sharpe = final_fitness[best_idx]
            self._current_best_chrom = best_chrom
            self._current_best_decoded = best_decoded

        self.logger.info(
            f"[GA] 完成 | Best={self._current_best_sharpe:.4f} | "
            f"Factors={self._current_best_decoded.get('factors', [])}"
        )

        self._save_checkpoint(force=True)

        # ========== 多重检验偏误矫正（过拟合检测）==========
# ========== 多重检验偏误矫正（过拟合检测）==========
        all_sharpes = [h['best_sharpe'] for h in self._fitness_history]
        # 收集所有代的全部染色体适应度（每代所有个体）
        all_chrom_sharpes = []
        for gen in range(len(self._fitness_history)):
            for chrom in (population if gen == len(self._fitness_history) - 1 else population):
                pass  # 已在 final_fitness 中
        # 也收集每代的 avg_sharpe 作为额外数据点
        avg_sharpes = [h['avg_sharpe'] for h in self._fitness_history]
        all_trial_sharpes = all_sharpes + avg_sharpes

        n_trials = len(all_trial_sharpes)

        # 加载配置并运行过拟合检测
        overfit_cfg = {}
        mt_cfg = {}
        if hasattr(self, 'config') and self.config:
            overfit_cfg = self.config.get('overfit_detection', {})
            mt_cfg = self.config.get('multiple_testing', {})

        overfit_enabled = overfit_cfg.get('enabled', True)
        overfit_report = {}
        if overfit_enabled and n_trials > 0:
            try:
                from src.core.overfit_detector import OverfitDetector
                detector = OverfitDetector(None, self.logger, self.config)
                # 多重检验矫正：使用所有代的 best_sharpe
                mt_result = detector.calc_multiple_testing_adjustment(
                    all_sharpes, n_trials=len(all_sharpes)
                )
                overfit_report = {
                    'multiple_testing': mt_result,
                    'n_trials': n_trials,
                }
                n_sig_bh = mt_result.get('n_significant_bh', 0)
                n_sig_bonf = mt_result.get('n_significant_bonferroni', 0)
                self.logger.info(
                    f"[GA-Overfit] 多重检验: Bonferroni显著={n_sig_bonf}/{len(all_sharpes)}, "
                    f"BH-FDR显著={n_sig_bh}/{len(all_sharpes)}"
                )
            except Exception as e:
                self.logger.warning(f"[GA-Overfit] 多重检验计算失败: {e}")

        return {
            'best_chromosome': best_chrom,
            'best_factors': best_decoded['factors'],
            'best_params': best_decoded['params'],
            'best_sharpe': float(final_fitness[best_idx]),
            'generation_history': self._fitness_history,
            # 过拟合检测结果
            'overfit_report': overfit_report,
            'all_trial_sharpes': all_trial_sharpes,
        }

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """从检查点恢复状态，返回已保存的检查点字典；无检查点则返回 {}"""
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return {}
        try:
            with open(self._checkpoint_path) as f:
                ckpt = json.load(f)
            # 恢复状态
            self._current_best_sharpe = ckpt.get('best_sharpe', -999)
            self._current_best_chrom = ckpt.get('best_chromosome')
            self._current_best_decoded = {
                'factors': ckpt.get('best_factors', []),
                'params': ckpt.get('best_params', {}),
            }
            self._fitness_history = ckpt.get('generation_history', [])
            self.logger.info(
                f"[GA] 检查点已加载 | n_gens_completed={ckpt.get('n_gens_completed', 0)}, "
                f"best_sharpe={self._current_best_sharpe:.4f}"
            )
            return ckpt
        except Exception as e:
            self.logger.warning(f"[GA] 检查点加载失败: {e}")
            return {}

    def _save_checkpoint(self, force: bool = False):
        """线程安全保存当前最优检查点（每代调用，force=True 时强制写盘）"""
        if self._checkpoint_path is None:
            return
        with self._checkpoint_lock:
            # 计算当前已完成代数（fitness_history 长度）
            n_done = len(self._fitness_history)
            ckpt = {
                'best_sharpe': self._current_best_sharpe,
                'best_chromosome': self._current_best_chrom,
                'best_factors': self._current_best_decoded.get('factors', []),
                'best_params': self._current_best_decoded.get('params', {}),
                'generation_history': self._fitness_history,
                # 续跑关键字段
                'n_gens_completed': n_done,
                'total_target': getattr(self, '_total_target', self.n_generations),
                'population': getattr(self, '_current_population', []),
            }
            tmp = str(self._checkpoint_path) + '.tmp'
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump(ckpt, f, default=str)
            Path(tmp).rename(self._checkpoint_path)

    def _init_population(self, initial_pop: List[Dict] = None) -> List[List]:
        """初始化种群"""
        population = []

        # 从初始种群转换（如果提供）
        if initial_pop:
            for indiv in initial_pop:
                chrom = self._encode_from_dict(indiv)
                population.append(chrom)

        # 随机填充至种群规模
        while len(population) < self.pop_size:
            population.append(self._create_chromosome())

        # 裁剪至种群规模
        return population[:self.pop_size]

    def _create_chromosome(self) -> List:
        """随机生成染色体"""
        chrom = []

        # 基因位 0-4: 因子选择（0=无, 1-7=具体因子）
        # 随机选1-3个不同因子
        n_factors = random.randint(1, 3)
        chosen = random.sample(range(1, len(self.FACTOR_MAP)), n_factors)
        # 不足5位补0
        factor_genes = sorted(chosen) + [0] * (5 - len(chosen))
        chrom.extend(factor_genes)

        # 基因位 5: holding_period
        chrom.append(random.randint(0, len(self.HOLDING_PERIOD_MAP) - 1))

        # 基因位 6: stop_loss
        chrom.append(random.randint(0, len(self.STOP_LOSS_MAP) - 1))

        # 基因位 7: n_stocks
        chrom.append(random.randint(0, len(self.N_STOCKS_MAP) - 1))

        return chrom

    def _encode_from_dict(self, indiv: Dict) -> List:
        """从字典编码为染色体"""
        chrom = [0] * self.CHROMOSOME_LEN

        factors = indiv.get('factors', [])
        for i, fac in enumerate(factors[:5]):
            for k, v in self.FACTOR_MAP.items():
                if v == fac:
                    chrom[i] = k
                    break

        params = indiv.get('params', {})
        for k, v in self.HOLDING_PERIOD_MAP.items():
            if v == params.get('holding_period'):
                chrom[5] = k
                break
        for k, v in self.STOP_LOSS_MAP.items():
            if abs(v - params.get('stop_loss', 0.1)) < 0.001:
                chrom[6] = k
                break
        for k, v in self.N_STOCKS_MAP.items():
            if v == params.get('n_stocks', 20):
                chrom[7] = k
                break

        return chrom

    def _decode(self, chrom: List) -> Dict:
        """解码染色体为 (factors, params)"""
        # 解析因子（去重 + 去除 None）
        factor_genes = chrom[:5]
        factors = []
        seen = set()
        for g in factor_genes:
            fac = self.FACTOR_MAP.get(g)
            if fac and fac not in seen:
                factors.append(fac)
                seen.add(fac)

        # 解析参数
        params = {
            'holding_period': self.HOLDING_PERIOD_MAP.get(chrom[5], 20),
            'stop_loss': self.STOP_LOSS_MAP.get(chrom[6], 0.10),
            'n_stocks': self.N_STOCKS_MAP.get(chrom[7], 20),
            'take_profit': 0.20,  # 默认
        }

        return {'factors': factors, 'params': params}

    def _evaluate(self, chrom: List) -> float:
        """评估染色体适应度（跑回测，返回夏普比率）"""
        decoded = self._decode(chrom)
        factors = decoded['factors']
        params = decoded['params']

        # 至少需要一个有效因子
        if not factors:
            return -999.0

        strategy = {
            'strategy_id': 'ga_chromosome',
            'strategy_name': 'ga_chromosome',
            'factors': factors,
        }

        try:
            result = self.backtester.run(
                strategy,
                params,
                self.start_date,
                self.end_date,
            )
            sharpe = result.get('sharpe_ratio', -999.0)
        except Exception as e:
            self.logger.warning(f"[GA] 评估失败: {e}")
            sharpe = -999.0

        return float(sharpe)

    def _select(
        self,
        population: List[List],
        fitness_scores: List[float],
    ) -> List:
        """锦标赛选择"""
        tournament_indices = random.sample(
            range(len(population)), min(self.tournament_size, len(population))
        )
        best_tournament_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_tournament_idx][:]

    def _crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """单点交叉"""
        if len(parent1) < 2:
            return parent1[:], parent2[:]

        pt = random.randint(1, len(parent1) - 1)
        c1 = parent1[:pt] + parent2[pt:]
        c2 = parent2[:pt] + parent1[pt:]
        return c1, c2

    def _mutate(self, chrom: List) -> List:
        """随机突变"""
        mutated = chrom[:]

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if i < 5:
                    # 因子基因突变：随机选择一个新因子
                    options = [x for x in range(len(self.FACTOR_MAP)) if x != chrom[i]]
                    if options:
                        mutated[i] = random.choice(options)
                elif i == 5:
                    mutated[i] = random.randint(0, len(self.HOLDING_PERIOD_MAP) - 1)
                elif i == 6:
                    mutated[i] = random.randint(0, len(self.STOP_LOSS_MAP) - 1)
                elif i == 7:
                    mutated[i] = random.randint(0, len(self.N_STOCKS_MAP) - 1)

        return mutated

    def _elitism(
        self,
        population: List[List],
        fitness_scores: List[float],
        offsprings: List[List],
        elite_ratio: float,
    ) -> List:
        """精英保留：将最优个体直接复制到下一代"""
        n_elite = max(1, int(len(population) * elite_ratio))

        # 合并父代和子代
        combined = list(zip(population, fitness_scores)) + list(
            zip(offsprings, [self._evaluate(ch) for ch in offsprings])
        )
        combined.sort(key=lambda x: x[1], reverse=True)

        new_pop = [chrom for chrom, _ in combined[:n_elite]]

        # 补充随机个体至种群规模
        while len(new_pop) < self.pop_size:
            new_pop.append(self._create_chromosome())

        return new_pop[: self.pop_size]
