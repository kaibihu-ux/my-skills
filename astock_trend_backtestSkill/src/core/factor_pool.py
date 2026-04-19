import json
from datetime import datetime
from typing import List, Dict


class FactorPoolManager:
    """因子池管理器"""
    
    def __init__(self, store, logger, config: Dict):
        self.store = store
        self.logger = logger
        self.config = config
        self.max_size = config['factor_pool']['max_size']
        self.promotion_ir = config['factor_pool']['promotion_threshold_ir']
        self.eviction_ir = config['factor_pool'].get('eviction_threshold_ir', 0.3)
        self.eviction_ic = config['factor_pool'].get('eviction_threshold_ic', 0.02)
    
    def add_factor(self, factor_eval: Dict):
        """添加或更新因子到池（DuckDB MERGE upsert）"""
        src_sql = """SELECT
            ? AS factor_name,
            ? AS avg_ic,
            ? AS avg_ir,
            ? AS ic_series,
            0 AS rank,
            'active' AS status,
            CURRENT_TIMESTAMP AS updated_at"""
        merge_sql = f"""MERGE INTO factor_pool AS target
        USING ({src_sql}) AS source
        ON target.factor_name = source.factor_name
        WHEN MATCHED THEN UPDATE SET
            avg_ic = source.avg_ic,
            avg_ir = source.avg_ir,
            ic_series = source.ic_series,
            status = source.status,
            updated_at = source.updated_at
        WHEN NOT MATCHED THEN INSERT (factor_name, avg_ic, avg_ir, ic_series, rank, status, updated_at)
            VALUES (source.factor_name, source.avg_ic, source.avg_ir, source.ic_series, source.rank, source.status, source.updated_at)"""
        self.store.execute(merge_sql, [
            factor_eval['factor_name'],
            factor_eval.get('ic_mean', 0),
            factor_eval.get('ir', 0),
            json.dumps(factor_eval)
        ])
    
    def get_top_factors(self, n: int = 20, include_evicted: bool = False) -> List[Dict]:
        """获取Top N因子（默认仅active，include_evicted=True时包含已淘汰因子）"""
        if include_evicted:
            sql = "SELECT * FROM factor_pool ORDER BY avg_ir DESC LIMIT ?"
        else:
            sql = "SELECT * FROM factor_pool WHERE status = 'active' ORDER BY avg_ir DESC LIMIT ?"
        df = self.store.df(sql, [n])
        return df.to_dict('records')
    
    def rebalance(self):
        """因子池重平衡 - IC和IR双重淘汰"""
        df = self.store.df("SELECT * FROM factor_pool ORDER BY avg_ir DESC")
        
        # 淘汰条件：IC < eviction_ic 且 IR < eviction_ir
        for _, row in df.iterrows():
            ic = row.get('avg_ic', 0)
            ir = row.get('avg_ir', 0)
            if ic < self.eviction_ic and ir < self.eviction_ir:
                self.store.execute(
                    "UPDATE factor_pool SET status = 'evicted' WHERE factor_name = ?",
                    [row['factor_name']]
                )
                self.logger.info(f"淘汰因子(低IC低IR): {row['factor_name']}, IC={ic:.4f}, IR={ir:.4f}")
        
        # 更新排名
        df = self.store.df("SELECT * FROM factor_pool WHERE status = 'active' ORDER BY avg_ir DESC")
        for i, (_, row) in enumerate(df.iterrows()):
            self.store.execute(
                "UPDATE factor_pool SET rank = ? WHERE factor_name = ?",
                [i+1, row['factor_name']]
            )
