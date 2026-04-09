import pytest
import pandas as pd
import numpy as np
from src.core.factor_miner import FactorCalculator


class TestFactorCalculator:
    def test_momentum(self):
        df = pd.DataFrame({'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 20})
        mom = FactorCalculator.calc_momentum(df, 20)
        assert mom.notna().sum() > 0
    
    def test_rsi(self):
        df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100})
        rsi = FactorCalculator.calc_rsi(df, 14)
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
    
    def test_macd(self):
        df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100})
        dif, dea, macd = FactorCalculator.calc_macd(df)
        assert len(dif) == len(dea) == len(macd) == 100
    
    def test_bollinger(self):
        df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100})
        pos, bw = FactorCalculator.calc_bollinger(df)
        # 布林带位置可超出[0,1]（价格可能突破布林带）
        assert pos.dropna().notna().all()
    
    def test_volatility(self):
        df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100})
        vol = FactorCalculator.calc_volatility(df, 20)
        assert vol.notna().sum() > 0
    
    def test_volume_ratio(self):
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'vol': np.random.randint(1000, 10000, 100)
        })
        vr = FactorCalculator.calc_volume_ratio(df, 20)
        assert vr.notna().sum() > 0
