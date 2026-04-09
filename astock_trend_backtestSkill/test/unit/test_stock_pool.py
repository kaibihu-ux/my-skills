import pytest
from src.core.stock_pool import AShareMainBoardFilter


class TestAShareMainBoardFilter:
    def test_include_codes(self):
        f = AShareMainBoardFilter()
        assert f.is_main_board('600519') == True
        assert f.is_main_board('601398') == True
        assert f.is_main_board('603288') == True
        assert f.is_main_board('000858') == True
        assert f.is_main_board('001696') == True
    
    def test_exclude_codes(self):
        f = AShareMainBoardFilter()
        assert f.is_main_board('688001') == False  # 科创板
        assert f.is_main_board('300750') == False  # 创业板
        assert f.is_main_board('830879') == False  # 北交所
    
    def test_filter(self):
        f = AShareMainBoardFilter()
        stocks = ['600519', '688001', '300750', '000858', '001696']
        filtered = f.filter(stocks)
        assert len(filtered) == 3
        assert '600519' in filtered
        assert '000858' in filtered
        assert '688001' not in filtered
