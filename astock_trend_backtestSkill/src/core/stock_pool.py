"""
股票池过滤器
"""


class AShareMainBoardFilter:
    """A股主板过滤器"""
    
    INCLUDE_CODES = ('600', '601', '603', '000', '001')
    EXCLUDE_CODES = ('688', '300', '8', '200')
    
    def filter(self, stock_list: list) -> list:
        """返回仅包含主板股票的列表"""
        return [s for s in stock_list 
                if any(s.startswith(prefix) for prefix in self.INCLUDE_CODES)
                and not any(s.startswith(prefix) for prefix in self.EXCLUDE_CODES)]
    
    def is_main_board(self, ts_code: str) -> bool:
        """判断是否为A股主板股票"""
        return (any(ts_code.startswith(p) for p in self.INCLUDE_CODES)
                and not any(ts_code.startswith(p) for p in self.EXCLUDE_CODES))
