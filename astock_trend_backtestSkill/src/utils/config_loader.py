import yaml
from pathlib import Path


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    def load(self) -> dict:
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.load()
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
