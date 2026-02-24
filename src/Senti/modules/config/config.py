import importlib.util
import os
from copy import deepcopy


class ConfigDict(dict):
    """Allows dot-notation access to dictionary keys."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

class Config:
    @staticmethod
    def fromfile(filename) -> ConfigDict:
        filename = os.path.abspath(filename)
        
        # 1. Execute the python file and capture variables
        with open(filename, 'r') as f:
            code = f.read()
        
        namespace = {}
        exec(code, namespace)
        
        # 2. Filter out internal python variables (__name__, etc)
        extracted = {k: v for k, v in namespace.items() if not k.startswith('__')}
        
        # 3. Handle Inheritance (_base_)
        final_cfg = ConfigDict()
        if '_base_' in extracted:
            base_files = extracted.pop('_base_')
            if isinstance(base_files, str):
                base_files = [base_files]
            
            for base_file in base_files:
                # Resolve path relative to current config file
                base_path = os.path.join(os.path.dirname(filename), base_file)
                base_cfg = Config.fromfile(base_path)
                Config._recursive_update(final_cfg, base_cfg)
        
        # 4. Merge current file variables into the base
        Config._recursive_update(final_cfg, extracted)
        return final_cfg

    @staticmethod
    def _recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                Config._recursive_update(d[k], v)
            else:
                d[k] = Config._to_config_dict(v)

    @staticmethod
    def _to_config_dict(obj):
        """Recursively converts dicts to ConfigDicts."""
        if isinstance(obj, dict):
            return ConfigDict({k: Config._to_config_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [Config._to_config_dict(v) for v in obj]
        else:
            return obj