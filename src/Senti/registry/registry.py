from typing import Any, Type


class Registry:
    """simple registry to store and build modules by name"""
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._module_dict: dict[str, Type] = {}

    def register_module(self, cls: Type | None = None, *, name: str | None = None):
        def _register(c: Type):
            key = name or c.__name__
            if key in self._module_dict:
                print(f"{key} already registered in {self._name}")
            self._module_dict[key] = c
            return c
        return _register(cls) if cls is not None else _register

    def get(self, name: str) -> Type:
        return self._module_dict[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def run(self, name:str, func:str, *args:Any, **kwargs:Any):
        cls = self.get(name)
        
        if hasattr(cls, func):
            attr = getattr(cls, func)
            if callable(attr):
                return attr(*args, **kwargs)
            raise TypeError(f"Attribute '{func}' in class '{name}' is not callable.")
        
        raise AttributeError(f"Class '{name}' has no method named '{func}'")

    def list_modules(self) -> list[str]:
        return list(self._module_dict.keys())
