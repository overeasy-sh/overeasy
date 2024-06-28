from typing import Callable, Any
from overeasy.types import DataAgent

class MapAgent(DataAgent):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def _execute(self, data: Any) -> Any:
        return self.fn(data)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fn={self.fn})"