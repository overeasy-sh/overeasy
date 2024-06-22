from typing import Callable, Any
from overeasy.types import DataAgent, ExecutionNode

class MapAgent(DataAgent):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def _execute(self, node: ExecutionNode) -> ExecutionNode:
        return ExecutionNode(node.image, self.fn(node.data))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fn={self.fn})"