from typing import Union, Callable, Any, List
from overeasy.types import DataAgent, ExecutionNode, Detections

class ToClassificationAgent(DataAgent):
    def __init__(self, fn: Union[Callable[[Any], str], Callable[[Any], List[str]]]):
        self.fn = fn
    
    def _execute(self, node: ExecutionNode) -> ExecutionNode:
        res = self.fn(node.data)
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            return ExecutionNode(node.image, Detections.from_classification(res))
        elif isinstance(res, str):
            return ExecutionNode(node.image, Detections.from_classification([res]))
        else:
            raise ValueError(f"{self.__class__.__name__} must return a string or list of strings")

    def __repr__(self):
        return f"{self.__class__.__name__}(fn={self.fn})"
