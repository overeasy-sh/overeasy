import itertools
from typing import Any, Dict, List, Union, Optional, Any
from abc import ABC, abstractmethod
from typing import List

from overeasy.logging import log_time
from .detections import Detections
from PIL import Image
from overeasy.visualize_utils import annotate, annotate_with_string
from dataclasses import dataclass
from collections import defaultdict

__all__ = [
    "OCRModel",
    "LLM",
    "MultimodalLLM",
    "BoundingBoxModel",
    "ExecutionNode",
    "NullExecutionNode",
    "Node",
    "ExecutionGraph",
    "Model",
    "Agent",
    "ImageAgent",
    "DetectionAgent",
    "TextAgent",
    "DataAgent",
    "ClassificationModel",
    "CaptioningModel",
]
    
@dataclass
class ExecutionNode:
    image: Image.Image
    data: Union[Detections, Any]
    parent_detection: Optional[Detections] = None
    
    def data_is_detections(self) -> bool:
        return isinstance(self.data, Detections)
    
    def visualize(self, seed: Optional[int] = None) -> Image.Image:
        if self.data_is_detections():
            return annotate(self.image, self.data, seed)
        else:
            return annotate_with_string(self.image, str(self.data))
    
    @property
    def id(self):
        return id(self)
    
@dataclass
class NullExecutionNode():
    parent_detection: Optional[Detections] = None
    
    @property
    def id(self):
        return id(self)
    
Node = Union[ExecutionNode, NullExecutionNode]
    
@dataclass
class ExecutionGraph:
    root: ExecutionNode
    edges: Dict[int, List[Node]]
    parent: Dict[int, List[Node]]
    
    def __init__(self, root: ExecutionNode):
        self.root = root
        self.edges = {}
        self.parent = {}
    
    def add_child(self, parent: Node, child: Node):
        assert isinstance(parent, ExecutionNode) or isinstance(parent, NullExecutionNode)
        assert isinstance(child, ExecutionNode) or isinstance(child, NullExecutionNode)
        
        if parent.id == child.id:
            raise ValueError("Cannot self loops to execution graph")
        
        # Add edges and reverse edges
        if parent.id not in self.edges:
            self.edges[parent.id] = []
        self.edges[parent.id].append(child)
        
        if child.id not in self.parent:
            self.parent[child.id] = []
        self.parent[child.id].append(parent)

    def ascii_graph(self):
        print("ExecutionGraph")
        id_counter = itertools.count()
        id_map = {}
        
        def get_mapped_id(original_id):
            if original_id not in id_map:
                id_map[original_id] = next(id_counter)
            return id_map[original_id]
        
        def print_node(node, prefix="", is_last=True):
            connector = "└── " if is_last else "├── "
            print(prefix + connector + f"Node(ID={get_mapped_id(node.id)})")
            if node.id in self.edges:
                children = self.edges[node.id]
                for i, child in enumerate(children):
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_node(child, next_prefix, i == len(children) - 1)
  
        print_node(self.root)
        
    
    def parent_of(self, node: Node) -> Node:
        if node.id not in self.parent:
            raise ValueError(f"Node {node.id} has no parent")
        parent_list = self.parent[node.id]
        if len(parent_list) > 1:
            print("Multiple parents", parent_list)
            raise ValueError("Node has multiple parents")
        return self.parent[node.id][0]
    
    def children(self, node: Node) -> List[Node]:
        if node.id not in self.edges:
            raise ValueError(f"Node {node.id} has no children")
        
        return self.edges[node.id]
    
    def __getitem__(self, node: Node) -> List[Node]:
        return self.edges[node.id]
    
    def __repr__(self):
        return f"ExecutionGraph(root={self.root}, edges={self.edges}, parent={self.parent})"
    
    # TODO: This does a lot of heavy lifting so it should have some more strict ordering properties                    
    def top_sort(self) -> List[List[Node]]:
        # Create a copy of the edges dictionary to avoid mutating the original
        if len(self.edges) == 0:
            return [[self.root]] 
        
        edges_copy = {node_id: neighbors.copy() for node_id, neighbors in self.edges.items()}
        
        # Initialize the in-degree dictionary
        in_degree : Dict[int, int] = defaultdict(int)
        for neighbors in edges_copy.values():
            for node in neighbors:
                in_degree[node.id] += 1
        
        # Initialize the queue with nodes having in-degree 0
        queue : List[Node] = [node for node in [self.root] if in_degree[node.id] == 0]
        
        # Initialize the sorted nodes list
        sorted_nodes = []
        
        # Perform the topological sort
        while queue:
            level_nodes = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level_nodes.append(node)
                
                # Decrement the in-degree of the neighbors and add them to the queue if in-degree becomes 0
                if node.id in edges_copy:
                    for neighbor in edges_copy[node.id]:
                        in_degree[neighbor.id] -= 1
                        if in_degree[neighbor.id] == 0:
                            queue.append(neighbor)
            
            sorted_nodes.append(level_nodes)
        
        return sorted_nodes

class Model(ABC):
    # Load model resources, such as allocating memory for weights, only once when this method is called.
    # Note: Weights can still be downloaded to disk, just not allocated to VRAM until load_resources is called.
    @abstractmethod
    def load_resources(self):
        pass
    
    # Free up the allocated resources, such as memory for weights, when this method is called.
    @abstractmethod
    def release_resources(self):
        pass
    
    def __del__(self):
        self.release_resources()

    
# OCRModel reads text from an image
class OCRModel(Model):
    @abstractmethod
    def parse_text(self, image: Image.Image) -> str:
        pass
    
class LLM(Model):    
    @abstractmethod
    def prompt(self, query: str) -> str:
        pass
    
#TODO: Add support for prompting with multiple images
class MultimodalLLM(LLM):
    @abstractmethod
    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        pass

class CaptioningModel(Model):
    @abstractmethod
    def caption(self, image: Image.Image) -> str:
        pass
    
class BoundingBoxModel(Model):
    @abstractmethod
    def detect(self, image: Image.Image, classes: List[str]) -> Detections:
        pass
     
class ClassificationModel(Model):
    @abstractmethod
    def classify(self, image: Image.Image, classes: list) -> Detections:
        pass

class Agent(ABC):
    pass
    
class ImageAgent(Agent):
    @abstractmethod
    def _execute(self, image: Image.Image) -> ExecutionNode:
        pass
    
    @log_time
    def execute(self, image: Image.Image) -> ExecutionNode:
        if not isinstance(image, Image.Image):
            raise ValueError("ImageAgent received non-image data")
        return self._execute(image)
    
class DetectionAgent(Agent):
    @abstractmethod
    def _execute(self, data: Detections) -> Detections:
        pass
    
    @log_time
    def execute(self, data: Detections) -> Detections:
        if not isinstance(data, Detections):
            raise ValueError("DetectionAgent received non-detection data")
        return self._execute(data)
    
class TextAgent(Agent):
    @abstractmethod
    def _execute(self, data: str) -> Any:
        pass
    
    @log_time
    def execute(self, data: str) -> Any:
        if not isinstance(data, str):
            raise ValueError("TextAgent received non-string data")
        return self._execute(data)

class DataAgent(Agent):
    @abstractmethod
    def _execute(self, data: Any) -> Any:
        pass
    
    @log_time
    def execute(self, data: Any) -> Any:
        return self._execute(data)