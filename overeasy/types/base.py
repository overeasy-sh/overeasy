import itertools
from typing import Any, Dict, List, Union, Optional, Any
from abc import ABC, abstractmethod
from typing import List
from .detections import Detections
from PIL import Image
from overeasy.visualize_utils import annotate
from dataclasses import dataclass
from collections import defaultdict

# OCRModel reads text from an image
class OCRModel(ABC):
    @abstractmethod
    def parse_text(self, image: Image.Image) -> str:
        pass

# TODO: Return type needs to be fleshed out
# class FaceRecognitionModel(ABC):
#     @abstractmethod
#     def detect_faces(self, image: Image.Image) -> list:
#         pass
    
class LLM(ABC):    
    @abstractmethod
    def prompt(self, query: str) -> str:
        pass
    
class MultimodalLLM(LLM):
    
    @abstractmethod
    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        pass
    

class BoundingBoxModel(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, classes: List[str]) -> Detections:
        pass
    
from matplotlib import pyplot as plt
import numpy as np
@dataclass
class ExecutionNode:
    image: Image.Image
    data: Union[Detections, Any]
    parent_detection: Optional[Detections] = None
    
    def data_is_detections(self) -> bool:
        return isinstance(self.data, Detections)
    
    
    def visualize(self) -> Image.Image:
        if self.data_is_detections():
            return annotate(self.image, self.data)
        else:
            fig, ax = plt.subplots()
            ax.imshow(np.array(self.image))
            ax.axis('off')
            data_str = str(self.data)
            plt.figtext(0.5, 0.01, data_str, wrap=True, horizontalalignment='center', fontsize=12)
            fig.canvas.draw()  
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # type: ignore
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)  
            image = Image.fromarray(data)
            return image.convert('RGB')
    
    @property
    def id(self):
        return id(self)
    
@dataclass
class ExecutionGraph:
    root: ExecutionNode
    edges: Dict[int, List[ExecutionNode]]
    parent: Dict[int, List[ExecutionNode]]
    
    def __init__(self, root: ExecutionNode):
        self.root = root
        self.edges = {}
        self.parent = {}
    
    def add_child(self, parent: ExecutionNode, child: ExecutionNode):
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
        
    
    def parent_of(self, node: ExecutionNode) -> ExecutionNode:
        parent_list = self.parent[node.id]
        if len(parent_list) > 1:
            print("Multiple parents", parent_list)
            raise ValueError("Node has multiple parents")
        return self.parent[node.id][0]
    
    def __getitem__(self, node: ExecutionNode) -> List[ExecutionNode]:
        return self.edges[node.id]
    
    def __repr__(self):
        return f"ExecutionGraph(root={self.root}, edges={self.edges}, parent={self.parent})"
    
                    
    def top_sort(self) -> List[List[ExecutionNode]]:
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
        queue = [node for node in [self.root] if in_degree[node.id] == 0]
        
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


class Agent(ABC):
    pass
    
class ImageAgent(Agent):
    @abstractmethod
    def execute(self, image: Image.Image) -> ExecutionNode:
        pass

class DetectionAgent(Agent):
    @abstractmethod
    def execute(self, data: Detections) -> Detections:
        pass
    
class TextAgent(Agent):
    @abstractmethod
    def execute(self, data: str) -> Any:
        pass

class DataAgent(Agent):
    @abstractmethod
    def execute(self, node: ExecutionNode) -> ExecutionNode:
        pass



class ClassificationModel(ABC):
    @abstractmethod
    def classify(self, image: Image.Image, classes: list) -> Detections:
        pass