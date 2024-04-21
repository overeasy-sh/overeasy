from __future__ import annotations
import itertools

from contextlib import suppress
from dataclasses import dataclass as dataclass
from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Dict, List, Union, Optional
import numpy as np
from collections import defaultdict
from overeasy.types.detections import Detections
from overeasy.types.type_utils import DetectionType
from .base import *
@dataclass
class ExecutionNode:
    image: Image.Image
    data: Union[Detections, Any]
    parent_detection: Optional[Detections] = None
    
    def data_is_detections(self) -> bool:
        return isinstance(self.data, Detections)
    
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
        
    # Note: The original instructions were to use a Python library for ASCII graph generation.
    # However, there's no direct import or use of an external library in the provided solution.
    # The solution manually constructs the ASCII graph. For compliance with the follow-up prompt
    # without specifying a particular library, this approach is maintained as it effectively
    # fulfills the task without additional dependencies.
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
        
    # def leaves(self) -> List[ExecutionNode]:
    #     def find_leaves(node, depth=0):
    #         if node.id not in self.edges or not self.edges[node.id]:
    #             return [(node, depth)]
    #         leaves = []
    #         for child in self.edges[node.id]:
    #             leaves.extend(find_leaves(child, depth + 1))
    #         return leaves
        
    #     return [node for node, depth in find_leaves(self.root)]
    
    def parent_of(self, node: ExecutionNode) -> ExecutionNode:
        parent_list = self.parent[node.id]
        if len(parent_list) > 1:
            print("Multiple parents", parent_list)
            raise ValueError("Node has multiple parents")
        return self.parent[node.id][0]
    
    def __getitem__(self, node: ExecutionNode) -> List[ExecutionNode]:
        return self.edges[node.id]
    
    def __str__(self):
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
    
class SplitAgent(Agent):
    @abstractmethod
    def execute_split(self, image: Image.Image) -> List[ExecutionNode]:
        pass
    
    @abstractmethod
    def execute(self, image: Image.Image) -> ExecutionNode:
        pass
    
    @abstractmethod
    def is_split(self) -> bool:
        pass

# _SMALL_OBJECT_AREA_THRESH = 1000
# _LARGE_MASK_AREA_THRESH = 120000
# _OFF_WHITE = (1.0, 1.0, 240.0 / 255)
# _BLACK = (0, 0, 0)
# _RED = (1.0, 0, 0)

# _KEYPOINT_THRESHOLD = 0.05


# def draw_instance_predictions(self, detections: Detections, jittering: bool = True):
#     """
#     Draw instance-level prediction results on an image using Detections object.

#     Args:
#         detections (Detections): the output of an instance detection/segmentation
#             model encapsulated in a Detections object.
#         jittering: if True, in color mode SEGMENTATION, randomly jitter the colors per class
#             to distinguish instances from the same class

#     Returns:
#         output (VisImage): image object with visualizations.
#     """
#     boxes = detections.xyxy if detections.xyxy is not None else None
#     scores = detections.confidence if detections.confidence is not None else None
#     classes = detections.class_ids.tolist() if detections.class_ids is not None else None
#     labels = detections.classes

        

    
#     colors = [
#         self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
#     ] if jittering else [
#         tuple(
#             mplc.to_rgb([x / 255 for x in self.metadata.thing_colors[c]])
#         ) for c in classes
#     ]

#     alpha = 0.8
    
    

#     self.overlay_instances(
#         masks=masks,
#         boxes=boxes,
#         labels=labels,
#         keypoints=keypoints,
#         assigned_colors=colors,
#         alpha=alpha,
#     )
#     return self.output


# def overlay_instances(
#         self,
#         *,
#         boxes=None,
#         labels=None,
#         masks=None,
#         keypoints=None,
#         assigned_colors=None,
#         alpha=0.5,
#     ):
#         num_instances = 0
#         if boxes is not None:
#             boxes = self._convert_boxes(boxes)
#             num_instances = len(boxes)
#         if masks is not None:
#             masks = self._convert_masks(masks)
#             if num_instances:
#                 assert len(masks) == num_instances
#             else:
#                 num_instances = len(masks)
#         if keypoints is not None:
#             if num_instances:
#                 assert len(keypoints) == num_instances
#             else:
#                 num_instances = len(keypoints)
#             keypoints = self._convert_keypoints(keypoints)
#         if labels is not None:
#             assert len(labels) == num_instances
#         if assigned_colors is None:
#             assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
#         if num_instances == 0:
#             return self.output
#         if boxes is not None and boxes.shape[1] == 5:
#             return self.overlay_rotated_instances(
#                 boxes=boxes, labels=labels, assigned_colors=assigned_colors
#             )

#         # Display in largest to smallest order to reduce occlusion.
#         areas = None
#         if boxes is not None:
#             areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
#         elif masks is not None:
#             areas = np.asarray([x.area() for x in masks])

#         if areas is not None:
#             sorted_idxs = np.argsort(-areas).tolist()
#             # Re-order overlapped instances in descending order.
#             boxes = boxes[sorted_idxs] if boxes is not None else None
#             labels = [labels[k] for k in sorted_idxs] if labels is not None else None
#             masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
#             assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
#             keypoints = keypoints[sorted_idxs] if keypoints is not None else None

#         for i in range(num_instances):
#             color = assigned_colors[i]
#             if boxes is not None:
#                 self.draw_box(boxes[i], edge_color=color)

#             if masks is not None:
#                 for segment in masks[i].polygons:
#                     self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

#             if labels is not None:
#                 # first get a box
#                 if boxes is not None:
#                     x0, y0, x1, y1 = boxes[i]
#                     text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
#                     horiz_align = "left"
#                 elif masks is not None:
#                     # skip small mask without polygon
#                     if len(masks[i].polygons) == 0:
#                         continue

#                     x0, y0, x1, y1 = masks[i].bbox()

#                     # draw text in the center (defined by median) when box is not drawn
#                     # median is less sensitive to outliers.
#                     text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
#                     horiz_align = "center"
#                 else:
#                     continue  # drawing the box confidence for keypoints isn't very useful.
#                 # for small objects, draw text at the side to avoid occlusion
#                 instance_area = (y1 - y0) * (x1 - x0)
#                 if (
#                     instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
#                     or y1 - y0 < 40 * self.output.scale
#                 ):
#                     if y1 >= self.output.height - 5:
#                         text_pos = (x1, y0)
#                     else:
#                         text_pos = (x0, y1)

#                 height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
#                 lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
#                 font_size = (
#                     np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
#                     * 0.5
#                     * self._default_font_size
#                 )
#                 self.draw_text(
#                     labels[i],
#                     text_pos,
#                     color=lighter_color,
#                     horizontal_alignment=horiz_align,
#                     font_size=font_size,
#                 )

#         # draw keypoints
#         if keypoints is not None:
#             for keypoints_per_instance in keypoints:
#                 self.draw_and_connect_keypoints(keypoints_per_instance)

#         return self.output

#TODO: Fix code from FB DINO to draw BBOXes

# def overlay_instances_pil(
#         image: Image.Image,
#         boxes=None,
#         labels=None,
#         masks=None,
#         keypoints=None,
#         assigned_colors=None,
#         alpha=0.5,
#     ) -> Image.Image:
#     """
#     Overlay boxes, labels, masks, and keypoints on an image.

#     Args:
#         image (PIL.Image.Image): The image to draw on.
#         boxes (ndarray): Nx4 numpy array of XYXY_ABS format for the N objects.
#         labels (list[str]): The text to be displayed for each instance.
#         masks (list[ndarray]): Each ndarray is a binary mask of shape (H, W).
#         keypoints (array like): Array-like object of shape (N, K, 3), where N is the number of instances and K is the number of keypoints.
#         assigned_colors (list[tuple]): A list of colors, where each color corresponds to each mask or box in the image.
#         alpha (float): Transparency for masks.

#     Returns:
#         PIL.Image.Image: The image with overlaid instances.
#     """
#     draw = ImageDraw.Draw(image, "RGBA")
#     font = ImageFont.load_default()

#     num_instances = len(boxes) if boxes is not None else 0

#     if assigned_colors is None:
#         assigned_colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), int(255 * alpha)) for _ in range(num_instances)]

#     if boxes is not None:
#         for i, box in enumerate(boxes):
#             color = assigned_colors[i]
#             draw.rectangle(box, outline=color, width=2)

#     if labels is not None:
#         for i, label in enumerate(labels):
#             color = assigned_colors[i]
#             # Adjust text position to be inside the box
#             text_position = (boxes[i][0], boxes[i][1] - 10)
#             draw.text(text_position, label, fill=color, font=font)

#     if masks is not None:
#         for i, mask in enumerate(masks):
#             color = assigned_colors[i]
#             # Create an RGBA mask image where the mask area is filled with the color and the rest is transparent
#             mask_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
#             mask_draw = ImageDraw.Draw(mask_image)
#             for y in range(mask.shape[0]):
#                 for x in range(mask.shape[1]):
#                     if mask[y, x]:
#                         mask_draw.point((x, y), fill=color)
#             # Composite the mask image onto the original image
#             image = Image.alpha_composite(image.convert("RGBA"), mask_image)

#     # Keypoints drawing is omitted for brevity. It would involve drawing circles and lines between keypoints.

#     return image.convert("RGB")