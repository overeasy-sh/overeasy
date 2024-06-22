from overeasy.types import Agent, ExecutionNode, Detections, DetectionType, ExecutionGraph
from overeasy.logging import log_time
from typing import List, Tuple
import numpy as np

class SplitAgent(Agent):
    @log_time
    def execute(self, node: ExecutionNode) -> List[ExecutionNode]:
        result : List[ExecutionNode] = []
        if not isinstance(node.data, Detections):
            raise ValueError("ExecutionNode data must be of type Detections")
        detections: Detections = node.data
        if detections.detection_type != DetectionType.BOUNDING_BOX:
            raise ValueError("Detection type must be BOUNDING_BOX")
            
        for split_detection in detections.split():
            parent_xyxy = split_detection.xyxy[0]
            child_image = node.image.crop(parent_xyxy)
            child_detection = Detections(
                xyxy=np.array([[0, 0, child_image.width, child_image.height]]),
                class_ids=split_detection.class_ids,
                confidence=split_detection.confidence,
                classes=split_detection.classes,
                data=split_detection.data,
                detection_type=DetectionType.CLASSIFICATION
            )
            result.append(ExecutionNode(child_image, child_detection, parent_detection=split_detection))
                    
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
"""
This function walks up the execution graph to find the immediate children
produced by a SplitAgent. This is useful because these nodes will have the
parent_detection attribute set to the original detections.

Graph Representation:
         Parent
            |
       SplitAgent
            |
  -----------------
  |       |       |
*Child1* *Child2* *Child3*
  |       |       |
 C1       C2      C3

- '*Child1*', '*Child2*', and '*Child3*' are the immediate children produced by the SplitAgent.
"""
def find_immediate_children(start: List[ExecutionNode], graph: ExecutionGraph):
    parents = [x for x in start]

    while True:
        if len(parents) == 0:
            raise ValueError("No parents found")
        some_have_parents = any(parent.parent_detection is not None for parent in parents)
        all_have_parents = all(parent.parent_detection is not None for parent in parents)
        
        if some_have_parents and not all_have_parents:
            raise ValueError("Some detections have parents, but not all")
        
        if all_have_parents:
            break
        
        parents = [graph.parent_of(item) for item in parents]

    return parents
        
def combine_detections(dets: List[Detections], parent_dets: List[Detections]) -> Detections:
    if not all(isinstance(x, Detections) for x in dets) or not all(isinstance(x, Detections) for x in parent_dets):
        raise ValueError("Cannot combine detections")
    
    det_types = [x.detection_type for x in dets]
    if not all(x == det_types[0] for x in det_types):
        raise ValueError("Cannot combine detections of different types")
    
    det_type = det_types[0]
    parent_detection_type = parent_dets[0].detection_type

    if det_type == DetectionType.CLASSIFICATION:
        if any(len(det.class_ids) > 1 for det in dets):
            raise ValueError("Detections with multiple classes are not supported for combination")
        
        uniq_classes = set()
        for det in dets:
            for cls in det.classes:
                uniq_classes.add(cls)
        classes = np.array(list(uniq_classes))
        class_id_map = {cls: idx for idx, cls in enumerate(classes)}
        xyxy = []
        class_ids = []
        confidence = []        
        
        
        for det, parent_det in zip(dets, parent_dets):
            confidence.append(det.confidence[0] if det.confidence is not None else None)
            cls = det.class_names[0]
            class_ids.append(class_id_map[cls])
            xyxy.append(parent_det.xyxy[0])
        
        return Detections(
            xyxy=np.array(xyxy),
            class_ids=np.array(class_ids),
            confidence=np.array(confidence),
            classes=classes,
            detection_type=parent_detection_type
        )
        
    elif det_type == DetectionType.BOUNDING_BOX:
        # TODO: Implement this
        pass
    elif det_type == DetectionType.SEGMENTATION:
        # TODO: Implement this
        pass
   
   
    return Detections.empty()


class JoinAgent(Agent):
    # Mutates the input graph by adding a new layer of child nodes
    @log_time
    def join(self, start: List[ExecutionNode], graph: ExecutionGraph) -> List[ExecutionNode]:
        leaves = []
        
        def merge_nodes(node_and_parent_det: List[Tuple[ExecutionNode, Detections]], parent: ExecutionNode):
            if len(node_and_parent_det) == 0:
                return
            nodes = [x[0] for x in node_and_parent_det]
            parent_dets = [x[1] for x in node_and_parent_det]

            original_data = [node.data for node in nodes]
            merged_node = ExecutionNode(parent.image, original_data)
            
            if all(isinstance(x, Detections) for x in original_data):
                merged_node.data = combine_detections(original_data, parent_dets)
                
            for node in nodes:
                graph.add_child(node, merged_node)
            leaves.append(merged_node)

        immediate_children = find_immediate_children(start, graph)
        
        parent_dets = [child.parent_detection for child in immediate_children]
        parents = [graph.parent_of(child) for child in immediate_children]

        current_parent_id = None
        current_group: List[Tuple[ExecutionNode, Detections]] = []
        for cur, parent_det, parent in sorted(zip(start, parent_dets, parents), key=lambda x: x[-1].id):
            if parent.id != current_parent_id:
                merge_nodes(current_group, parent)
                current_parent_id = parent.id
                current_group = [(cur, parent_det)]
            else:
                current_group.append((cur, parent_det))
        
        if len(current_group) > 0:
            merge_nodes(current_group, parent)
        
        return leaves

    def __repr__(self):
        return f"{self.__class__.__name__}()"