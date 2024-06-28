from overeasy.types import Agent, ExecutionNode, Detections, DetectionType, ExecutionGraph, NullExecutionNode
from overeasy.logging import log_time
from typing import List, Tuple, Union
import numpy as np

Node = Union[ExecutionNode, NullExecutionNode]

class SplitAgent(Agent):
    @log_time
    def execute(self, node: ExecutionNode) -> List[Node]:
        result : List[Node] = []
        if not isinstance(node.data, Detections):
            raise ValueError(f"ExecutionNode data must be of type Detections, got {type(node.data)}")
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
        
        if len(result) == 0:
            result.append(NullExecutionNode(parent_detection=detections))
        
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
     
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
            if len(parent_det.class_ids) == 0:
                continue
            confidence.append(det.confidence[0] if det.confidence is not None else None)
            cls = det.class_names[0]
            class_ids.append(class_id_map[cls])
            xyxy.append(parent_det.xyxy[0])

        return Detections(
            xyxy=np.array(xyxy) if len(xyxy) > 0 else np.zeros((0, 4)),
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
    def join(self, graph: ExecutionGraph, target_split: int) -> List[Node]:
        topsort = graph.top_sort()
        parents_all = topsort[target_split]
        
        if len(parents_all) == 1 and isinstance(parents_all[0], NullExecutionNode):
            if len(topsort[-1]) == 1 and isinstance(topsort[-1][0], NullExecutionNode):
                null_child = NullExecutionNode()
                graph.add_child(topsort[-1][0], null_child)
                return [null_child]
            else:
                raise ValueError("Graph is formatted incorrectly")
        
        leaves: List[Node] = []        
        def merge_nodes(node_and_parent_det: List[Tuple[Node, Detections]], parent: ExecutionNode):
            if len(node_and_parent_det) == 0:
                return
            nodes = [x[0] for x in node_and_parent_det]
            parent_dets = [x[1] for x in node_and_parent_det]

            original_data = [node.data if isinstance(node, ExecutionNode) else None for node in nodes]
            merged_node = ExecutionNode(parent.image, original_data)
            
            if all(isinstance(x, Detections) for x in original_data):
                merged_node.data = combine_detections(original_data, parent_dets) #type: ignore
                
            for node in nodes:
                graph.add_child(node, merged_node)
            leaves.append(merged_node)
        
        # Filter out non execution nodes
        parents = [p for p in parents_all if isinstance(p, ExecutionNode)]
        split_children = [graph.children(p) for p in parents]
        to_merge = topsort[-1]
        
        ind = 0
        for (parent_list, parent_node) in zip(split_children, parents):
            nodes: List[Tuple[Node, Detections]] = []
            for child in parent_list:
                if child.parent_detection is None:
                    raise ValueError("Parent detection is None")
                node = to_merge[ind]

                nodes.append((node, child.parent_detection))
                ind+=1
            if isinstance(parent_node, NullExecutionNode):
                raise ValueError("Parent node is NullExecutionNode")
            merge_nodes(nodes, parent_node)


        
        
        return leaves

    def __repr__(self):
        return f"{self.__class__.__name__}()"