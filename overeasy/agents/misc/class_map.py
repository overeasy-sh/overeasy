from overeasy.types import DetectionAgent, Detections
from typing import Dict

class ClassMapAgent(DetectionAgent):
    def __init__(self, class_map: Dict[str, str]):
        self.class_map = class_map
    
    def _execute(self, dets: Detections) -> Detections:
        unmapped_classes = set(dets.class_names) - set(self.class_map.keys())
        if unmapped_classes:
            raise ValueError(f"Class names {unmapped_classes} not mapped")
        class_names = [self.class_map[x] for x in dets.class_names]
            
        new_dets = Detections(
            xyxy=dets.xyxy,
            class_ids=dets.class_ids,
            confidence=dets.confidence,
            classes=class_names,
            data=dets.data,
            detection_type=dets.detection_type
        )
        return new_dets
    
    def __repr__(self):
        return f"{self.__class__.__name__}(class_map={self.class_map})"
