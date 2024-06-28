from dataclasses import dataclass
from overeasy.types import Detections, DetectionAgent
from typing import List, Optional
import numpy as np

@dataclass
class FilterClassesAgent(DetectionAgent):
    class_names: Optional[List[str]]
    class_ids: Optional[List[int]]
    
    """
    Filter detections by class name or class id.
    """
    def __init__(self, class_names: Optional[List[str]] = None, class_ids: Optional[List[int]] = None):
        self.class_names = class_names
        self.class_ids = class_ids
        
        if class_names is None and class_ids is None:
            raise ValueError("Must specify class_name or class_id")
        if class_names is not None and class_ids is not None:
            raise ValueError("Can only specify one of class_names or class_ids")

    
    def _execute(self, dets: Detections) -> Detections:
        if self.class_ids is not None:
            slice = np.isin(dets.class_ids, self.class_ids)
        elif self.class_names is not None:
            slice = np.isin(dets.class_names, self.class_names)
        else:
            raise ValueError("No filter specified")
        
        return dets[slice]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(confidence_threshold={self.confidence_threshold})"

