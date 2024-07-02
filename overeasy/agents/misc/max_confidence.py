from dataclasses import dataclass
from overeasy.types import Detections, DetectionAgent
from typing import List, Optional
import numpy as np

@dataclass
class ConfidenceFilterAgent(DetectionAgent):
    max_n: Optional[int] = None
    min_confidence: Optional[float] = None

    """
    Filter detections by confidence, either selecting the top N or those above a minimum confidence threshold.
    """
    def __init__(self, max_n: Optional[int] = None, min_confidence: Optional[float] = None):
        self.max_n = max_n
        self.min_confidence = min_confidence
        
        if max_n is None and min_confidence is None:
            raise ValueError("Must specify either max_n or min_confidence")
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be at least 1")
        if min_confidence is not None and (min_confidence < 0 or min_confidence > 1):
            raise ValueError("min_confidence must be between 0 and 1")

    def _execute(self, dets: Detections) -> Detections:
        if dets.confidence is None:
            raise ValueError("Detections must have confidence scores")
        
        indices = np.arange(len(dets.confidence))
        
        if self.min_confidence is not None:
            indices = indices[dets.confidence >= self.min_confidence]

        if self.max_n is not None:
            if len(indices) > self.max_n:
                sorted_indices = np.argsort(dets.confidence[indices])[-self.max_n:][::-1]
                indices = indices[sorted_indices]
            else:
                indices = np.argsort(dets.confidence[indices])[::-1]
        
        if len(indices) == 0:
            return Detections.empty()
        
        return dets[indices.astype(int)]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(max_n={self.max_n}, min_confidence={self.min_confidence})"
