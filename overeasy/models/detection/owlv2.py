

from typing import List
import numpy as np
from PIL import Image
from overeasy.types import BoundingBoxModel, Detections, DetectionType
from transformers import pipeline
import torch

class OwlV2(BoundingBoxModel):
    def __init__(self):
        self.checkpoint = "google/owlv2-base-patch16-ensemble"
    
    def load_resources(self):
        self.detector = pipeline(model=self.checkpoint, task="zero-shot-object-detection")
    
    def release_resources(self):
        self.detector = None
    
    def detect(self, image: Image.Image, classes: List[str], threshold: float = 0) -> Detections:
        predictions = self.detector(image, candidate_labels=classes,)
        num_preds = len(predictions)
        xyxy = np.zeros((num_preds, 4), dtype=np.int32)
        confidence = np.zeros(num_preds, dtype=np.float32)
        class_ids = np.zeros(num_preds, dtype=np.int64)
                
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2 = pred['box']['xmin'], pred['box']['ymin'], pred['box']['xmax'], pred['box']['ymax']

            xyxy[i] = [x1, y1, x2, y2]
            confidence[i] = pred['score']
            class_ids[i] = classes.index(pred['label'])
        
        slice = confidence > threshold
        xyxy = xyxy[slice, :]
        confidence = confidence[slice]
        class_ids = class_ids[slice]

        return Detections(
            xyxy=xyxy,
            confidence=confidence, 
            class_ids=class_ids,
            classes=classes,
            detection_type=DetectionType.BOUNDING_BOX
        )