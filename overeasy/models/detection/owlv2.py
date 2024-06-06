

from typing import List
import numpy as np
from PIL import Image
from overeasy.types import BoundingBoxModel, Detections, DetectionType
from transformers import pipeline

class OwlV2(BoundingBoxModel):
    
    def __init__(self):
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
        
    def detect(self, image: Image.Image, classes: List[str]) -> Detections:
        predictions = self.detector(image, candidate_labels=classes)
        num_preds = len(predictions)
        xyxy = np.zeros((num_preds, 4), dtype=np.int32)
        confidence = np.zeros(num_preds, dtype=np.float32)
        class_ids = np.zeros(num_preds, dtype=np.int64)
        
        width_ratio = 1.0
        height_ratio = 1.0
        width = image.width
        height = image.height
        if width > height:
            height_ratio = height / width
        elif height > width:
            width_ratio = width / height
                
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2 = pred['box']['xmin'], pred['box']['ymin'], pred['box']['xmax'], pred['box']['ymax']
            x1 = round(x1 / width_ratio)
            y1 = round(y1 / height_ratio)
            x2 = round(x2 / width_ratio)
            y2 = round(y2 / height_ratio)
            
            xyxy[i] = [x1, y1, x2, y2]
            confidence[i] = pred['score']
            class_ids[i] = classes.index(pred['label'])

        return Detections(
            xyxy=xyxy,
            confidence=confidence, 
            class_ids=class_ids,
            classes=classes,
            detection_type=DetectionType.BOUNDING_BOX
        )