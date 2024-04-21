import os
import torch
from typing import List, Union
from overeasy.types import BoundingBoxModel, Detections
import numpy as np
from PIL import Image
import supervision as sv
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLOWorldModel(BoundingBoxModel):
    
    def __init__(self):
        self.model = YOLOWorld( model_id="yolo_world/l")


    def detect(self, image: Image.Image, classes: List[str], box_threshold=0.35, text_threshold=0.25) -> Detections:
        results = self.model.infer(load_image(input), text=classes, conf=box_threshold)

        detections = sv.Detections.from_inference(results)

        detections = detections[detections.confidence > box_threshold]

        return detections