import os
import torch
from typing import List, Union
from overeasy.types import BoundingBoxModel, Detections
import numpy as np
from PIL import Image
from overeasy.download_utils import atomic_retrieve_and_rename
from ultralytics import YOLOWorld as YOLOWorld_ultralytics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


valid_models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8s-world", "yolov8m-world", "yolov8l-world"]
def get_yoloworld_model(model: str) -> str:
    local_model_path = os.path.join(os.path.expanduser("~/.overeasy"), model + ".pt")
    if os.path.exists(local_model_path):
        return local_model_path
    
    url = None
    if model in valid_models:
        url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model}.pt"
    else:
        raise ValueError(f"Model {model} not valid. Valid models are: {valid_models}")
    
    atomic_retrieve_and_rename(url, local_model_path)
    
    return local_model_path

# Load a pretrained YOLOv8s-worldv2 model
class YOLOWorld(BoundingBoxModel):
    def __init__(self, model: str = "yolov8l-worldv2"):
        self.model_name = model
    
    def load_resources(self):
        self.model_path = get_yoloworld_model(self.model_name)
        self.model = YOLOWorld_ultralytics(self.model_path)

    def release_resources(self):
        self.model = None

    def detect(self, image: Image.Image, classes: List[str]) -> Detections:
        self.model.set_classes(classes)
        results = self.model.predict(image, verbose=False)
        detections = Detections.from_ultralytics(results[0])
        return detections
    