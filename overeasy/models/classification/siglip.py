from PIL import Image
import torch
from typing import List
from overeasy.types import Detections, ClassificationModel
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

class SigLIP(ClassificationModel):
    def __init__(self, model_card="google/siglip-base-patch16-224"):
        self.model_card = model_card
        
    def load_resources(self):
        self.processor = AutoProcessor.from_pretrained(self.model_card)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_card)

    def release_resources(self):
        self.processor = None
        self.model = None

    def classify(self, image: Image.Image, classes: List[str]) -> Detections:
        return self.model(image)