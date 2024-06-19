from typing import Literal, Optional
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import numpy as np
from overeasy.types import Detections, DetectionType, ClassificationModel
import open_clip
import torch

class CLIP(ClassificationModel):
    def __init__(self, model_card: str = "openai/clip-vit-large-patch14"):
        self.model_card = model_card

    def load_resources(self):
        self.processor = AutoProcessor.from_pretrained(self.model_card)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_card)
        self.model.eval()

    def release_resources(self):
        self.model = None
        self.processor = None
        
    def classify(self, image: Image.Image, classes: list) -> Detections:
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs).logits_per_image
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=-1).detach().numpy()
        index = softmax_outputs.argmax()
        return Detections(
            xyxy=np.zeros((1, 4)),
            class_ids=np.array([index]),
            confidence= np.array([softmax_outputs[0, index]]),
            classes=np.array(classes),
            detection_type=DetectionType.CLASSIFICATION
        )
