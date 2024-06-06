from transformers import AutoImageProcessor, Dinov2ForImageClassification
from PIL import Image
import torch

from typing import Literal, List


class DinoV2Classifier():
    def __init__(self, size: Literal['small', 'base', 'large', 'giant']):
        model_name = f"dinov2-{size}-imagenet1k-1-layer"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2ForImageClassification.from_pretrained(model_name)
        self.model.eval()


    def classify(self, image: Image.Image, classes: List[str]) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()

        return self.model.config.id2label[predicted_label]