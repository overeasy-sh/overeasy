from typing import Literal, Optional
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import numpy as np
from overeasy.types import Detections, DetectionType, ClassificationModel
import open_clip
import torch

class CLIP(ClassificationModel):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

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

    
class OpenCLIPBase(ClassificationModel):
    def __init__(self, model_name):
        model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model
        self.preprocess = preprocess_val

    def classify(self, image: Image.Image, classes: list) -> Detections:
        image = self.preprocess(image).unsqueeze(0)
        text = self.tokenizer(classes, padding=True, return_tensors="pt")

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            index = text_probs.argmax()
            confidence = text_probs[0, index]
            return Detections(
                xyxy=np.zeros((1, 4)),
                class_ids=np.array([index]),
                confidence=np.array([confidence]),
                classes=np.array(classes),
                detection_type=DetectionType.CLASSIFICATION
            )

# models = ["laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]
class LaionCLIP(OpenCLIPBase):
    def __init__(self, model_name: Optional[Literal["laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]]):
        if model_name is None:
            model_name = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
        super().__init__(model_name)

class BiomedCLIP(OpenCLIPBase):
    def __init__(self):
        super().__init__('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')