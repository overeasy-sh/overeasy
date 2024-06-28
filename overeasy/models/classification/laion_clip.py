import open_clip
import torch
import numpy as np
from PIL import Image
from overeasy.types import Detections, DetectionType, ClassificationModel
from typing import List, Optional

class OpenCLIPBase(ClassificationModel):
    def __init__(self, model_card):
        self.model_card = model_card
        self.device = "cpu" # TODO: Work out the cuda impl
        
    def load_resources(self):
        model, _, preprocess_val = open_clip.create_model_and_transforms(self.model_card)
        self.tokenizer = open_clip.get_tokenizer(self.model_card)
        self.model = model
        self.model.to(self.device)
        self.preprocess = preprocess_val

    def release_resources(self):
        self.model = None
        self.tokenizer = None
        self.preprocess = None

    def classify(self, image: Image.Image, classes: List[str]) -> Detections:
        image = self.preprocess(image).to(self.device).unsqueeze(0)
        text = self.tokenizer(classes)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
            index = text_probs.argmax()
            confidence = text_probs[0, index]
            return Detections(
                xyxy=np.zeros((1, 4)),
                class_ids=np.array([index]),
                confidence=np.array([confidence]),
                classes=np.array(classes),
                detection_type=DetectionType.CLASSIFICATION
            )
    
    def batch_classify(self, images: List[Image.Image], classes: List[str], top_k: int = 1) -> List[Detections]:
        preprocessed_images = [self.preprocess(image).unsqueeze(0) for image in images]
        image_input = torch.cat(preprocessed_images)
        text_tokens = self.tokenizer(classes)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.cpu().topk(top_k, dim=-1)

            detections: List[Detections] = []
            for i in range(len(images)):
                detections.append(Detections(
                    xyxy=np.zeros((top_k, 4)),
                    class_ids=top_labels[i].numpy(),
                    confidence=top_probs[i].numpy(),
                    classes=np.array(classes)[top_labels[i].numpy()],
                    detection_type=DetectionType.CLASSIFICATION
                ))
            return detections

models = ["laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]
class LaionCLIP(OpenCLIPBase):
    def __init__(self, model_name: str = models[1]):
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        super().__init__(model_name)

class BiomedCLIP(OpenCLIPBase):
    def __init__(self):
        super().__init__('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')