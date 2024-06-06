from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
from overeasy.types import ClassificationModel
import open_clip
import torch

class CLIP(ClassificationModel):
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

    def classify(self, image: Image.Image, classes: list) -> str:
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs).logits_per_image
        # Get the class with the smallest distance
        min_distance_class = classes[outputs.argmin()]
        return min_distance_class
    
class OpenCLIPBase(ClassificationModel):
    def __init__(self, model_name):
        model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model
        self.preprocess = preprocess_val

    def classify(self, image: Image.Image, classes: list) -> str:
        image = self.preprocess(image).unsqueeze(0)
        text = self.tokenizer(classes, padding=True, return_tensors="pt")

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            min_distance_class = classes[text_probs.argmin()]

        return min_distance_class

class LaionCLIP(OpenCLIPBase):
    def __init__(self):
        super().__init__('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

class BiomedCLIP(OpenCLIPBase):
    def __init__(self):
        super().__init__('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')