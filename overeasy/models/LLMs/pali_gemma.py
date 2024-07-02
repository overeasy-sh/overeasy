from PIL import Image
from overeasy.types import MultimodalLLM
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from typing import Optional

# Make sure to set your HuggingFace token to use PaliGemma
class PaliGemma(MultimodalLLM):
    SIZES = [224, 448, 896] # Image sizes the model was trained on
    
    MODEL_OPTIONS = (
        [f"google/paligemma-3b-pt-{size}" for size in SIZES] +
        [f"google/paligemma-3b-mix-{size}" for size in SIZES] +
        [f"google/paligemma-3b-ft-vqav2-{size}" for size in SIZES] +  # Diagram Understanding - 85.64 Accuracy on VQAV2
        [f"google/paligemma-3b-ft-cococap-{size}" for size in SIZES] +  # COCO Captions - 144.6 CIDEr
        [f"google/paligemma-3b-ft-science-qa-{size}" for size in SIZES] +  # Science Question Answering - 95.93 Accuracy on ScienceQA Img subset with no CoT
        [f"google/paligemma-3b-ft-refcoco-seg-{size}" for size in SIZES] +  # Understanding References to Specific Objects in Images - 76.94 Mean IoU on refcoco, 72.18 Mean IoU on refcoco+, 72.22 Mean IoU on refcocog
        [f"google/paligemma-3b-ft-rsvqa-hr-{size}" for size in SIZES]  # Remote Sensing Visual Question Answering - 92.61 Accuracy on test, 90.58 Accuracy on test2
    )
    
    def __init__(self, model_card: str = "google/paligemma-3b-mix-448", device: Optional[str] = None):
        self.model_card = model_card
        if self.model_card not in self.MODEL_OPTIONS:
            raise ValueError(f"Model {self.model_id} not found. Please select a valid model from {self.MODEL_OPTIONS}.")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = None
        self.processor = None

    def load_resources(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_card).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_card)

    def release_resources(self):
        self.model = None
        self.processor = None

    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        if self.model is None or self.processor is None:
            raise ValueError("Model is not loaded. Please call load_resources() first.")
        
        inputs = self.processor(query, image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.processor.decode(output[0], skip_special_tokens=True)[len(query):]

    def prompt(self, query: str) -> str:
        if self.model is None or self.processor is None:
            raise ValueError("Model is not loaded. Please call load_resources() first.")
        
        inputs = self.processor(query, None, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.processor.decode(output[0], skip_special_tokens=True)[len(query):]