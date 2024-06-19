import os
from PIL import Image
from overeasy.types import MultimodalLLM, OCRModel, Model
from typing import Optional
import google.generativeai as genai
import warnings

models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro-vision",
]

class Gemini(MultimodalLLM, OCRModel, Model):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        if model not in models:
            raise ValueError(f"Model {model} not supported for Gemini. Please use one of the following: {models}")
        self.api_key = api_key if api_key is not None else os.getenv("GOOGLE_API_KEY")
        if self.api_key is None:
            warnings.warn("No API key found. Using Gemini free tier. For higher usage please provide an API key, or set the GOOGLE_API_KEY environment variable.")
        else:
            genai.configure(api_key=self.api_key)
        
    def load_resources(self):
        self.model = genai.GenerativeModel(self.model)

        
    def release_resources(self):
        self.model = None

    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        response = self.model.generate_content([query, image], stream=True)
        response.resolve()
        return response.text


    def prompt(self, query: str) -> str:
        response = self.model.generate_content([query])
        response.resolve()
        return response.text
    
    
    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image.")