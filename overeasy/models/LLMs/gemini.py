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

class Gemini(MultimodalLLM, OCRModel):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        if model not in models:
            warnings.warn(f"Model {model} not supported for Gemini. Please use one of the following: {models}")
        self.api_key = api_key if api_key is not None else os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        self.client = None
        
    def load_resources(self):
        if self.api_key is None:
            warnings.warn("No API key found. Using Gemini free tier. For higher usage please provide an API key, or set the GOOGLE_API_KEY environment variable.")
        else:
            genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model_name)

        
    def release_resources(self):
        self.client = None

    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        if self.client is None:
            raise ValueError("Client is not loaded. Please call load_resources() first.")
        response = self.client.generate_content([query, image], stream=True)
        response.resolve()
        return response.text


    def prompt(self, query: str) -> str:
        if self.client is None:
            raise ValueError("Client is not loaded. Please call load_resources() first.")
        response = self.client.generate_content([query])
        response.resolve()
        return response.text
    
    
    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image line by line only output the text.")