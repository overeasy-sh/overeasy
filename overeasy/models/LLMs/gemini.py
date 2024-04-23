import os
from PIL import Image
from overeasy.types import MultimodalLLM, OCRModel
from typing import Optional
import google.generativeai as genai


class Gemini(MultimodalLLM, OCRModel):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("GOOGLE_API_KEY")
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self.gemini_pro_vision = genai.GenerativeModel('gemini-pro-vision')
        self.gemini_pro = genai.GenerativeModel('gemini-pro')

    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        response = self.gemini_pro_vision.generate_content([query, image], stream=True)
        response.resolve()
        return response.text


    def prompt(self, query: str) -> str:
        response = self.gemini_pro.generate_content([query])
        response.resolve()
        return response.text
    
    
    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image.")