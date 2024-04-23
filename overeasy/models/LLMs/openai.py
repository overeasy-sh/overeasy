import openai
import os
from PIL import Image
from overeasy.types import MultimodalLLM, LLM, OCRModel
from typing import Optional
import io
import base64
import requests

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class GPT4Vision(MultimodalLLM, OCRModel):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the OPENAI_API_KEY environment variable.")
        self.model = "gpt-4-turbo" 
        
    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        base64_image = encode_image_to_base64(image)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        
        return response_json['choices'][0]['message']['content'].strip()


    def prompt(self, query: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "prompt": query,
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=payload)
        response_json = response.json()

        return response_json['choices'][0]['message']['content'].strip()
    
    
    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image.")