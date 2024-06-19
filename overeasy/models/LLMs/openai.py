import os
from PIL import Image
from overeasy.types import MultimodalLLM, LLM, OCRModel, Model
from typing import Literal, Optional
import io
import base64
import requests
import warnings
import backoff

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

current_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613"
]

class RateLimitError(Exception):
    pass

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=7)
def _post(self, url, headers, json):
    response = requests.post(url, headers=headers, json=json)
    response.raise_for_status()
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded, retrying...")

    return response.json()


def _prompt(self, query: str, model: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
    }
    payload = {
        "model": model,
        "prompt": query,
        "max_tokens": 500
    }

    response_json = _post("https://api.openai.com/v1/completions", headers=headers, json=payload)

    return response_json['choices'][0]['message']['content'].strip()

class GPT(LLM):
    def __init__(self, api_key: Optional[str] = None, model:str = "gpt-3.5-turbo"):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.model not in current_models:
            warnings.warn(f"Model {model} may not be supported. Please provide a valid model.")

    def prompt(self, query: str) -> str:
        return _prompt(self, query, self.model)
    
    def load_resources(self):
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the OPENAI_API_KEY environment variable.")
    
    def release_resources(self):
        super().release_resources()

class GPTVision(MultimodalLLM, OCRModel):
    def __init__(self, api_key: Optional[str] = None,
                 model : Literal["gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"] = "gpt-4o"
                 ):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
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
        
        response = _post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        
        return response_json['choices'][0]['message']['content'].strip()

    def prompt(self, query: str) -> str:
        return _prompt(self, query, self.model)
    
    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image.")
    
    def load_resources(self):
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the OPENAI_API_KEY environment variable.")
        super().load_resources()
    
    def release_resources(self):
        super().release_resources()