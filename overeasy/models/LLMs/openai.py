import os
from PIL import Image
from overeasy.types import MultimodalLLM, LLM, OCRModel, Model
from typing import Literal, Optional
import io
import base64
import requests
import warnings
import backoff
import openai

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


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=6)
def _prompt(query: str, model: str, client: openai.OpenAI, max_tokens: int = 1024) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ],
        max_tokens=max_tokens
    )

    result = response.choices[0].message.content
    if result is None:
        raise ValueError("No content found in response")
    
    return result



class GPT(LLM):
    def __init__(self, api_key: Optional[str] = None, model:str = "gpt-3.5-turbo"):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.model not in current_models:
            warnings.warn(f"Model {model} may not be supported. Please provide a valid model.")

    def prompt(self, query: str) -> str:
        if self.client is None:
            raise ValueError("Client is not loaded. Please call load_resources() first.")
        return _prompt(query, self.model, self.client)
    
    def load_resources(self):
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def release_resources(self):
        self.client = None

class GPTVision(MultimodalLLM, OCRModel, GPT):
    def __init__(self, api_key: Optional[str] = None,
                 model : Literal["gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"] = "gpt-4o"
                 ):
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
    
    # def load_resources(self):
    #     self.client = openai.OpenAI(api_key=self.api_key)
    
    # def release_resources(self):
    #     self.client = None
        
    def prompt_with_image(self, image: Image.Image, query: str, max_tokens: int = 1024) -> str:
        if self.client is None:
            raise ValueError("Client is not loaded. Please call load_resources() first.")
        
        base64_image = encode_image_to_base64(image)

        messages = [
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
        ]

        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("No content found in response")        
        
        return result

    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image line by line only output the text.")
    
