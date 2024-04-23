from PIL import Image
from overeasy.types import MultimodalLLM, OCRModel
from typing import Optional
import anthropic
import base64
import io
import os

#TODO: Add support for prompting with multiple images
class Claude(MultimodalLLM, OCRModel):
    def __init__(self, model: str = 'claude-3-opus-20240229', api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Support for shortened model names
        if model == "opus":
            model = "claude-3-opus-20240229"
        elif model == "haiku":
            model = "claude-3-haiku-20240307" 
        elif model == "sonnet":
            model = "claude-3-sonnet-20240229"
            
        self.model = model

    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": query
                        }
                    ],
                }
            ],
        )
        return message.content[0].text

    def prompt(self, query: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return message.content[0].text

    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image.")