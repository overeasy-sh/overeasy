from PIL import Image
from overeasy.types import MultimodalLLM, OCRModel, Model
from typing import Optional
import anthropic
import base64
import io
import os

class Claude(MultimodalLLM, OCRModel):
    models = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    def __init__(self, model: str = 'claude-3-5-sonnet-20240620', api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        # Support for shortened model names
        self.model = model
        self.client = None

    def load_resources(self):
        if self.api_key is None:
            raise ValueError("No API key found. Please provide an API key, or set the ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

        super().load_resources()
    
    def release_resources(self):
        self.client = None
        super().release_resources()

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
        return message.content[0].text  # type: ignore

    def prompt(self, query: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return message.content[0].text  # type: ignore

    def parse_text(self, image: Image.Image) -> str:
        return self.prompt_with_image(image, "Read the text from the image line by line only output the text.")
    
