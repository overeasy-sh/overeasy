from PIL import Image
from overeasy.models import *
from overeasy.types import *
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Union, Optional, Any
import instructor
import base64, io

__all__ = [
    "BoundingBoxSelectAgent",
    "VisionPromptAgent",
    "DenseCaptioningAgent",
    "TextPromptAgent",
    "BinaryChoiceAgent",
    "ClassificationAgent",
    "OCRAgent",
    "InstructorImageAgent",
    "InstructorTextAgent"
]

class BoundingBoxSelectAgent(ImageAgent):
    def __init__(self, classes: List[str], model: BoundingBoxModel = GroundingDINO()):
        self.classes = classes
        self.model = model
    
    def _execute(self, image: Image.Image) -> ExecutionNode:
        return ExecutionNode(image, self.model.detect(image, self.classes))
    
    def __repr__(self):
        model_name = self.model.__class__.__name__ if self.model else "None"
        return f"{self.__class__.__name__}(classes={self.classes}, model={model_name})" 

class VisionPromptAgent(ImageAgent):
    def __init__(self, query: str, model: MultimodalLLM = GPTVision()):
        self.query = query
        self.model = model

    def _execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""{self.query}"""
        response = self.model.prompt_with_image(image, prompt)
        return ExecutionNode(image, response)

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(query={self.query}, model={model_name})"
    
class DenseCaptioningAgent(ImageAgent):
    def __init__(self, model: Union[MultimodalLLM, CaptioningModel] = GPTVision()):
        self.model = model

    def _execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""Describe the following image in detail"""
        if isinstance(self.model, MultimodalLLM):
            response = self.model.prompt_with_image(image, prompt)
        else:
            response = self.model.caption(image)
            
        return ExecutionNode(image, response)

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(model={model_name})"

class TextPromptAgent(TextAgent):
    def __init__(self, query: str, model: LLM = GPT()):
        self.query = query
        self.model = model

    def _execute(self, text: str)-> str:
        prompt = f"""{text}\n{self.query}"""
        response = self.model.prompt(prompt)
        return response

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(query={self.query}, model={model_name})"

# Convert binary_choice into an agent class
class BinaryChoiceAgent(ImageAgent):
    def __init__(self, query: str, model: MultimodalLLM = GPTVision()):
        self.query = query
        self.model = model 

    def _execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""{self.query}"""
        response = self.model.prompt_with_image(image, prompt)
        truthy = "yes" in response.lower()        
        assigned_class = "yes" if truthy else "no"
        
        detection = Detections.from_classification([assigned_class], all_classes=['yes', 'no'])
        node = ExecutionNode(image, detection)

        return node

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(query={self.query}, model={model_name})"

class ClassificationAgent(ImageAgent):
    def __init__(self, classes, model: ClassificationModel = LaionCLIP()):
        self.classes = classes
        self.model = model 

    def _execute(self, image: Image.Image)-> ExecutionNode:
        selected_class = self.model.classify(image, self.classes)
        return ExecutionNode(image, selected_class)
    
    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(classes={self.classes}, model={model_name})"
    
class OCRAgent(ImageAgent):
    def __init__(self, model: Optional[OCRModel] = None):
        self.model = model if model is not None else GPTVision()

    def _execute(self, image: Image.Image)-> ExecutionNode:
        response = self.model.parse_text(image)
        return ExecutionNode(image, response)
    
    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(model={model_name})"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class InstructorImageAgent(ImageAgent):
    def __init__(self, response_model: type[BaseModel], model: Optional[str] = "gpt-4o"):
        self.model = model
        self.response_model = response_model

    def _execute(self, image: Image.Image)-> ExecutionNode:
        client = instructor.from_openai(OpenAI())
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Extract structured data from natural language
        structured_response: Any = client.chat.completions.create(
            model=self.model,
            response_model=self.response_model,
            messages=[{"role": "user", "content": [
                {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                }
                ]}],
        )

        return ExecutionNode(image, structured_response)
    
    def __repr__(self):
        model_name = self.model
        return f"{self.__class__.__name__}(response_model={self.response_model}, model={model_name})"
    
class InstructorTextAgent(TextAgent):
    def __init__(self, response_model: type[BaseModel], model: Optional[str] = "gpt-3.5-turbo"):
        self.model = model
        self.response_model = response_model

    def _execute(self, text: str)-> Any:
        client = instructor.from_openai(OpenAI())

        structured_response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": text},
            ],
            response_model=self.response_model,
        )

        return structured_response

    def __repr__(self):
        model_name = self.model
        return f"{self.__class__.__name__}(response_model={self.response_model}, model={model_name})"
 