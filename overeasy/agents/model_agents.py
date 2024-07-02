from PIL import Image
from torch import mode
from overeasy.models import *
from overeasy.types import *
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
import instructor
import base64, io
import google.generativeai as genai

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
    def __init__(self, classes, model: ClassificationModel = CLIP()):
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

options  = Union[GPTVision, Gemini, Claude]



class InstructorImageAgent(ImageAgent):

    def __init__(self, response_model: type[BaseModel], model: Union[GPTVision, Gemini, Claude] = GPTVision(), max_tokens: int = 4096, extra_context: Optional[List[Dict[str, str]]] = None):
        self.response_model = response_model
        self.model = model
        self.extra_context = extra_context if extra_context is not None else []
        self.max_tokens = max_tokens
        if not isinstance(self.model, GPTVision) and not isinstance(self.model, Gemini) and not isinstance(self.model, Claude):
            raise ValueError("Model must be a GPTVision, Gemini, or Claude")

    def _execute(self, image: Image.Image)-> ExecutionNode:
        model_name = ""
        model_client = self.model.client
        if model_client is None:
            raise ValueError("No client found. Please call load_resources() on model")
        
        if isinstance(self.model, GPTVision):
            client = instructor.from_openai(model_client)
            model_name = self.model.model
        elif isinstance(self.model, Gemini):
            client = instructor.from_gemini(model_client)
            model_name = self.model.model_name
        elif isinstance(self.model, Claude):
            client = instructor.from_anthropic(model_client)
            model_name = self.model.model
            
        
        if isinstance(self.model, Gemini):
            structured_response: Any = client.chat.completions.create(
                response_model=self.response_model,
                messages=[
                    *self.extra_context,
                    {
                        "role": "user",
                        "content": image
                    }
                ],
                max_tokens=self.max_tokens,
            )
        elif isinstance(self.model, GPTVision):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
            messages = [*self.extra_context, {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                    ]}]
            structured_response: Any = client.chat.completions.create(
                model=model_name,
                response_model=self.response_model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
        else:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
            messages = [*self.extra_context, {"role": "user", "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    }
                    ]}]
            
            structured_response: Any = client.chat.completions.create(
                model=model_name,
                response_model=self.response_model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            
            
        

        return ExecutionNode(image, structured_response)
    
    def __repr__(self):
        model_name = self.model
        return f"{self.__class__.__name__}(response_model={self.response_model}, model={repr(model_name)})"
    

class InstructorTextAgent(TextAgent):
    def __init__(self, response_model: type[BaseModel], model: Union[GPT, Gemini, Claude] = GPT(), max_tokens: int = 4096, extra_context: Optional[List[Dict[str, str]]] = None):
        self.response_model = response_model
        self.model = model
        self.extra_context = extra_context if extra_context is not None else []
        self.max_tokens = max_tokens
        if not isinstance(self.model, GPT) and not isinstance(self.model, Gemini) and not isinstance(self.model, Claude):
            raise ValueError("Model must be a GPT, Gemini, or Claude")
        
    def _execute(self, text: str)-> Any:
        model_client = self.model.client
        if model_client is None:
            raise ValueError("No client found. Please call load_resources() on model")
        
        if isinstance(self.model, GPT):
            client = instructor.from_openai(model_client)
            model_name = self.model.model
        elif isinstance(self.model, Gemini):
            client = instructor.from_gemini(model_client)
            model_name = self.model.model_name
        elif isinstance(self.model, Claude):
            client = instructor.from_anthropic(model_client)
            model_name = self.model.model

        if isinstance(self.model, Gemini):
            structured_response = client.chat.completions.create(
                response_model=self.response_model,
                messages=[*self.extra_context, {"role": "user", "parts": [
                    text
                ]}],
                max_tokens=self.max_tokens,
            )
        else:
            structured_response = client.chat.completions.create(
                model=model_name,
                messages=[*self.extra_context, {"role": "user", "content": text}],
                response_model=self.response_model,
                max_tokens=self.max_tokens,
            )
            

        return structured_response

    def __repr__(self):
        model_name = self.model
        return f"{self.__class__.__name__}(response_model={self.response_model}, model={repr(model_name)})"
 