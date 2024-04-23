
from abc import ABC, abstractmethod
from typing import List
from .detections import Detections
from PIL import Image

class OCRModel(ABC):
    """
    Abstract class representing an optical character recognition (OCR) model
    that can accept an image and return text.
    """
    
    @abstractmethod
    def parse_text(self, image: Image.Image) -> str:
        """
        Process an image and return text.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.

        Returns:
        A string representing the OCR text.
        """
        pass

class FaceRecognitionModel(ABC):
    """
    Abstract class representing a face recognition model that can accept an image
    and return a list of detected faces.
    """
    
    @abstractmethod
    def detect_faces(self, image: Image.Image) -> list:
        """
        Process an image and return a list of detected faces.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.

        Returns:
        A list of detected faces, where the specific format of each face
        representation (e.g., bounding box coordinates) is determined by the
        implementation.
        """
        pass
    
class LLM(ABC):
    """
    Abstract class representing a language model
    that can accept a text prompt and return a text output.
    """
    
    @abstractmethod
    def prompt(self, query: str) -> str:
        """
        Process a text query, returning a text response.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.
        - query: A string representing the text input.

        Returns:
        A string representing the model's text response.
        """
        pass
    
class MultimodalLLM(LLM):
    """
    Abstract class representing multimodal language and vision models
    that can accept a text prompt and an image and return a text output.
    """
    
    @abstractmethod
    def prompt_with_image(self, image: Image.Image, query: str) -> str:
        """
        Process an image and a text query, returning a text response.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.
        - query: A string representing the text input.

        Returns:
        A string representing the model's text response.
        """
        pass
    

class BoundingBoxModel(ABC):
    """
    Abstract class representing a detection model
    that can accept an image and return detection results.
    """
    
    @abstractmethod
    def detect(self, image: Image.Image, classes: List[str]) -> Detections:
        """
        Process an image and return detection results.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.
        
        Returns:
        A list of detection results, where each result is typically a dictionary
        containing details such as the bounding box, class, and confidence score.
        """
        pass
    

class ClassificationModel(ABC):
    """
    Abstract class representing a classification model
    that can accept an image and return a classification result.
    """
    
    @abstractmethod
    def classify(self, image: Image.Image, classes: list) -> str:
        """
        Process an image and return a classification result.

        Parameters:
        - image: An instance of PIL.Image.Image representing the input image.
        - classes: A list of strings representing the classes to classify the image into.
        
        Returns:
        A string representing the classification result.
        """
        pass