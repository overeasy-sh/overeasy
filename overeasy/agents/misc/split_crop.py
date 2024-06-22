from PIL import Image
from pydantic.dataclasses import dataclass
from typing import Tuple
from overeasy.types import ExecutionNode, ImageAgent, Detections, DetectionType
import numpy as np

class SplitCropAgent(ImageAgent):
    # WE HAVE PYDANTIC AT HOME
    def __init__(self, split: Tuple[int, int]):
        if len(split) != 2:
            raise ValueError("Split must be a tuple of two integers")
        if split[0] <= 0 or split[1] <= 0:
            raise ValueError("Split must be greater than 0")
        if type(split[0]) != int or type(split[1]) != int:
            raise ValueError("Split must be a tuple of two integers")
        
        self.rows = split[0]
        self.columns = split[1]


    def _execute(self, image: Image.Image) -> ExecutionNode:
        # Convert PIL Image to numpy array
        width, height = image.width, image.height
        
        # Calculate the size of each block
        block_height = height // self.rows
        block_width = width // self.columns
        
        left_over_height = height - block_height * self.rows
        left_over_width = width - block_width * self.columns
        
        # Initialize a list to hold the bounding boxes
        bounding_boxes = []

        # Create bounding boxes
        for row in range(self.rows):
            for col in range(self.columns):
                x1 = col * block_width + min(col, left_over_width)
                y1 = row * block_height + min(row, left_over_height)
                x2 = x1 + block_width + (1 if col < left_over_width else 0)
                y2 = y1 + block_height + (1 if row < left_over_height else 0)
                bounding_boxes.append((x1, y1, x2, y2))

        # Convert lists to numpy arrays
        bounding_boxes_np = np.array(bounding_boxes)
        confidence_np = np.ones(len(bounding_boxes_np))
        class_ids_np = np.arange(len(bounding_boxes_np))
        classes_np = np.array([f'split_{i+1}' for i in range(len(bounding_boxes_np))])

        # Create Detections object
        det = Detections(
            xyxy=bounding_boxes_np, 
            confidence=confidence_np,  # Confidence for each box
            class_ids=class_ids_np,  # Unique class ID for each box
            classes=classes_np,  # Class names
            detection_type=DetectionType.BOUNDING_BOX
        )

        # Convert numpy array back to PIL Image
        return ExecutionNode(image, det)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(split={self.split})"