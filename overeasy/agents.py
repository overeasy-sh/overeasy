from typing import List, Tuple, Optional
from PIL import Image 
from overeasy.models import YOLOWorld, QwenVL, GPT4Vision, CLIP
from dataclasses import dataclass, field
from overeasy.models import OwlV2, GroundingDINO
from overeasy.types import *
import numpy as np
from pydantic import BaseModel
import instructor
from openai import OpenAI

class BoundingBoxSelectAgent(SplitAgent):
    def __init__(self, classes: List[str], model: Optional[BoundingBoxModel] = None, split: bool = False):
        self.classes = classes
        self.model = model if model is not None else GroundingDINO()
        self.split = split      
    
    def execute(self, image: Image.Image) -> ExecutionNode:
        return ExecutionNode(image, self.model.detect(image, self.classes))
    
    def execute_split(self, image: Image.Image) -> List[ExecutionNode]:
        detections = self.model.detect(image, self.classes)
        result : List[ExecutionNode] = []
        for detection in  detections.split():
            parent_xyxy=detection.xyxy[0]
            child_image = image.crop(parent_xyxy)
            child_detection = Detections(
                xyxy=np.array([[0, 0, child_image.width, child_image.height]]),
                class_ids=detection.class_ids,
                confidence=detection.confidence,
                classes=detection.classes,
                data=detection.data,
                detection_type=DetectionType.CLASSIFICATION
            )
            
            result.append(ExecutionNode(image.crop(detection.xyxy[0]), child_detection, parent_detection=detection))
            
        return result 
    
    def is_split(self) -> bool:
        return self.split

    def __repr__(self):
        code_snippet = f"workflow.add_step(BoundingBoxSelectAgent(classes=[{', '.join(self.classes)}], "
        code_snippet += f"model={self.model}" if self.model else ""
        code_snippet += f", split={self.split}))"
            
        return f"workflow.add_step(BoundingBoxSelectAgent(classes=[{', '.join(self.classes)}], model={self.model}, split={self.split}))"

    
class VisionPromptAgent(ImageAgent):
    
    def __init__(self, query: str, model: Optional[MultimodalLLM] = None):
        self.query = query
        self.model = model if model is not None else GPT4Vision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""{self.query}"""
        response = self.model.prompt_with_image(image, prompt)
        return ExecutionNode(image, response)

    
# @dataclass
# class TextPromptAgent(Agent):
#     query: str
#     model: MultimodalLLM = GPT4Vision()
#     # TODO check this
#     def execute(self, image: Image.Image)-> ExecutionNode:
#         prompt = f"""{self.query}"""
#         response = self.model.prompt_with_image(image, prompt)
#         return ExecutionNode(image, response)


class DenseCaptioningAgent(ImageAgent):

    def __init__(self, model: Optional[MultimodalLLM] = None):
        self.model = model if model is not None else GPT4Vision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""Describe the following image in detail"""
        response = self.model.prompt_with_image(image, prompt)
        return ExecutionNode(image, response)

# Convert binary_choice into an agent class
class BinaryChoiceAgent(ImageAgent):
    def __init__(self, query: str, model:Optional[MultimodalLLM] = None):
        self.query = query
        self.model = model if model is not None else QwenVL()

    def execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""{self.query}"""
        response = self.model.prompt_with_image(image, prompt)
        truthy = "yes" in response.lower()
        
        detection_type = DetectionType.CLASSIFICATION
        class_ids = [0] if truthy else [1]
        classes = ['yes', 'no']
        confidence = [1.0]  
        
        full_image_box = np.array([0, 0, image.width, image.height]).reshape(1, -1)
        
        detection = Detections(
            xyxy=full_image_box,  # No bounding box for classification
            detection_type=detection_type,
            confidence=np.array(confidence),
            class_ids=np.array(class_ids),
            classes=classes,
            data={'response': [response]}
        )
        node = ExecutionNode(image, detection)

        return node

    def __repr__(self):
        return f"workflow.add_step(BinaryChoiceAgent('{self.query}'))"

class ClassificationAgent(ImageAgent):
    def __init__(self, classes, model: Optional[ClassificationModel] = None):
        self.classes = classes
        self.model = model if model is not None else CLIP()

    def execute(self, image: Image.Image)-> ExecutionNode:
        selected_class = self.model.classify(image, self.classes)
        return ExecutionNode(image, selected_class)

class OCRAgent(ImageAgent):
    def __init__(self, model: Optional[OCRModel] = None):
        self.model = model if model is not None else GPT4Vision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        response = self.model.parse_text(image)
        return ExecutionNode(image, response)

class FacialRecognitionAgent(ImageAgent):
    def __init__(self, model: Optional[MultimodalLLM] = None):
        self.prompt = "Identify the person in the image."
        self.model = model if model is not None else GPT4Vision()
        
    def execute(self, image: Image.Image)-> ExecutionNode:
        return ExecutionNode(image, None)

import base64, io        
class JSONAgent(ImageAgent):
    def __init__(self, response_model: type[BaseModel], model: Optional[MultimodalLLM] = None):
        self.model = model if model is not None else GPT4Vision()
        self.response_model = response_model

    def execute(self, image: Image.Image)-> ExecutionNode:
        client = instructor.from_openai(OpenAI())
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Extract structured data from natural language
        structured_response: Any = client.chat.completions.create(
            model="gpt-4o",
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
    
    
# class FunctionAgent(Agent):
#     def __init__(self, function: Callable[[Image.Image], Any]):
#         self.function = function

#     def execute(self, image: Image.Image)-> ExecutionNode:
#         response = self.function(image)
#         return ExecutionNode(image, response)


class JoinAgent(Agent):
    def __init__(self):
        self.info = {}

    # Mutates the input graph by adding a new layer of child nodes
    def join(self, start: List[ExecutionNode], input:ExecutionGraph) -> List[ExecutionNode]:
        
        parents = [x for x in start]
        while True:
            parents = [input.parent_of(item) for item in parents]
            if len(parents) == 0:
                raise ValueError("No parents found")
            some_have_parents = any(parent.parent_detection is not None for parent in parents)
            all_have_parents = all(parent.parent_detection is not None for parent in parents)
            
            if some_have_parents and not all_have_parents:
                raise ValueError("Some detections have parents, but not all")
            
            if all_have_parents:
                break
        

        leaves = []
        
        def merge_nodes(nodes: List[ExecutionNode], parent: ExecutionNode):
            if len(nodes) == 0:
                return
            
            # Create a new node with the merged data
            merged_node = ExecutionNode(parent.image, None)
            
            # Point all original nodes to the new merged node
            for node in nodes:
                input.add_child(node, merged_node)

            leaves.append(merged_node)

        
        # parent_ids : List[int] = [input.parent_of(item).id for item in parents]
        parents = [input.parent_of(item) for item in parents]
        current_parent_id = None
        current_group : List[ExecutionNode] = []
        for cur, parent in sorted(zip(start, parents), key=lambda x: x[1].id):
            if parent.id != current_parent_id:
                merge_nodes(current_group, parent)
                current_parent_id = parent.id
                current_group = [cur]
            else:
                current_group.append(cur)
        
        if len(current_group) > 0:
            merge_nodes(current_group, parent)
        

        return leaves

    def __repr__(self):
        return "workflow.add_step(JoinAgent())"
  


@dataclass
class Workflow:
    steps: List[Agent] = field(default_factory=list)

    def add_step(self, agent: Agent):
        """Add a processing step to the workflow."""
        self.steps.append(agent)

    # Return leaves of the graph and the graph itself
    def execute(self, input_image: Image.Image) -> Tuple[List[ExecutionNode], ExecutionGraph]:
        if input_image is None:
            raise ValueError("Input image is None")
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        elif not isinstance(input_image, Image.Image):
            raise ValueError("Input image is not a valid image format")
        
        root = ExecutionNode(input_image, None)
        graph = ExecutionGraph(root)

        intermediate_results = [root]
        for ind, agent in enumerate(self.steps):
            if isinstance(agent, JoinAgent):
                intermediate_results = agent.join(intermediate_results, graph)
            elif isinstance(agent, SplitAgent):
                if agent.is_split():
                    res = []
                    for node in intermediate_results:
                        children = agent.execute_split(node.image)
                        res.extend(children)
                        for child in children:
                            graph.add_child(node, child)
                    intermediate_results = res
                else:
                    res = []
                    for node in intermediate_results:
                        child = agent.execute(node.image)
                        graph.add_child(node, child)
                        res.append(child)
                    intermediate_results = res
            elif isinstance(agent, ImageAgent):
                res = []
                for node in intermediate_results:
                    child = agent.execute(node.image)
                    graph.add_child(node, child)
                    res.append(child)
                intermediate_results = res
            else:
                raise TypeError(f"Unsupported agent type: {type(agent)}")
                              
        return intermediate_results, graph






# def plot_boxes(image: Image.Image, detection: Detections):
#     """Plot the image and bounding boxes."""
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches

#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     ax.imshow(image)
#     n, _ = detection.xyxy.shape
#     confidence = [None for _ in range(n)] if detection.confidence is None else detection.confidence
#     for box, cls, score in zip(detection.xyxy, detection.class_names, confidence):
#         x, y, w, h = box
#         rect = patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor="r", facecolor="none")
#         ax.add_patch(rect)
#         ax.text(x, y, f"{cls} {score:.2f}", color="r")

#     plt.savefig("res.png")
