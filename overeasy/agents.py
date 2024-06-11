from typing import List, Tuple, Optional, Callable, Any
from PIL import Image
from overeasy.models import GPT, QwenVL, GPTVision, CLIP
from dataclasses import dataclass
from overeasy.models import OwlV2, GroundingDINO
from overeasy.types import *
import numpy as np
from pydantic import BaseModel
import instructor
from openai import OpenAI
import gradio as gr
from math import sqrt
from .visualize_utils import annotate
import base64, io, os

class BoundingBoxSelectAgent(ImageAgent):
    def __init__(self, classes: List[str], model: Optional[BoundingBoxModel] = None):
        self.classes = classes
        self.model = model if model is not None else GroundingDINO()
    
    def execute(self, image: Image.Image) -> ExecutionNode:
        return ExecutionNode(image, self.model.detect(image, self.classes))
    
    def __repr__(self):
        model_name = self.model.__class__.__name__ if self.model else "None"
        return f"{self.__class__.__name__}(classes={self.classes}, model={model_name})" 

class VisionPromptAgent(ImageAgent):
    def __init__(self, query: str, model: Optional[MultimodalLLM] = None):
        self.query = query
        self.model = model if model is not None else GPTVision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""{self.query}"""
        response = self.model.prompt_with_image(image, prompt)
        return ExecutionNode(image, response)

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(query={self.query}, model={model_name})"
    
class DenseCaptioningAgent(ImageAgent):
    def __init__(self, model: Optional[MultimodalLLM] = None):
        self.model = model if model is not None else GPTVision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        prompt = f"""Describe the following image in detail"""
        response = self.model.prompt_with_image(image, prompt)
        return ExecutionNode(image, response)

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(model={model_name})"

class TextPromptAgent(TextAgent):
    def __init__(self, query: str, model: Optional[LLM] = None):
        self.query = query
        self.model = model if model is not None else GPT()

    def execute(self, text: str)-> str:
        prompt = f"""{text}\n{self.query}"""
        response = self.model.prompt(prompt)
        return response

    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(query={self.query}, model={model_name})"

# Convert binary_choice into an agent class
class BinaryChoiceAgent(ImageAgent):
    def __init__(self, query: str, model:Optional[MultimodalLLM] = None):
        self.query = query
        self.model = model if model is not None else GPTVision()

    def execute(self, image: Image.Image)-> ExecutionNode:
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
    def __init__(self, classes, model: Optional[ClassificationModel] = None):
        self.classes = classes
        self.model = model if model is not None else CLIP()

    def execute(self, image: Image.Image)-> ExecutionNode:
        selected_class = self.model.classify(image, self.classes)
        return ExecutionNode(image, selected_class)
    
    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(classes={self.classes}, model={model_name})"
    
class ClassMapAgent(DetectionAgent):
    def __init__(self, name_map: Dict[str, str]):
        self.name_map = name_map
    
    def execute(self, dets: Detections) -> Detections:
        class_names = []
        for x in dets.class_names:
            if x not in self.name_map:
                raise ValueError(f"Class name '{x}' not mapped")
            class_names.append(self.name_map[x])
            
        new_dets = Detections(
            xyxy=dets.xyxy,
            class_ids=dets.class_ids,
            confidence=dets.confidence,
            classes=class_names,
            data=dets.data,
            detection_type=dets.detection_type
        )
        return new_dets
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name_map={self.name_map})"

class MapAgent(DataAgent):
    def __init__(self, fn: Callable[[Any], Any]):
        self.fn = fn

    def execute(self, node: ExecutionNode) -> ExecutionNode:
        return ExecutionNode(node.image, self.fn(node.data))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fn={self.fn})"

class ToClassificationAgent(DataAgent):
    def __init__(self, fn: Union[Callable[[Any], str], Callable[[Any], List[str]]]):
        self.fn = fn
    
    def execute(self, node: ExecutionNode) -> ExecutionNode:
        res = self.fn(node.data)
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            return ExecutionNode(node.image, Detections.from_classification(res))
        elif isinstance(res, str):
            return ExecutionNode(node.image, Detections.from_classification([res]))
        else:
            raise ValueError(f"{self.__class__.__name__} must return a string or list of strings")

    def __repr__(self):
        return f"{self.__class__.__name__}(fn={self.fn})"

class OCRAgent(ImageAgent):
    def __init__(self, model: Optional[OCRModel] = None):
        self.model = model if model is not None else GPTVision()

    def execute(self, image: Image.Image)-> ExecutionNode:
        response = self.model.parse_text(image)
        return ExecutionNode(image, response)
    
    def __repr__(self):
        model_name = self.model.__class__.__name__
        return f"{self.__class__.__name__}(model={model_name})"

class InstructorImageAgent(ImageAgent):
    def __init__(self, response_model: type[BaseModel], model: Optional[str] = "gpt-4o"):
        self.model = model
        self.response_model = response_model

    def execute(self, image: Image.Image)-> ExecutionNode:
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

    def execute(self, text: str)-> Any:
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
    
class SplitAgent(Agent):
    def execute(self, node: ExecutionNode) -> List[ExecutionNode]:
        result : List[ExecutionNode] = []
        if not isinstance(node.data, Detections):
            raise ValueError("ExecutionNode data must be of type Detections")
        detections: Detections = node.data
        if detections.detection_type != DetectionType.BOUNDING_BOX:
            raise ValueError("Detection type must be BOUNDING_BOX")
            
        for split_detection in detections.split():
            parent_xyxy = split_detection.xyxy[0]
            child_image = node.image.crop(parent_xyxy)
            child_detection = Detections(
                xyxy=np.array([[0, 0, child_image.width, child_image.height]]),
                class_ids=split_detection.class_ids,
                confidence=split_detection.confidence,
                classes=split_detection.classes,
                data=split_detection.data,
                detection_type=DetectionType.CLASSIFICATION
            )
            result.append(ExecutionNode(child_image, child_detection, parent_detection=split_detection))
                    
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
"""
This function walks up the execution graph to find the immediate children
produced by a SplitAgent. This is useful because these nodes will have the
parent_detection attribute set to the original detections.

Graph Representation:
         Parent
            |
       SplitAgent
            |
  -----------------
  |       |       |
*Child1* *Child2* *Child3*
  |       |       |
 C1       C2      C3

- '*Child1*', '*Child2*', and '*Child3*' are the immediate children produced by the SplitAgent.
"""
def find_immediate_children(start: List[ExecutionNode], graph: ExecutionGraph):
    parents = [x for x in start]
    while True:
        parents = [graph.parent_of(item) for item in parents]
        if len(parents) == 0:
            raise ValueError("No parents found")
        some_have_parents = any(parent.parent_detection is not None for parent in parents)
        all_have_parents = all(parent.parent_detection is not None for parent in parents)
        
        if some_have_parents and not all_have_parents:
            raise ValueError("Some detections have parents, but not all")
        
        if all_have_parents:
            break
    return parents
        
def combine_detections(dets: List[Detections], parent_dets: List[Detections]) -> Detections:
    if not all(isinstance(x, Detections) for x in dets) or not all(isinstance(x, Detections) for x in parent_dets):
        raise ValueError("Cannot combine detections")
    
    det_types = [x.detection_type for x in dets]
    if not all(x == det_types[0] for x in det_types):
        raise ValueError("Cannot combine detections of different types")
    
    det_type = det_types[0]
    parent_detection_type = parent_dets[0].detection_type

    if det_type == DetectionType.CLASSIFICATION:
        if any(len(det.class_ids) > 1 for det in dets):
            raise ValueError("Detections with multiple classes are not supported for combination")
        
        uniq_classes = set()
        for det in dets:
            for cls in det.classes:
                uniq_classes.add(cls)
        classes = np.array(list(uniq_classes))
        class_id_map = {cls: idx for idx, cls in enumerate(classes)}
        xyxy = []
        class_ids = []
        confidence = []        
        
        
        for det, parent_det in zip(dets, parent_dets):
            confidence.append(det.confidence[0] if det.confidence is not None else None)
            cls = det.class_names[0]
            class_ids.append(class_id_map[cls])
            xyxy.append(parent_det.xyxy[0])
        
        return Detections(
            xyxy=np.array(xyxy),
            class_ids=np.array(class_ids),
            confidence=np.array(confidence),
            classes=classes,
            detection_type=parent_detection_type
        )
        
    elif det_type == DetectionType.BOUNDING_BOX:
        # TODO: Implement this
        pass
    elif det_type == DetectionType.SEGMENTATION:
        # TODO: Implement this
        pass
   
   
    return Detections.empty()
    
class JoinAgent(Agent):
    # Mutates the input graph by adding a new layer of child nodes
    def join(self, start: List[ExecutionNode], graph:ExecutionGraph) -> List[ExecutionNode]:
        leaves = []
        def merge_nodes(node_and_parent_det: List[Tuple[ExecutionNode, Detections]], parent: ExecutionNode):
            if len(node_and_parent_det) == 0:
                return
            nodes = [x[0] for x in node_and_parent_det]
            parent_dets = [x[1] for x in node_and_parent_det]

            original_data = [node.data for node in nodes]
            merged_node = ExecutionNode(parent.image, original_data)
            
            if all(isinstance(x, Detections) for x in original_data):
                merged_node.data = combine_detections(original_data, parent_dets)
                
            for node in nodes:
                graph.add_child(node, merged_node)
            leaves.append(merged_node)

        
        immediate_children = find_immediate_children(start, graph)
        parent_dets = [child.parent_detection for child in immediate_children]
        parents = [graph.parent_of(child) for child in immediate_children]

        current_parent_id = None
        current_group : List[Tuple[ExecutionNode, Detections]] = []
        for cur, parent_det, parent in sorted(zip(start, parent_dets, parents), key=lambda x: x[-1].id):
            if parent.id != current_parent_id:
                merge_nodes(current_group, parent)
                current_parent_id = parent.id
                current_group = [(cur, parent_det)]
            else:
                current_group.append((cur, parent_det))
        
        if len(current_group) > 0:
            merge_nodes(current_group, parent)
        
        return leaves

    def __repr__(self):
        return f"{self.__class__.__name__}()"
  


def _visualize_layer(layer: List[ExecutionNode]) -> List[Tuple[Image.Image, str]]:
    images: List[Tuple[Image.Image, str]] = []
    if all(isinstance(node.data, Detections) for node in layer):
        for node in layer:
            detection = node.data
            if detection.detection_type == DetectionType.BOUNDING_BOX:
                images.append((annotate(node.image, detection), "Bounding Box"))
            elif detection.detection_type == DetectionType.SEGMENTATION:
                images.append((annotate(node.image, detection), "Segmentation"))
            elif detection.detection_type == DetectionType.CLASSIFICATION:
                labels = [f"{name} {score:.2f}" if score is not None else f"{name}" for name, score in zip(detection.class_names, detection.confidence_scores)]
                stringified = labels[0] if len(labels)==1 else str(labels)
                images.append((node.image, stringified))
    else:
        images = [(x.image, str(x.data)) for x in layer]
    return images

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "./assets/favicon.ico")
@dataclass(frozen=True)
class Workflow:
    steps: List[Agent]
    
    def __repr__(self):
        return f"Workflow(steps={self.steps})"

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
                res = []
                for node in intermediate_results:
                    children : List[ExecutionNode] = agent.execute(node)
                    res.extend(children)
                    for _child in children:
                        graph.add_child(node, _child)
                intermediate_results = res
            elif isinstance(agent, ImageAgent):
                res = []
                for node in intermediate_results:
                    child: ExecutionNode = agent.execute(node.image)
                    graph.add_child(node, child)
                    res.append(child)
                intermediate_results = res
            elif isinstance(agent, TextAgent):
                res = []
                for node in intermediate_results:
                    if isinstance(node.data, Detections):
                        raise ValueError("TextAgent cannot be used with must have stringable input")
                    response = agent.execute(str(node.data))
                    child = ExecutionNode(node.image, response)
                    graph.add_child(node, child)
                    res.append(child)
                intermediate_results = res
            elif isinstance(agent, DetectionAgent):
                res = []
                for node in intermediate_results:
                    output_detection: Detections = agent.execute(node.data)
                    child = ExecutionNode(node.image, output_detection)
                    graph.add_child(node, child)
                    res.append(child)
                intermediate_results = res
            elif isinstance(agent, DataAgent):
                res = []
                for node in intermediate_results:
                    child = agent.execute(node)
                    graph.add_child(node, child)
                    res.append(child)
                intermediate_results = res
            else:
                raise TypeError(f"Unsupported agent type: {type(agent)}")
                              
        return intermediate_results, graph
    

    
    def to_steps(self, graph: ExecutionGraph) -> List[Tuple[str, List[Tuple[Image.Image, str]], str]]:
        workflow_steps_names = ["Input Image"]
        code_snippets = [""]
        for i in range(len(self.steps)):
            workflow_steps_names.append(self.steps[i].__class__.__name__)
            code_snippets.append(repr(self.steps[i]))        
        
        layers = graph.top_sort()
        steps = []
        for i, layer in enumerate(layers):
            code = f"```python\n{code_snippets[i]}\n```"
            layer_images = _visualize_layer(layer)
            steps.append((f"Step {i+1}: {workflow_steps_names[i]}", layer_images, code))
        
        return steps

    def visualize(self, graph: ExecutionGraph):
        steps = self.to_steps(graph)
        css = """
        .gradio-container .single-image {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        """
        with gr.Blocks(css=css) as demo:
            for step_title, images, code in steps:
                with gr.Group():
                    gr.Markdown(f"### {step_title}")
                    gr.Markdown(f"{code}")
                    if len(images) > 1:
                        gr.Gallery(
                            value=images,
                            height="max-content",
                            object_fit="contain",
                            show_label=False, 
                            columns=max(3, min(9, round(sqrt(len(images))))),
                        )
                    else:
                        [image] = images
                        with gr.Row():
                            with gr.Column(scale=1):
                                pass
                            with gr.Column(scale=4):
                                gr.Image(
                                    width=640,
                                    value=image[0],
                                    label=image[1],
                                )
                            with gr.Column(scale=1):
                                pass
                            

        demo.launch(favicon_path=FAVICON_PATH)


