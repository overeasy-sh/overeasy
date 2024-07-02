
from re import split
from overeasy.types import *
from overeasy.agents import SplitAgent, JoinAgent
from overeasy.visualize_utils import annotate
from typing import List, Tuple
from PIL import Image
import numpy as np
import gradio as gr
from math import fabs, sqrt
from tqdm import tqdm
from dataclasses import field, dataclass
import os
from typing import Optional, Any, Dict, Union

def _visualize_layer(layer: List[Node]) -> List[Tuple[Optional[Image.Image], str]]:
    images: List[Tuple[Optional[Image.Image], str]] = []
    
    if all(isinstance(node, ExecutionNode) and isinstance(node.data, Detections) for node in layer):
        for node in layer:
            assert isinstance(node, ExecutionNode)
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
        images = [(x.image, str(x.data)) if isinstance(x, ExecutionNode) else (None, "None") for x in layer]
    return images

def handle_node(node: ExecutionNode, agent: Agent) -> List[ExecutionNode]:    
    if isinstance(agent, SplitAgent):
        return agent.execute(node)
    elif isinstance(agent, ImageAgent):
        return [agent.execute(node.image)]
    elif isinstance(agent, TextAgent) or isinstance(agent, DetectionAgent) or isinstance(agent, DataAgent):
        return [ExecutionNode(node.image, agent.execute(node.data))]
    else:
        raise ValueError(f"Agent {agent} is not a valid agent type")
    
 

FAVICON_PATH = os.path.join(os.path.dirname(__file__), "./assets/favicon.ico")
@dataclass(frozen=True)
class Workflow:
    steps: List[Agent]
    join_to_split_map: Dict[int, int] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        splits = []
        for i, agent in enumerate(self.steps):
            if isinstance(agent, SplitAgent):
                splits.append(i)
            elif isinstance(agent, JoinAgent):
                if len(splits) == 0:
                    raise ValueError(f"JoinAgent at index {i} has no matching SplitAgent")
                self.join_to_split_map[i] = splits.pop()
        
    
    def __repr__(self):
        return f"Workflow(steps={self.steps})"

    # Return leaves of the graph and the graph itself
    def execute(self, input_image: Image.Image, data: Optional[Any] = None) -> Tuple[List[ExecutionNode], ExecutionGraph]:
        
        if input_image is None:
            raise ValueError("Input image is None")
        elif isinstance(input_image, np.ndarray):
            raise ValueError("Input image is a numpy array, please convert it to a PIL.Image first using Image.fromarray(), make sure to convert your color channels to RGB!")            
        elif not isinstance(input_image, Image.Image):
            raise ValueError("Input image is not a valid image format, must be a PIL.Image")
        
        input_image = input_image.convert("RGB")

        root = ExecutionNode(input_image, data)
        graph = ExecutionGraph(root)

        intermediate_results : List[Node] = [root]
        for ind, agent in enumerate(self.steps):
            if hasattr(agent, 'model') and isinstance(agent.model, Model):
                agent.model.load_resources()
            try:
                if isinstance(agent, JoinAgent):
                    target_split = self.join_to_split_map[ind]
                    intermediate_results = agent.join(graph, target_split)
                    continue

                next_results: List[Node] = []
                for node in intermediate_results:
                    if isinstance(node, NullExecutionNode):
                        null_node = NullExecutionNode()
                        graph.add_child(node, null_node)
                        next_results.append(null_node)
                        continue
                    # Now node must be a ExecutionNode
                    assert isinstance(node, ExecutionNode)
                    res = handle_node(node, agent)
                    for child in res:
                        graph.add_child(node, child)
                    next_results.extend(res)
                    
                intermediate_results = next_results
            finally:
                if hasattr(agent, 'model') and isinstance(agent.model, Model):
                    agent.model.release_resources()

        return [node for node in intermediate_results if isinstance(node, ExecutionNode)], graph
    

    def execute_multiple(self, input_images: List[Image.Image]) -> Tuple[List[List[ExecutionNode]], List[ExecutionGraph]]:
        if not all(isinstance(img, Image.Image) for img in input_images):
            raise ValueError("All input images must be of type PIL.Image")
        # Normalize image format
        input_images = [img.convert("RGB") for img in input_images]
        
        all_graphs = [ExecutionGraph(ExecutionNode(img, None)) for img in input_images]
        intermediate_results: List[List[Node]] = [[graph.root] for graph in all_graphs]

        for ind, agent in enumerate(tqdm(self.steps, desc="Processing steps")):
            try:
                if hasattr(agent, 'model') and isinstance(agent.model, Model):
                    agent.model.load_resources()

                if isinstance(agent, JoinAgent):
                    target_split = self.join_to_split_map[ind]
                    for i, graph in enumerate(all_graphs):
                        intermediate_results[i] = agent.join(graph, target_split)
                else:
                    for i in range(len(all_graphs)):
                        next_results: List[Node] = []
                        for node in intermediate_results[i]:
                            if isinstance(node, NullExecutionNode):
                                null_node = NullExecutionNode()
                                all_graphs[i].add_child(node, null_node)
                                next_results.append(null_node)
                            elif isinstance(node, ExecutionNode):
                                res = handle_node(node, agent)
                                for child in res:
                                    all_graphs[i].add_child(node, child)
                                next_results.extend(res)
                        intermediate_results[i] = next_results

            finally:
                if hasattr(agent, 'model') and isinstance(agent.model, Model):
                    agent.model.release_resources()

        all_leaves = [
            [node for node in results if isinstance(node, ExecutionNode)]
            for results in intermediate_results
        ]

        return all_leaves, all_graphs    
    
    def to_steps(self, graph: ExecutionGraph) -> List[Tuple[str, List[Tuple[Optional[Image.Image], str]], str]]:
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

    def visualize(self, graph: ExecutionGraph, share: bool = False):
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
                    filtered_images = [(img, label) for img, label in images if img is not None]
                    if len(filtered_images) > 1:
                        gr.Gallery(
                            value=filtered_images,
                            height="max-content",
                            object_fit="contain",
                            show_label=False, 
                            columns=max(3, min(9, round(sqrt(len(images))))),
                        )
                    elif len(filtered_images) == 1:
                        [image] = filtered_images
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
                    elif len(filtered_images) == 0:
                        with gr.Row():
                            with gr.Column(scale=1):
                                pass
                            with gr.Column(scale=4):
                                gr.Text("No image to display skipped because previous split was empty")
                            with gr.Column(scale=1):
                                pass
                            

        demo.launch(favicon_path=FAVICON_PATH, share=share)


