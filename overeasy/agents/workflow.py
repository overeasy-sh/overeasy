
from overeasy.types import *
from overeasy.agents import SplitAgent, JoinAgent
from overeasy.visualize_utils import annotate
from typing import List, Tuple
from PIL import Image
import numpy as np
import gradio as gr
from math import sqrt
from tqdm import tqdm
from dataclasses import dataclass
import os

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
            raise ValueError("Input image is a numpy array, please convert it to a PIL.Image first using Image.fromarray(), make sure to convert your color channels to RGB!")            
        elif not isinstance(input_image, Image.Image):
            raise ValueError("Input image is not a valid image format, must be a PIL.Image")
        
        root = ExecutionNode(input_image, None)
        graph = ExecutionGraph(root)

        intermediate_results = [root]
        for ind, agent in enumerate(self.steps):
            if hasattr(agent, 'model') and isinstance(agent.model, Model):
                agent.model.load_resources()
            try:
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
            finally:
                if hasattr(agent, 'model') and isinstance(agent.model, Model):
                    agent.model.release_resources()
        return intermediate_results, graph
    

    def execute_multiple(self, input_images: List[Image.Image]) -> Tuple[List[List[ExecutionNode]], List[ExecutionGraph]]:
        if not all(isinstance(img, Image.Image) for img in input_images):
            raise ValueError("All input images must be of type PIL.Image")

        graphs = [ExecutionGraph(ExecutionNode(img, None)) for img in input_images]
        intermediate_results = [[graph.root] for graph in graphs]

        for ind, agent in enumerate(self.steps):
            if hasattr(agent, 'model') and isinstance(agent.model, Model):
                agent.model.load_resources()

            try:
                if isinstance(agent, JoinAgent) or isinstance(agent, SplitAgent):
                    for i, graph in enumerate(tqdm(graphs)):
                        if isinstance(agent, JoinAgent):
                            intermediate_results[i] = agent.join(intermediate_results[i], graph)
                        elif isinstance(agent, SplitAgent):
                            res = []
                            for node in intermediate_results[i]:
                                children = agent.execute(node)
                                res.extend(children)
                                for child in children:
                                    graph.add_child(node, child)
                            intermediate_results[i] = res
                else:
                    new_intermediate_results: List[List[ExecutionNode]] = [[] for _ in graphs]
                    for i, graph in enumerate(tqdm(graphs)):
                        for node in intermediate_results[i]:
                            if isinstance(agent, ImageAgent):
                                child = agent.execute(node.image)
                                graph.add_child(node, child)
                                new_intermediate_results[i].append(child)
                            elif isinstance(agent, TextAgent):
                                if isinstance(node.data, Detections):
                                    raise ValueError("TextAgent cannot be used with must have stringable input")
                                response = agent.execute(str(node.data))
                                child = ExecutionNode(node.image, response)
                                graph.add_child(node, child)
                                new_intermediate_results[i].append(child)
                            elif isinstance(agent, DetectionAgent):
                                output_detection = agent.execute(node.data)
                                child = ExecutionNode(node.image, output_detection)
                                graph.add_child(node, child)
                                new_intermediate_results[i].append(child)
                            elif isinstance(agent, DataAgent):
                                child = agent.execute(node)
                                graph.add_child(node, child)
                                new_intermediate_results[i].append(child)
                    intermediate_results = new_intermediate_results
            finally:
                if hasattr(agent, 'model') and isinstance(agent.model, Model):
                    agent.model.release_resources()

        return intermediate_results, graphs
    
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


