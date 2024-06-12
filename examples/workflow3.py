from overeasy import *
from PIL import Image
import os

workflow = Workflow([
    # Detect each head in the input image
    BoundingBoxSelectAgent(classes=["person's head"], model=OwlV2()),  
    # Applies Non-Maximum Suppression to the detected heads
    # It's reccomended to use NMS when using OwlV2
    NMSAgent(iou_threshold=0.5),  
    # Splits the input image into an image of each detected head
    SplitAgent(),
    # Classifies PPE using CLIP
    ClassificationAgent(classes=["hard hat", "no hard hat"]),  
    # Maps the returned class names 
    ClassMapAgent({"hard hat": "has ppe", "no hard hat": "no ppe"}),
    # Combines results back into a BoundingBox Detection
    JoinAgent()  
])

image_path = os.path.join(os.path.dirname(__file__), "construction.jpg")
image = Image.open(image_path)
result, graph = workflow.execute(image)
workflow.visualize(graph)