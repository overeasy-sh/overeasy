import os
from overeasy import *
from PIL import Image
workflow = Workflow([
  BoundingBoxSelectAgent(classes=["butterfly kite"], model=OwlV2()),
  NMSAgent(iou_threshold=0.8, score_threshold=0.6),
])

image_path = os.path.join(os.path.dirname(__file__), "kites.jpg")
image = Image.open(image_path)
results, graph = workflow.execute(image)
workflow.visualize(graph)