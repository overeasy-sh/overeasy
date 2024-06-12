from overeasy import Workflow, BoundingBoxSelectAgent, NMSAgent, JoinAgent, SplitAgent, OwlV2, ClassificationAgent, ClassMapAgent
from PIL import Image

workflow = Workflow([
    BoundingBoxSelectAgent(classes=["person's head"], model=OwlV2()),
    NMSAgent(iou_threshold=0.5, score_threshold=0),
    SplitAgent(),
    ClassificationAgent(classes=["hard hat", "no hard hat"]),
    ClassMapAgent({"hard hat": "has ppe", "no hard hat": "no ppe"}),
    JoinAgent(),
])

image_path = os.path.join(os.path.dirname(__file__), "construction.jpg")
image = Image.open(image_path)
result, graph = workflow.execute(image)
workflow.visualize(graph)
