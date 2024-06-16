import os
from overeasy import Workflow, BoundingBoxSelectAgent, ToClassificationAgent, JoinAgent, SplitAgent, InstructorImageAgent, ClassMapAgent
from pydantic import BaseModel
from PIL import Image

class PPE(BaseModel):
    hardhat: bool
    vest: bool
    boots: bool


workflow = Workflow([
    BoundingBoxSelectAgent(classes=["person"]),
    SplitAgent(),
    InstructorImageAgent(response_model=PPE),
    ToClassificationAgent(fn=lambda x: "has ppe" if x.hardhat else "no ppe"),
    # ClassMapAgent({"has ppe": "has ppe", "no ppe": "no ppe"}),
    JoinAgent(),
])

image_path = os.path.join(os.path.dirname(__file__), "construction_workers.jpg")
image = Image.open(image_path)
result, graph = workflow.execute(image)
workflow.visualize(graph)
