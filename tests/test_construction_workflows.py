import pytest
from overeasy import *
from overeasy.models import OwlV2
from pydantic import BaseModel
from PIL import Image
import os

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

class PPE(BaseModel):
    hardhat: bool
    vest: bool
    boots: bool

@pytest.fixture
def construction_image():
    image_path = os.path.join(os.path.dirname(ROOT), "examples", "construction_workers.jpg")
    return Image.open(image_path)

@pytest.fixture
def ppe_instructor_workflow():
    return Workflow([
        BoundingBoxSelectAgent(classes=["person"]),
        SplitAgent(),
        InstructorImageAgent(response_model=PPE),
        ToClassificationAgent(fn=lambda x: "has ppe" if x.hardhat else "no ppe"),
        JoinAgent(),
    ])

@pytest.fixture
def owl_v2_workflow():
    return Workflow([
        BoundingBoxSelectAgent(classes=["person's head"], model=OwlV2()),
        NMSAgent(iou_threshold=0.5, score_threshold=0),
        SplitAgent(),
        ClassificationAgent(classes=["hard hat", "no hard hat"]),
        ClassMapAgent({"hard hat": "has ppe", "no hard hat": "no ppe"}),
        JoinAgent(),
    ])

def test_ppe_detection(construction_image, ppe_instructor_workflow):
    result, graph = ppe_instructor_workflow.execute(construction_image)
    assert result is not None
    assert graph is not None
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "construction_ppe_instructor.png"))


def test_ppe_owl_v2(construction_image, owl_v2_workflow):
    result, graph = owl_v2_workflow.execute(construction_image)
    assert result is not None
    assert graph is not None
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "construction_ppe_owlv2.png"))

