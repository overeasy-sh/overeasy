import pytest
from PIL import Image
from overeasy import *
from overeasy.models import *
from overeasy.types import Detections
import os
import numpy as np

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

@pytest.fixture
def split_worflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        SplitAgent(),
    ])
    return workflow

@pytest.fixture
def split_join_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        SplitAgent(),
        JoinAgent()
    ])
    return workflow

@pytest.fixture
def no_split_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        SplitAgent(),
        ClassificationAgent(classes=["whole egg", "cracked egg"]),
        JoinAgent()
    ])
    return workflow

def test_split_agent(split_worflow: Workflow, count_eggs_image):
    result, graph = split_worflow.execute(count_eggs_image)
    assert all(isinstance(x.data, Detections) for x in result), "Split didn't return detections"
    assert isinstance(result, list), "Split didn't return a list"
    assert len(result) > 0, "Didn't return a list of detections"

def test_split_join_agent(split_join_workflow: Workflow, no_split_workflow: Workflow, count_eggs_image):
    result, graph = split_join_workflow.execute(count_eggs_image)
    result2, graph2 = no_split_workflow.execute(count_eggs_image)
    detections = result[0].data
    detections2 = result2[0].data
    assert isinstance(detections, Detections)  
    assert isinstance(detections2, Detections)
    assert detections == detections2, "Split join produced incorrect output"