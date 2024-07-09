import pytest
from PIL import Image
from overeasy import *
from overeasy.types import Detections
from overeasy.models import *
from typing import List
import numpy as np
import os

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

def test_pad_crop_agent(pad_crop_workflow: Workflow, count_eggs_image):
    width, height = count_eggs_image.size
    result, graph = pad_crop_workflow.execute(count_eggs_image)
    image = result[0].image
    assert image.size == (width//2 + 20, height//2 + 20), "Incorrect pad crop size"

def test_split_crop_agent(split_crop_workflow: Workflow, count_eggs_image):
    result, graph = split_crop_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) == 4, "Incorrect number of split crops"

def test_nms_agent(nms_workflow: Workflow, count_eggs_image):
    result, graph = nms_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) < 10, "NMS should reduce number of detections"

def test_class_map_agent(class_map_workflow: Workflow, count_eggs_image):
    result, graph = class_map_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert "egg" in detections.class_names, "Class map failed"

def test_map_agent(map_agent_workflow: Workflow, count_eggs_image):
    result, graph = map_agent_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, np.ndarray)
    assert len(response) > 0, "Map agent failed"

def test_to_classification_agent(to_classification_workflow: Workflow, count_eggs_image):
    result, graph = to_classification_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, Detections)
    assert response.class_names[0] == "eggs detected", "To classification agent failed"

def test_filter_classes_agent(filter_classes_workflow: Workflow, count_eggs_image):
    result, graph = filter_classes_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    uniq_names = list(set(detections.class_names))
    assert np.array_equal(uniq_names, ["a single egg"]), "Filter classes agent failed"

def test_confidence_filter_agent(confidence_filter_workflow: Workflow, count_eggs_image):
    result, graph = confidence_filter_workflow.execute(count_eggs_image)
    detections = result[0].data 
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) < 10, "Confidence filter should reduce detections"

@pytest.fixture
def pad_crop_workflow() -> Workflow:
    workflow = Workflow([
        SplitCropAgent(split=(2,2)),
        PadCropAgent.from_uniform_padding(10),
        SplitAgent()
    ])
    return workflow

@pytest.fixture
def split_crop_workflow() -> Workflow:
    workflow = Workflow([
        SplitCropAgent(split=(2,2))
    ])
    return workflow

@pytest.fixture
def nms_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        NMSAgent(score_threshold=0.5, iou_threshold=0.5)
    ])
    return workflow

@pytest.fixture
def class_map_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        ClassMapAgent(class_map={"a single egg": "egg"})
    ])
    return workflow

@pytest.fixture
def map_agent_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["egg"], model=GroundingDINO()),
        MapAgent(fn=lambda det: det.xyxy)
    ])
    return workflow

@pytest.fixture
def to_classification_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        ToClassificationAgent(fn=lambda det: "eggs detected" if len(det.xyxy) > 0 else "no eggs detected")
    ])
    return workflow

@pytest.fixture
def filter_classes_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg", "carton"], model=GroundingDINO()),
        FilterClassesAgent(class_names=["a single egg"])
    ])
    return workflow

@pytest.fixture
def confidence_filter_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
        ConfidenceFilterAgent(min_confidence=0.1, max_n=1)
    ])
    return workflow