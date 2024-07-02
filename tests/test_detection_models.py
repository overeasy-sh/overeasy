import pytest
from PIL import Image
from regex import W
from overeasy import Workflow, BoundingBoxSelectAgent, NMSAgent
from overeasy.models import GroundingDINO, YOLOWorld, OwlV2, DETIC
from overeasy.types import Detections
import os
import sys

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

@pytest.fixture
def grounding_dino_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=GroundingDINO()),
    ])
    return workflow

@pytest.fixture
def owlvit_v2_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=OwlV2()),
    ])
    return workflow

@pytest.fixture
def owlvit_v2_nms_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["one egg"], model=OwlV2()),
        NMSAgent(score_threshold=0, iou_threshold=0.5),
    ])
    return workflow

@pytest.fixture
def yoloworld_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"], model=YOLOWorld(model="yolov8s-worldv2")),
    ])
    return workflow

@pytest.fixture
def detic_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["egg"], model=DETIC()),
    ])
    return workflow

def test_grounding_dino_detection(count_eggs_image, grounding_dino_workflow: Workflow):
    result, graph = grounding_dino_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) > 0, "No detections found"
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "grounding_dino_detection_output.png"))

def test_yoloworld_detection(count_eggs_image, yoloworld_workflow: Workflow):
    result, graph = yoloworld_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) > 0, "No detections found"
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "yoloworld_detection_output.png"))
    

def test_owlvit_v2_detection(count_eggs_image, owlvit_v2_workflow: Workflow):
    result, graph = owlvit_v2_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) > 0, "No detections found"
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "owlv2_detection_output.png"))

# @pytest.mark.skipif(sys.platform == "darwin", reason="Detic is not working on macOS")
def test_detic_detection(count_eggs_image, detic_workflow: Workflow):
    result, graph = detic_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) > 0, "No detections found"
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "detic_detection_output.png"))

def test_nms_detection(count_eggs_image, owlvit_v2_nms_workflow: Workflow):
    result, graph = owlvit_v2_nms_workflow.execute(count_eggs_image)
    detections = result[0].data
    assert isinstance(detections, Detections)
    assert len(detections.xyxy) > 0, "No detections found"
    result[0].visualize().save(os.path.join(OUTPUT_DIR, "nms_detection_output.png"))

