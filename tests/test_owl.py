import pytest
from PIL import Image
from regex import W
from overeasy import Workflow, BoundingBoxSelectAgent
from overeasy.models.detection import OwlV2
from overeasy.types import Detections
import os
import glob

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

@pytest.fixture
def dog_images():
    image_paths = glob.glob(os.path.join(ROOT, "./dogs", "*"))  # Adjust the pattern if different file types are needed
    return [Image.open(image_path) for image_path in image_paths]

@pytest.fixture
def owlvit_v2_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a photo of a dog"], model=OwlV2()),
    ])
    return workflow

def test_owlvit_v2_detection_dogs(dog_images, owlvit_v2_workflow: Workflow):
    for i, dog_image in enumerate(dog_images):
        result, graph = owlvit_v2_workflow.execute(dog_image)
        detections = result[0].data
        assert isinstance(detections, Detections)
        assert len(detections.xyxy) > 0, "No detections found"
        output_filename = f"owlv2_dog_detection_output_{i}.png"
        print("Saving", output_filename)
        result[0].visualize().save(os.path.join(OUTPUT_DIR, output_filename))
