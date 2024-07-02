import pytest
from PIL import Image
from overeasy import *
from overeasy.models import *
from overeasy.types import Detections
from pydantic import BaseModel
import sys
import os
import torch

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

classification_models = [
    LaionCLIP(),
]

if torch.cuda.is_available():
    multimodal_llms = [
        PaliGemma("google/paligemma-3b-mix-224"),
        QwenVL(model_type="base"),
        QwenVL(model_type="int4"),
    ]
else:
    print("CUDA is not available, skipping QwenVL LLM tests.")
    multimodal_llms = [PaliGemma()]


@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

@pytest.fixture
def license_plate_image():
    image_path = os.path.join(ROOT, "plate.jpg")
    return Image.open(image_path)

@pytest.fixture(params=multimodal_llms)
def vision_prompt_workflow(request) -> Workflow:
    model = request.param
    workflow = Workflow([
        VisionPromptAgent(query="How many eggs are in this image?", model=model)
    ])
    return workflow

@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on macOS")
def local_vision_prompt_workflow() -> Workflow:  
    workflow = Workflow([
        VisionPromptAgent(query="How many eggs are in this image?", model=QwenVL(model_type="base"))
    ])
    return workflow

@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on macOS")
def local_vision_prompt_workflow_int4() -> Workflow:  
    workflow = Workflow([
        VisionPromptAgent(query="How many eggs are in this image?", model=QwenVL(model_type="int4"))
    ])
    return workflow

@pytest.fixture(params=classification_models)
def classification_workflow(request) -> Workflow:
    model = request.param
    workflow = Workflow([
        ClassificationAgent(classes=["0-5 eggs", "6-10 eggs", "11+ eggs"], model=model)
    ])
    return workflow


@pytest.fixture
def blank_image():
    return Image.new('RGB', (100, 100), color = 'white')


def test_vision_prompt_agent(vision_prompt_workflow: Workflow, count_eggs_image):
    result, graph = vision_prompt_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, str)
    name = (vision_prompt_workflow.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"vision_prompt_{name}.png"))
    
    del result, graph

@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on macOS")
def test_local_vision_prompt_agent(local_vision_prompt_workflow: Workflow, count_eggs_image):
    result, graph = local_vision_prompt_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, str)
    name = (local_vision_prompt_workflow.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"local_vision_prompt_{name}.png"))

@pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on macOS")
def test_local_vision_prompt_agent_int4(local_vision_prompt_workflow_int4: Workflow, count_eggs_image):
    result, graph = local_vision_prompt_workflow_int4.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, str)
    name = (local_vision_prompt_workflow_int4.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"local_vision_prompt_{name}.png"))

def test_classification_agent(classification_workflow: Workflow, count_eggs_image):
    result, graph = classification_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, Detections)

    name = (classification_workflow.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"classification_{name}.png"))