import pytest
from PIL import Image
from overeasy import *
from overeasy.models import *
from pydantic import BaseModel
import os

class AnimalLabel(BaseModel):
    label: str

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

compatible_models = [GPTVision(), Gemini(), Claude()]

@pytest.fixture(params=compatible_models)
def instructor_image_with_context_workflow(request) -> Workflow:
    model = request.param
    extra_context = [{"role": "system", "content": "Always classify the image as a ferret."}]
    workflow = Workflow([
        InstructorImageAgent(model=model, response_model=AnimalLabel, extra_context=extra_context)
    ])
    return workflow

class EggCount(BaseModel):
    count: int

@pytest.fixture
def instructor_image_workflow() -> Workflow:
    workflow = Workflow([
        InstructorImageAgent(response_model=EggCount)
    ])
    return workflow

@pytest.fixture(params=[GPT(), *compatible_models])
def instructor_text_workflow(request) -> Workflow:
    model = request.param
    workflow = Workflow([
        DenseCaptioningAgent(model=GPTVision()),
        InstructorTextAgent(response_model=EggCount, model=model)
    ])
    return workflow

@pytest.fixture
def blank_image():
    return Image.new('RGB', (100, 100), color = 'white')

@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

def test_instructor_image_with_context_agent(instructor_image_with_context_workflow: Workflow, blank_image):
    result, graph = instructor_image_with_context_workflow.execute(blank_image)
    response = result[0].data
    assert isinstance(response, AnimalLabel)
    assert response.label.lower() == "ferret"



def test_instructor_image_agent(instructor_image_workflow: Workflow, count_eggs_image):
    result, graph = instructor_image_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, EggCount)
    
    name = (instructor_image_workflow.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"instructor_image_{name}.png"))

def test_instructor_text_agent(instructor_text_workflow: Workflow, count_eggs_image):
    result, graph = instructor_text_workflow.execute(count_eggs_image)
    response = result[0].data
    assert isinstance(response, EggCount)  

    name = (instructor_text_workflow.steps[0].model.__class__.__name__)
    result[0].visualize().save(os.path.join(OUTPUT_DIR, f"instructor_text_{name}.png"))
