import pytest
from overeasy import Workflow
from overeasy.agents import BoundingBoxSelectAgent, SplitAgent, InstructorImageAgent, ToClassificationAgent, JoinAgent
from PIL import Image
import os
from pydantic import BaseModel

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

class PPE(BaseModel):
    hardhat: bool
    vest: bool
    boots: bool

@pytest.fixture
def construction_image():
    image_path = os.path.join(os.path.dirname(ROOT), "examples", "construction.jpg")
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

def test_save_visualization_to_html(construction_image, ppe_instructor_workflow):
    result, graph = ppe_instructor_workflow.execute(construction_image)
    assert result is not None
    assert graph is not None
    
    output_file = os.path.join(OUTPUT_DIR, "construction_ppe_instructor_visualization.html")
    ppe_instructor_workflow.visualize_to_file(graph, output_file)
    
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert '<html' in content
        assert '</html>' in content
        assert 'Step 1: Input Image' in content
        
