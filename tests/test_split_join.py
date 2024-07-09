import pytest
from PIL import Image
from overeasy import *
from overeasy.models import *
from overeasy.types import Detections, DetectionType, ExecutionNode
import os
import numpy as np

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, "outputs")

@pytest.fixture
def count_eggs_image():
    image_path = os.path.join(ROOT, "count_eggs.jpg")
    return Image.open(image_path)

@pytest.fixture
def construction_image():
    image_path = os.path.join(ROOT, "../", "examples", "construction_workers.jpg")
    return Image.open(image_path)

@pytest.fixture
def dense_street_images():
    image_path = os.path.join(ROOT, "../", "examples", "dense_street1.jpg")
    image_path2 = os.path.join(ROOT, "../", "examples", "dense_street2.jpg")
    image_path3 = os.path.join(ROOT, "../", "examples", "dense_street3.jpg")

    return [Image.open(image_path), Image.open(image_path2), Image.open(image_path3)]


@pytest.fixture
def split_workflow() -> Workflow:
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
    ])
    return workflow



def test_execute_multiple(split_workflow: Workflow, count_eggs_image):
    # Test with a single image
    single_result, single_graph = split_workflow.execute(count_eggs_image)
    
    # Test with multiple copies of the same image
    multi_results, multi_graphs = split_workflow.execute_multiple([count_eggs_image, count_eggs_image])
    
    assert len(multi_results) == 2, "execute_multiple should return results for 2 images"
    assert len(multi_graphs) == 2, "execute_multiple should return graphs for 2 images"
    
    for multi_result, multi_graph in zip(multi_results, multi_graphs):
        # Compare number of detections
        assert len(single_result) == len(multi_result), "Number of detections should be the same"
        
        # Compare detection data
        for single_node, multi_node in zip(single_result, multi_result):
            assert np.array_equal(single_node.data.xyxy, multi_node.data.xyxy), "Bounding boxes should be the same"
            assert np.array_equal(single_node.data.confidence_scores, multi_node.data.confidence_scores), "Confidence scores should be the same"
            assert single_node.data.class_names == multi_node.data.class_names, "Class names should be the same"
        
        # Compare graph structure
        single_layers = single_graph.top_sort()
        multi_layers = multi_graph.top_sort()
        assert len(single_layers) == len(multi_layers), "Graph structure should be the same"
        for single_layer, multi_layer in zip(single_layers, multi_layers):
            assert len(single_layer) == len(multi_layer), "Each layer should have the same number of nodes"

def test_split_agent(split_workflow: Workflow, count_eggs_image):
    result, graph = split_workflow.execute(count_eggs_image)
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

def images_are_equal(img1: Image.Image, img2: Image.Image) -> bool:
    # Ensure both images are in the same mode
    if img1.mode != img2.mode:
        img1 = img1.convert(img2.mode)
    
    # Ensure both images are the same size
    if img1.size != img2.size:
        return False
    
    # Compare pixel data
    np_img1 = np.array(img1)
    np_img2 = np.array(img2)
    
    return np.array_equal(np_img1, np_img2)

def test_splitting_empty_detection(split_workflow: Workflow):
    empty_detections = Detections.empty()
    empty_image = Image.new('RGB', (640, 640))

    result, graph = split_workflow.execute(empty_image, empty_detections)
    
    assert len(result) == 0, "SplitAgent should return an empty list when there are no detections"

def test_splitting_and_joining_empty_detection():
    workflow = Workflow([
        SplitAgent(),
        JoinAgent()
    ])
    
    empty_detections = Detections.empty()
    empty_image = Image.new('RGB', (640, 640))
    result, graph = workflow.execute(empty_image, empty_detections)
    
    assert len(result) == 1
    assert np.array_equal(result[0].data, [None]), "JoinAgent should return null data"
    assert images_are_equal(result[0].image, empty_image), "JoinAgent should return original image"
    assert [len(layer)==1 for layer in graph.top_sort()], "Graph should have one node per layer"

@pytest.fixture
def filter_split_join_workflow() -> Workflow:
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg", "carton"], model=GroundingDINO()),
        SplitAgent(),
        JoinAgent(),
        FilterClassesAgent(class_names=["a single egg"]),
        SplitAgent(),
        JoinAgent()
    ])
    return workflow

def test_many_split_joins1(count_eggs_image):
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["person"], model=OwlV2()),
        SplitAgent(),
        BoundingBoxSelectAgent(classes=["head"]),
        SplitAgent(),
        ClassificationAgent(classes=["hard hat", "head"]),
        JoinAgent(),
        JoinAgent()
    ])
    result, graph = workflow.execute(count_eggs_image)

    assert all(len(x) == 1 for x in graph.top_sort()), "Graph should have one node per layer"
    assert isinstance(result, list), "Complex Split-Join didn't return a list"
    assert len(result) > 0, "Complex Split-Join didn't return a list of detections"
    assert images_are_equal(result[0].image, count_eggs_image), "Image should be the same"

def test_many_split_joins2(count_eggs_image):
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["person"], model=OwlV2()),
        SplitAgent(),
        SplitAgent(),
        SplitAgent(),
        ClassificationAgent(classes=["hard hat", "head"]),
        JoinAgent(),
        JoinAgent(),
        JoinAgent()
    ])
    result, graph = workflow.execute(count_eggs_image)

    assert all(len(x) == 1 for x in graph.top_sort()), "Graph should have one node per layer"
    assert isinstance(result, list), "Complex Split-Join didn't return a list"
    assert len(result) > 0, "Complex Split-Join didn't return a list of detections"
    assert images_are_equal(result[0].image, count_eggs_image), "Image should be the same"


def test_many_split_joins3(dense_street_images):
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["person"]),
        PadCropAgent.from_uniform_padding(padding=25),
        ConfidenceFilterAgent(max_n=5), # this is to save time
        SplitAgent(),
        BoundingBoxSelectAgent(classes=["head"]),
        ConfidenceFilterAgent(max_n=1),
        SplitAgent(),
        BoundingBoxSelectAgent(classes=["glasses"], model=OwlV2()),
        ConfidenceFilterAgent(max_n=1),
        SplitAgent(),
        ClassificationAgent(classes=["sunglasses", "glasses"]),
        JoinAgent(),
        JoinAgent(),
        JoinAgent()
    ])

    result, _ = workflow.execute(dense_street_images[0])
    
    assert isinstance(result, list), "Multi Split-Join didn't return a list"
    assert len(result) > 0, "Multi Split-Join didn't return a list of detections"
    assert images_are_equal(result[0].image, dense_street_images[0]), "Image should be the same"


def test_filter_split_join_workflow(filter_split_join_workflow: Workflow, count_eggs_image):
    result, graph = filter_split_join_workflow.execute(count_eggs_image)
    assert all(isinstance(x.data, Detections) for x in result), "Filter Split-Join didn't return detections"
    assert isinstance(result, list), "Filter Split-Join didn't return a list"
    assert len(result) > 0, "Filter Split-Join didn't return a list of detections"
    
    
def test_mismatched_split_join():
    # This should be fine
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"]),
        SplitAgent(),
        JoinAgent(),
        SplitAgent(),
    ])
    try:
        workflow = Workflow([
            BoundingBoxSelectAgent(classes=["a single egg"]),
            JoinAgent(),
            SplitAgent(),
            SplitAgent(),
        ])
        assert False, "Mismatched number of join has no corresponding split"
    except ValueError as e:
        pass
    
    workflow = Workflow([
        BoundingBoxSelectAgent(classes=["a single egg"]),
        SplitAgent(),
        SplitAgent(),
        JoinAgent(),
        JoinAgent(),
    ])

