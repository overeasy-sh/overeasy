# Lift imports from subdirs
from .detection import *
from .recognition import *
from .LLMs import *
from .classification import *


def warmup_models():
    try:
        qwen = QwenVL()
        qwen.load_resources()
        del qwen
    except:
        print("Skipping QwenVL")
    
    try:
        detic = DETIC()
        detic.load_resources()
        detic.set_classes(["hi"])
        del detic
    except:
        print("Skipping DETIC")
    
    bounding_box_models = [    
    GroundingDINO(type=GroundingDINOModel.Pretrain_1_8M),
    GroundingDINO(type=GroundingDINOModel.SwinB),
    GroundingDINO(type=GroundingDINOModel.SwinT),
    YOLOWorld(model="yolov8s-worldv2"),
    YOLOWorld(model="yolov8m-worldv2"),
    YOLOWorld(model="yolov8l-worldv2"),
    YOLOWorld(model="yolov8s-world"),
    YOLOWorld(model="yolov8m-world"),
    YOLOWorld(model="yolov8l-world"),
    OwlV2(),
]
    
    for bounding_box_model in bounding_box_models:
        bounding_box_model.load_resources()
        del bounding_box_model
        
    
    
    clip = CLIP()
    del clip
    laionclip = LaionCLIP()
    del laionclip
    bio = BiomedCLIP()
    del bio

    pali_gemma = PaliGemma()
    pali_gemma.load_resources()
    del pali_gemma