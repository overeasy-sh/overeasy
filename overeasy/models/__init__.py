# Lift imports from subdirs
from .detection import *
from .recognition import *
from .LLMs import *
from .classification import *




def warmup_models():
    qwen = QwenVL()
    del qwen
    dino = GroundingDINO()
    del dino
    detic = DETIC()
    del detic
    owlv2 = OwlV2()
    del owlv2
    yoloworld = YOLOWorld()
    del yoloworld
    clip = CLIP()
    del clip
    laionclip = LaionCLIP()
    del laionclip
    bio = BiomedCLIP()
    del bio

