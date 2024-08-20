import os
from re import T
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Union
from overeasy.types import Detections, DetectionType
from enum import Enum 
from overeasy.types import BoundingBoxModel
import warnings
import sys, io
from overeasy.download_utils import atomic_retrieve_and_rename
from typing import Any

# Ignore the specific UserWarning about torch.meshgrid
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated and will be removed in v5 of Transformers.")
# Suppress UserWarning about use_reentrant parameter in torch.utils.checkpoint
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.", category=UserWarning, module='torch.utils.checkpoint')
# Suppress UserWarning about None of the inputs having requires_grad=True
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None", category=UserWarning, module='torch.utils.checkpoint')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GroundingDINOModel(Enum):
    SwinB = "swinb"
    SwinT = "swint"

mapping = {
    "swinb": {
        "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "checkpoint": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    },
    "swint": {
        "config": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "checkpoint": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    },
    "mmdet_swinb_zero": {
        "config": "./mmdetection/configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"
    },
    "mmdet_swinl_zero": {
        "config": "./mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
    },
    "mmdet_swinb": {
        "config": "./mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth"
    },
    "mmdet_swinl": {
        "config": "./mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py",
        "checkpoint": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth"
    }

}

def download_and_cache_grounding_dino(model: GroundingDINOModel):
    OVEREASY_CACHE_DIR = os.path.expanduser("~/.overeasy")
    GROUNDING_DINO_CACHE_DIR = os.path.join(OVEREASY_CACHE_DIR, "groundingdino")

    if not os.path.exists(GROUNDING_DINO_CACHE_DIR):
        os.makedirs(GROUNDING_DINO_CACHE_DIR)

    model_key = model.value
    if model_key not in mapping:
        raise ValueError(f"Unsupported model type: {model_key}")

    config_url = mapping[model_key]["config"]
    checkpoint_url = mapping[model_key]["checkpoint"]

    config_file = os.path.basename(config_url)
    checkpoint_file = os.path.basename(checkpoint_url)

    config_path = os.path.join(GROUNDING_DINO_CACHE_DIR, config_file)
    checkpoint_path = os.path.join(GROUNDING_DINO_CACHE_DIR, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        atomic_retrieve_and_rename(checkpoint_url, checkpoint_path)

    if not os.path.exists(config_path):
        atomic_retrieve_and_rename(config_url, config_path)

    return config_path, checkpoint_path



def load_grounding_dino(model: GroundingDINOModel):
    from groundingdino.util.inference import Model

    config_path, checkpoint_path = download_and_cache_grounding_dino(model)
    
    instantiate = lambda: Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=DEVICE,
    )

    try:
        grounding_dino_model = instantiate()
        return grounding_dino_model
    except Exception:


        grounding_dino_model = instantiate()

        return grounding_dino_model
    
def combine_detections(detections_list: List[Detections], classes: List[str], overwrite_class_ids=None):
    if len(detections_list) == 0:
        return Detections.empty()
    
    if not all(isinstance(detection, Detections) for detection in detections_list):
        raise TypeError("All elements in detections_list must be instances of Detections.")

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(detections_list):
        raise ValueError("Length of overwrite_class_ids must match the length of detections_list.")

    # Initialize lists to collect combined attributes
    xyxy = []
    confidence = []
    class_ids = []
    data = []
    masks = []

    detection_types = [detection.detection_type for detection in detections_list]
    if len(set(detection_types)) > 1:
        raise ValueError("All detections in the list must have the same type.")


    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_ids is not None:
            if overwrite_class_ids is not None:
                class_ids.append(np.full_like(detection.class_ids, overwrite_class_ids[idx], dtype=np.int32))
            else:
                class_ids.append(detection.class_ids)
        if detection.masks is not None:
            masks.append(detection.masks)

        # Merge custom data from each detection
        data.append(detection.data)

    return Detections(
        xyxy=np.vstack(xyxy),
        masks=np.hstack(masks) if masks else None,  # Assuming masks are not handled in this function
        classes = classes,
        confidence=np.hstack(confidence) if confidence else None,
        class_ids=np.hstack(class_ids) if class_ids else None,
        detection_type=detections_list[0].detection_type
    )

class GroundingDINO(BoundingBoxModel):
    grounding_dino_model: Any
    box_threshold: float
    text_threshold: float

    def __init__(
        self, type: GroundingDINOModel = GroundingDINOModel.SwinB,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_type = type
        
    def load_resources(self):
        # if DEVICE.type != "cuda":
        #     warnings.warn("CUDA not available. GroundingDINO may run slowly.")

        download_and_cache_grounding_dino(self.model_type)
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.grounding_dino_model = load_grounding_dino(model=self.model_type)
        except Exception as e:
            print(sys.stdout.getvalue())
            print(f"Error loading GroundingDINO model: {e}")
            raise e
        finally:
            sys.stdout = original_stdout
            
    def release_resources(self):
        self.grounding_dino_model = None

    # def detect_multiple(self, images: List[np.ndarray], class_groups: List[List[str]], box_threshold) -> List[Detections]:
    #     cv2_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) if isinstance(image, Image.Image) else image for image in images]
        
    #     raise ValueError("Unsupported model type")

    def detect(self, image: Union[np.ndarray, Image.Image], classes: List[str], box_threshold=None, text_threshold=None) -> Detections:
        if box_threshold is None:
            box_threshold = self.box_threshold
        if text_threshold is None:
            text_threshold = self.text_threshold

            
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        sv_detection = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        valid_indexes = [i for i, class_id in enumerate(sv_detection.class_id) if class_id is not None]
        sv_detection = sv_detection[valid_indexes]
        detections = Detections.from_supervision_detection(sv_detection, classes=classes)
 
        return detections