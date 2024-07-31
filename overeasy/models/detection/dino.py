import os
from re import T
import cv2
import numpy as np
import torch
from groundingdino.util.inference import Model
from PIL import Image
from typing import List, Union
from overeasy.types import Detections, DetectionType
from enum import Enum 
from overeasy.types import BoundingBoxModel
import warnings
import sys, io
from overeasy.download_utils import atomic_retrieve_and_rename


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
    mmdet_SwinL_zero = "mmdet_swinl_zero"
    mmdet_SwinB_zero = "mmdet_swinb_zero"
    mmdet_SwinL = "mmdet_swinl"
    mmdet_SwinB = "mmdet_swinb"


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

def download_mmdet_config(model: GroundingDINOModel):
    from transformers import BertConfig, BertModel
    from transformers import AutoTokenizer
    import nltk

    config = BertConfig.from_pretrained("bert-base-uncased")
    _ = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    to_store = os.path.expanduser("~/nltk_data")
    if not os.path.exists(to_store):
        os.makedirs(to_store)
    nltk.download('punkt', download_dir=to_store)
    nltk.download('averaged_perceptron_tagger', download_dir=to_store)

    OVEREASY_CACHE_DIR = os.path.expanduser("~/.overeasy")
    GROUNDING_DINO_CACHE_DIR = os.path.join(OVEREASY_CACHE_DIR, "groundingdino_mmdet")
    
    if not os.path.exists(GROUNDING_DINO_CACHE_DIR):
        os.makedirs(GROUNDING_DINO_CACHE_DIR)

    model_key = model.value
    if model_key not in mapping:
        raise ValueError(f"Unsupported model type: {model_key}")


    config_path = os.path.join(GROUNDING_DINO_CACHE_DIR, mapping[model.value]["config"])

    checkpoint_url = mapping[model_key]["checkpoint"]
    checkpoint_file = os.path.basename(checkpoint_url)
    checkpoint_path = os.path.join(GROUNDING_DINO_CACHE_DIR, checkpoint_file)

    if not os.path.exists(checkpoint_path):
        atomic_retrieve_and_rename(checkpoint_url, checkpoint_path)

    if not os.path.exists(config_path):
        # Clone mmdetection repository
        import subprocess
        mmdet_repo_url = "https://github.com/AnirudhRahul/mmdetection"
        mmdet_repo_path = os.path.join(GROUNDING_DINO_CACHE_DIR, "mmdetection")
        if not os.path.exists(mmdet_repo_path):
            subprocess.run(["git", "clone", mmdet_repo_url, mmdet_repo_path], check=True)
    
    return config_path, checkpoint_path


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


def load_mmdet_grounding_dino(model: GroundingDINOModel):
    from mmdet.apis import DetInferencer

    config_path, checkpoint_path = download_mmdet_config(model)
    inferencer =  DetInferencer(model=config_path, weights=checkpoint_path, device=DEVICE, palette='none', show_progress=False )
    inferencer.model.test_cfg.chunked_size = -1
    return inferencer


def load_grounding_dino(model: GroundingDINOModel):
    if model.value.startswith("mmdet"):
        return load_mmdet_grounding_dino(model)

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
    grounding_dino_model: Model
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
        if self.model_type.value.startswith("mmdet"):
            download_mmdet_config(self.model_type)
        else:
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

    def detect_mmdet(self, image: np.ndarray, classes: List[str], box_threshold) -> Detections:
        import sys
        import io

        # Redirect stdout to a StringIO object
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            res = self.grounding_dino_model(
                inputs=image,
                out_dir='outputs',
                texts=[tuple(classes)],
                pred_score_thr=box_threshold,
                batch_size=1,
                show=False,
                no_save_vis=True,
                no_save_pred=True,
                print_result=False,
                custom_entities=False,
                tokens_positive=None
            )
        except Exception as e:
            # If there's an error, print the captured stdout
            print("Error occurred during model inference:")
            print(sys.stdout.getvalue())
            raise e
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout

        preds = res["predictions"][0]
        scores = np.array(preds['scores'])
        class_ids = np.array(preds['labels'])
        bboxes = np.array(preds['bboxes'])
        
        slicer = np.where(scores > box_threshold)
        dets = Detections(
            xyxy=bboxes[slicer],
            class_ids=class_ids[slicer],
            confidence=scores[slicer],
            classes=classes,
            detection_type=DetectionType.BOUNDING_BOX
        )

        return dets
    
    def detect_multiple_mmdet(self, images: List[np.ndarray], class_groups: List[List[str]], box_threshold) -> List[Detections]:
        batch_size = len(images)
        
        res = self.grounding_dino_model(
            inputs=images,
            out_dir='outputs',
            texts=[tuple(classes) for classes in class_groups],
            pred_score_thr=box_threshold,
            batch_size=batch_size,
            show=False,
            no_save_vis=True,
            no_save_pred=True,
            print_result=False,
            custom_entities=False,
            tokens_positive=None
        )
        
        all_detections = []
        
        for i, (preds, classes) in enumerate(zip(res["predictions"], class_groups)):
            scores = np.array(preds['scores'])
            class_ids = np.array(preds['labels'])
            bboxes = np.array(preds['bboxes'])
            
            slicer = np.where(scores > box_threshold)
            dets = Detections(
                xyxy=bboxes[slicer],
                class_ids=class_ids[slicer],
                confidence=scores[slicer],
                classes=classes,
                detection_type=DetectionType.BOUNDING_BOX
            )
            
            all_detections.append(dets)
    
        return all_detections

    def detect_multiple(self, images: List[np.ndarray], class_groups: List[List[str]], box_threshold) -> List[Detections]:
        cv2_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) if isinstance(image, Image.Image) else image for image in images]
        
        if self.model_type.value.startswith("mmdet"):
            return self.detect_multiple_mmdet(cv2_images, class_groups, box_threshold)
        
        raise ValueError("Unsupported model type")

    def detect(self, image: Union[np.ndarray, Image.Image], classes: List[str], box_threshold=None, text_threshold=None) -> Detections:
        if box_threshold is None:
            box_threshold = self.box_threshold
        if text_threshold is None:
            text_threshold = self.text_threshold

            
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if self.model_type.value.startswith("mmdet"):
            return self.detect_mmdet(image, classes, box_threshold)

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