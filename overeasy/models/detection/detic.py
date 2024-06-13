import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from typing import Any, Union
import numpy as np
import torch
from overeasy.types import Detections, DetectionType
from typing import List
from PIL import Image
import cv2
from overeasy.types import BoundingBoxModel
import subprocess

VOCAB = "custom"
CONFIDENCE_THRESHOLD = 0.3


def setup_cfg(args):
    from centernet.config import add_centernet_config
    from detic.config import add_detic_config
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu" if args.cpu else "cuda"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False
    cfg.freeze()
    return cfg


def load_detic_model(classes : List[str]):
    mp.set_start_method("spawn", force=True)

    args = argparse.Namespace()

    args.confidence_threshold = CONFIDENCE_THRESHOLD
    args.vocabulary = VOCAB
    args.opts = []
    args.config_file = "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    args.cpu = False if torch.cuda.is_available() else True
    args.opts.append("MODEL.WEIGHTS")
    args.opts.append("models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    args.output = None
    args.webcam = None
    args.video_input = None
    args.custom_vocabulary = ", ".join(classes).rstrip(",")
    args.pred_all_class = False
    cfg = setup_cfg(args)
    print("SETUP CONFIGIIGGIG")

    from detic.predictor import VisualizationDemo
    # https://github.com/facebookresearch/Detic/blob/main/detic/predictor.py#L39
    demo = VisualizationDemo(cfg, args)

    return demo


HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_dependencies():
    try:
        import detectron2
    except:
        subprocess.run(
        ["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"]
    )
    overeasy_dir = os.path.expanduser("~/.overeasy")
    os.makedirs(overeasy_dir, exist_ok=True)

    os.chdir(overeasy_dir)

    models_dir = os.path.join(overeasy_dir, "Detic", "weights")
    os.makedirs(models_dir, exist_ok=True)

    fname = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    download_dir = os.path.join(models_dir, fname)
    if not os.path.exists(download_dir):
        model_url = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        subprocess.run(["wget", model_url, "-O", download_dir])







class DETIC(BoundingBoxModel):
    def __init__(self):
        check_dependencies()
        self.classes = None
        
    def set_classes(self, classes: List[str]):
        self.classes = classes
        self.detic_model = load_detic_model(classes)

    def detect(self, image: Union[np.ndarray, Image.Image], classes: List[str], box_threshold=0.35, text_threshold=0.25) -> Detections:
        if self.classes is None:
            self.set_classes(classes)
        elif not np.array_equal(self.classes, classes):
            self.set_classes(classes)
        
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        predictions, visualized_output = self.detic_model.run_on_image(np.array(image))
        pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        pred_classes = predictions["instances"].pred_classes.cpu().numpy()
        pred_scores = predictions["instances"].scores.cpu().numpy()


        if len(pred_classes) == 0:
            return Detections.empty()

        return Detections(
            xyxy=np.array(pred_boxes),
            detection_type=DetectionType.BOUNDING_BOX,
            class_ids=np.array(pred_classes),
            confidence=np.array(pred_scores),
            classes=self.classes,
        )

