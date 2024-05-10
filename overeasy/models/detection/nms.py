from overeasy.types import Detections, DetectionType
import cv2
import numpy as np

def do_nms(dets: Detections, nms_threshold: float, score_threshold : float = 0.0) -> Detections:
    if dets.detection_type != DetectionType.BOUNDING_BOX:
        raise ValueError("Only bounding box detections are supported for NMS.")
    
    bboxes = dets.xyxy
    scores = dets.confidence
    if scores is None:
        raise ValueError("Confidence scores are required for NMS.")
    
    indices = list(cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold))

    return dets[indices]



