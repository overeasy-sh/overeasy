from overeasy.types import Detections, DetectionType, DetectionAgent
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

class NMSAgent(DetectionAgent):
    def __init__(self, iou_threshold: float, score_threshold: float):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
    def execute(self, dets: Detections) -> Detections:
        if dets.detection_type != DetectionType.BOUNDING_BOX:
            raise ValueError("Only bounding box detections are supported for NMS.")
        return do_nms(dets, self.iou_threshold, self.score_threshold)
