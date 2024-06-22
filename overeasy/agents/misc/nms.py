from overeasy.types import Detections, DetectionType, DetectionAgent
import cv2

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
    # IoU (Intersection over Union) Threshold: Determines the minimum overlap between two bounding boxes to consider them as the same object.
    # Score Threshold: Filters out detections that have a confidence score below this value before applying NMS.
    def __init__(self, iou_threshold: float, score_threshold: float = 0.0):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
    def _execute(self, dets: Detections) -> Detections:
        if dets.detection_type != DetectionType.BOUNDING_BOX:
            raise ValueError("Only bounding box detections are supported for NMS.")
        return do_nms(dets, self.iou_threshold, self.score_threshold)

    def __repr__(self):
        return f"NMSAgent(iou_threshold={self.iou_threshold}, score_threshold={self.score_threshold})"
