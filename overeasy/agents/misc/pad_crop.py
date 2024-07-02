from overeasy.types import Detections, DetectionType, DetectionAgent
from pydantic.dataclasses import dataclass
import numpy as np

@dataclass
class PadCropAgent(DetectionAgent):
    x1_pad: int
    y1_pad: int
    x2_pad: int
    y2_pad: int

    @classmethod
    def from_uniform_padding(cls, padding):
        return cls(padding, padding, padding, padding)

    @classmethod
    def from_xy_padding(cls, x_pad, y_pad):
        return cls(x_pad, y_pad, x_pad, y_pad)

    def _execute(self, dets: Detections) -> Detections:
        if dets.detection_type != DetectionType.BOUNDING_BOX:
            raise ValueError("Only bounding box detections are supported for padding.")
        
        padded_bboxes = []
        for bbox in dets.xyxy:
            x1, y1, x2, y2 = bbox
            padded_bbox = [
                x1 - self.x1_pad,
                y1 - self.y1_pad,
                x2 + self.x2_pad,
                y2 + self.y2_pad
            ]
            padded_bboxes.append(padded_bbox)
        
        dets.xyxy = np.array(padded_bboxes)
        dets.__post_init__()
        return dets

    def __repr__(self):
        return f"{self.__class__.__name__}(x1_pad={self.x1_pad}, y1_pad={self.y1_pad}, x2_pad={self.x2_pad}, y2_pad={self.y2_pad})"