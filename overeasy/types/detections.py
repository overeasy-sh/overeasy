from pydantic.dataclasses import dataclass as pydantic_dataclass
import pydantic_numpy.typing as pnd
from pydantic import Field
import supervision as sv
import numpy as np
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple

from supervision.detection.utils import (
    extract_ultralytics_masks,
)

from .type_utils import ( 
    validate_detections_fields, 
    is_data_equal, 
    get_data_item,
    DetectionType
)

ORIENTED_BOX_COORDINATES="oriented_box_coordinates"

@pydantic_dataclass
class Detections:
    """
    Represents a collection of detection data including bounding boxes, segmentation masks,
    class IDs, and additional custom data. It supports operations like filtering, splitting,
    and merging detections.
    
    Attributes:
        xyxy (np.ndarray[np.float32]): Coordinates of bounding boxes `[x1, y1, x2, y2]` for each detection.
        masks (Optional[np.ndarray[np.float32]]): Segmentation masks for each detection, required for segmentation types.
        class_ids (Optional[np.ndarray[np.int32]]): Class IDs for each detection, indexing into `classes`.
        classes (Optional[np.ndarray[np.object_]]): Array of class names, indexed by `class_ids`.
        confidence (Optional[np.ndarray[np.float32]]): Confidence scores for each detection.
        data (Dict[str, Union[np.ndarray, List]]): Additional custom data related to detections.
        detection_type (DetectionType): Type of detection (e.g., bounding box, segmentation, classification).
    """
    
    xyxy: pnd.NpNDArrayFp32
    detection_type: DetectionType
    class_ids: pnd.NpNDArrayInt32  
    classes: pnd.NpNDArray  
    masks: Optional[pnd.NpNDArrayFp32] = None  
    confidence: Optional[pnd.NpNDArrayFp32] = None
    data: Dict[str, Union[pnd.NpNDArray, List]] = Field(default_factory=dict)
    
    @property
    def confidence_scores(self) -> np.ndarray:
        if self.confidence is None:
            return np.array([None] * self.xyxy.shape[0])
        return self.confidence
    
    
    def split(self) -> List['Detections']:
        """
        Split the Detections object into a list of Detections objects, each containing
        a single detection.

        Returns:
            List[Detections]: A list of Detections objects, each containing a single
                detection.
        """
        rows, _ = np.shape(self.xyxy)
        if self.detection_type == DetectionType.CLASSIFICATION:
            raise ValueError("Splitting is not supported for classification detections as they are considered a single entity.")
        else:
            return [Detections(
                xyxy=self.xyxy[i:i+1],
                masks=self.masks[i:i+1] if self.masks is not None else None,
                class_ids=self.class_ids[i:i+1] if self.class_ids is not None else None,
                classes=self.classes,
                confidence=self.confidence[i:i+1] if self.confidence is not None else None,
                data={key: [value[i]] for key, value in self.data.items()} if self.data else {},
                detection_type=self.detection_type
            ) for i in range(rows)]
    
    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            detection_type=self.detection_type,
            masks=self.masks,  
            confidence=self.confidence,
            class_ids=self.class_ids,
            classes= self.classes,
            data=self.data,
        )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)
    
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, 
                                         DetectionType,
                                         Optional[List[np.ndarray]], 
                                         Optional[List[int]], 
                                         Optional[List[str]], 
                                         Optional[np.ndarray]]]:
        """
        Iterates over the Detections object and yields a tuple of
        `(xyxy, masks, class_ids, classes, confidence)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.detection_type,
                self.masks[i] if self.masks is not None else None,
                self.class_ids if self.class_ids is not None else None,
                self.classes if self.classes is not None else None,
                self.confidence[i] if self.confidence is not None else None,
            )
    
    
    
    def __eq__(self, other: object):
        if not isinstance(other, Detections):
            raise NotImplementedError("Only Detections objects can be compared.")

        def array_equal(array1: Optional[np.ndarray], array2: Optional[np.ndarray]) -> bool:
                if array1 is None and array2 is None:
                    return True
                elif array1 is None or array2 is None:
                    return False
                else:
                    return np.array_equal(array1, array2)
        
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                array_equal(self.masks, other.masks),
                array_equal(self.class_ids, other.class_ids),
                array_equal(self.confidence, other.confidence),
                array_equal(self.classes, other.classes),
                is_data_equal(self.data, other.data),
                self.detection_type == other.detection_type
            ]
        )
        
    @classmethod
    def from_classification(cls, assigned_classes: List[str], all_classes: Optional[List[str]] = None) -> 'Detections':
        if all_classes is None:
            return cls(
                xyxy=np.zeros((len(assigned_classes), 4)),
                class_ids=np.arange(len(assigned_classes)),
                classes=np.array(assigned_classes),
                detection_type=DetectionType.CLASSIFICATION
            )
        else:
            return cls(
                xyxy=np.zeros((len(assigned_classes), 4)),
                class_ids=np.array([all_classes.index(c) for c in assigned_classes]),
                classes=np.array(all_classes),
                detection_type=DetectionType.CLASSIFICATION
            )

    @classmethod
    def from_yolov5(cls, yolov5_results) -> 'Detections':
        """
        Creates a Detections instance from a
        [YOLOv5](https://github.com/ultralytics/yolov5) inference result.

        Args:
            yolov5_results (yolov5.models.common.Detections):
                The output Detections instance from YOLOv5

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import torch
            import overeasy as ov

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            result = model(image)
            detections = ov.Detections.from_yolov5(result)
            ```
        """
        yolov5_detections_predictions = yolov5_results.pred[0].cpu().cpu().numpy()
        classes = yolov5_results.names.values()
        return cls(
            xyxy=yolov5_detections_predictions[:, :4],
            confidence=yolov5_detections_predictions[:, 4],
            class_ids=yolov5_detections_predictions[:, 5].astype(int),
            classes=np.array(classes),
            detection_type=DetectionType.BOUNDING_BOX
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> 'Detections':
        """
        Creates a Detections instance from a
            [YOLOv8](https://github.com/ultralytics/ultralytics) inference result.

        !!! Note

            `from_ultralytics` is compatible with
            [detection](https://docs.ultralytics.com/tasks/detect/),
            [segmentation](https://docs.ultralytics.com/tasks/segment/), and
            [OBB](https://docs.ultralytics.com/tasks/obb/) models.

        Args:
            ultralytics_results (ultralytics.yolo.engine.results.Results):
                The output Results instance from YOLOv8

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import overeasy as ov
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = ov.Detections.from_ultralytics(result)
            ```
        """  # noqa: E501 // docs

        if ultralytics_results.obb is not None:
            class_id = ultralytics_results.obb.cls.cpu().numpy().astype(int)
            class_names = np.array([ultralytics_results.names[i] for i in class_id])
            oriented_box_coordinates = ultralytics_results.obb.xyxyxyxy.cpu().numpy()
            return cls(
                xyxy=ultralytics_results.obb.xyxy.cpu().numpy(),
                confidence=ultralytics_results.obb.conf.cpu().numpy(),
                class_ids=class_id,
                classes = class_names,
                data={
                    ORIENTED_BOX_COORDINATES: oriented_box_coordinates,
                },
                detection_type=DetectionType.BOUNDING_BOX
            )
            

        masks = extract_ultralytics_masks(ultralytics_results)
        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        
        if isinstance(ultralytics_results.names, dict):
            class_names = np.array(list(ultralytics_results.names.values()))
        else:
            class_names = np.array(ultralytics_results.names)
        
        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_ids=class_id,
            masks=masks,
            classes = class_names,
            data={},
            detection_type=DetectionType.BOUNDING_BOX if masks is None else DetectionType.SEGMENTATION
        )
    
    @classmethod
    def from_supervision_detection(
        cls, 
        sv_detection, 
        classes: Optional[List[str]] = None, 
        **kwargs
    ) -> 'Detections':

        xyxy = sv_detection.xyxy
        class_ids = np.array(sv_detection.class_id) if sv_detection.class_id is not None else None
        masks = np.array(sv_detection.mask) if sv_detection.mask is not None else None
        confidence = np.array(sv_detection.confidence) if sv_detection.confidence is not None else None
  
        return cls(
            xyxy=xyxy, 
            class_ids=class_ids, 
            masks=masks, 
            classes=np.array(classes) if classes is not None else None, 
            confidence=confidence, 
            detection_type=DetectionType.BOUNDING_BOX if masks is None else DetectionType.SEGMENTATION, 
            **kwargs
        )
    
    @classmethod
    def empty(cls) -> 'Detections':
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            classes=np.array([], dtype='object'),
            class_ids=np.array([], dtype=np.int32),
            confidence=np.array([], dtype=np.float32),
            detection_type=DetectionType.BOUNDING_BOX,
            data={}
        )


    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> 'Detections':
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            masks=self.masks[index] if self.masks is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_ids=self.class_ids[index],
            classes=self.classes,
            data=get_data_item(self.data, index),
            detection_type=self.detection_type
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @property
    def area(self) -> np.ndarray:
        if self.masks is not None:
            return np.array([np.sum(mask) for mask in self.masks])
        else:
            return self.box_area

    @property
    def box_area(self) -> np.ndarray:
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])
    
    @property
    def class_names(self) -> List[str]:
        try:
            return [self.classes[class_id] for class_id in self.class_ids]
        except IndexError as e:
            raise IndexError(f"One or more class_ids are out of bounds for the available classes: {e}")
   
   
    def to_supervision(self) -> sv.Detections:
        # Extract class names and class IDs if they exist in the data dictionary

        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.confidence,
            class_id=self.class_ids,
            mask=self.masks,
            data=self.data  
        )


        

        


