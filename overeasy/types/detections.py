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
            import supervision as sv

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            result = model(image)
            detections = sv.Detections.from_yolov5(result)
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
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)
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
        class_names = np.array([ultralytics_results.names[i] for i in class_id])
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
    def from_detectron2(cls, detectron2_results) -> 'Detections':
        """
        Create a Detections object from the
        [Detectron2](https://github.com/facebookresearch/detectron2) inference result.

        Args:
            detectron2_results: The output of a
                Detectron2 model containing instances with prediction data.

        Returns:
            (Detections): A Detections object containing the bounding boxes,
                class IDs, and confidences of the predictions.

        Example:
            ```python
            import cv2
            import supervision as sv
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg


            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            cfg = get_cfg()
            cfg.merge_from_file(<CONFIG_PATH>)
            cfg.MODEL.WEIGHTS = <WEIGHTS_PATH>
            predictor = DefaultPredictor(cfg)

            result = predictor(image)
            detections = sv.Detections.from_detectron2(result)
            ```
        """

        return cls(
            xyxy=detectron2_results["instances"].pred_boxes.tensor.cpu().numpy(),
            confidence=detectron2_results["instances"].scores.cpu().numpy(),
            class_ids=detectron2_results["instances"]
            .pred_classes.cpu()
            .numpy()
            .astype(int),
            classes=None,
            detection_type=DetectionType.BOUNDING_BOX
        )
 


    @classmethod
    def from_azure_analyze_image(
        cls, azure_result: dict
    ) -> 'Detections':
        """
        Creates a Detections instance from [Azure Image Analysis 4.0](
        https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/
        concept-object-detection-40).

        Args:
            azure_result (dict): The result from Azure Image Analysis. It should
                contain detected objects and their bounding box coordinates.
            class_map (Optional[Dict[int, str]]): A mapping ofclass IDs (int) to class
                names (str). If None, a new mapping is created dynamically.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import requests
            import supervision as sv

            image = open(input, "rb").read()

            endpoint = "https://.cognitiveservices.azure.com/"
            subscription_key = ""

            headers = {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": subscription_key
             }

            response = requests.post(endpoint,
                headers=self.headers,
                data=image
             ).json()

            detections = sv.Detections.from_azure_analyze_image(response)
            ```
        """
        if "error" in azure_result:
            raise ValueError(
                f'Azure API returned an error {azure_result["error"]["message"]}'
            )

        xyxy, confidences, class_ids = [], [], []
        class_map: Dict[str, int] = {}
        all_class_names = []
        def assign_class_id(class_name: str) -> int:
            if class_name in class_map:
                return class_map[class_name]
            else:
                all_class_names.append(class_name)
                class_id = len(class_map)
                class_map[class_name] = class_id
                return class_id
        
        for detection in azure_result["objectsResult"]["values"]:
            bbox = detection["boundingBox"]

            tags = detection["tags"]

            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = x0 + bbox["w"]
            y1 = y0 + bbox["h"]

            for tag in tags:
                confidence = tag["confidence"]
                class_name = tag["name"]
                class_id = assign_class_id(class_name)
                
                xyxy.append([x0, y0, x1, y1])
                confidences.append(confidence)
                class_ids.append(class_id)
                


        if len(xyxy) == 0:
            return Detections.empty()

        return cls(
            xyxy=np.array(xyxy),
            class_ids=np.array(class_ids),
            confidence=np.array(confidences),
            detection_type=DetectionType.BOUNDING_BOX,
            classes=np.array(all_class_names)
        )
         
    @classmethod
    def from_supervision_detection(
        cls, 
        sv_detection, 
        classes: Optional[List[str]] = None, 
        **kwargs
    ) -> 'Detections':
        """
        Converts detections from the Supervision format to the Overeasy Detections format.

        Args:
            sv_detections (List[Dict[str, Any]]): A list of detections in the supervision format.
            classes (Optional[List[str]]): An optional list of class names.

        Returns:
            Detections: An instance of the Detections class.
        """
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
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.

        Example:
            ```python
            from supervision import Detections

            empty_detections = Detections.empty()
            ```
        """
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
        """
        Get a subset of the Detections object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the Detections or an item from the data.

        Returns:
            Union[Detections, Any]: A subset of the Detections object or an item from
                the data field.

        Example:
            ```python
            import supervision as sv

            detections = sv.Detections()

            first_detection = detections[0]
            first_10_detections = detections[0:10]
            some_detections = detections[[0, 2, 4]]
            class_0_detections = detections[detections.class_id == 0]
            high_confidence_detections = detections[detections.confidence > 0.5]

            feature_vector = detections['feature_vector']
            ```
        """
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
        """
        Set a value in the data dictionary of the Detections object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections['names'] = [
                 model.model.names[class_id]
                 for class_id
                 in detections.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each detection in the set of object detections.
        If masks field is defined property returns are of each mask.
        If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection
            in the format of `(area_1, area_2, , area_n)`,
            where n is the number of detections.
        """
        if self.masks is not None:
            return np.array([np.sum(mask) for mask in self.masks])
        else:
            return self.box_area

    @property
    def box_area(self) -> np.ndarray:
        """
        Calculate the area of each bounding box in the set of object detections.

        Returns:
            np.ndarray: An array of floats containing the area of each bounding
                box in the format of `(area_1, area_2, , area_n)`,
                where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])
    
    @property
    def class_names(self) -> List[str]:
        try:
            return [self.classes[class_id] for class_id in self.class_ids]
        except IndexError as e:
            raise IndexError(f"One or more class_ids are out of bounds for the available classes: {e}")
   
   
    def to_supervision(self) -> sv.Detections:
        """
        Convert this Detections instance into an sv.Detections instance.

        Returns:
            An instance of sv.Detections.
        """
        # Extract class names and class IDs if they exist in the data dictionary

        return sv.Detections(
            xyxy=self.xyxy,
            confidence=self.confidence,
            class_id=self.class_ids,
            mask=self.masks,
            data=self.data  
        )


        

        


