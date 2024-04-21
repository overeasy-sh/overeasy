from pydantic.dataclasses import dataclass as pydantic_dataclass
import pydantic_numpy.typing as pnd
from pydantic import Field
import supervision as sv
import numpy as np
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple

from supervision.detection.utils import (
    extract_ultralytics_masks,
    process_roboflow_result,
    xywh_to_xyxy,
)

from .type_utils import ( 
    validate_detections_fields, 
    is_data_equal, 
    merge_data, 
    get_data_item,
    DetectionType
)

CLASS_NAME_DATA_FIELD="class_name"
ORIENTED_BOX_COORDINATES="oriented_box_coordinates"

@pydantic_dataclass
class Detections:
    class Config:
        arbitrary_types_allowed = True
    """
    Represents a collection of detections, including bounding boxes, segmentation masks,
    classification IDs, and additional custom data. This class provides a structured
    way to handle detection results from various sources, supporting operations like
    filtering, splitting, and merging detections.
    
    Attributes:
        xyxy (np.ndarray[np.float32]): An array of shape `(n, 4)` containing the coordinates of
            bounding boxes in the format `[x1, y1, x2, y2]`, where `n` is the number
            of detections. This is a required field for all detection types.
        masks (Optional[np.ndarray[np.float32]]): An optional 3D array of shape `(n, H, W)`,
            containing the segmentation masks for each detection. Each mask in the array
            corresponds to a bounding box in `xyxy`, with `n` being the number of detections,
            and `H` and `W` representing the height and width of the mask respectively.
            This field is applicable and required only for segmentation detection types.
        class_ids (Optional[np.ndarray[np.int32]]): An optional array of shape `(n,)` representing the class IDs
            for each detection, where `n` is the number of detections. Each element in the array
            is an index into the `classes` array, indicating the class of each detection.
        classes (Optional[np.ndarray[np.object_]]): An optional array of strings representing the class names
            available for detection. The length of this array corresponds to the number of unique
            classes. The `class_ids` field indexes into this array to provide a human-readable class
            name for each detection.
        confidence (Optional[np.ndarray[np.float32]]): An optional array of shape `(n,)` containing the
            confidence scores for each detection, where `n` is the number of detections.
            This field provides a measure of confidence for each detection and is useful
            for filtering detections based on a confidence threshold.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary for storing additional
            custom data related to detections. Each key is a string representing the
            data type, and the value is either a NumPy array or a list of data
            corresponding to each detection. This field is designed to accommodate any
            additional data that does not fit into the predefined attributes, allowing
            for flexibility in handling diverse detection data requirements.
        detection_type (DetectionType): An enum value indicating the type of detection
            (e.g., bounding box, segmentation, classification). This field helps in
            distinguishing between different detection modes and applying relevant
            operations based on the detection type.

    Note:
        The `data` field is designed to accommodate any additional data that does not
        fit into the predefined attributes. It allows for flexibility in handling
        diverse detection data requirements.
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

        return cls(
            xyxy=yolov5_detections_predictions[:, :4],
            confidence=yolov5_detections_predictions[:, 4],
            class_ids=yolov5_detections_predictions[:, 5].astype(int),
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
                class_id=class_id,
                classes = class_names,
                data={
                    ORIENTED_BOX_COORDINATES: oriented_box_coordinates,
                },
            )

        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])
        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_ids=class_id,
            masks=extract_ultralytics_masks(ultralytics_results),
            classes = class_names,
            data={},
        )

    @classmethod
    def from_mmdetection(cls, mmdet_results) -> 'Detections':
        """
        Creates a Detections instance from a
        [mmdetection](https://github.com/open-mmlab/mmdetection) and
        [mmyolo](https://github.com/open-mmlab/mmyolo) inference result.

        Args:
            mmdet_results (mmdet.structures.DetDataSample):
                The output Results instance from MMDetection.

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import supervision as sv
            from mmdet.apis import init_detector, inference_detector

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = init_detector(<CONFIG_PATH>, <WEIGHTS_PATH>, device=<DEVICE>)

            result = inference_detector(model, image)
            detections = sv.Detections.from_mmdetection(result)
            ```
        """  # noqa: E501 // docs

        return cls(
            xyxy=mmdet_results.pred_instances.bboxes.cpu().numpy(),
            confidence=mmdet_results.pred_instances.scores.cpu().numpy(),
            class_ids=mmdet_results.pred_instances.labels.cpu().numpy().astype(int),
        )

    @classmethod
    def from_transformers(cls, transformers_results: dict) -> 'Detections':
        """
        Creates a Detections instance from object detection
        [transformer](https://github.com/huggingface/transformers) inference result.

        Returns:
            Detections: A new Detections object.
        """

        return cls(
            xyxy=transformers_results["boxes"].cpu().numpy(),
            confidence=transformers_results["scores"].cpu().numpy(),
            class_id=transformers_results["labels"].cpu().numpy().astype(int),
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
        )
 
    @classmethod
    def from_sam(cls, sam_result: List[dict]) -> 'Detections':
        """
        Creates a Detections instance from
        [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
        inference result.

        Args:
            sam_result (List[dict]): The output Results instance from SAM

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import supervision as sv
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator
             )

            sam_model_reg = sam_model_registry[MODEL_TYPE]
            sam = sam_model_reg(checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
            mask_generator = SamAutomaticMaskGenerator(sam)
            sam_result = mask_generator.generate(IMAGE)
            detections = sv.Detections.from_sam(sam_result=sam_result)
            ```
        """

        sorted_generated_masks = sorted(
            sam_result, key=lambda x: x["area"], reverse=True
        )

        xywh = np.array([mask["bbox"] for mask in sorted_generated_masks])
        mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])

        if np.asarray(xywh).shape[0] == 0:
            return cls.empty()

        xyxy = xywh_to_xyxy(boxes_xywh=xywh)
        return cls(xyxy=xyxy, masks=mask)

    @classmethod
    def from_azure_analyze_image(
        cls, azure_result: dict, class_map: Optional[Dict[int, str]] = None
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

        is_dynamic_mapping = class_map is None
        if is_dynamic_mapping:
            class_map = {}

        class_map = {value: key for key, value in class_map.items()}

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
                class_id = class_map.get(class_name, None)

                if is_dynamic_mapping and class_id is None:
                    class_id = len(class_map)
                    class_map[class_name] = class_id

                if class_id is not None:
                    xyxy.append([x0, y0, x1, y1])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        if len(xyxy) == 0:
            return Detections.empty()

        return cls(
            xyxy=np.array(xyxy),
            class_ids=np.array(class_ids),
            confidence=np.array(confidences),
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
            # masks=masks, 
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

        
    @classmethod
    def merge(cls, detections_list: List['Detections']) -> 'Detections':
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object. If all elements in a field are not
        `None`, the corresponding field will be stacked.
        Otherwise, the field will be set to `None`.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.

        Example:
            ```python
            import numpy as np
            import supervision as sv

            detections_1 = sv.Detections(
                xyxy=np.array([[15, 15, 100, 100], [200, 200, 300, 300]]),
                class_id=np.array([1, 2]),
                data={'feature_vector': np.array([0.1, 0.2)])}
             )

            detections_2 = sv.Detections(
                xyxy=np.array([[30, 30, 120, 120]]),
                class_id=np.array([1]),
                data={'feature_vector': [np.array([0.3])]}
             )

            merged_detections = Detections.merge([detections_1, detections_2])

            merged_detections.xyxy
            array([[ 15,  15, 100, 100],
                   [200, 200, 300, 300],
                   [ 30,  30, 120, 120]])

            merged_detections.class_id
            array([1, 2, 1])

            merged_detections.data['feature_vector']
            array([0.1, 0.2, 0.3])
            ```
        """
        if len(detections_list) == 0:
            return Detections.empty()

        for detections in detections_list:
            validate_detections_fields(
                xyxy=detections.xyxy,
                detection_type=detections.detection_type,
                masks=detections.masks,
                confidence=detections.confidence,
                class_ids=detections.class_ids,
                classes=detections.classes,
                data=detections.data,
            )

        xyxy = np.vstack([d.xyxy for d in detections_list])

        def stack_or_none(name: str):
            if all(d.__getattribute__(name) is None for d in detections_list):
                return None
            if any(d.__getattribute__(name) is None for d in detections_list):
                raise ValueError(f"All or none of the '{name}' fields must be None")
            return (
                np.vstack([d.__getattribute__(name) for d in detections_list])
                if name == "mask"
                else np.hstack([d.__getattribute__(name) for d in detections_list])
            )

        mask = stack_or_none("mask")
        confidence = stack_or_none("confidence")
        class_ids = stack_or_none("class_ids")

        data = merge_data([d.data for d in detections_list])

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_ids=class_ids,
            data=data,
        )

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union['Detections', List, np.ndarray, None]:
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
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            masks=self.masks[index] if self.masks is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_ids=self.class_ids[index] if self.class_id is not None else None,
            data=get_data_item(self.data, index),
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
        if self.mask is not None:
            return np.array([np.sum(mask) for mask in self.mask])
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

    # @property
    
    
    @property
    def class_name(self) -> str:
        id = self.class_id
        if id < 0 or id >= len(self.classes):
            raise IndexError(f"Class ID {id} is out of bounds for available classes.")
        return self.classes[id]
    
    @property
    def class_names(self) -> List[str]:
        try:
            return [self.classes[class_id] for class_id in self.class_ids]
        except IndexError as e:
            raise IndexError(f"One or more class_ids are out of bounds for the available classes: {e}")
    
    
    @classmethod
    def from_bounding_boxes(
        cls, 
        xyxy: np.ndarray, 
        classes: Optional[List[str]] = None, 
        **kwargs
    ) -> 'Detections':
        return cls(
            xyxy=xyxy, 
            classes=np.array(classes) if classes is not None else None, 
            detection_type=DetectionType.BOUNDING_BOX, 
            **kwargs
        )

    @classmethod
    def from_segmentations(
        cls, 
        masks: np.ndarray, 
        classes: Optional[List[str]] = None, 
        **kwargs
    ) -> 'Detections':
        return cls(
            xyxy=np.empty((len(masks), 4), dtype=np.float32), 
            masks=masks, 
            classes=np.array(classes) if classes is not None else None, 
            detection_type=DetectionType.SEGMENTATION, 
            **kwargs
        )

    @classmethod
    def from_classification(
        cls, 
        class_ids: List[int], 
        classes: Optional[List[str]] = None, 
        **kwargs
    ) -> 'Detections':
        return cls(
            xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32), 
            class_ids=np.array(class_ids, dtype=np.int32), 
            classes=np.array(classes) if classes is not None else None, 
            detection_type=DetectionType.CLASSIFICATION, 
            **kwargs
        )
   
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
        
    @staticmethod
    def join_detections(detections_list: List['Detections'], node_list: List['ExecutionNode']):
        if len(detections_list) == 0:
            return Detections.empty()

        # Initialize lists to collect combined attributes
        xyxy: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        confidence: List[float] = []
        class_ids: List[int] = []
        classes = []
        data = []
        
        for idx, detection in enumerate(detections_list):
            parent_xyxy = node_list[idx].parent_xyxy  
            if detection.detection_type == DetectionType.CLASSIFICATION:
                # Convert classifications to bounding boxes in the parent's reference frame
                # Assuming a default bounding box size or area
                for class_id in detection.class_ids:
                    bbox = detection.xyxy[0]
                    bbox = np.array([
                        bbox[0] + parent_xyxy[0],
                        bbox[1] + parent_xyxy[1],
                        bbox[2] + parent_xyxy[0],
                        bbox[3] + parent_xyxy[1]])
                    xyxy.append(bbox)
                    
                    class_ids.append(class_id)
                    if detection.classes is not None:
                        classes.extend(detection.classes.tolist())

            elif detection.detection_type == DetectionType.BOUNDING_BOX:
                # Add bounding boxes directly
                for box in detection.xyxy:
                    adjusted_box = np.array([
                        parent_xyxy[0] + box[0] , 
                        parent_xyxy[1] + box[1] , 
                        parent_xyxy[0] + box[2] , 
                        parent_xyxy[1] + box[3] 
                    ])
                    xyxy.append(adjusted_box)

                if detection.class_ids is not None:
                    class_ids.extend(detection.class_ids)
                if detection.classes is not None:
                    classes.extend(detection.classes.tolist())
            # TODO: Handle joining segmentation detections
            elif detection.detection_type == DetectionType.SEGMENTATION:
                raise BaseException("GG not handled yet")


            if detection.confidence is not None:
                confidence.extend(detection.confidence)

            if detection.data:
                data.append(detection.data)

        # Stack or concatenate attributes

        # Merge data using the merge_data function from the provided context
        merged_data = merge_data(data)

        return Detections(
            xyxy=np.vstack(xyxy) if xyxy else None,
            masks=np.vstack(masks) if masks else None,
            confidence=np.hstack(confidence) if confidence else None,
            class_ids=np.hstack(class_ids) if class_ids else None,
            classes=np.unique(classes) if classes else None,
            data=merged_data,
            detection_type=DetectionType.BOUNDING_BOX  # or appropriate type based on the logic
        )

        

        


