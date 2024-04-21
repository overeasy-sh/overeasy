from typing import Any, Dict, Union, List
from itertools import chain
import numpy as np
from enum import Enum

class DetectionType(Enum):
    BOUNDING_BOX = "bounding_box"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    
def validate_data(data: Dict[str, Any], n: int) -> None:
    for key, value in data.items():
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(f"Length of list for key '{key}' must be {n}")
        elif isinstance(value, np.ndarray):
            if value.ndim == 1 and value.shape[0] != n:
                raise ValueError(f"Shape of np.ndarray for key '{key}' must be ({n},)")
            elif value.ndim > 1 and value.shape[0] != n:
                raise ValueError(
                    f"First dimension of np.ndarray for key '{key}' must have size {n}"
                )
        else:
            raise ValueError(f"Value for key '{key}' must be a list or np.ndarray")
        
def validate_detections_fields(
    xyxy: Any,
    masks: Any,
    class_ids: Any,
    classes: Any,
    confidence: Any,
    data: Dict[str, Any],
    detection_type: Any,
) -> None:
    expected_shape_xyxy = "(_, 4)"
    actual_shape_xyxy = str(getattr(xyxy, "shape", None))
    is_valid_xyxy = isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
    if not is_valid_xyxy:
        raise ValueError(
            f"xyxy must be a 2D np.ndarray with shape {expected_shape_xyxy}, but got shape "
            f"{actual_shape_xyxy}"
        )

    n = len(xyxy)
    if masks is not None:
        expected_shape_masks = f"({n}, H, W)"
        actual_shape_masks = str(getattr(masks, "shape", None))
        is_valid_masks = isinstance(masks, np.ndarray) and len(masks.shape) == 3 and masks.shape[0] == n
        if not is_valid_masks:
            raise ValueError(
                f"masks must be a 3D np.ndarray with shape {expected_shape_masks}, but got shape "
                f"{actual_shape_masks}"
            )

    if class_ids is not None:
        expected_shape_class_ids = f"({n},)"
        actual_shape_class_ids = str(getattr(class_ids, "shape", None))
        is_valid_class_ids = isinstance(class_ids, np.ndarray) and class_ids.shape == (n,)
        if not is_valid_class_ids:
            raise ValueError(
                f"class_ids must be a 1D np.ndarray with shape {expected_shape_class_ids}, but got "
                f"shape {actual_shape_class_ids}"
            )

    if classes is not None:
        is_valid_classes = isinstance(classes, np.ndarray) and classes.dtype.kind in {'U', 'O'}
        if not is_valid_classes:
            raise ValueError(
                "classes must be a np.ndarray of strings."
            )

    if confidence is not None:
        expected_shape_confidence = f"({n},)"
        actual_shape_confidence = str(getattr(confidence, "shape", None))
        is_valid_confidence = isinstance(confidence, np.ndarray) and confidence.shape == (n,)
        if not is_valid_confidence:
            raise ValueError(
                f"confidence must be a 1D np.ndarray with shape {expected_shape_confidence}, but got "
                f"shape {actual_shape_confidence}"
            )

    if not isinstance(detection_type, DetectionType):
        raise ValueError(
            "detection_type must be an instance of DetectionType enum."
        )

    validate_data(data, n)
    
def is_data_equal(data_a: Dict[str, Union[np.ndarray, List]], data_b: Dict[str, Union[np.ndarray, List]]) -> bool:
    """
    Compares the data payloads of two Detections instances.

    Args:
        data_a, data_b: The data payloads of the instances.

    Returns:
        True if the data payloads are equal, False otherwise.
    """
    return set(data_a.keys()) == set(data_b.keys()) and all(
        np.array_equal(data_a[key], data_b[key]) for key in data_a
    )
    
def get_data_item(
    data: Dict[str, Union[np.ndarray, List]],
    index: Union[int, slice, List[int], np.ndarray],
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Retrieve a subset of the data dictionary based on the given index.

    Args:
        data: The data dictionary of the Detections object.
        index: The index or indices specifying the subset to retrieve.

    Returns:
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data : Dict[str, Union[np.ndarray, List]] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            subset_data[key] = value[index]
        elif isinstance(value, list):
            if isinstance(index, slice):
                subset_data[key] = value[index]
            elif isinstance(index, (list, np.ndarray)):
                subset_data[key] = [value[i] for i in index]
            elif isinstance(index, int):
                subset_data[key] = [value[index]]
            else:
                raise TypeError(f"Unsupported index type: {type(index)}")
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    return subset_data
    
    
def merge_data(
    maps_to_merge: List[Dict[str, Union[np.ndarray, List]]],
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Merges the data payloads of a list of Detections instances.

    Args:
        data_list: The data payloads of the instances.

    Returns:
        A single data payload containing the merged data, preserving the original data
            types (list or np.ndarray).

    Raises:
        ValueError: If data values within a single object have different lengths or if
            dictionaries have different keys.
    """
    if not maps_to_merge:
        return {}

    all_keys_sets = [set(data.keys()) for data in maps_to_merge]
    if not all(keys_set == all_keys_sets[0] for keys_set in all_keys_sets):
        raise ValueError("All data dictionaries must have the same keys to merge.")

    for data in maps_to_merge:
        lengths = [len(value) for value in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                "All data values within a single object must have equal length."
            )

    def merge_list(key: str, to_merge: List[Union[np.ndarray, List]]) -> Union[np.ndarray, List]:
        if all(isinstance(item, list) for item in to_merge):
            return list(chain.from_iterable(to_merge))
        elif all(isinstance(item, np.ndarray) for item in to_merge):
            ndim = -1
            if isinstance(to_merge[0], np.ndarray):
                ndim = to_merge[0].ndim 
            if ndim == 1:
                return np.hstack(to_merge)
            elif ndim > 1:
                return np.vstack(to_merge)
            else:
                raise ValueError(f"Unexpected array dimension for input '{key}'.")
        else:
            raise ValueError(
                f"Inconsistent data types for key '{key}'. Only np.ndarray and list "
                f"types are allowed."
            )
            
    return {key: merge_list(key, [data[key] for data in maps_to_merge if key in data]) for key in all_keys_sets[0]}


