import numpy as np
from PIL import Image
import random
import cv2
from overeasy.types.detections import Detections, DetectionType
random.seed(42)

def generate_random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def annotate(scene: Image.Image, detection: Detections) -> Image.Image:

    def draw_bounding_boxes(image: Image.Image, boxes, class_ids, class_names):
        cv2_image = np.array(image)
        image_height, image_width = cv2_image.shape[:2]
        scale = np.sqrt(image_height * image_width) / 200
        
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = generate_random_color()
            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, 2)
            label = class_names[class_id]
            font_scale = max(min((x2 - x1) / (scale * 50), 0.9), 0.5)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, 1)[0]
            text_x = x1
            text_y = max(y1, label_size[1] + 10)

            cv2.rectangle(cv2_image, (text_x, text_y - label_size[1] - 10), (text_x + label_size[0], text_y), color, -1)
            cv2.putText(cv2_image, label, (text_x, text_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, (0, 0, 0), 1)

        return Image.fromarray(cv2_image)
    
    if detection.detection_type == DetectionType.BOUNDING_BOX:
        return draw_bounding_boxes(scene, detection.xyxy, detection.class_ids, detection.classes)
    elif detection.detection_type == DetectionType.SEGMENTATION:
        raise NotImplementedError("Segmentation detections are not yet supported.")
    elif detection.detection_type == DetectionType.CLASSIFICATION:
        raise NotImplementedError("Classification detections are not yet supported.")


