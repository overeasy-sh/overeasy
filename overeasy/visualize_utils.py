import numpy as np
from PIL import Image
import random
import cv2
from overeasy.types.detections import Detections, DetectionType
from typing import Optional
import matplotlib.pyplot as plt

random.seed(42)

def generate_random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def annotate_with_string(image: Image.Image, data_str: str) -> Image.Image:
    fig, ax = plt.subplots()
    ax.imshow(np.array(image))
    ax.axis('off')
    plt.figtext(0.5, 0.01, data_str, wrap=True, horizontalalignment='center', fontsize=12)
    fig.canvas.draw()  
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # type: ignore
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    image = Image.fromarray(data)
    return image.convert('RGB')

def annotate(scene: Image.Image, detection: Detections, seed: Optional[int] = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)

    def draw_bounding_boxes(image: Image.Image, boxes, class_ids, class_names):
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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
        class_names = detection.class_names
        if len(class_names) == 1:
            return annotate_with_string(scene, class_names[0])
        else:
            return annotate_with_string(scene, str(class_names))


