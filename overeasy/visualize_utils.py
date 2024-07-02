import numpy as np
from PIL import Image
import random
import cv2
from PIL import ImageDraw, ImageFont
import textwrap
from overeasy.types.detections import Detections, DetectionType
from typing import Optional

random.seed(42)

def generate_random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def annotate_with_string(image: Image.Image, data_str: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.load_default()
    max_width = image.width - 20
    lines = textwrap.wrap(data_str, width=(max_width // font.getlength(' ')))
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines])
    
    # Extend the image height to fit the wrapped text
    new_height = image.height + text_height + 20
    new_image = Image.new("RGB", (image.width, new_height), (0, 0, 0))
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    
    y_text = image.height + 10
    for line in lines:
        text_width, line_height = draw.textbbox((0, 0), line, font=font)[2:4]
        text_x = (new_image.width - text_width) / 2
        draw.text((text_x, y_text), line, font=font, fill="white")
        y_text += line_height
    
    return new_image

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

        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    
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


