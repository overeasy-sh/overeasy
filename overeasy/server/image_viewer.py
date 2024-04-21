
from flask import Flask, send_file, jsonify, redirect
import io
import base64
import overeasy as ov
from PIL import Image
import supervision as sv
import numpy as np 
import cv2
from typing import List, Tuple, Any
from overeasy.types import Agent, Detections, ExecutionNode, ExecutionGraph
from typing import Union, Any

def image_to_base64(image: Image.Image):
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
   
    return base64.b64encode(img_io.getvalue()).decode()

def annotate_image(detection: ov.types.Detections, image: Image.Image) -> Image.Image:
    draw_image = cv2.cvtColor(np.array(image.copy()), cv2.COLOR_RGB2BGR)
    bounding_box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    sv_detection = detection.to_supervision()
    
    if detection.detection_type == ov.types.DetectionType.BOUNDING_BOX:
        draw_image = bounding_box_annotator.annotate(scene=draw_image, detections=sv_detection)
    elif detection.detection_type == ov.types.DetectionType.SEGMENTATION:
        draw_image = mask_annotator.annotate(scene=draw_image, detections=sv_detection)

    return Image.fromarray(cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB))

def classification_to_html(class_names: List[str]) -> str:
    if len(class_names) > 1:
        return "<ul>" + "".join([f"<li>{name}</li>" for name in class_names]) + "</ul>"
    else:
        return f"<p>{class_names[0]}</p>"

def result_html(detection: Union[ov.types.Detections, Any], image: Image.Image):

    if not isinstance(image, Image.Image):
        raise TypeError("image must be an instance of PIL.Image.Image")

    if isinstance(detection, ov.types.Detections):
        if detection.detection_type in [ov.types.DetectionType.BOUNDING_BOX, ov.types.DetectionType.SEGMENTATION]:
            annotated_image = annotate_image(detection, image)
            return f'<div style="display: flex; justify-content: center; align-items: center; height: 100%;"><img src="data:image/jpeg;base64,{image_to_base64(annotated_image)}" /></div>'

        elif detection.detection_type == ov.types.DetectionType.CLASSIFICATION:
            return classification_to_html(detection.class_names)
    else:
        detection_str = str(detection)
        return f'<div style="display: flex; flex-direction: column; align-items: center;"><img src="data:image/jpeg;base64,{image_to_base64(image)}" /><p>{detection_str}</p></div>'


def make_server(graph: ExecutionGraph, steps: List[Agent]):
 
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def get_result():
            return redirect('/image/0', code=302)
        
        @app.route('/image/<int:n>', methods=['GET'])
        def get_nth_image(n):
            try:
                node, data = result[n]
                
                
                def fprint_agent(agent):
                    name = agent.__class__.__name__
                    res = "<strong>" + name + "</strong>"
                    if name == "BoundingBoxSelectAgent" or name == "ClassificationAgent" :
                        for c in agent.classes:
                            res += "<li>" + c + "</li>"
                    elif name == "BinaryChoiceAgent" or name == "PromptAgent":
                        res += "<li>" + agent.query + "</li>"
                    
                    return res
                

                reversed_data_list_items = ''
                for item, agent in zip(reversed(data), reversed(steps)):
                    split = False
                    # TODO: GG this part is kinda fucked you need to be able to
                    # tell if the split was joined later down the line of not 💀💀💀
                    
                    # if hasattr(agent, 'split') and agent.split:
                    #     node = node.parent
                    #     split = True

                    split_indicator = "<td style='color: red; font-weight: bold;'>SPLIT</td>" if split else "<td></td>"
                    reversed_data_list_items += f"""<tr>
                                    {split_indicator}
                                    <td style="padding: 10px; border: 1px solid #ddd; word-wrap: break-word; min-width: 300px;">{fprint_agent(agent)}</td>
                                    <td style="padding: 10px; border: 1px solid #ddd; word-wrap: break-word; max-width: 600px;">{result_html(item, node.image)}</td>
                                </tr>"""
  
                                    
                data_list_items = ''.join(reversed(reversed_data_list_items.splitlines(True)))
                
                 
                next_image_index = n + 1 if n + 1 < len(result) else 0 
                prev_image_index = n - 1 if n - 1 >= 0 else len(result)-1 

                return f"""
                    <html>
                        <head>
                            <style>
                                body {{
                                    font-family: 'Arial', sans-serif;
                                    text-align: center;
                                    margin: 0;
                                    padding: 0;
                                }}
                                table {{
                                    margin: 20px auto;
                                    border-collapse: collapse;
                                }}
                                th, td {{
                                    padding: 10px;
                                    border: 1px solid #ddd;
                                }}
                                th {{
                                    background-color: #f2f2f2;
                                }}
                                img {{
                                    max-width: 80%;
                                    height: auto;
                                    margin: 20px auto;
                                }}
                            </style>
                            <script>
                                document.addEventListener('keydown', function(event) {{
                                    if (event.key === 'ArrowRight') {{
                                        window.location.href = '/image/{next_image_index}';
                                    }}
                                    if (event.key === 'ArrowLeft') {{
                                        window.location.href = '/image/{prev_image_index}';
                                    }}
                                    
                                }});
                            </script>
                        </head>
                        <body>

                            <table>
                                <tr>
                                    <th>Annotated Output</th>
                                    <th>Agent Config</th>
                                </tr>
                                {data_list_items}
                            </table>
                        </body>
                    </html>
                    """
            except IndexError:
                return "Image not found", 404
        
        print("Running on http://localhost:3000")
        
        
        
        return app