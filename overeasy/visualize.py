from typing import Optional, Union, List, Dict
from overeasy.types import ExecutionNode, ExecutionGraph, Detections, DetectionType
from collections import defaultdict
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure, Figure
import matplotlib.gridspec as gridspec

def visualize_graph(graph) -> Figure:
    layers = graph.top_sort()  
    fig_height = 5 * len(layers) 
    fig = plt.figure(figsize=(8, fig_height), constrained_layout=True)  
    subfigs = fig.subfigures(nrows=len(layers), ncols=1) 

    for i, layer in enumerate(layers):
        subfig = subfigs[i]
        subfig.suptitle(f'Layer {i}', y=0.95)  
        visualize_layer(layer, subfig)  
        

    fig.tight_layout(pad=10.0)  
    return fig

def visualize_layer(nodes: List[ExecutionNode], parent_fig: Optional[SubFigure]=None) -> SubFigure:
    """Visualize detections on an image or a list of nodes."""

    if parent_fig is None:
        parent_fig = plt.figure(figsize=(10, 10)).subfigures(1, 1)[0]
    
    if not isinstance(parent_fig, SubFigure):
        print("parent_fig must be a matplotlib Figure object", type(parent_fig))
        raise TypeError("parent_fig must be a matplotlib Figure object")



    if all(isinstance(node.data, Detections) for node in nodes):
        detections: List[Detections] = [node.data for node in nodes]
        num_nodes = len(nodes)
        gs = gridspec.GridSpec(1, num_nodes, figure=parent_fig)

        colors = ['g', 'r', 'b', 'c', 'm', 'y']
        detection: Detections = Detections.empty()

        for idx, (node, detection) in enumerate(zip(nodes, detections)):
            ax = parent_fig.add_subplot(gs[0, idx])
            ax.imshow(np.array(node.image))
            if detection.detection_type == DetectionType.BOUNDING_BOX:
                for i, (box, cls_name, score) in enumerate(zip(detection.xyxy, detection.class_names, detection.confidence_scores)):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none')
                    ax.add_patch(rect)
                    
                    text_bg_color = (1, 1, 1, 0.7)  
                    text_color = 'black'
                    
                    if score is not None:
                        label = f"{cls_name} {score:.2f}"
                    else:
                        label = f"{cls_name}"
                        
                    text_bbox = dict(facecolor=text_bg_color, edgecolor='none', boxstyle='round,pad=0.3')
                    font_size = 10
                    text_x = x1
                    text_y = y1 - 3 * font_size
                    
                    ax.text(text_x, text_y, label, color=text_color, fontsize=font_size, verticalalignment='top', bbox=text_bbox)
                                        
            elif detection.detection_type == DetectionType.SEGMENTATION:
                if detection.masks is None:
                    raise ValueError("Masks are required for segmentation detections.")
                
                for i, (mask, cls_name, score) in enumerate(zip(detection.masks, detection.class_names, detection.confidence_scores)):
                    mask_image = np.zeros_like(node.image)  # Assuming node.image is a numpy array
                    mask_image[mask] = colors[i % len(colors)]
                    ax.imshow(mask_image, alpha=0.5)
                    if score is not None:
                        ax.text(10, 30 * (i+1), f"{cls_name} {score:.2f}", color=colors[i % len(colors)], fontsize=12)
                    else:
                        ax.text(10, 30 * (i+1), f"{cls_name}", color=colors[i % len(colors)], fontsize=12)

            elif detection.detection_type == DetectionType.CLASSIFICATION:
                labels = [f"{name} {score:.2f}" if score is not None else f"{name}" for name, score in zip(detection.class_names, detection.confidence_scores)]

                for i, (label, class_id) in enumerate(zip(labels, detection.class_ids)):
                    
                    ax.text(10, -50, label, color=colors[class_id % len(colors)], fontsize=12, verticalalignment='top')
                
                ax.axis('off')

        return parent_fig
    else:
        axs = parent_fig.subplots(1, len(nodes), squeeze=False)

        for idx, node in enumerate(nodes):
            ax = axs[0][idx]
            ax.imshow(np.array(node.image))
            ax.axis('off')
            data_str = str(node.data)
            ax.text(0.5, 0, data_str, transform=ax.transAxes, ha="center")  

        return parent_fig


    
   