<h1 align="center"> 🥚 Overeasy
<br/>
<span align="center">
   <a href="https://github.com/overeasy-sh/overeasy/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/overeasy-sh/overeasy" alt="Github Stars"></a>
   <a href="https://pypi.org/project/overeasy/" target="_blank"><img src="https://img.shields.io/pypi/v/overeasy.svg?style=flat-square&label=PyPI+Overeasy" alt="Issues"></a>
   <a href="https://github.com/overeasy-sh/overeasy/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
    <a href="https://docs.overeasy.sh"><img src="https://img.shields.io/badge/Docs-informational" alt="Docs"></a>
    <a href="https://colab.research.google.com/drive/1Mkx9S6IG5130wiP9WmwgINiyw0hPsh3c?usp=sharing#scrollTo=L0_U27WJaTNO""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Demo"></a>
</span>
 </h1>




> **Overeasy is a framework for creating powerful computer vision models with no data**. 

By leveraging and chaining zero-shot models, Overeasy enables you to create custom end-to-end pipelines for tasks like:

- 📦 Bounding Box Detection
- 🏷️ Classification
- 🖌️ Segmentation (Coming Soon!)

All of this can be achieved without needing to collect and annotate large training datasets. 

Overeasy makes it simple to combine pre-trained zero-shot models to build powerful custom computer vision solutions.


## Installation
It's as easy as
```bash
conda create -n overeasy python=3.10
conda activate overeasy
pip install overeasy
```

For installing extras refer to our [Docs](https://docs.overeasy.sh/installation/installing-extras).

## Key Features
- `🤖 Agents`: Specialized tools that perform specific image processing tasks.
- `🧩 Workflows`: Define a sequence of Agents to process images in a structured manner.
- `🔗 Execution Graphs`: Manage and visualize the image processing pipeline.
- `🔎 Detections`: Represent bounding boxes, segmentation, and classifications.

## Documentation 
For more details on types, library structure, and available models please refer to our [Docs](https://docs.overeasy.sh).

## Example Usage 

Here is some example Overeasy code:

Download example image
```bash
!wget https://github.com/overeasy-sh/overeasy/blob/73adbaeba51f532a7023243266da826ed1ced6ec/examples/construction.jpg?raw=true -O construction.jpg
```

```python
from overeasy import *
from overeasy.models import OwlV2
from PIL import Image

workflow = Workflow([
    # Detect each head in the input image
    BoundingBoxSelectAgent(classes=["person's head"], model=OwlV2()),
    # Applies Non-Maximum Suppression to remove overlapping bounding boxes
    NMSAgent(iou_threshold=0.5, score_threshold=0),
    # Splits the input image into images of each detected head
    SplitAgent(),
    # Classifies the split images using CLIP
    ClassificationAgent(classes=["hard hat", "no hard hat"]),
    # Maps the returned class names
    ClassMapAgent({"hard hat": "has ppe", "no hard hat": "no ppe"}),
    # Combines results back into a BoundingBox Detection
    JoinAgent()
])

image = Image.open("./construction.jpg")
result, graph = workflow.execute(image)
workflow.visualize(graph)
```
And you should see an output like [this](https://htmlpreview.github.io/?https://github.com/overeasy-sh/overeasy/blob/main/gradio_example.html)



If you don't have a local GPU, you can run our examples by making a copy of this [Colab notebook](https://colab.research.google.com/drive/1Mkx9S6IG5130wiP9WmwgINiyw0hPsh3c?usp=sharing#scrollTo=L0_U27WJaTNO).

## Support
If you have any questions or need assistance, please open an issue or reach out to us at help@overeasy.sh.


Let's build amazing vision models together 🍳!