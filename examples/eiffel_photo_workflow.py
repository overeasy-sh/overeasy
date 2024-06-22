import os
from overeasy import *
from overeasy.models import GPTVision
from PIL import Image

workflow = Workflow([
    DenseCaptioningAgent(model=GPTVision(model="gpt-4o"))
])
image_path = os.path.join(os.path.dirname(__file__), "eiffel_photo.png")
image = Image.open(image_path)
result, graph = workflow.execute(image)
print(result[0].data)