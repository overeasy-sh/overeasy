import os
from overeasy import *
from overeasy.models import GPTVision
from PIL import Image
from pydantic import BaseModel

class License(BaseModel):
    license_number: str
    expiration_date: str
    state: str
    data_of_birth: str
    address: str
    first_name: str
    last_name: str

workflow = Workflow([
    InstructorImageAgent(response_model=License, model=GPTVision())
])

image_path = os.path.join(os.path.dirname(__file__), "drivers_license.jpg")
image = Image.open(image_path)
result, graph = workflow.execute(image)
print(repr(result[0].data))