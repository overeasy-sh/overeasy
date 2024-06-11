import boto3
from PIL import Image
from overeasy.types import OCRModel
import io

class Textract(OCRModel):
    def __init__(self):
        self.client = boto3.client('textract')

    def parse_text(self, image: Image.Image):
        """
        Analyzes a document image using Amazon Textract and returns the detected text.

        :param image: A PIL Image object of the document to analyze.
        :return: The detected text as a string.
        """
        # Convert the PIL Image to bytes
        buffer : io.BytesIO = io.BytesIO()
        image.save(buffer, format=image.format)
        image_bytes : bytes = buffer.getvalue()

        response = self.client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=["FORMS", "TABLES"]  # You can customize this based on your needs
        )

        # Extracting text from blocks
        text = []
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text.append(item['Text'])

        return ' '.join(text)