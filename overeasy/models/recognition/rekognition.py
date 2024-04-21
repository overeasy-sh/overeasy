import boto3

class RekognitionModel:
    def __init__(self):
        self.client = boto3.client('rekognition')

    def detect_faces(self, image_path):
        with open(image_path, 'rb') as image:
            response = self.client.detect_faces(
                Image={'Bytes': image.read()},
                Attributes=['ALL']
            )
        return response['FaceDetails']

# Example usage
if __name__ == "__main__":
    rekognition_model = RekognitionModel()
    image_path = 'path/to/your/image.jpg'
    faces = rekognition_model.detect_faces(image_path)
    print(f"Detected {len(faces)} faces in the image.")