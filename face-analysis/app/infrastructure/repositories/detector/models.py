from cv2 import CascadeClassifier
from keras.models import load_model


class CascadeDetectorModel:

    @staticmethod
    def load(path: str) -> CascadeClassifier:
        return CascadeClassifier(path)

class RecognitionModel:

    @staticmethod
    def load_model(path: str):
        return load_model(path)