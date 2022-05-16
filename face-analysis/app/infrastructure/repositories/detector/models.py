from cv2 import CascadeClassifier


class CascadeDetectorModel:

    @staticmethod
    def load(path: str) -> CascadeClassifier:
        return CascadeClassifier(path)
