import cv2

from app.core.config import settings
from app.infrastructure.repositories.detector.models import CascadeDetectorModel


class HaarcascadeDetector(CascadeDetectorModel):

    def __init__(self) -> None:
        super().__init__()
        self.__cascade_face = self.load(settings.HAARCASCADE_FACE)
        self.__cascade_smile = self.load(settings.HAARCASCADE_SMILE)
        self.__cascade_eye = self.load(settings.HAARCASCADE_EYE)

    def run(self, source_image):
        gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        face = self.__cascade_face.detectMultiScale(gray_image, 1.3, 5)
        smile = self.__cascade_smile.detectMultiScale(gray_image, 1.3, 20)
        eye = self.__cascade_eye.detectMultiScale(gray_image, 1.3, 1)

        for (x, y, w, h) in face:
            cv2.rectangle(source_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (xO, yO, wO, hO) in eye:
                if (x <= xO) and (y <= yO) and (x + w >= xO + wO) and (y + h >= yO + hO):
                    cv2.rectangle(source_image, (xO, yO), (xO + wO, yO + hO), (255, 255, 255), 2)
            for (xS, yS, wS, hS) in smile:
                if (x <= xS) and (y <= yS) and (x + w >= xS + wS) and (y + h >= yS + hS):
                    cv2.rectangle(source_image, (xS, yS), (xS + wS, yS + hS), (0, 255, 0), 2)
