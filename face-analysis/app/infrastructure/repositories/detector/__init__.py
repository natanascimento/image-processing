import cv2
import keras
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import numpy as np

from app.core.config import settings
from app.infrastructure.repositories.detector.models import CascadeDetectorModel, RecognitionModel


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

        cv2.imshow("Haarcascade Detector", source_image)

    @staticmethod
    def stop():
        cv2.destroyAllWindows()

class SentimentDetector(CascadeDetectorModel, RecognitionModel):

    def __init__(self) -> None:
        super().__init__()
        self.__cascade_face = self.load(settings.HAARCASCADE_FACE)
        self.__recognition_model = self.load_model(settings.RECOGNITION_MODEL)

    def run(self, source_image):
        gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        faces = self.__cascade_face.detectMultiScale(gray_image, 1.32, 5)

        for x, y, w, h in faces:
            print(f"w:{w}")
            print(f"h:{h}")
            cv2.rectangle(source_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_image[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            predictions = self.__recognition_model.predict(img_pixels)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(source_image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sentiment Detector", source_image)
        
    @staticmethod
    def stop():
        cv2.destroyAllWindows()