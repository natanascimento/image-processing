import time
from datetime import datetime

from imutils import resize
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
from tqdm import tqdm

from app.core.config import settings


class CameraCapture: 
  
    def __init__(self) -> None:
        self.__cascade_face = self.__load_model(settings.HAARCASCADE_FACE)
        self.__cascade_smile = self.__load_model(settings.HAARCASCADE_SMILE)
        self.__cascade_eye = self.__load_model(settings.HAARCASCADE_EYE)
  
    @staticmethod
    def __load_model(path:str):
      return cv2.CascadeClassifier(path)
  
    @staticmethod
    def __loading():
        for i in tqdm(range(int(100)), ncols=100):
            time.sleep(0.02)


    def run(self):
        print("[INFO] Reading camera image")
        video = VideoStream(src=0).start()
        self.__loading()
        print("[INFO] Starting FPS Counter")
        fps = FPS().start()
        
        while True:
            frame = video.read()
            image = resize(frame, width=720)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Image Captured", image)
            cv2.imshow("Image Captured", gray_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Detector was been paused!")
                break
                    
        fps.stop()
        cv2.destroyAllWindows()

        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))