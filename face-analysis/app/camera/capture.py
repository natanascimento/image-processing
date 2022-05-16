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

            face = self.__cascade_face.detectMultiScale(gray_image, 1.3, 5)
            smile = self.__cascade_smile.detectMultiScale(gray_image, 1.3, 20)
            eye = self.__cascade_eye.detectMultiScale(gray_image, 1.3, 1)

            for (x,y,w,h) in face:
                cv2.rectangle(image,(x,y),(x + w,y + h),(255,0,0),2)
                for (xO,yO,wO,hO) in eye:
                    if( (x <= xO) and (y <= yO) and ( x + w >= xO + wO) and ( y + h >= yO + hO)):
                        cv2.rectangle(image, (xO,yO),(xO + wO,yO + hO),(255,255,255),2)
                for (xS,yS,wS,hS) in smile:
                    if( (x <= xS) and (y <= yS) and ( x + w >= xS + wS) and ( y + h >= yS + hS)):
                        cv2.rectangle(image, (xS, yS),(xS + wS, yS + hS),(0,255,0),2)            

            cv2.imshow("Image Captured", image)
            cv2.imshow("Image Captured Gray", gray_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Detector was been paused!")
                break
                    
        fps.stop()
        cv2.destroyAllWindows()

        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))