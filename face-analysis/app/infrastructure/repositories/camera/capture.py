import time

from imutils import resize
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
from tqdm import tqdm

from app.infrastructure.repositories.detector import HaarcascadeDetector


class CameraCapture:

    def __init__(self) -> None:
        self.__detector = HaarcascadeDetector()

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

            self.__detector.run(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Detector was been paused!")
                break

        fps.stop()
        self.__detector.stop()

        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
