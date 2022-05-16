import time

from imutils import resize
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
from tqdm import tqdm

from app.infrastructure.repositories.detector import HaarcascadeDetector


class CameraCapture:

    @staticmethod
    def __loading():
        for i in tqdm(range(int(100)), ncols=100):
            time.sleep(0.02)

    @staticmethod
    def run():
        print("[INFO] Reading camera image")
        video = VideoStream(src=0).start()
        CameraCapture.__loading()
        print("[INFO] Starting FPS Counter")
        fps = FPS().start()

        while True:
            frame = video.read()
            image = resize(frame, width=720)

            HaarcascadeDetector().run(image)

            cv2.imshow("Image Captured", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Detector was been paused!")
                break

        fps.stop()
        cv2.destroyAllWindows()

        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
