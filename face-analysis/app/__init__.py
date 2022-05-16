from app.camera.capture import CameraCapture
from app.core.config import settings


def main():
  print(settings.HAARCASCADE_EYE)
  print(settings.HAARCASCADE_FACE)
  print(settings.HAARCASCADE_SMILE)
  #CameraCapture().run()