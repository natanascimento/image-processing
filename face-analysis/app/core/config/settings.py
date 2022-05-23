from os.path import abspath, dirname, join


class Settings: 

    ROOT_PATH = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
    
    PROJECT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
    
    MODELS_PATH = join(ROOT_PATH, "models")
    
    HAARCASCADE_EYE = join(MODELS_PATH, "haarcascade_eye.xml")
    HAARCASCADE_FACE = join(MODELS_PATH, "haarcascade_frontalface_default.xml")
    HAARCASCADE_SMILE = join(MODELS_PATH, "haarcascade_smile.xml")

    RECOGNITION_MODEL = join(MODELS_PATH, "recognition_model.h5")
