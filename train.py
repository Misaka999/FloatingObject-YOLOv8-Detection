import os
from ultralytics import YOLO
from multiprocessing import freeze_support
from dotenv import load_dotenv


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv('.env')


def train_model():
    pre_trained_model = os.getenv('PRE_TRAINED_MODEL')
    model = YOLO(pre_trained_model)

    data_path = os.getenv('DATA_PATH')
    model.train(data=data_path, epochs=100, imgsz=640)


if __name__ == '__main__':
    freeze_support()
    train_model()
