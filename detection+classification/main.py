"""
Combines the detection and classification of objects in video frames, using YOLOv8. 
Detection process uses a combination of YOLO and SAHI models, results are written to a queue for further classification. 
Classification process filters the results from detection, increase precision rate.

Environment Variables:
- VIDEO_SAVE: Whether to save the video with annotations (0 or 1).
- DETECTION_INTERVAL: Number of frames between detections.
- START_TIME: Start detection from the specified second in the video.
- VIDEO_PATH: Path to the input video file.
- OUTPUT_PATH: Path to the output video file.
- YOLO_MODEL: Path to the YOLO model file.
- SAHI_MODEL: Path to the SAHI model file.
- SAHI_CONF_THRESHOLD: Confidence threshold for SAHI model.
- YOLO_CONF_THRESHOLD: Confidence threshold for YOLO model.
- DEVICE: Device to run the models on ('cpu' or 'cuda:0').
- COOL_SECONDS: Cool down period in seconds for re-detection of the same object.
"""


import os
from multiprocessing import freeze_support, Process, Queue, Event
from dotenv import load_dotenv
from detection import Detect
from classification import Classify
from config import Config


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv('.env')

config = Config()


def write_to_q(config, q, stop_event, data_event):
    # Call detector. Write to queue.
    print("Process to write: %s" % os.getpid())

    detector = Detect(config.detect_config, q, stop_event, data_event)
    detector.track()


def read_from_q(config, q, stop_event, data_event):
    # Call classifier. Read from queue.
    print("Process to read: %s" % os.getpid())

    classifier = Classify(config.classify_config, q, stop_event, data_event)
    classifier.run()


if __name__ == '__main__':
    freeze_support()
    q = Queue()
    stop_event = Event()
    data_event = Event()

    p_detect = Process(target=write_to_q, args=(config, q, stop_event, data_event))
    p_cls = Process(target=read_from_q, args=(config, q, stop_event, data_event))

    p_detect.start()
    p_cls.start()
    p_detect.join()
    q.put(None)
    p_cls.join()
    print("All processes terminated.")
