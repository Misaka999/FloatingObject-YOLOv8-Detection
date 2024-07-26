from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    detect_config = {
        "yolo_det_model_path": os.getenv('YOLO_MODEL'),
        "yolo_conf_threshold": float(os.getenv('YOLO_CONF_THRESHOLD', 0.35)),
        "sahi_model_path": os.getenv('SAHI_MODEL'),
        "sahi_conf_threshold": float(os.getenv('SAHI_CONF_THRESHOLD', 0.4)),
        "device": os.getenv('DEVICE'),
        "detection_interval": int(os.getenv('DETECTION_INTERVAL')),
        "video_path": os.getenv('VIDEO_PATH'),
        "start_time": float(os.getenv('START_TIME'))
    }

    classify_config = {
        "yolo_cls_model_path": os.getenv('YOLO_CLS_MODEL'),
        "video_save": bool(int(os.getenv('VIDEO_SAVE'))),
        "output_path": os.getenv('OUTPUT_PATH'),
        "cool_seconds": int(os.getenv('COOL_SECONDS')),
        "detection_interval": int(os.getenv('DETECTION_INTERVAL')),
        "video_path": os.getenv('VIDEO_PATH'),
        "start_time": float(os.getenv('START_TIME'))
    }
