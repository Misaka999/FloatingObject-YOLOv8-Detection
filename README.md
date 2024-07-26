## Introduction
Video processing system to detect and track floating objects in river using YOLOv8

Two sub-projects:
### 1. detection/
This project focuses on detecting and tracking objects in video frames using a combination of the YOLO (You Only Look Once) 
model and the SAHI (Slicing Aided Hyper Inference) approach. 
####  structure
```angular2html
│  train_detection.py
│  yolov8n.pt
│  requirements.txt
│  readme.md
│
├─data/
│   │  data.yaml
│   │
│   ├─test/
│   │  │
│   │  ├─images/
│   │  └─labels/
│   │
│   ├─train/
│   │  │
│   │  ├─images/
│   │  └─labels/
│   │
│   ├─valid/
│   │   │
│   │   ├─images/
│   │   └─labels/

```

### 2. detection+classification/
Extends the functionality of object detection by incorporating object classification. 

Utilizes the YOLOv8 model and the SAHI approach for object detection and tracking, and an additional YOLO classification model to classify the detected objects. The results are then saved as an annotated video and in a CSV file.

For each detected object, cropped the corresponding region from the frame and passed it to the classifier. The classifier then determined the object’s class. If the object was classified as water surface, it was discarded. Otherwise, it was retained and its details (bounding box, class, confidence) were added to the results;

Leveraged Python’s multiprocessing capabilities to run detection and classification processes in parallel. The detection process continuously fed detected objects into a queue, while the classification process consumed these objects, classified them, and updated the results.

Implemented real-time visualization of detected objects. The system displayed bounding boxes and confidence scores for each object, with different colors representing different classes.

####  structure
```angular2html
│  train_detection.py
│  train_classification.py
│  classification.py
│  detection.py
│  main.py
│  config.py
│  yolov8n.pt
│  yolov8n-cls.pt
│  requirements.txt
│  readme.md
│
├─detect_data/
│   │  data.yaml
│   │
│   ├─test/
│   │   │
│   │   ├─images/
│   │   └─labels/
│   │
│   ├─train/
│   │   │
│   │   ├─images/
│   │   └─labels/
│   │
│   ├─valid/
│   │   │
│   │   ├─images/
│   │   └─labels/
│
├─cls_data/
│   │
│   ├─test/
│   │   │
│   │   ├─garbage/
│   │   ├─ship/
│   │   └─water/
│   │
│   ├─train/
│   │   │
│   │   ├─garbage/
│   │   ├─ship/
│   │   └─water/
│   │
│   ├─val/
│   │   │
│   │   ├─garbage/
│   │   ├─ship/
│   │   └─water/

```
