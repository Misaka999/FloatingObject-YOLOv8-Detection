# FloatingObject-YOLOv8-Detection
##  项目目录
```angular2html
项目目录/
│  
│  train_detection.py
│  train_classification.py
│  yolov8n.pt
│  yolov8n-cls.pt
│  main.py
│  detection.py
│  classification.py
│  config.py
│  requirements.txt
│  readme.md
│  
├─detect_data/
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
│
├─cls_data/
│   │
│   ├─test/
│   │  │
│   │  ├─garbage/
│   │  └─water/
│   │
│   ├─train/
│   │  │
│   │  ├─garbage/
│   │  └─water/
│   │
│   ├─val/
│   │   │
│   │   ├─garbage/
│   │   └─water/
