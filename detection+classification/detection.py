import uuid
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


class Detect:
    def __init__(self, config, queue, stop_event, data_event):
        self.yolo_det_model = YOLO(config['yolo_det_model_path'])
        self.yolo_conf_threshold = config['yolo_conf_threshold']
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=config['sahi_model_path'],
            confidence_threshold=config['sahi_conf_threshold'],
            device=config['device']
        )
        self.video_path = config['video_path']
        self.detection_interval = config['detection_interval']
        self.start_time = config['start_time']
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.combined_results = []
        self.queue = queue
        self.stop_event = stop_event
        self.data_event = data_event

    def get_sahi_predictions(self, frame):
        # Use SAHI to slice the original video (2560*1440) into four regions and detect
        # 用SAHI把原视频（2560*1440）切成四个区域并检测
        sahi_results = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=int(self.height/1.6),
            slice_width=int(self.width/1.6),
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        object_prediction_list = sahi_results.object_prediction_list
        sahi_boxes = []
        sahi_clss = []
        sahi_confs = []
        sahi_ids = []
        for ind, _ in enumerate(object_prediction_list):
            boxes2 = (
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
            )
            clss2 = object_prediction_list[ind].category.name
            conf2 = object_prediction_list[ind].score.value
            sahi_boxes.append(boxes2)
            sahi_clss.append(clss2)
            sahi_confs.append(conf2)
            sahi_ids.append(str(uuid.uuid4()))

        return sahi_boxes, sahi_clss, sahi_confs, sahi_ids

    def get_yolo_predictions(self, frame):
        # Detect using yolov8 without SAHI  不用SAHI，单纯用yolov8检测
        yolo_results = self.yolo_det_model.track(frame, persist=True, conf=self.yolo_conf_threshold, iou=0.5)
        yolo_boxes = yolo_results[0].boxes.xyxy.tolist()
        yolo_classes = yolo_results[0].boxes.cls.tolist()
        yolo_names = yolo_results[0].names
        yolo_confs = yolo_results[0].boxes.conf.tolist()
        if yolo_results[0].boxes.id is not None:
            yolo_ids = yolo_results[0].boxes.id.tolist()
        else:
            yolo_ids = [str(uuid.uuid4())for _ in yolo_boxes]

        return yolo_boxes, yolo_classes, yolo_names, yolo_confs, yolo_ids

    def merge_predictions(self, sahi_preds, yolo_preds):
        sahi_boxes, sahi_clss, sahi_confs, sahi_ids = sahi_preds
        yolo_boxes, yolo_classes, yolo_names, yolo_confs, yolo_ids = yolo_preds
        # Compare and merge results using IoU and confidence scores
        # 用iou和置信值比较并合并两种检测结果
        self.combined_results = []
        for s_box, s_cls, s_conf, s_id in zip(sahi_boxes, sahi_clss, sahi_confs, sahi_ids):
            best_match_idx = -1
            best_match_iou = 0
            for idx, (y_box, y_cls, y_conf, y_id) in enumerate(zip(yolo_boxes, yolo_classes, yolo_confs, yolo_ids)):
                if yolo_names[int(y_cls)] == s_cls:
                    iou = compute_iou(s_box, y_box)
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = idx

            if best_match_idx != -1 and best_match_iou > 0.5:
                if yolo_confs[best_match_idx] > s_conf:
                    self.combined_results.append((yolo_boxes[best_match_idx],
                                                  yolo_names[int(yolo_classes[best_match_idx])],
                                                  yolo_confs[best_match_idx],
                                                  yolo_ids[best_match_idx]))
                else:
                    self.combined_results.append((s_box, s_cls, s_conf, yolo_ids[best_match_idx]))
            else:
                self.combined_results.append((s_box, s_cls, s_conf, s_id))

        for y_box, y_cls, y_conf, y_id in zip(yolo_boxes, yolo_classes, yolo_confs, yolo_ids):
            if not any(compute_iou(y_box, s_box) > 0.5 for s_box in sahi_boxes):
                self.combined_results.append((y_box, yolo_names[int(y_cls)], y_conf, y_id))

    def process_frame(self, frame):
        self.combined_results = []
        sahi_preds = self.get_sahi_predictions(frame)
        yolo_preds = self.get_yolo_predictions(frame)
        self.merge_predictions(sahi_preds, yolo_preds)

        def custom_sort_key(result):
            _, cls, conf, _ = result
            # Define priority: "soft" and "hard" come first
            # 定义优先级：soft和hard在前
            priority = {"soft": 1, "hard": 1}
            # Get priority, default value 0 (other categories: ship, buoy)
            # 获取优先级，默认值为0（表示其他类别: 船，浮标）
            cls_priority = priority.get(cls, 0)
            return cls_priority, conf

        # Sort according to custom sorting rules  按自定义排序规则排序
        self.combined_results.sort(key=custom_sort_key, reverse=True)

        # Save results for each detection interval 保存每一检测间隔的结果
        if self.queue:
            self.queue.put((self.frame_count, self.combined_results))
            # Set event when there are 20 data in queue 当队列中有20条数据时设置事件 
            if self.queue.qsize() >= 20: 
                self.data_event.set()

    def track(self):
        start_frame = int(self.start_time * self.fps)
        if start_frame > self.total_frames:
            print("Error: Start time exceeds video length.")
            return None, None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while self.cap.isOpened() and not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:   # Video over 视频结束
                break
            if self.frame_count % self.detection_interval == 0:
                self.process_frame(frame)
            self.frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()
