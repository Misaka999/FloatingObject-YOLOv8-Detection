"""
Track objects in video frames using a combination of YOLO and SAHI models.
The detection results are processed and displayed on the video frames, which can be saved to an output file.
The module also allows for interaction during the video playback, such as pausing, fast forwarding, and rewinding.

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
"""

import os
import cv2
import csv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from multiprocessing import freeze_support
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv('.env')


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


class VideoTracker:
    def __init__(self,
                 yolo_model_path,
                 sahi_model_path,
                 video_path,
                 output_path,
                 device,
                 detection_interval=5,
                 sahi_conf_threshold=0.35,
                 yolo_conf_threshold=0.3
                 ):
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_conf_threshold = yolo_conf_threshold
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=sahi_model_path,
            confidence_threshold=sahi_conf_threshold,
            device=device
        )
        self.video_path = video_path
        self.output_path = output_path
        self.detection_interval = detection_interval
        self.colors = {
            "hard": (128, 0, 128),
            "soft": (0, 0, 255),
            "mark": (0, 165, 255),
            "ship": (255, 0, 0)
        }
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.combined_results = []
        self.all_results = []
        self.paused = False

    def process_frame(self, frame):
        if self.frame_count % self.detection_interval == 0:
            # Use SAHI to slice the original video (2560*1440) into four regions and detect
            # 用SAHI把原视频（2560*1440）切成四个区域并检测
            sahi_results = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=900,
                slice_width=1600,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            object_prediction_list = sahi_results.object_prediction_list
            sahi_boxes = []
            sahi_clss = []
            sahi_confs = []
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

            # Detect using yolov8 without SAHI  不用SAHI，单纯用yolov8检测
            yolo_results = self.yolo_model.track(frame, persist=True, conf=self.yolo_conf_threshold, iou=0.5)
            yolo_boxes = yolo_results[0].boxes.xyxy.tolist()
            yolo_classes = yolo_results[0].boxes.cls.tolist()
            yolo_names = yolo_results[0].names
            yolo_confs = yolo_results[0].boxes.conf.tolist()

            # Compare and merge results using IoU and confidence scores
            # 用iou和置信值比较并合并两种检测结果
            self.combined_results = []
            for s_box, s_cls, s_conf in zip(sahi_boxes, sahi_clss, sahi_confs):
                best_match_idx = -1
                best_match_iou = 0
                for idx, (y_box, y_cls, y_conf) in enumerate(zip(yolo_boxes, yolo_classes, yolo_confs)):
                    if yolo_names[int(y_cls)] == s_cls:
                        iou = compute_iou(s_box, y_box)
                        if iou > best_match_iou:
                            best_match_iou = iou
                            best_match_idx = idx

                if best_match_idx != -1 and best_match_iou > 0.5:
                    if yolo_confs[best_match_idx] > s_conf:
                        self.combined_results.append((yolo_boxes[best_match_idx],
                                                      yolo_names[int(yolo_classes[best_match_idx])],
                                                      yolo_confs[best_match_idx]))
                    else:
                        self.combined_results.append((s_box, s_cls, s_conf))
                else:
                    self.combined_results.append((s_box, s_cls, s_conf))

            for y_box, y_cls, y_conf in zip(yolo_boxes, yolo_classes, yolo_confs):
                if not any(compute_iou(y_box, s_box) > 0.5 for s_box in sahi_boxes):
                    self.combined_results.append((y_box, yolo_names[int(y_cls)], y_conf))

            # Save results for each detection interval 保存每一检测间隔的结果
            for box, cls, conf in self.combined_results:
                self.all_results.append((self.frame_count, box, cls, conf))

        # Display detection boxes and confidence scores on the frame
        # 在画面中展示检测框和置信值
        for box, cls, conf in self.combined_results:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(self.width, int(x2)), min(self.height, int(y2))
            color = self.colors.get(cls, (56, 56, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 2), 0, 0.6,
                        [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        return frame

    def track(self, video_save, start_time):
        start_frame = int(start_time * self.fps)
        if start_frame > self.total_frames:
            print("Error: Start time exceeds video length.")
            return None, None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while self.cap.isOpened():
            if not self.paused:
                success, frame = self.cap.read()
                if not success:   # Video over 视频结束
                    break

                frame = self.process_frame(frame)

                cv2.imshow("floating", frame)

                if video_save:
                    self.video_writer.write(frame)

                self.frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press "q" to quit  按“q”退出
                break
            elif key == ord("p"):   # Press "p" to pause  按“p”暂停
                self.paused = not self.paused
            elif key == ord("f"):    # Press "f" to fast forward 5 seconds  按“f”快进5秒
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame_position = min(current_frame + self.fps * 5, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)
            elif key == ord("b"):   # Press "b" to rewind 5 seconds  按“b"后退5秒
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_frame_position = max(current_frame - self.fps * 5, 0)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

        # Output detection boxes in csv format  以csv形式输出检测框
        results_path = self.output_path.replace('.mp4', '_results.csv')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x1', 'y1', 'x2', 'y2'])
            for result in self.all_results:
                frame_number, box, cls, conf = result
                x1, y1, x2, y2 = box
                writer.writerow((frame_number, [x1, y1, x2, y2], cls, conf))


if __name__ == '__main__':
    freeze_support()
    video_save = bool(int(os.getenv('VIDEO_SAVE')))
    detection_interval = int(os.getenv('DETECTION_INTERVAL'))
    start_time = float(os.getenv('START_TIME'))
    video_path = os.getenv('VIDEO_PATH')
    output_path = os.getenv('OUTPUT_PATH')
    yolo_model_path = os.getenv('YOLO_MODEL')
    sahi_model_path = os.getenv('SAHI_MODEL')
    sahi_conf_threshold = float(os.getenv('SAHI_CONF_THRESHOLD'))
    yolo_conf_threshold = float(os.getenv('YOLO_CONF_THRESHOLD'))
    device = os.getenv('DEVICE')

    tracker = VideoTracker(yolo_model_path, sahi_model_path, video_path, output_path, device,
                           detection_interval, sahi_conf_threshold, yolo_conf_threshold)
    tracker.track(video_save, start_time)
