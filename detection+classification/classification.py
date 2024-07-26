import cv2
from collections import defaultdict
from ultralytics import YOLO


class Classify:
    def __init__(self, config, queue, stop_event, data_event):
        self.queue = queue
        self.cls_model_path = config['yolo_cls_model_path']
        self.video_save = config['video_save']
        self.video_path = config['video_path']
        self.output_path = config['output_path']
        self.detection_interval = config['detection_interval']
        self.colors = {
            "hard": (128, 0, 128),
            "soft": (0, 0, 255),
            "mark": (0, 165, 255),
            "ship": (255, 0, 0)
        }
        self.classifier = YOLO(self.cls_model_path)
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        self.tracked_objects = defaultdict(lambda: -1)
        self.cool_seconds = config['cool_seconds']
        self.start_time = config['start_time']
        self.new_results = []
        self.frame_count = 0
        self.stop_event = stop_event
        self.data_event = data_event

    def run(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(self.start_time * self.fps)
        if start_frame > total_frames:
            print("Error: Start time exceeds video length.")
            return None, None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+5)

        self.frame_count = 0
        self.new_results = []

        # Wait until there are enough data in the queue
        self.data_event.wait()

        while self.cap.isOpened() and not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                break

            if self.frame_count % self.detection_interval == 0:
                self.process_frame(frame)

            frame = self.display(frame)
            cv2.imshow("floating", frame)

            if self.video_save:
                self.video_writer.write(frame)

            self.frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop_event.set()
                break

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

    def classify_object(self, all_objects, frame, curr_frame):
        for ob in all_objects:
            box, cls, conf, iid = ob
            x1, y1, x2, y2 = box

            # Skip objects that have been recently tracked
            # 跳过最近已经跟踪过的物体
            cool_frames = int(self.cool_seconds * self.fps)
            if (self.tracked_objects[iid] != -1 and
                    self.frame_count - self.tracked_objects[iid] < cool_frames):
                continue

            # Classify the detection box area to confirm if it is water surface
            # 对检测框区域进行分类，确认是否是水面
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            classification_results = self.classifier(cropped_image)
            names_dict = classification_results[0].names
            name_index = classification_results[0].probs.top1

            if cls == "ship":
                if classification_results[0].names and names_dict[name_index] == "water":
                    print(f"Frame {curr_frame}: Detected water surface, discarding detection of {cls}")
                elif classification_results[0].names and names_dict[name_index] == "garbage":
                    print(f"Frame {curr_frame}: Mismatching, discarding detection of {cls}")
                else:
                    print(f"Frame {curr_frame}: Detected {cls} with confidence {conf}")
                    self.new_results.append((box, cls, conf))
                    self.tracked_objects[iid] = self.frame_count
                    break

            elif cls == "soft" or cls == "hard":
                if classification_results[0].names and names_dict[name_index] == "water":
                    print(f"Frame {curr_frame}: Detected water surface, discarding detection of {cls}")
                elif classification_results[0].names and names_dict[name_index] == "ship":
                    print(f"Frame {curr_frame}: Mismatching, discarding detection of {cls}")
                else:
                    print(f"Frame {curr_frame}: Detected {cls} with confidence {conf}")
                    self.new_results.append((box, cls, conf))
                    self.tracked_objects[iid] = self.frame_count
                    break

    def process_frame(self, frame):
        self.new_results = []
        value = self.queue.get(True)

        curr_frame, all_objects = value
        print(f"Get {curr_frame} from queue.")
        print("value:", value)

        if self.frame_count == curr_frame:
            self.classify_object(all_objects, frame, curr_frame)

    def display(self, frame):
        # Display detection boxes and confidence scores on the frame
        # 在画面中展示检测框和置信值
        for box, cls, conf in self.new_results:
            x1, y1, x2, y2 = box
            color = self.colors.get(cls, (56, 56, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1,
                        lineType=cv2.LINE_AA)

        return frame
