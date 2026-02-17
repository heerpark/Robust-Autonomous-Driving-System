# yolo_worker.py

import threading
import time
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, MIN_AREA

class YoloWorker(threading.Thread):
    def __init__(self, shared, lock, interval=0.15):
        super().__init__(daemon=True)
        self.shared = shared
        self.lock = lock
        self.interval = interval
        self.model = YOLO(YOLO_MODEL_PATH)
        self.running = True

    def run(self):
        while self.running:
            frame = None
            with self.lock:
                if self.shared["frame"] is not None:
                    frame = self.shared["frame"].copy()

            detections = []
            biggest = None
            max_area = 0

            if frame is not None:
                results = self.model(frame, verbose=False)[0]

                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls)
                    name = self.model.names[cls]
                    conf = float(box.conf)
                    area = (x2 - x1) * (y2 - y1)

                    detections.append(
                        (name, conf, area, (int(x1),int(y1),int(x2),int(y2)))
                    )

                    if area > max_area:
                        max_area = area
                        biggest = name

            with self.lock:
                self.shared["detections"] = detections
                self.shared["sign"] = biggest if max_area >= MIN_AREA else None

            time.sleep(self.interval)

    def stop(self):
        self.running = False
