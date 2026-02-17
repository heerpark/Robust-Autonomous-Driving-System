# main.py

import cv2
import threading
import mycamera

from motor import Motor
from buzzer import Buzzer
from rule_lane import preprocess, find_line, control
from yolo_worker import YoloWorker
from state_manager import StateManager
from visualizer import draw

def main():
    cam = mycamera.MyPiCamera(640,480)
    motor = Motor()
    buzzer = Buzzer(12)
    sm = StateManager(buzzer=buzzer)

    shared = {"frame": None, "sign": None, "detections": []}
    lock = threading.Lock()

    yolo = YoloWorker(shared, lock)
    yolo.start()

    last_cx = 320

    try:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, -1)
            h, w, _ = frame.shape

            with lock:
                shared["frame"] = frame.copy()
                sign = shared["sign"]
                detections = shared["detections"]

            mode = sm.update(sign)

            mask = preprocess(frame)
            lane = find_line(mask, sm.follow_side)

            if mode == "STOP":
                motor.stop()
            elif mode == "STRAIGHT":
                motor.go(sm.current_speed)
            elif mode == "TRUMPET":
                motor.go(sm.current_speed)
            else:
                if lane:
                    cx, cy, area = lane
                    act, last_cx = control(cx, w, last_cx, sm.follow_side)
                    if act == "straight": motor.go(sm.current_speed)
                    elif act == "left": motor.left(sm.current_speed)
                    else: motor.right(sm.current_speed)
                else:
                    motor.go(0.3)

            vis = draw(frame, lane, detections, sign, mode)
            cv2.imshow("Frame", vis)
            cv2.imshow("Mask", preprocess(frame))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        motor.stop()
        yolo.stop()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
