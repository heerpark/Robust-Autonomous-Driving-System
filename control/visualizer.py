# visualizer.py
import cv2

def draw(frame, lane, dets, sign, mode):
    h, w, _ = frame.shape

    if lane is not None:
        cx, cy, _ = lane
        cv2.circle(frame, (cx, cy + h//2), 5, (0,255,255), -1)

    for name, conf, area, (x1,y1,x2,y2) in dets:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(frame, f"Mode: {mode}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    if sign:
        cv2.putText(frame, f"Sign: {sign}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    return frame
