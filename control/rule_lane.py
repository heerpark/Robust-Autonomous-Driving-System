# rule_lane.py
import cv2
import numpy as np

def preprocess(image):
    h, _, _ = image.shape
    roi = image[int(h/2):, :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 80, 80])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, 1)
    mask = cv2.dilate(mask, None, 2)
    return mask

def find_line(mask, side="right"):
    h, w = mask.shape
    row0 = int(h * 0.6)

    roi = mask[row0:h]
    if side == "right":
        roi = roi[:, w//2:]
        offset_x = w//2
    else:
        roi = roi[:, :w//2]
        offset_x = 0

    contours, _ = cv2.findContours(
        roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    cm = np.zeros_like(roi)
    cv2.drawContours(cm, [c], -1, 255, 1)

    ys, xs = np.where(cm > 0)
    if len(xs) == 0:
        return None

    edge_x = (np.min(xs) if side=="right" else np.max(xs)) + offset_x
    edge_y = int(np.mean(ys)) + row0
    area = cv2.contourArea(c)

    return edge_x, edge_y, area

def control(cx, width, last_cx, side, base_offset=240, dead_zone=10):
    if cx is None:
        cx = last_cx

    offset = -base_offset if side=="right" else base_offset
    target = cx + offset
    error = target - (width//2)

    if abs(error) < dead_zone:
        return "straight", cx
    elif error > 0:
        return "right", cx
    else:
        return "left", cx
