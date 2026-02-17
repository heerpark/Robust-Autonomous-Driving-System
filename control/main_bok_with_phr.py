import mycamera
import cv2
import numpy as np
import time
from collections import deque, Counter
from gpiozero import DigitalOutputDevice, PWMOutputDevice, TonalBuzzer
from ultralytics import YOLO


# ==================================================
# MODE TIMING SETTINGS (초 단위)
# ==================================================
STOP_DURATION = 3.0          # STOP 동안 정지
STOP_LOCK = 3.0              # STOP 후 잠금 (움직임 금지)

TRUMPET_DURATION = 3.0       # 트럼펫 울린 후 재울림 금지 시간

STRAIGHT_DELAY = 2.0         # STRAIGHT 표지판 보고 몇 초 후 강제 직진 시작?
STRAIGHT_DURATION = 10.0      # 강제 직진 유지 시간
STRAIGHT_LOCK = 0.0          # 강제 직진 끝난 후 잠금 시간

class ObjectFilter:
	def __init__(self, model_names, n=10, m=7, k=6, min_conf=0.5, min_size=100):
			"""
			model_names: 클래스 ID와 매핑되는 이름 리스트 (e.g., ['stop', 'left_arrow'])
			n: 전체 히스토리 버퍼 크기
			... (나머지 인자)
			"""
			self.history = deque(maxlen=n)
			self.m = m
			self.k = k
			self.min_conf = min_conf
			self.min_size = min_size
			# 2. model_names를 객체의 멤버 변수로 저장
			self.model_names = model_names
            
	def push_and_decide(self, results, image):
            detected_cls = None
            max_conf = 0.0
            
            # 1. 현재 프레임에서 가장 Confidence가 높고 크기가 일정 이상인 객체 1개만 추출
            if results and len(results[0].boxes) > 0:
                
                # 모든 박스를 순회하며 조건을 만족하는 가장 높은 Conf 객체 찾기
                best_box = None
                
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    
                    # Bounding Box 좌표 (xyxy 형식)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # box_area = (x2 - x1) * (y2 - y1) # 면적 계산
                    width = abs(x2 - x1)

                    # 신뢰도와 면적 조건 모두 만족
                    if conf > self.min_conf and width >= self.min_size:
                        # 현재까지 찾은 가장 높은 Conf보다 높으면 업데이트
                        if conf > max_conf:
                            max_conf = conf
                            best_box = box
                
                if best_box:
                    detected_cls = self.model_names[int(best_box.cls[0])]
                    print("detected_cls", detected_cls)
                    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,255), 2)
                    cv2.putText(image, detected_cls, (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)


            # 2. 히스토리에 저장 및 3. 최근 m개 데이터 슬라이싱 (기존 로직과 동일)
            self.history.append(detected_cls)

            recent_frames = list(self.history)[-self.m:]
            valid_frames = [cls for cls in recent_frames if cls is not None]
            
            if not valid_frames:
                return None

            # 4. 카운팅 및 5. K개 이상이면 확정 (기존 로직과 동일)
            count_result = Counter(valid_frames).most_common(1)
            
            if count_result:
                most_common_cls, count = count_result[0]
                if count >= self.k:
                    return most_common_cls
            
            return None


# ==================================================
# Buzzer
# ==================================================
class Buzzer:
    def __init__(self, pin=12):
        try:
            self.buzzer = TonalBuzzer(pin)
            self.working = True
        except:
            self.buzzer = None
            self.working = False

    def beep(self, freq=261, duration=0.2):
        if self.working:
            try:
                self.buzzer.play(freq)
                time.sleep(duration)
            finally:
                self.buzzer.stop()
        else:
            print("(Virtual Beep)")


my_buzzer = Buzzer(pin=12)
def beep_horn(duration=0.2):
    my_buzzer.beep(261, duration)


# ==================================================
# Motor
# ==================================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)


def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_left(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed * 0.3
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_right(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed * 0.3


def motor_stop():
    PWMA.value = 0
    PWMB.value = 0
    AIN1.value = AIN2.value = 0
    BIN1.value = BIN2.value = 0


# ==================================================
# Vision (lane)
# ==================================================
def img_preprocess(image):
    h, _, _ = image.shape
    roi = image[h//2:, :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask



def find_line(mask, side="right"):
    h, w = mask.shape
    row0 = int(h * 0.5)
    roi = mask[row0:, :]

    # 왼/오른쪽 절반 선택
    if side == "right":
        roi = roi[:, w//2:]
        offset_x = w//2
    else:
        roi = roi[:, :w//2]
        offset_x = 0

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 400:
        return None

    # 윤곽선에서 모든 점 추출
    contour_pts = c.reshape(-1, 2)

    # y가 가장 큰 지점만 추출
    max_y = np.max(contour_pts[:, 1])
    candidate_xs = contour_pts[contour_pts[:, 1] == max_y][:, 0]

    if len(candidate_xs) == 0:
        return None

    # 사이드별 엣지 판단
    if side == "right":
        edge_x = np.min(candidate_xs)
    else:
        edge_x = np.max(candidate_xs)

    cx = int(edge_x + offset_x)

    # y좌표 원본 이미지로 환산
    cy = int(h + (row0 + max_y))

    return cx, cy



# ==================================================
# Control Logic
# ==================================================
def control_logic(cx, width, follow_side, base_offset=130, dead_zone=20, speed=0.4):
    if cx is None:
        return "lost"

    offset = -base_offset if follow_side == "right" else base_offset
    target_cx = cx + offset
    error = target_cx - (width//2)

    if abs(error) < dead_zone:
        motor_go(speed)
        return "straight"

    if error > 0:
        motor_right(speed)
        return "right"

    motor_left(speed)
    return "left"


# ==================================================
# MAIN
# ==================================================
def main():
    print("Loading YOLO model...")
    model = YOLO("best_strong4.pt")
    camera = mycamera.MyPiCamera(640, 480)
    obj_filter = ObjectFilter(model_names=model.names)
    follow_side = "left"
    base_offset = 140
    run_speed = 0.5

    # -----------------------------
    # LOCK VARIABLES
    # -----------------------------
    stop_active_until = 0
    stop_lock_until = 0

    trumpet_lock_until = 0

    straight_delay_until = 0
    straight_active_until = 0
    straight_lock_until = 0

    frame_count = 0
    yolo_interval = 3
    MIN_WIDTH = 110

    try:
        while camera.isOpened():
            ret, image = camera.read()
            if not ret:
                break
            
            image = cv2.flip(image, -1)
            now = time.time()
            frame_count += 1
            h_img, w_img, _ = image.shape
            
            base_offset = 140 if follow_side == "right" else 160
            mode = "GO"
            yolo_detect = "None"

            # ======================================================
            # STOP ACTIVE
            # ======================================================
            if now < stop_active_until:
                mode = f"STOP {stop_active_until - now:.1f}s"
                motor_stop()
                draw_debug(image, None, None, follow_side, mode, yolo_detect, base_offset)
                show(image)
                continue

            # ======================================================
            # STOP LOCK (정지는 끝났지만 움직이면 안 됨)
            # ======================================================
            if stop_active_until <= now < stop_lock_until:
                mode = f"STOP_LOCK {stop_lock_until - now:.1f}s"
                motor_stop()
                draw_debug(image, None, None, follow_side, mode, yolo_detect, base_offset)
                show(image)
                continue

            # ======================================================
            # YOLO DETECT
            # ======================================================
            if frame_count % yolo_interval == 0:
                results = model(image, imgsz=320, conf=0.5, verbose=False)
                cls = obj_filter.push_and_decide(results, image)
                print("sequential detect !!!!!!", cls)

                # ---------------- STOP ------------------
                if cls in ["stop","traffic_red"]:
                    stop_active_until = now + STOP_DURATION
                    stop_lock_until = stop_active_until + STOP_LOCK

                # ---------------- SLOW ------------------
                elif cls in ["traffic_yellow","slow"]:
                    mode = "SLOW"

                # ---------------- TRUMPET ----------------
                elif cls == "trumpet":
                    if now > trumpet_lock_until:
                        beep_horn(0.2)
                        trumpet_lock_until = now + TRUMPET_DURATION

                # ---------------- LEFT / RIGHT ----------
                elif cls == "left":
                    follow_side = "left"
                elif cls == "right":
                    follow_side = "right"

                # ---------------- STRAIGHT ---------------
                elif cls == "straight":
                    straight_delay_until = now + STRAIGHT_DELAY
                    straight_active_until = straight_delay_until + STRAIGHT_DURATION
                    straight_lock_until = straight_active_until + STRAIGHT_LOCK

            # ======================================================
            # STRAIGHT MODES
            # ======================================================

            if now < straight_delay_until:
                mode = "GO"

            elif straight_delay_until <= now < straight_active_until:
                mode = f"FORCE_STRAIGHT {straight_active_until - now:.1f}s"
                motor_go(run_speed)
                draw_debug(image, None, None, follow_side, mode, yolo_detect, base_offset)
                show(image)
                continue

            elif straight_active_until <= now < straight_lock_until:
                mode = f"STRAIGHT_LOCK {straight_lock_until - now:.1f}s"
                motor_go(run_speed)
                draw_debug(image, None, None, follow_side, mode, yolo_detect, base_offset)
                show(image)
                continue

            # ======================================================
            # LANE FOLLOWING
            # ======================================================
            mask = img_preprocess(image)
            found = find_line(mask, follow_side)

            if found:
                cx, cy = found
                direction = control_logic(cx, w_img, follow_side,
                                          base_offset=base_offset,
                                          dead_zone=5,
                                          speed=run_speed)
            else:
                cx, cy = None, None
                direction = "lost"
                motor_go(0.1)

            mode = direction.upper()

            draw_debug(image, cx, cy, follow_side, mode, yolo_detect, base_offset)
            show(image)
            cv2.imshow("mask",mask)
            
            key = cv2.waitKey(1) & 0xFF

            if key == 81:     # ← 왼쪽 방향키
                follow_side = "left"

            elif key == 83:   # → 오른쪽 방향키
                follow_side = "right"



    finally:
        motor_stop()
        camera.release()
        cv2.destroyAllWindows()


# ==================================================
# Debug Visualization
# ==================================================
def draw_debug(image, cx, cy, follow_side, mode, yolo_detect, base_offset):
    h, w, _ = image.shape

    cv2.line(image, (w//2, 0), (w//2, h), (255,255,255), 2)
    cv2.putText(image, f"Side: {follow_side}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(image, f"Detect: {yolo_detect}", (10,70), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,200),2)
    cv2.putText(image, f"Mode: {mode}", (10,100), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3)

    if cx is not None and cy is not None:
        cv2.circle(image, (cx, cy), 6, (0,255,0), -1)
        cv2.putText(image, f"cx={cx}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        offset = -base_offset if follow_side == "right" else base_offset
        target = int(np.clip(cx + offset, 0, w-1))
        cv2.circle(image, (target, cy), 6, (255,0,0), -1)
        cv2.putText(image, f"target={target}", (target+10, cy+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)


def show(image):
    cv2.imshow("Frame", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
