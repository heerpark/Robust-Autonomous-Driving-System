import mycamera
import cv2
import numpy as np
from gpiozero import DigitalOutputDevice, PWMOutputDevice

## offeset이 너무 커서 방향 결정 오류가 나는 문제 수정 + 라인 미검출 시 안전 탐색 로직 추가
## 실제 주행 없이도 동작 확인 가능 (카메라 프리뷰, 콘솔 로그)

# ==================================================
# ✅ 모터 설정
# ==================================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

def motor_go(speed: float):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_left(speed,l,r):
    left_speed  = speed * l      # 천천히
    right_speed = speed * r      # 빠르게

    # 왼쪽 (AIN1=0, AIN2=1 이 전진이면 그대로)
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = left_speed

    # 오른쪽
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = right_speed
    
def motor_right(speed,l,r):
    left_speed  = speed * l
    right_speed = speed * r

    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = left_speed

    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = right_speed


def motor_stop():
    PWMA.value = 0.0
    PWMB.value = 0.0
    AIN1.value = 0
    AIN2.value = 0
    BIN1.value = 0
    BIN2.value = 0

# ==================================================
# ✅ 노란색 차선 검출
#   - 원본 하단 50%만 사용
#   - HSV에서 노란색 마스크
# ==================================================
def img_preprocess(image):
    height, _, _ = image.shape
    roi = image[int(height/2):, :, :]  # 하단 절반만
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

# ==================================================
# ✅ 라인 검출
# 검출 영역에서 최대한 끝점을 기준으로 기준점 선택
# ==================================================

def find_line(mask, side="right"):
    height, width = mask.shape
    row0 = int(height * 0.6)
    roi = mask[row0:height, :]

    # 절반 영역 선택
    if side == "right":
        roi = roi[:, width // 2 :]
        offset_x = width // 2
    else:
        roi = roi[:, : width // 2]
        offset_x = 0

    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 가장 큰 윤곽 선택
    c = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(roi)
    cv2.drawContours(contour_mask, [c], -1, 255, 1)

    ys, xs = np.where(contour_mask > 0)
    if len(xs) == 0:
        return None

    # ✅ 엣지 기반 기준점 선택
    if side == "left":
        edge_x = np.max(xs) + offset_x   # 오른쪽 가장자리
    else:
        edge_x = np.min(xs) + offset_x   # 왼쪽 가장자리

    edge_y = int(np.mean(ys)) + row0     # 대략적인 높이 (중심선 부근)
    area = cv2.contourArea(c)
    return (edge_x, edge_y, area)


# ==================================================
# ✅ 제어 로직
#   - side에 따라 base_offset 부호 반영
#   - error dead-zone으로 직진/좌/우 결정
# ==================================================
def control_logic(cx, width, last_known, side, base_offset=120, dead_zone=10, speed=0.4):
    if cx is None:
        cx = last_known

    offset = -base_offset if side == "right" else base_offset
    target_cx = cx + offset
    error = target_cx - (width // 2)

    if abs(error) < dead_zone:
        motor_go(speed)
        action = "Go straight"
    elif error > 0:
        motor_right(speed,1,0.4)
        action = "Turn Right"
    else:
        motor_left(speed,0.4,1)
        action = "Turn Left"

    return action, cx, offset, error

# ==================================================
# ✅ 메인 루프
#   - 라인 미검출 시 안전 탐색:
#       * 잠깐(예: 5프레임) 끊기면: 저속 직진 유지
#       * 그 이상 끊기면: follow_side 방향으로 천천히 드리프트하며 탐색
#   - 찾으면 즉시 정상 주행 복귀
# ==================================================
def main():
    print("🚗 Line Following (← or → to switch line side)")
    camera = mycamera.MyPiCamera(640, 480)

    # 기본 상태값
    last_known = 320                 # 과거 중심값(라인 유실 시 임시 사용)
    follow_side = "right"            # 시작은 오른쪽 라인 기준
    base_offset = 240                # 라인과의 목표 거리
    dead_zone = 10                   # error 허용 범위(px)
    run_speed = 0.5                 # 정상 주행 속도
    search_speed = 0.5           # 탐색 시 저속 드리프트 속도

    # 안전 탐색 파라미터
    lost_count = 0                   # 연속 미검출 프레임 수
    hold_frames = 1                  # 이 이하로는 직진 유지(일시 끊김 허용) 우선 0으로 설정
    hard_lost_stop = 250              # 너무 오래 못찾으면 정지(프레임 기준, 선택)

    try:
        while camera.isOpened():
            ret, image = camera.read()
            if not ret:
                break

            # 카메라 상하반전 쓰는 경우 유지
            image = cv2.flip(image, -1)

            mask = img_preprocess(image)
            found = find_line(mask, side=follow_side)

            h_img, w_img, _ = image.shape

            if found:
                cx, cy, area = found
                action, last_known, offset, error = control_logic(
                    cx, w_img, last_known, follow_side,
                    base_offset=base_offset, dead_zone=dead_zone, speed=run_speed
                )
                lost_count = 0  # 복구
                mode = f"FOLLOW {follow_side.upper()}"

                # 시각화
                # mask는 원본 하단 50% 기준이므로, 표시 시 y보정: 이미지 하단 절반 기준이 화면의 아래쪽에 위치
                cv2.circle(image, (cx, cy + h_img // 2), 5, (0, 255, 255), -1)
                cv2.line(image,
                         (cx + offset, cy + h_img // 2 - 10),
                         (cx + offset, cy + h_img // 2 + 10),
                         (255, 0, 0), 2)

            else:
                # 라인 미검출
                lost_count += 1
                cx, cy, offset, error = None, None, 0, 0

                if lost_count <= hold_frames:
                    # 잠깐 끊긴 경우: 저속 직진
                    motor_go(0.3)
                    action = f"Line lost short — keep straight"
                    mode = f"STABILIZE ({follow_side})"
                else:
                    # 지속 미검출: follow_side 방향으로 천천히 탐색 드리프트
                    if follow_side == "left":
                        motor_left(search_speed,0,1)
                        action = f"Searching LEFT line..."
                    else:
                        motor_right(search_speed,1,0)
                        action = f"Searching RIGHT line..."
                    mode = f"RECOVERY ({follow_side})"

                    # 너무 오래 못 찾으면 완전 정지(선택)
                    if lost_count >= hard_lost_stop:
                        motor_stop()
                        action = "Hard lost — STOP"
                        mode = "EMERGENCY STOP"

            # 정보 표시
            cv2.putText(image, f"Mode: {mode}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(image, f"Action: {action}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(image, f"Side: {follow_side} | Lost: {lost_count} | Offset: {base_offset} | DZ: {dead_zone}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

            cv2.imshow("Frame", image)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 81:  # ← 왼쪽 화살표
                follow_side = "left"
                lost_count = 0  # 전환 시 카운트 리셋
                print("↩️ Switched to LEFT line (will recover by drifting LEFT if lost)")
            elif key == 83:  # → 오른쪽 화살표
                follow_side = "right"
                lost_count = 0
                print("↪️ Switched to RIGHT line (will recover by drifting RIGHT if lost)")

    except KeyboardInterrupt:
        print("🛑 Interrupted manually.")
    finally:
        motor_stop()
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Motors stopped and camera released safely.")

if __name__ == "__main__":
    main()
