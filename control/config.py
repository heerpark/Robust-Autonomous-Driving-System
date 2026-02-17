# config.py

YOLO_MODEL_PATH = "best.pt"

# YOLO 박스가 이 면적 이상이면 실제 표지판으로 취급
MIN_AREA = 8000

# 속도 세팅
NORMAL_SPEED = 0.5
SLOW_SPEED = 0.3

# straight 표지판 유지 프레임
STRAIGHT_HOLD = 120

# stop/red 정지 시간 (초)
STOP_TIME = 5