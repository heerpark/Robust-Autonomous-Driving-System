# state_manager.py

import time
from config import *

class StateManager:
    def __init__(self, buzzer=None):
        self.follow_side = "right"
        self.straight_mode = False
        self.straight_frames = 0
        self.stop_until = 0
        self.current_speed = NORMAL_SPEED
        self.buzzer = buzzer

    def update(self, sign):
        now = time.time()

        if now < self.stop_until:
            return "STOP"

        if self.straight_mode:
            if self.straight_frames < STRAIGHT_HOLD:
                self.straight_frames += 1
                return "STRAIGHT"
            else:
                self.straight_mode = False

        # ---------- 표지판 처리 ----------
        if sign == "left":
            self.follow_side = "left"

        elif sign == "right":
            self.follow_side = "right"

        elif sign == "slow":
            self.current_speed = SLOW_SPEED

        elif sign == "stop":
            self.stop_until = now + STOP_TIME
            return "STOP"

        elif sign == "straight":
            self.straight_mode = True
            self.straight_frames = 0
            return "STRAIGHT"

        elif sign == "traffic_green":
            self.current_speed = NORMAL_SPEED

        elif sign == "traffic_red":
            self.stop_until = now + STOP_TIME
            return "STOP"

        elif sign == "trumpet":
            if self.buzzer:
                self.buzzer.beep()
            return "TRUMPET"

        return "FOLLOW"
