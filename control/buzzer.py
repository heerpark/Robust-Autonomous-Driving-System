# buzzer.py
from gpiozero import TonalBuzzer
import time

class Buzzer:
    def __init__(self, pin=12):
        self.buzzer = TonalBuzzer(pin)

    def beep(self, freq=261, duration=1.0):
        """도(C4:261Hz) 기본"""
        try:
            self.buzzer.play(freq)
            time.sleep(duration)
        finally:
            self.buzzer.stop()
