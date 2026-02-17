# motor.py
from gpiozero import DigitalOutputDevice, PWMOutputDevice

class Motor:
    def __init__(self):
        self.PWMA = PWMOutputDevice(18)
        self.AIN1 = DigitalOutputDevice(22)
        self.AIN2 = DigitalOutputDevice(27)

        self.PWMB = PWMOutputDevice(23)
        self.BIN1 = DigitalOutputDevice(25)
        self.BIN2 = DigitalOutputDevice(24)

        self.speed = 0.5

    def set_speed(self, s):
        self.speed = s

    def go(self, speed=None):
        if speed is None: speed = self.speed
        self._forward_left(speed)
        self._forward_right(speed)

    def left(self, speed=None):
        if speed is None: speed = self.speed
        self._forward_left(speed * 0.4)
        self._forward_right(speed * 1.0)

    def right(self, speed=None):
        if speed is None: speed = self.speed
        self._forward_left(speed * 1.0)
        self._forward_right(speed * 0.4)

    def stop(self):
        self.PWMA.value = 0
        self.PWMB.value = 0
        self.AIN1.value = self.AIN2.value = 0
        self.BIN1.value = self.BIN2.value = 0

    def _forward_left(self, speed):
        self.AIN1.value = 0
        self.AIN2.value = 1
        self.PWMA.value = speed

    def _forward_right(self, speed):
        self.BIN1.value = 0
        self.BIN2.value = 1
        self.PWMB.value = speed
