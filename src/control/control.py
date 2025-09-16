#!/usr/bin/env python3
#机器狗基本控制
import socket
import struct
import time
import threading


class HeartBeat:  # 心跳包
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ('192.168.1.120', 43893)
        self.heart_task = threading.Thread(target=self.send_heartbeat)
        self.heart_task.start()

    def send_heartbeat(self):
        while True:
            data = struct.pack("<3i", 0x21040001, 0, 0)
            self.udp_socket.sendto(data, self.send_addr)
            time.sleep(0.5)  # sending command frequency not lower than 2HZ


def time_display() -> str:
    time_hour = str(time.localtime().tm_hour)
    time_min = str(time.localtime().tm_min)
    time_sec = str(time.localtime().tm_sec)
    return time_hour + ":" + time_min + ":" + time_sec


class RobotController:
    def __init__(self):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ('192.168.1.120', 43893)

    def send_data(self, code, value, ctype):
        data = struct.pack("<3i", code, value, ctype)
        self.udp_socket.sendto(data, self.send_addr)

    def stand_up(self):  # 起立
        self.send_data(0x21010202, 0, 0)
        time.sleep(3)

    def move_left(self, duration, value: int = -20000):  # 左平移,ps:移动指令给两个参数，duration限定运动时间，value限定距离
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010131, value, 0)
            time.sleep(0.1)

    def move_right(self, duration, value: int = 20000):  # 右平移
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010131, value, 0)
            time.sleep(0.1)

    def forward(self, duration, value: int = 13000):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010130, value, 0)
            time.sleep(0.1)

    def back(self, duration, value: int = -13000):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010130, value, 0)
            time.sleep(0.1)

    def move_turn_right(self, duration, value: int = 10000):  # 移动时右转
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    def move_turn_left(self, duration, value: int = -10000):  # 移动时左转
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    # 语音指令发送
    def voice_turn_left_90(self):
        self.send_data(0x21010C0A, 13, 0)
        time.sleep(3)

    def voice_turn_right_90(self):
        self.send_data(0x21010C0A, 14, 0)
        time.sleep(3)

    def voice_turn_right_180(self):
        self.send_data(0x21010C0A, 15, 0)
        time.sleep(3)

    # 扭身体
    def twist(self):
        self.send_data(0x21010204, 0, 0)
        time.sleep(3)

    # 太空步
    def moonwalk(self):
        self.send_data(0x2101030C, 0, 0)
        time.sleep(3)

    def move_turn_360(self, duration = 6, value = 24000):  # 转360需要计算时间与速度，从而使得刚好等于360°，同理也可以传参使其转90或者180
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    def move_turn_180(self, duration = 3, value = 16000):  # 转180需要计算时间与速度，从而使得刚好等于180°，同理也可以传参使其转90或者180
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    def move_turn_right_90(self, duration = 1.5, value = 12000):  # 右转90
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    def move_turn_left_90(self, duration = 1.5, value = -12000):  # 左转90
        start_time = time.time()
        while time.time() - start_time < duration:
            self.send_data(0x21010135, value, 0)
            time.sleep(0.1)

    def change_flat_slow_mode(self):  # 切换平地低速
        self.send_data(0x21010300, 0, 0)
        time.sleep(2)

    def change_flat_midlle_mode(self):  # 切换平地中速
        self.send_data(0x21010307, 0, 0)
        time.sleep(2)

    def change_flat_high_mode(self):  # 切换平地高速
        self.send_data(0x21010303, 0, 0)
        time.sleep(2)

    def change_normal_creep_mode(self):  # 切换正常匍匐
        self.send_data(0x21010306, 0, 0)
        time.sleep(2)

    def change_manual(self):
        self.send_data(0x21010C02, 0, 0)
        time.sleep(2)

    def change_autonomic(self):
        self.send_data(0x21010C03, 0, 0)
        time.sleep(2)

if __name__ == '__main__':
    h = HeartBeat()
    time.sleep(1)  # waiting for heart beat

    robot = RobotController()

    robot.stand_up()
    print("stand_up")
    time.sleep(3)  # waiting for standing up

    robot.move_turn_left_90()
    print("twist")
    time.sleep(3)

    robot.stand_up()
    print("stand_up")
    time.sleep(3)

    robot.stand_up()
    print("stand_up")
    time.sleep(3)  # waiting for standing up