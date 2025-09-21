# noise_detector.py
import pyaudio
import numpy as np
import threading
import time
from collections import deque
import wave


class NoiseDetector:
    def __init__(self, memory_manager, sensitivity=0.7, sample_rate=16000, chunk_size=1024):
        """
        异常噪声检测器

        参数:
            memory_manager: 记忆管理器实例，用于触发事件
            sensitivity: 检测敏感度 (0.0-1.0)
            sample_rate: 音频采样率
            chunk_size: 每次处理的音频块大小
        """
        self.memory_manager = memory_manager
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None

        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 1

        # 噪声检测参数
        self.energy_threshold = 500  # 能量阈值
        self.energy_window = deque(maxlen=20)  # 能量窗口用于计算动态阈值
        self.min_abnormal_duration = 0.2  # 异常噪声最短持续时间(秒)
        self.cooldown_period = 5  # 检测冷却时间(秒)
        self.last_detection_time = 0

        # 初始化PyAudio
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"无法初始化PyAudio: {e}")
            self.audio = None

    def calculate_energy(self, data):
        """计算音频数据的能量"""
        # 将字节数据转换为numpy数组
        audio_data = np.frombuffer(data, dtype=np.int16)
        # 计算能量 (平方和)
        energy = np.sum(audio_data.astype(np.float32) ** 2) / len(audio_data)
        return energy

    def is_abnormal_noise(self, energy):
        """判断是否为异常噪声"""
        current_time = time.time()

        # 冷却期检查
        if current_time - self.last_detection_time < self.cooldown_period:
            return False

        # 更新能量窗口
        self.energy_window.append(energy)

        # 计算动态阈值 (平均能量的倍数)
        if len(self.energy_window) > 10:
            avg_energy = np.mean(list(self.energy_window)[:-5])  # 排除最近5个值
            dynamic_threshold = avg_energy * (1 + self.sensitivity * 10)
        else:
            dynamic_threshold = self.energy_threshold * (1 + self.sensitivity * 5)

        # 检查是否超过阈值
        return energy > dynamic_threshold

    def detect_noise(self):
        """噪声检测主循环"""
        if not self.audio:
            print("音频设备未初始化，无法进行噪声检测")
            return

        try:
            # 打开音频流
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print("开始噪声检测")
            abnormal_count = 0
            normal_count = 0

            while self.running:
                # 读取音频数据
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"读取音频数据错误: {e}")
                    time.sleep(0.1)
                    continue

                # 计算能量
                energy = self.calculate_energy(data)

                # 检测异常噪声
                if self.is_abnormal_noise(energy):
                    abnormal_count += 1
                    normal_count = 0

                    # 持续一段时间才确认为异常
                    if abnormal_count >= int(self.min_abnormal_duration * self.sample_rate / self.chunk_size):
                        print(f"检测到异常噪声! 能量: {energy}")
                        self.last_detection_time = time.time()

                        # 通过记忆管理器触发事件
                        self.memory_manager.trigger_event("abnormal_noise_detected", {
                            "energy": energy,
                            "timestamp": time.time()
                        })

                        # 重置计数
                        abnormal_count = 0
                else:
                    normal_count += 1
                    if normal_count > 5:  # 连续几个正常帧后重置异常计数
                        abnormal_count = 0

                # 短暂休眠
                time.sleep(0.01)

            # 停止时关闭流
            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"噪声检测错误: {e}")

    def start(self):
        """启动噪声检测"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.detect_noise)
        self.thread.daemon = True
        self.thread.start()
        print("噪声检测器已启动")

    def stop(self):
        """停止噪声检测"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("噪声检测器已停止")

    def set_sensitivity(self, sensitivity):
        """设置检测敏感度"""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        print(f"噪声检测敏感度设置为: {self.sensitivity}")

    def __del__(self):
        """析构函数"""
        self.stop()
        if self.audio:
            self.audio.terminate()


if __name__ == "__main__":
    # 测试代码
    class MockMemoryManager:
        def trigger_event(self, event_type, event_data):
            print(f"触发事件: {event_type}, 数据: {event_data}")


    memory_manager = MockMemoryManager()
    detector = NoiseDetector(memory_manager, sensitivity=0.7)

    try:
        detector.start()
        print("噪声检测测试中...按Ctrl+C停止")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()