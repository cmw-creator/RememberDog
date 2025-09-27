import numpy as np
import pyaudio
import threading
import time
import queue
from collections import deque


class NoiseDetector:
    def __init__(self, memory_manager, sample_rate=16000, chunk_size=1024):
        self.memory_manager = memory_manager
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.audio_queue = queue.Queue()

        # 音频参数
        self.format = pyaudio.paInt16
        self.channels = 1

        # 能量检测参数
        self.energy_threshold = 1000  # 基础能量阈值
        self.silence_limit = 3  # 持续静音帧数
        self.silence_count = 0

        # 音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        """启动噪声检测"""
        if self.running:
            return

        self.running = True
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("噪声检测器已启动")
        except Exception as e:
            print(f"启动音频流失败: {e}")
            self.running = False

    def stop(self):
        """停止噪声检测"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.running and in_data is not None and len(in_data) > 0:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def calculate_energy(self, audio_data):
        """修复版：计算音频能量"""
        if len(audio_data) == 0:
            return 0.0

        try:
            # 确保音频数据是numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_array) == 0:
                return 0.0

            # 计算RMS能量（均方根）
            squared = np.square(audio_array.astype(np.float64))
            mean_squared = np.mean(squared)
            rms_energy = np.sqrt(mean_squared)

            # 调试输出
            if hasattr(self, 'energy_debug_count'):
                self.energy_debug_count += 1
            else:
                self.energy_debug_count = 0

            if self.energy_debug_count % 50 == 0:  # 每50帧输出一次调试信息
                print(f"能量计算调试: 数据长度={len(audio_data)}, 数组长度={len(audio_array)}, RMS能量={rms_energy}")

            return rms_energy

        except Exception as e:
            print(f"能量计算错误: {e}")
            return 0.0