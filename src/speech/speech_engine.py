# speech_engine.py - 语音事件处理器
from .import speech_service
import threading
import time
import queue


class SpeechEngine:
    def __init__(self, memory_manager):
        """初始化语音事件处理器"""
        self.memory_manager = memory_manager
        self.speech_queue = queue.PriorityQueue()
        self.running = True

        # 注册事件回调
        self.memory_manager.register_event_callback(
            "medicine_detected",
            self.handle_medicine_event,
            "SpeechEventHandler"
        )
        self.memory_manager.register_event_callback(
            "unknown_medicine_detected",
            self.handle_unknown_medicine_event,
            "SpeechEventHandler"
        )
        self.memory_manager.register_event_callback(
            "face_detected",
            self.handle_face_event,
            "SpeechEventHandler"
        )
        self.memory_manager.register_event_callback(
            "photo_detected",
            self.handle_photo_event,
            "SpeechEventHandler"
        )
        self.memory_manager.register_event_callback(
            "abnormal_noise_detected",
            self.handle_abnormal_noise_event,
            "SpeechEventHandler"
        )

        # 启动语音处理线程
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        print("语音事件处理器初始化完成")

    def handle_medicine_event(self, event_data):
        """处理药品检测事件"""
        if event_data and "speak_text" in event_data:
            self.add_to_queue(event_data["speak_text"], priority=2)

    def handle_unknown_medicine_event(self, event_data):
        """处理未知药品事件"""
        if event_data and "speak_text" in event_data:
            self.add_to_queue(event_data["speak_text"], priority=1)

    def handle_face_event(self, event_data):
        """处理人脸检测事件"""
        if event_data and "speak_text" in event_data:
            self.add_to_queue(event_data["speak_text"], priority=2)

    def handle_photo_event(self, event_data):
        """处理照片检测事件"""
        if event_data and "speak_text" in event_data:
            self.add_to_queue(event_data["speak_text"], priority=2)

    def handle_abnormal_noise_event(self, event_data):
        """处理异常噪声事件"""
        if event_data:
            energy = event_data.get("energy", 0)
            print(f"检测到异常噪声，能量级别: {energy}")

            # 根据噪声级别选择不同的响应
            if energy > 10000:  # 高能量噪声
                response = "检测到巨大声响！您还好吗？需要帮助吗？"
            else:  # 中等能量噪声
                response = "检测到异常声响，请问发生了什么？"

            self.add_to_queue(response, priority=3)  # 高优先级

    def add_to_queue(self, text, priority=0):
        """添加语音到队列"""
        print(f"添加语音到队列:{text},priority:{priority}")
        self.speech_queue.put((priority, text))

    def _process_speech_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                # 获取最高优先级的语音
                if not self.speech_queue.empty():
                    priority, text = self.speech_queue.get()

                    # 等待语音服务空闲
                    while speech_service.is_busy():
                        time.sleep(0.1)

                    # 发送到语音服务
                    speech_service.add_speech(text, priority)

                # 短暂休眠以减少CPU使用
                time.sleep(0.1)

            except Exception as e:
                print(f"语音处理错误: {e}")
                time.sleep(1)

    def stop(self):
        """停止语音事件处理器"""
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)


if __name__ == "__main__":
    """测试函数"""


    class MemoryManager:
        def __init__(self):
            self.event_handlers = {}

        def register_event_callback(self, event_type, callback, handler_id):
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append((handler_id, callback))
            print(f"已注册事件处理器: {event_type} -> {handler_id}")

        def trigger_event(self, event_type, event_data):
            if event_type in self.event_handlers:
                for handler_id, callback in self.event_handlers[event_type]:
                    print(f"触发事件: {event_type} -> {handler_id}")
                    callback(event_data)


    memory_manager = MemoryManager()
    speech_engine = SpeechEngine(memory_manager)

    try:
        print("语音事件处理器测试开始...")

        # 测试事件触发
        medicine_data = {
            "speak_text": "检测到药品阿司匹林，请按说明书服用。",
            "medicine_name": "阿司匹林",
            "confidence": 0.95
        }
        memory_manager.trigger_event("medicine_detected", medicine_data)

        # 等待语音处理
        time.sleep(5)

    except KeyboardInterrupt:
        print("\n用户中断测试...")
    finally:
        speech_engine.stop()
        speech_service.stop()
        print("测试结束")