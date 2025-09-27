# speech_engine.py - 语音事件处理器
import threading
import time
import queue


class SpeechEngine:
    def __init__(self, memory_manager):
        """初始化语音事件处理器"""
        self.memory_manager = memory_manager
        self.speech_queue = queue.PriorityQueue()
        self.running = True

        # 延迟导入，避免循环依赖
        self.speech_service = None

        # 注册事件回调
        self._register_event_handlers()

        # 启动语音处理线程
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        print("语音事件处理器初始化完成")

    def _get_speech_service(self):
        """延迟获取语音服务实例"""
        if self.speech_service is None:
            try:
                from .speech_service import speech_service
                self.speech_service = speech_service
            except ImportError as e:
                print(f"导入语音服务失败: {e}")
                return None
        return self.speech_service

    def _register_event_handlers(self):
        """注册事件处理器"""
        event_handlers = {
            "medicine_detected": self.handle_medicine_event,
            "unknown_medicine_detected": self.handle_unknown_medicine_event,
            "face_detected": self.handle_face_event,
            "photo_detected": self.handle_photo_event,
            "abnormal_noise_detected": self.handle_abnormal_noise_event,
            "urgent_noise_alert": self.handle_urgent_noise_alert,
        }

        for event_type, handler in event_handlers.items():
            self.memory_manager.register_event_callback(
                event_type, handler, "SpeechEventHandler"
            )

    def handle_medicine_event(self, event_data):
        """处理药品检测事件"""
        medicine_name = event_data.get("medicine_name", "未知药品")
        self.add_to_queue(f"检测到药品{medicine_name}，请按说明书服用", priority=2)

    def handle_unknown_medicine_event(self, event_data):
        """处理未知药品事件"""
        self.add_to_queue("检测到未知药品，请注意安全", priority=1)

    def handle_face_event(self, event_data):
        """处理人脸检测事件"""
        name = event_data.get("name", "访客")
        confidence = event_data.get("confidence", 0)
        if confidence > 0.7:
            self.add_to_queue(f"看到{name}来了，真高兴！", priority=2)

    def handle_photo_event(self, event_data):
        """处理照片检测事件"""
        description = event_data.get("description", "照片")
        self.add_to_queue(f"识别到{description}", priority=2)

    def handle_abnormal_noise_event(self, event_data):
        """处理异常噪声事件"""
        noise_type = event_data.get("noise_type", "")
        risk_level = event_data.get("risk_level", "")
        energy = event_data.get("energy", 0)

        print(f"检测到异常噪声: {noise_type}, 风险级别: {risk_level}")

        responses = {
            'high_pitch': {
                'medium': "检测到异常声响，请问发生了什么？",
                'high': "检测到大声喊叫，您需要帮助吗？"
            },
            'glass_break': {
                'high': "检测到玻璃破碎声！请小心碎片",
                'critical': "紧急！检测到玻璃破碎！"
            },
            'impact': {
                'high': "检测到重物落地声，您还好吗？",
                'critical': "检测到跌倒声！需要帮助吗？"
            },
            'alarm_sound': {
                'critical': "检测到报警器声响！请检查安全状况"
            },
            'moaning_crying': {
                'medium': "听到您似乎不舒服，需要帮助吗？",
                'high': "检测到痛苦声音，您还好吗？"
            }
        }

        response = responses.get(noise_type, {}).get(risk_level, "检测到异常声音")
        priority = 4 if risk_level in ['high', 'critical'] else 3
        self.add_to_queue(response, priority=priority)

    def handle_urgent_noise_alert(self, event_data):
        """处理紧急噪声警报"""
        noise_type = event_data.get("noise_type", "")
        urgent_responses = {
            'impact': "紧急！检测到可能跌倒的声音！您还好吗？",
            'glass_break': "紧急！检测到玻璃破碎！请小心！",
            'alarm_sound': "紧急！检测到安全警报！"
        }

        response = urgent_responses.get(noise_type, "紧急情况！需要帮助吗？")
        self.add_to_queue(response, priority=5)

    def add_to_queue(self, text, priority=0):
        """添加语音到队列"""
        print(f"添加语音到队列: {text}, 优先级: {priority}")
        self.speech_queue.put((priority, text))

    def _process_speech_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                if not self.speech_queue.empty():
                    priority, text = self.speech_queue.get()
                    speech_service = self._get_speech_service()

                    if speech_service:
                        # 等待语音服务空闲
                        while speech_service.is_busy():
                            time.sleep(0.1)

                        speech_service.add_speech(text, priority)

                    self.speech_queue.task_done()

                time.sleep(0.1)

            except Exception as e:
                print(f"语音处理错误: {e}")
                time.sleep(1)

    def stop(self):
        """停止语音事件处理器"""
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)