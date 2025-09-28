# speech_service.py
import pyttsx3
import queue
import threading
import time


class SpeechService:
    def __init__(self, rate=150, volume=0.9):
        """初始化语音服务"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
        except Exception as e:
            print(f"语音引擎初始化失败: {e}")
            self.engine = None

        # 语音队列和线程控制
        self.speech_queue = queue.PriorityQueue()  # 改为优先级队列
        self.is_speaking = False
        self.running = True

        # 启动语音处理线程
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        print("语音服务初始化完成")

    def _process_speech_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                if not self.speech_queue.empty():
                    self.is_speaking = True
                    priority, text = self.speech_queue.get()

                    if self.engine:
                        self.engine.say(text)
                        self.engine.runAndWait()

                    self.is_speaking = False
                    self.speech_queue.task_done()

                time.sleep(0.1)

            except Exception as e:
                print(f"语音处理错误: {e}")
                self.is_speaking = False
                time.sleep(1)

    def add_speech(self, text, priority=0):
        """添加语音到队列"""
        self.speech_queue.put((priority, text))

    def stop(self):
        """停止语音服务"""
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)

    def is_busy(self):
        """检查语音服务是否正在说话"""
        return self.is_speaking or not self.speech_queue.empty()


# 全局语音服务实例
speech_service = SpeechService()