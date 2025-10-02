#!/usr/bin/env python3
# speech_event_handler.py - 语音事件处理器

import pyttsx3
import threading
import time
import queue
import os
from playsound import playsound   # 用于播放已有音频文件

class SpeechEngine:
    def __init__(self, memory_manager, rate=150, volume=0.9):
        """初始化语音事件处理器"""
        self.memory_manager = memory_manager
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)  # 语速
        self.engine.setProperty('volume', volume)  # 音量
        
        # 语音队列和线程控制
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.running = True
        
        # 注册事件回调
        self.memory_manager.register_event_callback("speak_event", self.speak_event, "SpeechEventHandler")
        self.memory_manager.register_event_callback("medicine_detected", self.handle_medicine_event, "SpeechEventHandler")
        self.memory_manager.register_event_callback("face_detected", self.handle_face_event, "FaceEventHandler")
        self.memory_manager.register_event_callback("photo_detected", self.handle_photo_event, "FaceEventHandler")
        
        # 启动语音处理线程
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        print("语音事件处理器初始化完成")

    def speak_event(self, event_data):
        if event_data:
            self.add_to_queue(event_data, priority=0)

    def handle_medicine_event(self, event_data):
        if event_data:
            self.add_to_queue(event_data, priority=2)

    def handle_face_event(self, event_data):
        if event_data:
            self.add_to_queue(event_data, priority=2)

    def handle_photo_event(self, event_data):
        if event_data:
            self.add_to_queue(event_data, priority=2)

    def add_to_queue(self, event_data, priority=0):
        """添加语音或音频到队列"""
        text = event_data.get("speak_text", "")
        audio_file = event_data.get("audio_file", None)
        print(f"添加到队列: text={text}, audio_file={audio_file}, priority={priority}")
        self.speech_queue.put((priority, text, audio_file))
    
    def _process_speech_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                highest_priority = None
                highest_text = None
                highest_audio = None

                # 挑选最高优先级
                for _ in range(self.speech_queue.qsize()):
                    try:
                        priority, text, audio_file = self.speech_queue.get_nowait()
                        if highest_priority is None or priority > highest_priority:
                            highest_priority = priority
                            highest_text = text
                            highest_audio = audio_file
                        self.speech_queue.put((priority, text, audio_file))
                    except queue.Empty:
                        break

                if highest_text is not None or highest_audio is not None:
                    # 从队列中移除最高优先级的项目
                    for _ in range(self.speech_queue.qsize()):
                        priority, text, audio_file = self.speech_queue.get_nowait()
                        if text != highest_text or audio_file != highest_audio:
                            self.speech_queue.put((priority, text, audio_file))
                    
                    # 播放
                    self.is_speaking = True
                    if highest_audio and os.path.exists(highest_audio):
                        print(f"播放音频文件: {highest_audio}")
                        try:
                            playsound(highest_audio)
                        except Exception as e:
                            print(f"播放音频失败，改为TTS: {e}")
                            if highest_text:
                                self.engine.say(highest_text)
                                self.engine.runAndWait()
                    else:
                        if highest_text:
                            print(f"TTS 发声: {highest_text}")
                            self.engine.say(highest_text)
                            self.engine.runAndWait()
                    self.is_speaking = False
                
                time.sleep(0.1)

            except Exception as e:
                print(f"语音处理错误: {e}")
                self.is_speaking = False
                time.sleep(1)
    
    def stop(self):
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
    
    def is_busy(self):
        return self.is_speaking
