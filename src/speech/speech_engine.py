#!/usr/bin/env python3
# speech_event_handler.py - 语音事件处理器

import pyttsx3
import threading
import time
import queue

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
        # 通用的发声时间 #
        self.memory_manager.register_event_callback(
            "medicine_detected2", 
            self.handle_medicine_event,
            "SpeechEventHandler2"
        )
        # 二维码 #
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
        # 人脸识别 #
        self.memory_manager.register_event_callback(
            "face_detected", 
            self.handle_face_event,
            "FaceEventHandler"
        )
        # 照片识别 #
        self.memory_manager.register_event_callback(
            "photo_detected", 
            self.handle_photo_event,
            "FaceEventHandler"
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

    def add_to_queue(self, text, priority=0):
        """添加语音到队列"""
        print(f"添加语音到队列:{text},priority:{priority}")
        self.speech_queue.put((priority, text))
    
    def _process_speech_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                # 获取最高优先级的语音
                highest_priority = None
                highest_text = None
                
                # 检查队列中的所有项目
                for _ in range(self.speech_queue.qsize()):
                    try:
                        priority, text = self.speech_queue.get_nowait()
                        if highest_priority is None or priority > highest_priority:
                            highest_priority = priority
                            highest_text = text
                        # 将非最高优先级的项目重新放回队列
                        self.speech_queue.put((priority, text))
                    except queue.Empty:
                        break
                
                # 如果有最高优先级的语音，则播放
                if highest_text is not None:
                    # 从队列中移除最高优先级的项目
                    for _ in range(self.speech_queue.qsize()):
                        priority, text = self.speech_queue.get_nowait()
                        if text != highest_text:
                            self.speech_queue.put((priority, text))
                    
                    # 播放语音
                    self.is_speaking = True
                    self.engine.say(highest_text)
                    self.engine.runAndWait()
                    self.is_speaking = False
                
                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
                
            except Exception as e:
                print(f"语音处理错误: {e}")
                self.is_speaking = False
                time.sleep(1)
    
    def stop(self):
        """停止语音事件处理器"""
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)
    
    def is_busy(self):
        """检查语音引擎是否正在说话"""
        return self.is_speaking

if __name__ == "__main__":
    """
    测试函数，演示SpeechEngine的基本用法
    """
    # 创建一个简单的模拟内存管理器类
    class MemoryManager:
        def __init__(self):
            self.event_handlers = {}
            
        def register_event_callback(self, event_type, callback, handler_id):
            """模拟注册事件回调"""
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append((handler_id, callback))
            print(f"已注册事件处理器: {event_type} -> {handler_id}")
            
        def trigger_event(self, event_type, event_data):
            """模拟触发事件"""
            if event_type in self.event_handlers:
                for handler_id, callback in self.event_handlers[event_type]:
                    print(f"触发事件: {event_type} -> {handler_id}")
                    callback(event_data)
    
    # 创建模拟内存管理器实例
    memory_manager = MemoryManager()
    
    # 创建语音引擎实例
    speech_engine = SpeechEngine(memory_manager, rate=180, volume=0.8)
    
    try:
        print("语音事件处理器测试开始...")
        print("=" * 50)
        
        # 测试1: 直接添加语音到队列
        print("测试1: 直接添加语音到队列")
        speech_engine.add_to_queue("这是一条普通优先级语音", priority=0)
        speech_engine.add_to_queue("这是一条高优先级语音", priority=2)
        speech_engine.add_to_queue("这是一条中等优先级语音", priority=1)
        
        # 等待一段时间让语音队列处理
        time.sleep(3)
        
        # 测试2: 通过事件触发语音
        print("\n测试2: 通过事件触发语音")
        print("-" * 30)
        
        # 触发已知药品事件
        medicine_data = {
            "speak_text": "检测到药品阿司匹林，请按说明书服用。",
            "medicine_name": "阿司匹林",
            "confidence": 0.95
        }
        memory_manager.trigger_event("medicine_detected", medicine_data)
        
        # 触发未知药品事件
        unknown_medicine_data = {
            "speak_text": "警告！检测到未知药品，请谨慎使用。",
            "medicine_name": "未知药品",
            "confidence": 0.75
        }
        memory_manager.trigger_event("unknown_medicine_detected", unknown_medicine_data)
        
        # 等待语音处理
        print("\n等待语音处理完成...")
        for i in range(10):
            if speech_engine.is_busy():
                print("语音引擎正在说话...")
            else:
                print("语音引擎空闲")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n用户中断测试...")
    finally:
        # 停止语音引擎
        speech_engine.stop()
        print("测试结束，语音引擎已停止")