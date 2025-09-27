#!/usr/bin/env python3
# 增强版本地语音助手 - 使用本地识别和处理
from .speech_service import speech_service
import threading
import json
import time
import os
import re
from datetime import datetime, timedelta
import wave
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer

class VoiceAssistant:
    def __init__(self, memory_manager):
        # 初始化记忆管理器
        self.memory_manager = memory_manager

        # 语音识别模型路径
        self.model_path = "assets/voice_models/vosk-model-small-cn-0.22"

        # 延迟初始化语音识别
        self.model = None
        self.recognizer = None
        self._init_voice_recognition()
        # 音频参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000

        # 运行状态
        self.running = True
        self.listening = True

        # 本地命令数据库
        self.commands_db = self.load_commands_db()

        # 注册事件回调
        self._register_event_handlers()
        
    def setup_voice_recognition(self):
        """设置本地语音识别"""
        try:
            if os.path.exists(self.model_path):
                self.model = Model(self.model_path)
                self.recognizer = KaldiRecognizer(self.model, 16000)
                print("本地语音识别模型加载成功")
            else:
                print(f"警告: 语音模型路径不存在: {self.model_path}")
                self.model = None
                self.recognizer = None
        except Exception as e:
            print(f"语音识别初始化失败: {str(e)}")
            self.model = None
            self.recognizer = None

    def _init_voice_recognition(self):
        """延迟初始化语音识别"""
        try:
            from vosk import Model, KaldiRecognizer
            model_path = "assets/voice_models/vosk-model-small-cn-0.22"

            if os.path.exists(model_path):
                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, 16000)
                print("本地语音识别模型加载成功")
            else:
                print(f"语音模型路径不存在: {model_path}")
                # 使用备用方案
                self._init_fallback_recognition()

        except ImportError as e:
            print(f"无法导入Vosk: {e}")
            self._init_fallback_recognition()
        except Exception as e:
            print(f"语音识别初始化失败: {e}")
            self._init_fallback_recognition()

    def _init_fallback_recognition(self):
        """备用语音识别方案"""
        print("使用备用语音识别方案")
        # 这里可以添加其他识别库或简化版本

    def _register_event_handlers(self):
        """注册事件处理器"""
        if self.memory_manager:
            self.memory_manager.register_event_callback(
                "medicine_detected", self.handle_medicine_event, "VoiceAssistant"
            )
            self.memory_manager.register_event_callback(
                "face_recognized", self.handle_face_event, "VoiceAssistant"
            )

    def listen_command(self):
        """监听语音指令"""
        if not self.recognizer:
            print("语音识别未就绪")
            return

        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            print("语音助手开始监听...")
            while self.running and self.listening:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '')

                        if text and len(text.strip()) > 0:
                            print(f"识别结果: {text}")
                            self.process_command(text)

                except Exception as e:
                    print(f"语音处理错误: {e}")
                    time.sleep(0.1)

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            print(f"音频流错误: {e}")
    def load_commands_db(self):
        """加载本地命令数据库"""
        commands_path = "assets/voice_commands/commands.json"
        default_commands = {
            "commands": [
                {"pattern": "过来", "action": "come"},
                {"pattern": "走开", "action": "leave"},
                {"pattern": "吃药", "action": "medicine_remind"},
                {"pattern": "提醒", "action": "add_reminder"},
                {"pattern": "问题", "action": "add_question"},
                {"pattern": "停止", "action": "stop"},
                {"pattern": "开始", "action": "start"},
                {"pattern": "帮助", "action": "help"}
            ],
            "responses": {
                "greeting": ["你好", "您好", "嗨", "hello"],
                "unknown": ["我没有听懂，请再说一次", "抱歉，我不明白", "请重复一遍"]
            }
        }

        if os.path.exists(commands_path):
            try:
                with open(commands_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print("无法加载命令数据库，使用默认命令")
                return default_commands
        else:
            # 创建默认命令文件
            os.makedirs(os.path.dirname(commands_path), exist_ok=True)
            with open(commands_path, 'w', encoding='utf-8') as f:
                json.dump(default_commands, f, ensure_ascii=False, indent=2)
            return default_commands

    def recognize_speech_offline(self, audio_data):
        """使用本地模型识别语音"""
        if self.recognizer is None:
            return None

        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                return result.get('text', '')
            return None
        except Exception as e:
            print(f"语音识别错误: {str(e)}")
            return None
    
    def record_audio(self, duration=5):
        """录制音频"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        print("正在录音...")
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        print("录音结束")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return b''.join(frames)
    
    def listen_command(self):
        """持续监听语音指令"""
        
        p = pyaudio.PyAudio()
        # 列出可用音频设备
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        print("语音助手已启动，等待唤醒词...")
        frame_count = 0
        recognition_attempts = 0
        while self.running:
            #if not self.listening:
            #    time.sleep(0.5)
            #    continue
                
            try:
                # 读取音频数据
                data = stream.read(self.chunk, exception_on_overflow=False)
                frame_count += 1
            
                # 每处理一定数量的帧打印一次调试信息
                if frame_count % 50 == 0:
                    print(f"已处理 {frame_count} 帧音频数据，尝试识别 {recognition_attempts} 次")
                # 使用本地模型识别
                if self.recognizer.AcceptWaveform(data):
                    recognition_attempts += 1
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    
                    if text:
                        print(f"识别结果: {text}")
                        
                        # 处理特殊命令
                        if any(cmd in text for cmd in ["提醒", "设置提醒"]):
                            self.handle_add_reminder(text)
                        elif any(cmd in text for cmd in ["问题", "设置问题"]):
                            self.handle_add_question(text)
                        else:
                            # 处理普通命令
                            self.process_command(text)
                
            except Exception as e:
                print(f"语音处理错误: {str(e)}")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def process_command(self, text):
        """处理语音命令"""
        # 首先检查问候语
        for greeting in self.commands_db["responses"]["greeting"]:
            if greeting in text:
                self.speak_random(["你好！", "您好！", "嗨，有什么可以帮您？"])
                return
        
        # 检查命令
        command_matched = False
        for command in self.commands_db["commands"]:
            if re.search(command["pattern"], text):
                self.execute_action(command["action"], text)
                command_matched = True
                break
        
        # 如果没有匹配的命令
        if not command_matched:
            self.speak_random(self.commands_db["responses"]["unknown"])
    
    def execute_action(self, action, text):
        """执行对应动作"""
        if action == "add_reminder":
            self.handle_add_reminder(text)
        elif action == "add_question":
            self.handle_add_question(text)
        elif action == "stop":
            self.listening = False
            self.speak("已停止监听")
        elif action == "start":
            self.listening = True
            self.speak("开始监听")
        elif action == "help":
            self.speak("我可以帮您添加提醒、设置问题、控制机器狗行动")
        else:
            # 其他动作处理
            self.speak(f"执行命令: {action}")
    
    def parse_time_from_text(self, text):
        """从文本中解析时间信息"""
        # 时间模式匹配
        time_patterns = [
            r'(\d+)点(\d+)分',
            r'(\d+)点',
            r'下午(\d+)点',
            r'上午(\d+)点',
            r'明天(\d+)点',
            r'后天(\d+)点'
        ]
        
        hour, minute = None, 0
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 2:
                    hour, minute = int(match.group(1)), int(match.group(2))
                else:
                    hour = int(match.group(1))
                break
        
        # 处理上午/下午
        if "下午" in text and hour < 12:
            hour += 12
        elif "上午" in text and hour == 12:
            hour = 0
            
        # 处理明天/后天
        day_offset = 0
        if "明天" in text:
            day_offset = 1
        elif "后天" in text:
            day_offset = 2
            
        return hour, minute, day_offset
    
    def handle_add_reminder(self, text):
        """处理添加提醒命令"""
        try:
            # 解析时间
            hour, minute, day_offset = self.parse_time_from_text(text)
            
            if hour is not None:
                # 生成时间字符串
                time_str = f"{hour:02d}:{minute:02d}"
                
                # 提取事件描述 - 简单提取"提醒我"之后的内容
                event_match = re.search(r'提醒[我你]*([^，。！？]*)', text)
                if event_match:
                    event = event_match.group(1).strip()
                else:
                    # 如果找不到"提醒"关键词，尝试提取其他内容
                    event = text
                
                # 添加到提醒系统
                self.memory_manager.add_reminder(time_str, event)
                self.speak(f"已添加提醒: {event} 在 {time_str}")
            else:
                self.speak("未识别到时间信息，请说明具体时间")
                
        except Exception as e:
            self.speak("添加提醒失败，请重试")
            print(f"添加提醒错误: {str(e)}")
    
    def handle_add_question(self, text):
        """处理添加问题命令"""
        try:
            # 简单解析问题
            question_match = re.search(r'问题[是：:]*([^，。！？]*)', text)
            answer_match = re.search(r'答案[是：:]*([^，。！？]*)', text)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                # 添加到家庭信息库
                self.memory_manager.add_question(question, answer)
                self.speak(f"已添加问题: {question}")
            else:
                self.speak("请说明问题和答案，例如：问题是我的生日，答案是1月1日")
                
        except Exception as e:
            self.speak("添加问题失败，请重试")
            print(f"添加问题错误: {str(e)}")

    def speak(self, text):
        """语音合成 - 使用全局语音服务"""
        speech_service.add_speech(text, priority=1)

    def speak_random(self, texts):
        """随机选择一段文本进行语音合成"""
        if texts:
            text = np.random.choice(texts)
            self.speak(text)
    
    def start_listening(self):
        """开始监听"""
        self.listening = True
        self.speak("语音助手已启动")
    
    def stop_listening(self):
        """停止监听"""
        self.listening = False
        self.speak("语音助手已停止")
    
    def start(self):
        """启动语音助手"""
        print("启动本地语音助手")
        self.listening = True
        listen_thread = threading.Thread(target=self.listen_command)
        listen_thread.daemon = True
        listen_thread.start()
    
    def stop(self):
        """停止语音助手"""
        self.running = False
        self.listening = False
    def test_recognition(self, duration=10):
        """测试录音和识别功能"""
        print(f"\n开始 {duration} 秒测试录音...")
        audio_data = self.record_audio(duration)
        print("测试录音完成")
        
        # 保存录音用于调试
        test_file = "test_recording.wav"
        with wave.open(test_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)
        print(f"测试录音已保存为: {test_file}")
        
        # 尝试识别
        print("尝试识别测试录音...")
        result = self.recognize_speech_offline(audio_data)
        if result:
            print(f"测试识别结果: {result}")
        else:
            print("测试识别失败: 无返回结果")
            # 检查模型是否加载
            if self.model is None:
                print("模型未正确加载")
            else:
                print("模型已加载，但识别无结果")
        
        return result
    def handle_medicine_event(self, event_data):
        """处理药品检测事件"""
        medicine_info = event_data.get("info", "")
        self.speak(f"提醒您，刚才识别到了药品：{medicine_info}")
    
    def handle_face_event(self, event_data):
        """处理人脸识别事件"""
        name = event_data.get("name", "")
        confidence = event_data.get("confidence", 0)
        if confidence > 0.7:  # 高置信度
            self.speak(f"看到{name}来了，真高兴！")

if __name__ == "__main__":
    # 测试代码
    class MockMemoryManager:
        def add_reminder(self, time_str, event):
            print(f"添加提醒: {event} 在 {time_str}")
        
        def add_question(self, question, answer):
            print(f"添加问题: {question} 答案: {answer}")
    
    memory_manager = MockMemoryManager()
    assistant = VoiceAssistant(memory_manager)
    #assistant.test_recognition()
    assistant.start()
    time.sleep(1000)
    
