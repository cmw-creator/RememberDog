#!/usr/bin/env python3
# 增强版本地语音助手 - 集成降噪和自动增益控制

import threading
import json
import time
import os
import re
import numpy as np
import wave
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer
import noisereduce as nr
from scipy import signal

class VoiceAssistant:
    def __init__(self, memory_manager=None):
        # 初始化记忆管理器
        self.memory_manager = memory_manager
        
        # 语音识别模型路径
        self.model_path = "assets/voice_models/vosk-model-small-cn-0.22"
        
        # 音频参数
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
        # 音频增强参数
        self.noise_profile = None
        self.noise_profile_length = 2.0  # 用于噪声分析的秒数
        self.agc_factor = 5.0  # 自动增益因子
        self.noise_reduction_enabled = True
        self.agc_enabled = True
        self.noise_gate_threshold = 0.05  # 噪声门限阈值
        
        # 初始化语音识别
        self.setup_voice_recognition()
        
        # 初始化语音合成
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # 运行状态
        self.running = True
        self.listening = True
        
        # 本地命令数据库
        self.commands_db = self.load_commands_db()
        
        # 初始化噪声配置文件
        self.initialize_noise_profile()
        
        # 启动音频质量监控
        self.start_quality_monitor()
        
    def setup_voice_recognition(self):
        """设置本地语音识别"""
        try:
            if os.path.exists(self.model_path):
                self.model = Model(self.model_path)
                self.recognizer = KaldiRecognizer(self.model, self.rate)
                print("本地语音识别模型加载成功")
            else:
                print(f"警告: 语音模型路径不存在: {self.model_path}")
                print("请从 https://alphacephei.com/vosk/models 下载中文模型")
                self.model = None
                self.recognizer = None
        except Exception as e:
            print(f"语音识别初始化失败: {str(e)}")
            self.model = None
            self.recognizer = None
    
    def load_commands_db(self):
        """加载本地命令数据库"""
        commands_path = "assets/voice_commands/commands.json"
        with open(commands_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    

    ### 音频增强和降噪 ###
    def initialize_noise_profile(self):
        """录制初始噪声样本用于噪声分析[6](@ref)"""
        print("录制噪声样本用于降噪...")
        try:
            noise_sample = self.record_audio(duration=1.0)
            noise_sample_np = np.frombuffer(noise_sample, dtype=np.int16)
            self.noise_profile = noise_sample_np.astype(np.float32) / 32768.0
            print("噪声样本采集完成")
        except Exception as e:
            print(f"噪声样本采集失败: {str(e)}")
            self.noise_profile = None
    
    def apply_audio_enhancement(self, audio_data):
        """
        应用音频增强处理：降噪 + 自动增益控制[6,7](@ref)
        """
        # 转换为numpy数组并归一化
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # 应用降噪
        if self.noise_reduction_enabled and self.noise_profile is not None and len(audio_float) > 0:
            try:
                # 使用noisereduce库进行降噪[7](@ref)
                reduced_noise = nr.reduce_noise(
                    y=audio_float, 
                    sr=self.rate, 
                    y_noise=self.noise_profile,
                    prop_decrease=0.8,  # 噪声减少比例
                    n_fft=1024,
                    hop_length=256
                )
                audio_float = reduced_noise
            except Exception as e:
                print(f"降噪处理错误: {str(e)}")
        
        # 应用自动增益控制 (AGC)
        if self.agc_enabled and len(audio_float) > 0:
            audio_float = self.apply_agc(audio_float)
        
        # 应用噪声门限
        if len(audio_float) > 0:
            rms = np.sqrt(np.mean(audio_float**2))
            if rms < self.noise_gate_threshold:
                # 低于阈值，认为是噪声，静音处理
                audio_float = np.zeros_like(audio_float)
        
        # 转换回16位整数格式
        enhanced_audio = (audio_float * 32767.0).astype(np.int16)
        return enhanced_audio.tobytes()
    
    def apply_agc(self, audio_data):
        """应用自动增益控制"""
        if len(audio_data) == 0:
            return audio_data
            
        # 计算RMS（均方根）值
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            # 计算增益因子（目标RMS为0.1）
            target_rms = 0.1
            gain = target_rms / rms
            
            # 限制最大增益以避免失真
            max_gain = 5.0
            gain = min(gain, max_gain)
            
            # 应用增益
            audio_data = audio_data * gain
            
            # 限制幅值避免削波
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data
    
    def calculate_audio_level(self, audio_data):
        """计算音频电平"""
        if len(audio_data) == 0:
            return 0
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_np**2))
    
    def monitor_audio_quality(self):
        """监控音频质量并自动调整参数"""
        while self.running:
            try:
                time.sleep(5)  # 每5秒检查一次
                if not self.listening:
                    continue
                    
                # 录制短暂样本
                sample = self.record_audio(duration=0.5)
                level = self.calculate_audio_level(sample)
                
                # 根据音频电平动态调整AGC
                if level < 100:  # 安静环境
                    self.agc_factor = 4.0
                elif level > 1000:  # 嘈杂环境
                    self.agc_factor = 1.5
                else:  # 正常环境
                    self.agc_factor = 2.0
                    
                #print(f"音频电平: {level}, AGC因子: {self.agc_factor}")
                
            except Exception as e:
                print(f"音频监控错误: {str(e)}")
                time.sleep(1)
    
    def start_quality_monitor(self):
        """启动音频质量监控"""
        monitor_thread = threading.Thread(target=self.monitor_audio_quality)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def record_audio(self, duration=5):
        """录制音频[1,2](@ref)"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return b''.join(frames)
    
    def record_enhanced_audio(self, duration=5):
        """录制并增强音频"""
        raw_audio = self.record_audio(duration)
        enhanced_audio = self.apply_audio_enhancement(raw_audio)
        return enhanced_audio
    
    def test_enhancement(self, duration=3):
        """测试音频增强效果"""
        print("测试音频增强功能...")
        print("录制原始音频...")
        raw_audio = self.record_audio(duration)
        
        print("应用音频增强...")
        enhanced_audio = self.apply_audio_enhancement(raw_audio)
        
        # 计算并显示音频电平
        raw_level = self.calculate_audio_level(raw_audio)
        enhanced_level = self.calculate_audio_level(enhanced_audio)
        
        print(f"原始音频电平: {raw_level}")
        print(f"增强后音频电平: {enhanced_level}")
        print(f"增益: {enhanced_level/raw_level if raw_level > 0 else '无穷大'}")
        
        # 保存音频用于比较
        with wave.open("test_raw.wav", 'wb') as f:
            f.setnchannels(self.channels)
            f.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            f.setframerate(self.rate)
            f.writeframes(raw_audio)
            
        with wave.open("test_enhanced.wav", 'wb') as f:
            f.setnchannels(self.channels)
            f.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            f.setframerate(self.rate)
            f.writeframes(enhanced_audio)
            
        print("测试音频已保存为 test_raw.wav 和 test_enhanced.wav")
        
        return enhanced_audio
    ### 音频增强和降噪结束 ###



    ### 语音识别 ###
    def recognize_speech_offline(self, audio_data):
        """使用本地模型识别语音"""
        if self.recognizer is None:
            return None
            
        try:
            # 重置识别器状态
            self.recognizer.Reset()
            
            # 分块处理音频数据
            chunk_size = 4000  # Vosk推荐的处理块大小
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # 最后一块，用零填充
                    chunk += b'\x00' * (chunk_size - len(chunk))
                self.recognizer.AcceptWaveform(chunk)
            
            # 获取最终结果
            result = json.loads(self.recognizer.FinalResult())
            return result.get('text', '')
        except Exception as e:
            print(f"语音识别错误: {str(e)}")
            return None
    
    def run_listen(self):
        """持续监听语音指令"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        print("语音助手已启动，等待指令...")
        
        # 创建缓冲区用于存储连续的音频数据
        audio_buffer = bytes()
        buffer_duration = 1  # 缓冲0.5秒音频
        buffer_size = int(self.rate * buffer_duration * 2)  # 16位 = 2字节
        
        while self.running:
            # 启用终端输入代替语音识别测试 #
            if True:
                text=input("测试：请输入要说的话：")
                self.process_command(text)
                continue

            if not self.listening:
                time.sleep(0.1)
                continue
                
            try:
                # 读取音频数据
                data = stream.read(self.chunk, exception_on_overflow=False)
                
                # 添加到缓冲区
                audio_buffer += data
                
                # 当缓冲区有足够数据时进行处理
                if len(audio_buffer) >= buffer_size:
                    # 应用音频增强
                    enhanced_audio = self.apply_audio_enhancement(audio_buffer)
                    
                    # 使用增强后的音频进行识别
                    text = self.recognize_speech_offline(enhanced_audio)
                    
                    if text and len(text.strip()) > 0:
                        print(f"识别结果: {text}")
                        self.process_command(text)
                    
                    # 清空缓冲区
                    audio_buffer = bytes()
                    
            except Exception as e:
                print(f"语音处理错误: {str(e)}")
                audio_buffer = bytes()  # 发生错误时清空缓冲区
        
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
            pass
    
    def execute_action(self, action, text):
        """执行对应动作"""
        if action == "add_reminder":
            self.handle_add_reminder(text)
        elif action == "add_question":
            self.handle_add_question(text)
        elif action == "stop":
            self.listening = False
            print("已停止监听")
        elif action == "start":
            self.listening = True
            print("开始监听")
        elif action == "help":
            print("发出声音：我可以帮您添加提醒、设置问题、控制机器狗行动")
            #self.speak("我可以帮您添加提醒、设置问题、控制机器狗行动")
        else:
            print("xxx")
            #self.speak(f"执行命令: {action}")
    ### 语音识别结束 ###

    ###  处理各种指令 ###
    def handle_add_reminder(self, text):
        """处理添加提醒命令"""
        try:
            # 解析时间
            hour, minute, day_offset = self.parse_time_from_text(text)
            
            if hour is not None:
                # 生成时间字符串
                time_str = f"{hour:02d}:{minute:02d}"
                
                # 提取事件描述
                event_match = re.search(r'提醒[我你]*([^，。！？]*)', text)
                if event_match:
                    event = event_match.group(1).strip()
                else:
                    event = text
                
                # 添加到提醒系统
                print(f"尝试添加提醒: {event} 在 {time_str}")
                self.memory_manager.add_reminder(time_str, event)
                #self.speak(f"已添加提醒: {event} 在 {time_str}")
            else:
                print("未识别到时间信息，请说明具体时间")
                #self.speak("未识别到时间信息，请说明具体时间")
                
        except Exception as e:
            #self.speak("添加提醒失败，请重试")
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
                if self.memory_manager:
                    self.memory_manager.add_question(question, answer)
                #self.speak(f"已添加问题: {question}")
                print(f"已添加问题: {question}")
            else:
                #self.speak("请说明问题和答案，例如：问题是我的生日，答案是1月1日")
                print("发出声音：请说明问题和答案，例如：问题是我的生日，答案是1月1日")
                
        except Exception as e:
            
            #self.speak("添加问题失败，请重试")
            print(f"添加问题错误: {str(e)}")
    
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
    
    def speak(self, text):
        """语音合成[1,2,3](@ref)"""
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run).start()
    
    def speak_random(self, texts):
        """随机选择一段文本进行语音合成"""
        if texts:
            text = np.random.choice(texts)
            print("语音输出：",text)
            #self.speak(text)
    
    def start(self):
        """启动语音助手"""
        print("启动增强版语音助手（带降噪功能）")
        self.listening = True
        
        # 启动监听线程
        listen_thread = threading.Thread(target=self.run_listen)
        listen_thread.daemon = True
        listen_thread.start()

        print("语音助手已启动，降噪功能已启用")
    
    def stop(self):
        """停止语音助手"""
        self.running = False
        self.listening = False
        print("语音助手已停止")
        #self.speak("语音助手已停止")
    
    
    
# 使用示例
if __name__ == "__main__":
    import sys
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from vision.camera_manager  import  CameraManager
    from memory.memory_manager import MemoryManager

    memory_manager = MemoryManager()
    # 创建语音助手实例
    assistant = VoiceAssistant(memory_manager)
    
    # 测试音频增强功能
    assistant.test_enhancement()
    
    # 启动语音助手
    assistant.start()
    
    # 运行一段时间
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        print("语音助手已停止")