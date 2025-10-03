#!/usr/bin/env python3
import wave
# 本地语音助手 - 集成降噪和自动增益控制
from pyrnnoise import RNNoise
from scipy.signal import resample_poly  # 高质量有理数重采样
import threading
import json
import time
import os
import re
import numpy as np
import wave
import pyaudio
#import pyttsx3
from vosk import Model, KaldiRecognizer
import noisereduce as nr
from scipy import signal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from scipy.signal import resample
#from control import nav_client



class VoiceAssistant:
    def __init__(self, memory_manager,robot_controller):
        # 初始化记忆管理器
        self.memory_manager = memory_manager
        self.robot_controller = robot_controller
        
        # 语音识别模型路径
        self.model_path = "assets/voice_models/vosk-model-small-cn-0.22"
        
        # 音频参数
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.device_index = 0
        self.hw_rate = 16000   # 硬件录音采样率（麦克风一般支持）
        self.rate = 16000      # 语音识别采样率
        
        # 音频增强参数
        self.noise_profile = None
        self.noise_profile_length = 2.0  # 用于噪声分析的秒数
        self.agc_factor = 10.0  # 自动增益因子
        self.noise_reduction_enabled = True
        self.agc_enabled = True
        self.noise_gate_threshold = 0.02  # 噪声门限阈值
        
        # 初始化语音识别
        self.setup_voice_recognition()
        
        # 初始化语音合成
        #self.engine = pyttsx3.init()
        #self.engine.setProperty('rate', 150)
        #self.engine.setProperty('volume', 0.9)
        
        # 运行状态
        self.running = True
        self.listening = True
        
        # 本地命令数据库
        self.commands_db = self.load_commands_db()
        
        # 初始化噪声配置文件
        self.initialize_noise_profile()
        
        # 启动音频质量监控
        self.start_quality_monitor()


        # 初始化 Q&A 管理器，之后建议放在记忆管理器中
        self.qa_manager = memory_manager.qa_manager
        self.use_online = False   # True: 在线 deepseek / False: 本地 qwen3
        self.rnnoise = RNNoise(sample_rate=48000)  # RNNoise 按 48kHz 工作

        
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
        RNNoise 前端降噪 + 平滑 AGC + 噪声门
        - 入口：int16 bytes @ 16kHz
        - 内部：上采样到 48kHz，送 RNNoise（10ms 帧，480 样本/帧）
        - 出口：再下采样回 16kHz，按你原有流程做 AGC 与噪声门
        """
        if not audio_data:
            return audio_data

        # --- bytes -> int16 ndarray ---
        x16 = np.frombuffer(audio_data, dtype=np.int16)
        if x16.size == 0:
            return audio_data

        # --- 上采样到 48kHz（16k -> 48k = 1:3 upsample）---
        # 用 resample_poly 提升质量和稳定性
        x48 = resample_poly(x16.astype(np.float32), up=3, down=1).astype(np.int16)

        # --- 组帧并送入 RNNoise ---
        # RNNoise 的 Python API 支持一次性 chunk：shape [channels, samples]，int16
        # 你是单声道 -> [1, N]
        try:
            mono48 = x48.reshape(1, -1).astype(np.int16)
            denoised_frames = []

            # 逐帧产出：denoise_chunk 会按 480 样本帧返回
            for speech_prob, denoised in self.rnnoise.denoise_chunk(mono48):
                # denoised: shape [1, 480] int16
                denoised_frames.append(denoised[0])

            if denoised_frames:
                x48_denoised = np.concatenate(denoised_frames).astype(np.int16)
            else:
                x48_denoised = x48  # 回退
        except Exception as e:
            print(f"RNNoise 降噪失败，回退到原始音频：{e}")
            x48_denoised = x48

        # --- 下采样回 16kHz（48k -> 16k = 3:1 downsample）---
        y16 = resample_poly(x48_denoised.astype(np.float32), up=1, down=3)

        # --- 转 float 做后续处理 ---
        y = (y16 / 32768.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        # --- 你的 AGC（沿用原有实现） ---
        if self.agc_enabled and y.size > 0:
            y = self.apply_agc(y)

        # --- 噪声门（沿用你的全段 RMS，但建议后续替换成滞回门限/VAD）---
        if y.size > 0:
            rms = float(np.sqrt(max(1e-12, np.mean(y ** 2))))
            if rms < self.noise_gate_threshold:
                y[:] = 0.0

        # --- 回写 int16 bytes ---
        y = np.clip(y, -1.0, 1.0)
        out16 = (y * 32767.0).astype(np.int16)
        return out16.tobytes()
    
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
            max_gain = 10.0
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
                        rate=self.hw_rate,
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


    def downsample_to_16k(self, audio_bytes):
        x48 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        # 48k -> 16k (down=3)
        y16 = resample_poly(x48, up=1, down=3)
        return (y16.astype(np.int16)).tobytes()
    def test_speech_recognition(self):
        """测试语音识别功能"""
        print("测试语音识别功能...")

        # 录制音频并进行识别
        try:
            # 录制一段原始音频
            raw_audio = self.record_audio(duration=3)
            print("原始音频录制完成，开始识别...")

            # 保存原始音频到本地
            with wave.open("test_raw_audio.wav", 'wb') as f:
                f.setnchannels(self.channels)
                f.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
                f.setframerate(self.rate)
                f.writeframes(raw_audio)
            print("原始音频已保存为 test_raw_audio.wav")

            # 应用音频增强（如果有的话）
            enhanced_audio = self.apply_audio_enhancement(raw_audio)

            #audio_data_16k = self.downsample_to_16k(enhanced_audio)
            recognized_text = self.recognize_speech_offline(enhanced_audio)
            # 使用增强后的音频进行语音识别
            #recognized_text = self.recognize_speech_offline(enhanced_audio)

            if recognized_text:
                print(f"识别结果: {recognized_text}")
            else:
                print("未能识别语音内容")
        except Exception as e:
            print(f"语音识别测试失败: {str(e)}")
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
        """持续监听语音指令（AcceptWaveform 分段，低敏版）"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.hw_rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        print("语音助手已启动，等待指令...")

        # --- 可调参数（降低敏感度） ---
        silence_timeout = 1.2  # 句尾静音超时（秒），越大越不易截断
        rms_voice_threshold = 50  # 能量阈值（int16 RMS），越大越不敏感
        min_utter_duration = 0.60  # 一句话最短时长（秒），短于此不立刻结算
        ema_alpha = 0.2  # RMS 指数平滑系数（0~1），越小越平滑
        # ---------------------------

        self.recognizer.Reset()
        audio_index = 1
        utter_buf = bytearray()
        last_voice_ts = time.time()
        smoothed_rms = 0.0

        # 辅助：保存 wav
        def _save_wav(b, idx, tag=""):
            name = f"log/enhanced_segment_{idx}{tag}.wav"
            with wave.open(name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b)
            print(f"已保存分段音频: {name}")

        while self.running:
            if not self.listening:
                time.sleep(0.05)
                continue

            try:
                raw = stream.read(self.chunk)

                
                enhanced = self.apply_audio_enhancement(raw)

                # 累积增强后的音频到当前句
                if enhanced:
                    utter_buf += enhanced

                # 计算平滑 RMS（为空则当 0）
                if len(enhanced) > 0:
                    arr = np.frombuffer(enhanced, dtype=np.int16)
                    # 防 NaN
                    if arr.size == 0:
                        inst_rms = 0.0
                    else:
                        inst_rms = float(np.sqrt(max(1e-12, np.mean(arr.astype(np.float32) ** 2))))
                    smoothed_rms = (1 - ema_alpha) * smoothed_rms + ema_alpha * inst_rms
                    if smoothed_rms > rms_voice_threshold:
                        last_voice_ts = time.time()

                # 计算当前句已累积时长
                utt_dur = len(utter_buf) / (2 * self.rate)  # 2 字节/样本，单声道

                # 1) 首选：由 Vosk 判断一句话结束
                accepted = self.recognizer.AcceptWaveform(enhanced) if len(enhanced) > 0 else False
                if accepted and utt_dur >= min_utter_duration:
                    text = json.loads(self.recognizer.Result()).get("text", "").strip()

                    if len(utter_buf) > 0:
                        #_save_wav(utter_buf, audio_index)
                        audio_index += 1

                    if text:
                        print(f"识别结果: {text}")
                        self.process_command(text)

                    utter_buf = bytearray()
                    self.recognizer.Reset()
                    last_voice_ts = time.time()
                    continue  # 进入下一轮

                # 2) 兜底：句尾静音超时（防止迟迟不结束或被截断）
                if (time.time() - last_voice_ts) > silence_timeout and len(utter_buf) > 0:
                    # 仅当满足最短句时长才结算，避免半句被截断
                    if utt_dur >= min_utter_duration:
                        final_text = json.loads(self.recognizer.FinalResult()).get("text", "").strip()

                        #_save_wav(utter_buf, audio_index, tag="_timeout")
                        audio_index += 1

                        if final_text:
                            print(f"识别结果(超时结算): {final_text}")
                            self.process_command(final_text)

                    # 无论是否输出文本，都重置缓冲，避免越积越多
                    utter_buf = bytearray()
                    self.recognizer.Reset()
                    smoothed_rms = 0.0
                    last_voice_ts = time.time()

            except Exception as e:
                print(f"语音处理错误: {str(e)}")
                utter_buf = bytearray()  # 出错时清空当前句缓存

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

        ### QA记忆的相关代码写在这里 ###
        # 走问答知识库
        answer, score = self.qa_manager.query(text, top_k=1, threshold=0.5)
        print(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer}")

        # 交给生成式模型润色
        final_answer=answer
        # 触发语音事件，兼容 speak_text / audio_file
        event_payload = {
            "speak_text": final_answer,
            "timestamp": time.time(),
            "audio_file":""
        }
        self.memory_manager.trigger_event("speak_event", event_payload)
        # final_answer = self.generate_answer(text, answer)


        print(f"最终回答: {final_answer}")
        #self.speak(final_answer)
    
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
        elif action == "goto":
            print("开始导航去XXX")
            pass #这里写真的导航
            #nav_client.go_to(1.0, 2.0, 0.0)
        elif action == "come":
            self.robot_controller.forward(0.3)
        elif action == "leave":
            self.robot_controller.back(0.3)
        else:
            print(f"执行命令: {action}")
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
            else:
                print("未识别到时间信息，请说明具体时间")
                
        except Exception as e:
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
                print(f"已添加问题: {question}")
            else:
                print("发出声音：请说明问题和答案，例如：问题是我的生日，答案是1月1日")
                
        except Exception as e:
            
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
            print("说话：",text)
            #self.engine.say(text)
            #self.engine.runAndWait()
        threading.Thread(target=run).start()
    
    def speak_random(self, texts):
        """随机选择一段文本进行语音合成"""
        if texts:
            text = np.random.choice(texts)
            print("语音输出：",text)
            #self.speak(text)
    
    def generate_answer(self, user_input, retrieved_answer=""):
        prompt = f"用户提问: {user_input}相关知识: {retrieved_answer}请用简洁友好的语气回答："
        answer = self.llama_cli.generate(prompt)
        return answer

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
        self.llama_cli.stop()
        print("语音助手已停止")
        #self.speak("语音助手已停止")


    ### QA相关函数 ###
    def generate_answer2(self, user_input, retrieved_answer):
        """结合知识库答案 + llama.cpp 生成最终回答"""
        prompt = f"你是一只阿尔茨海默病患者的辅助治疗机器狗，你的名字是小影。\n用户提问: {user_input}\n参考回答: {retrieved_answer}\n请用简洁友好的语气回答。<think> </think>"
        #prompt = f"用户提问:{user_input}\n请用简洁准确的方式回答这个问题。只需要给出核心答案，不需要解释或扩展。<think> </think>"
        print("prompt:",prompt)
        if self.use_online:
            # 在线 deepseek API（保持不变）
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
                    json={
                        "model": "deepseek-v3",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                    },
                )
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"在线生成失败，降级使用知识库: {e}")
                return retrieved_answer
        else:
            # 本地调用 qwen3 (示例，用 transformers pipeline)
            self._local_pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                # 添加以下关键参数
                max_new_tokens=20,  # 控制生成文本的最大长度
                do_sample=True,     # 启用采样，与temperature和top_p配合使用
                eos_token_id=self.tokenizer.eos_token_id,  # 设置结束符，帮助模型正确断句
                pad_token_id=self.tokenizer.eos_token_id   # 某些模型可能需要设置pad_token
            )
            result = self._local_pipe(prompt, max_new_tokens=50)
            print(result)
            return result[0]["generated_text"]
            return result[0]["generated_text"].replace(prompt, "").strip()
    ### QA相关函数结束 ###
# 使用示例
if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from vision.camera_manager import CameraManager
    from memory.memory_manager import MemoryManager
    from memory.qa_manager import QAManager
    from speech.speech_engine import SpeechEngine
    from control.control import RobotController

    memory_manager = MemoryManager()
    robot_controller = RobotController()
    qa_manager = QAManager()
    #创建语音引擎
    speech_engine = SpeechEngine(memory_manager)

    # 创建语音助手实例
    assistant = VoiceAssistant(memory_manage,robot_controllerr)

    ###
    '''
    while True:
        text = input("你说: ")
        if text.lower() in ["exit", "quit"]:
            break

        answer, score = assistant.qa_manager.query(text, top_k=1, threshold=0.5)
        print(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer}")

        # 交给生成式模型润色
        final_answer=answer
        #final_answer = assistant.generate_answer2(text, answer)
        print(f"最终回答: {final_answer}")

    assistant.stop()
    '''
    ###
    # 测试音频增强功能
    # assistant.test_enhancement()
    
    # 启动语音助手
    assistant.start()

    # 运行一段时间
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        print("语音助手已停止")