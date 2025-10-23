#!/usr/bin/env python3
import wave
from pyrnnoise import RNNoise
from scipy.signal import resample_poly
import threading
import json
import time
import os
import re
import numpy as np
import pyaudio
import pyttsx3
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
        self.rate = 16000

        # 音频增强参数
        self.noise_profile = None
        self.noise_profile_length = 2.0
        self.agc_factor = 5.0
        self.noise_reduction_enabled = True
        self.agc_enabled = True
        self.noise_gate_threshold = 0.05

        # === 重要：先初始化运行状态 ===
        self.running = True
        self.listening = True
        # ==============================

        # 初始化语音识别
        self.setup_voice_recognition()

        # === 修改：使用多进程队列管理 TTS ===
        self.speech_queue = MPQueue()  # 进程间队列
        self.tts_rate = 150
        self.tts_volume = 0.9

        # 启动独立 TTS 进程
        self.tts_process = Process(
            target=self._tts_process_worker,
            args=(self.speech_queue, self.tts_rate, self.tts_volume),
            daemon=True
        )
        self.tts_process.start()
        # ================================

        # 本地命令数据库
        self.commands_db = self.load_commands_db()

        # 初始化噪声配置文件
        self.initialize_noise_profile()

        # 启动音频质量监控
        self.start_quality_monitor()

        # 初始化 Q&A 管理器
        self.qa_manager = memory_manager.qa_manager
        self.use_online = False
        self.rnnoise = RNNoise(sample_rate=48000)

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

    # ========== TTS 进程工作函数 ==========
    @staticmethod
    def _tts_process_worker(queue, rate, volume):
        """独立进程的 TTS 工作线程 - 每次重新创建引擎"""
        import pyttsx3
        import time

        print("[TTS进程] 已启动,等待任务...")

        while True:
            try:
                text = queue.get()
                if text is None:
                    print("[TTS进程] 收到退出信号")
                    break

                print(f"[TTS进程] 开始播放: {text}")

                # 每次都创建新引擎
                engine = pyttsx3.init()
                engine.setProperty('rate', rate)
                engine.setProperty('volume', volume)

                engine.say(text)
                engine.runAndWait()

                # 显式销毁引擎
                del engine

                # 短暂延迟确保资源释放
                time.sleep(0.1)

                print(f"[TTS进程] 播放完成")

            except Exception as e:
                print(f"[TTS进程] 播放错误: {e}")
                import traceback
                traceback.print_exc()

    def speak(self, text):
        """将文本放入队列供 TTS 进程播放"""
        if not text:
            return
        print(f"[SPEAK] 添加到队列: {text}")
        try:
            self.speech_queue.put(text, timeout=1)
        except Exception as e:
            print(f"[SPEAK] 队列错误: {e}")

    def speak_random(self, texts):
        """随机选择一段文本进行语音合成"""
        if not texts:
            return
        text = np.random.choice(texts)
        print(f"[SPEAK] 随机选择并添加到队列: {text}")
        self.speak(text)

    # ========================================

    ### 音频增强和降噪 ###
    def initialize_noise_profile(self):
        """录制初始噪声样本用于噪声分析"""
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
        """RNNoise 前端降噪 + 平滑 AGC + 噪声门"""
        if not audio_data:
            return audio_data

        x16 = np.frombuffer(audio_data, dtype=np.int16)
        if x16.size == 0:
            return audio_data

        x48 = resample_poly(x16.astype(np.float32), up=3, down=1).astype(np.int16)

        try:
            mono48 = x48.reshape(1, -1).astype(np.int16)
            denoised_frames = []

            for speech_prob, denoised in self.rnnoise.denoise_chunk(mono48):
                denoised_frames.append(denoised[0])

            if denoised_frames:
                x48_denoised = np.concatenate(denoised_frames).astype(np.int16)
            else:
                x48_denoised = x48
        except Exception as e:
            print(f"RNNoise 降噪失败，回退到原始音频：{e}")
            x48_denoised = x48

        y16 = resample_poly(x48_denoised.astype(np.float32), up=1, down=3)
        y = (y16 / 32768.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.agc_enabled and y.size > 0:
            y = self.apply_agc(y)

        if y.size > 0:
            rms = float(np.sqrt(max(1e-12, np.mean(y ** 2))))
            if rms < self.noise_gate_threshold:
                y[:] = 0.0

        y = np.clip(y, -1.0, 1.0)
        out16 = (y * 32767.0).astype(np.int16)
        return out16.tobytes()

    def apply_agc(self, audio_data):
        """应用自动增益控制"""
        if len(audio_data) == 0:
            return audio_data

        rms = np.sqrt(np.mean(audio_data ** 2))

        if rms > 0:
            target_rms = 0.1
            gain = target_rms / rms
            max_gain = 5.0
            gain = min(gain, max_gain)
            audio_data = audio_data * gain
            audio_data = np.clip(audio_data, -1.0, 1.0)

        return audio_data

    def calculate_audio_level(self, audio_data):
        """计算音频电平"""
        if len(audio_data) == 0:
            return 0
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_np ** 2))

    def monitor_audio_quality(self):
        """监控音频质量并自动调整参数"""
        while self.running:
            try:
                time.sleep(5)
                if not self.listening:
                    continue

                sample = self.record_audio(duration=0.5)
                level = self.calculate_audio_level(sample)

                if level < 100:
                    self.agc_factor = 4.0
                elif level > 1000:
                    self.agc_factor = 1.5
                else:
                    self.agc_factor = 2.0

            except Exception as e:
                print(f"音频监控错误: {str(e)}")
                time.sleep(1)

    def start_quality_monitor(self):
        """启动音频质量监控"""
        monitor_thread = threading.Thread(target=self.monitor_audio_quality)
        monitor_thread.daemon = True
        monitor_thread.start()

    def record_audio(self, duration=5):
        """录制音频"""
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

        raw_level = self.calculate_audio_level(raw_audio)
        enhanced_level = self.calculate_audio_level(enhanced_audio)

        print(f"原始音频电平: {raw_level}")
        print(f"增强后音频电平: {enhanced_level}")
        print(f"增益: {enhanced_level / raw_level if raw_level > 0 else '无穷大'}")

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

    def test_speech_recognition(self):
        """测试语音识别功能"""
        print("测试语音识别功能...")

        try:
            raw_audio = self.record_audio(duration=3)
            print("原始音频录制完成，开始识别...")

            with wave.open("test_raw_audio.wav", 'wb') as f:
                f.setnchannels(self.channels)
                f.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
                f.setframerate(self.rate)
                f.writeframes(raw_audio)
            print("原始音频已保存为 test_raw_audio.wav")

            enhanced_audio = self.apply_audio_enhancement(raw_audio)
            recognized_text = self.recognize_speech_offline(enhanced_audio)

            if recognized_text:
                print(f"识别结果: {recognized_text}")
            else:
                print("未能识别语音内容")
        except Exception as e:
            print(f"语音识别测试失败: {str(e)}")

    def recognize_speech_offline(self, audio_data):
        """使用本地模型识别语音"""
        if self.recognizer is None:
            return None

        try:
            self.recognizer.Reset()

            chunk_size = 4000
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                self.recognizer.AcceptWaveform(chunk)

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

        silence_timeout = 1.2
        rms_voice_threshold = 250
        min_utter_duration = 0.60
        ema_alpha = 0.2

        self.recognizer.Reset()
        audio_index = 1
        utter_buf = bytearray()
        last_voice_ts = time.time()
        smoothed_rms = 0.0

        while self.running:
            if not self.listening:
                time.sleep(0.05)
                continue

            try:
                raw = stream.read(self.chunk)

                
                #enhanced = self.apply_audio_enhancement(raw)
                enhanced = raw

                if enhanced:
                    utter_buf += enhanced

                '''
                # 计算平滑 RMS（为空则当 0）
                if len(enhanced) > 0:
                    arr = np.frombuffer(enhanced, dtype=np.int16)
                    if arr.size == 0:
                        inst_rms = 0.0
                    else:
                        inst_rms = float(np.sqrt(max(1e-12, np.mean(arr.astype(np.float32) ** 2))))
                    smoothed_rms = (1 - ema_alpha) * smoothed_rms + ema_alpha * inst_rms
                    if smoothed_rms > rms_voice_threshold:
                        last_voice_ts = time.time()
                '''
                # 计算当前句已累积时长
                utt_dur = len(utter_buf) / (2 * self.rate)  # 2 字节/样本，单声道

                accepted = self.recognizer.AcceptWaveform(enhanced) if len(enhanced) > 0 else False
                if accepted and utt_dur >= min_utter_duration:
                    text = json.loads(self.recognizer.Result()).get("text", "").strip()

                    if len(utter_buf) > 0:
                        audio_index += 1

                    if text:
                        print(f"识别结果: {text}")
                        self.process_command(text)

                    utter_buf = bytearray()
                    self.recognizer.Reset()
                    last_voice_ts = time.time()
                    continue

                if (time.time() - last_voice_ts) > silence_timeout and len(utter_buf) > 0:
                    if utt_dur >= min_utter_duration:
                        final_text = json.loads(self.recognizer.FinalResult()).get("text", "").strip()

                        audio_index += 1

                        if final_text:
                            print(f"识别结果(超时结算): {final_text}")
                            self.process_command(final_text)

                    utter_buf = bytearray()
                    self.recognizer.Reset()
                    smoothed_rms = 0.0
                    last_voice_ts = time.time()

            except Exception as e:
                print(f"语音处理错误: {str(e)}")
                utter_buf = bytearray()

        stream.stop_stream()
        stream.close()
        p.terminate()

    def process_command(self, text):
        """处理语音命令"""
        text=text.replace(" ", "")#去除所有空格
        if(text<=3): 
            return #太少的字就忽略调了
        # 首先检查问候语
        for greeting in self.commands_db["responses"]["greeting"]:
            if greeting in text:
                self.speak_random(["你好！", "您好！", "嗨，有什么可以帮您？"])
                return

        command_matched = False
        for command in self.commands_db["commands"]:
            if re.search(command["pattern"], text):
                self.execute_action(command["action"], text)
                command_matched = True
                break

        if not command_matched:
            self.speak_random(self.commands_db["responses"]["unknown"])

        ### QA记忆的相关代码写在这里 ###
        # 走问答知识库
        if len(text)<2:
            pass
        answer, score, audio_path = self.qa_manager.query(text, top_k=1, threshold=0.5)
        print(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer},音频文件：{audio_path}")

        # 交给生成式模型润色
        final_answer=answer
        # 触发语音事件，兼容 speak_text / audio_file
        event_payload = {
            "speak_text": final_answer,
            "timestamp": time.time(),
            "audio_file":audio_path
        }
        self.memory_manager.trigger_event("speak_event", event_payload)
        # final_answer = self.generate_answer(text, answer)


        print(f"最终回答: {final_answer}")

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

    def handle_add_reminder(self, text):
        """处理添加提醒命令"""
        try:
            hour, minute, day_offset = self.parse_time_from_text(text)

            if hour is not None:
                time_str = f"{hour:02d}:{minute:02d}"

                event_match = re.search(r'提醒[我你]*([^，。！？]*)', text)
                if event_match:
                    event = event_match.group(1).strip()
                else:
                    event = text

                print(f"尝试添加提醒: {event} 在 {time_str}")
                self.memory_manager.add_reminder(time_str, event)
            else:
                print("未识别到时间信息，请说明具体时间")

        except Exception as e:
            print(f"添加提醒错误: {str(e)}")

    def handle_add_question(self, text):
        """处理添加问题命令"""
        try:
            question_match = re.search(r'问题[是：:]*([^，。！？]*)', text)
            answer_match = re.search(r'答案[是：:]*([^，。！？]*)', text)

            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()

                if self.memory_manager:
                    self.memory_manager.add_question(question, answer)
                print(f"已添加问题: {question}")
            else:
                print("发出声音：请说明问题和答案，例如：问题是我的生日，答案是1月1日")

        except Exception as e:
            print(f"添加问题错误: {str(e)}")

    def parse_time_from_text(self, text):
        """从文本中解析时间信息"""
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

        if "下午" in text and hour < 12:
            hour += 12
        elif "上午" in text and hour == 12:
            hour = 0

        day_offset = 0
        if "明天" in text:
            day_offset = 1
        elif "后天" in text:
            day_offset = 2

        return hour, minute, day_offset

    def start(self):
        """启动语音助手"""
        print("启动增强版语音助手（带降噪功能）")
        self.listening = True

        listen_thread = threading.Thread(target=self.run_listen)
        listen_thread.daemon = True
        listen_thread.start()

        print("语音助手已启动，降噪功能已启用")

    def stop(self):
        """停止语音助手"""
        print("[停止] 正在关闭语音助手...")
        self.running = False
        self.listening = False

        # 发送停止信号到 TTS 进程
        try:
            self.speech_queue.put(None, timeout=1)
        except:
            pass

        # 等待 TTS 进程结束
        if self.tts_process.is_alive():
            self.tts_process.terminate()
            self.tts_process.join(timeout=2)

        print("语音助手已停止")

    ### QA相关函数 ###
    def generate_answer(self, user_input, retrieved_answer):
        """结合知识库答案 + 生成式模型生成最终回答"""
        prompt = f"用户提问: {user_input}\n相关知识: {retrieved_answer}\n请用简洁友好的语气回答。"

        if self.use_online:
            try:
                import requests
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
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                model_id = "assets/Qwen3-0.6B-GPTQ-Int8"

                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True
                )

                self._local_pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto"
                )
                result = self._local_pipe(prompt, max_new_tokens=100)
                return result[0]["generated_text"].replace(prompt, "").strip()
            except Exception as e:
                print(f"本地生成失败，降级使用知识库: {e}")
                return retrieved_answer


# 使用示例
if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

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
    assistant = VoiceAssistant(memory_manager,robot_controller)

    ###
    '''
    while True:
        text = input("你说: ")
        if text.lower() in ["exit", "quit"]:
            break

        answer, score = assistant.qa_manager.query(text, top_k=1, threshold=0.5)
        print(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer}")

    assistant = VoiceAssistant(memory_manager, None)

    assistant.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        print("语音助手已停止")
