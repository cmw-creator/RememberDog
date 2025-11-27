#!/usr/bin/env python3
import logging

from src.track.person_follow_yolo import PersonFollower

logger = logging.getLogger(name='Log')
logger.info("开始加载语音助手")
import json
import time
import threading
import queue
import os
import re
#from multiprocessing import Process, Queue as MPQueue
import dashscope
from dashscope.audio.asr import VocabularyService, Recognition
import pyaudio
import numpy as np
from scipy.signal import resample_poly
#import pyttsx3
#from pyrnnoise import RNNoise
#from dashscope import Generation
# ========== 引入阿里云 DashScope SDK ==========
import dashscope
from openai import OpenAI
logger.info("语音助手导入完成")

# API Key 设置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "sk-f648e9bd318443998ae4df272d479aa0")

from dashscope.audio.asr import Recognition, RecognitionCallback
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class VoiceAssistant:
    def __init__(self, memory_manager ,robot_controller, cam_manager):
        self.memory_manager = memory_manager
        self.robot_controller = robot_controller
        # 音频参数(与 Paraformer 兼容)
        self.rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.play_mode = False
        # 音频增强参数
        #self.rnnoise = RNNoise(sample_rate=48000)
        self.agc_enabled = True
        self.noise_gate_threshold = 0.05
        self.agc_target_rms = 0.1
        self.max_gain = 5.0
        self.pos_state = False
        # TTS 合成队列(使用进程间队列)
        # self.speech_queue = MPQueue()
        #
        # # 启动独立 TTS 进程
        # self.tts_process = Process(target=self._tts_process_worker, args=(self.speech_queue,), daemon=True)
        # self.tts_process.start()

        # 运行状态
        self.running = True
        self.listening = True

        # 本地命令数据库
        self.commands_db = self.load_commands_db()

        # QA 管理器
        self.qa_manager = memory_manager.qa_manager
        self.conversation_history = []

        # OpenAI 客户端初始化
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", "sk-51a6a596303e466f9054520ec297af09"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        follower = PersonFollower(
            robot=robot_controller,
            camera_manager=cam_manager,
            model_path="yolov8n.pt"  # 确保本地能加载，或让 ultralytics 自动下载
        )
        self.follower_thread = threading.Thread(target=follower.run, name="Follower_Thread")


    def load_commands_db(self):
        """加载本地命令数据库"""
        path = "assets/commands.json"
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _tts_process_worker(queue):
        """独立进程的 TTS 工作线程 - 每次重新创建引擎"""
        import pyttsx3
        import time

        #print("[TTS进程] 已启动,等待任务...")
        logger.info("[TTS进程] 已启动,等待任务...")

        while True:
            try:
                text = queue.get()
                if text is None:
                    #print("[TTS进程] 收到退出信号")
                    logger.info("[TTS进程] 收到退出信号")
                    break

                #print(f"[TTS进程] 开始播放: {text}")
                logger.info(f"[TTS进程] 开始播放: {text}")

                # 每次都创建新引擎
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)

                engine.say(text)
                engine.runAndWait()

                # 显式销毁引擎
                del engine

                # 短暂延迟确保资源释放
                time.sleep(0.1)

                #print(f"[TTS进程] 播放完成")
                logger.info(f"[TTS进程] 播放完成")

            except Exception as e:
                #print(f"[TTS进程] 播放错误: {e}")
                logger.error(f"[TTS进程] 播放错误: {e}")
                import traceback
                traceback.print_exc()

    def speak(self, text):
        """将文本放入队列供 TTS 进程播放"""
        #print(f"[SPEAK] 要朗读: {text}")
        logger.info(f"[SPEAK] 要朗读: {text}")
        try:
            self.speech_queue.put(text, timeout=1)
            #print(f"[SPEAK] 已加入队列")
            logger.info(f"[SPEAK] 已加入队列")
        except Exception as e:
            #print(f"[SPEAK] 队列错误: {e}")
            logger.error(f"[SPEAK] 队列错误: {e}")

    def speak_random(self, texts):
        if texts:
            t = np.random.choice(texts)
            self.speak(t)

    ### ========== 音频增强 / 降噪 ========== ###
    def apply_audio_enhancement(self, audio_bytes: bytes) -> bytes:
        """RNNoise 降噪 + AGC + 噪声门"""
        if not audio_bytes:
            return audio_bytes

        x16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if x16.size == 0:
            return audio_bytes

        # 上采样到 48kHz
        x48 = resample_poly(x16.astype(np.float32), up=3, down=1).astype(np.int16)

        # RNNoise 降噪
        try:
            mono48 = x48.reshape(1, -1).astype(np.int16)
            denoised_frames = []
            for speech_prob, denoised in self.rnnoise.denoise_chunk(mono48):
                denoised_frames.append(denoised[0])
            x48_denoised = np.concatenate(denoised_frames).astype(np.int16) if denoised_frames else x48
        except Exception as e:
            #print(f"RNNoise 出错,回退: {e}")
            logger.error(f"RNNoise 出错,回退: {e}")
            x48_denoised = x48

        # 下采样回 16kHz
        y16 = resample_poly(x48_denoised.astype(np.float32), up=1, down=3)
        y = (y16 / 32768.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        # AGC
        if self.agc_enabled and y.size > 0:
            rms = np.sqrt(np.mean(y ** 2))
            if rms > 0:
                gain = min(self.agc_target_rms / rms, self.max_gain)
                y = np.clip(y * gain, -1.0, 1.0)

        # 噪声门
        if y.size > 0:
            rms2 = float(np.sqrt(np.mean(y ** 2)))
            if rms2 < self.noise_gate_threshold:
                y[:] = 0.0

        out16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
        return out16.tobytes()

    def calculate_audio_level(self, audio_data):
        """计算音频电平"""
        if len(audio_data) == 0:
            return 0
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        return np.sqrt(np.mean(audio_np ** 2))

    ### ========== Paraformer 流式语音识别 ========== ###
    class _ASRCallback(RecognitionCallback):
        """内部回调类,处理 Paraformer 实时识别结果"""

        def __init__(self, outer):
            self.outer = outer
            self.last_text = ""

        def on_open(self):
            #print("Paraformer 回调: on_open")
            logger.info("Paraformer 回调: on_open")

        def on_event(self, result):
            s = result.get_sentence()
            current_text = s.get("text", "").strip() if s is not None else ""
            #print("get_sentence:",current_text)
            logger.info(f"get_sentence: {current_text}")

            if not current_text:
                return

            # 如果和上次识别内容完全相同, 跳过
            if current_text == self.last_text:
                return

            # 只处理包含句末标点的完整句子
            if any(punct in current_text for punct in ['。', '?','？', '!','！']):
                # 提取所有完整句子(按标点分割)
                sentences = re.split(r'([。?？!！])', current_text)

                complete_sentences = []
                for i in range(0, len(sentences) - 1, 2):
                    if i + 1 < len(sentences):
                        sentence = sentences[i] + sentences[i + 1]
                        if sentence.strip():
                            complete_sentences.append(sentence.strip())

                # 找出“新出现”的完整句子（相对于上次）
                new_sentences = []
                for sentence in complete_sentences:
                    if sentence not in getattr(self, "handled_sentences", set()):
                        new_sentences.append(sentence)

                # 只处理未处理过的句子
                for sentence in new_sentences:
                    #print(f"识别到: {sentence}")
                    logger.info(f"识别到: {sentence}")
                    self.outer._on_recognized(sentence)

                # 更新已处理句子集合
                if not hasattr(self, "handled_sentences"):
                    self.handled_sentences = set()
                # self.handled_sentences.update(new_sentences)

            # 更新最近识别文本
            self.last_text = current_text
        def on_complete(self):
            #print("Paraformer 回调: on_complete")
            logger.info("Paraformer 回调: on_complete")
            self.last_text = ""

        def on_error(self, error):
            #print("Paraformer 回调: on_error:", error)
            logger.error(f"Paraformer 回调: on_error: {error}")
            self.last_text = ""

        def on_close(self):
            #print("Paraformer 回调: on_close")
            logger.info("Paraformer 回调: on_close")
            self.last_text = ""

    def _on_recognized(self, text: str):
        """识别到完整句子后的动作"""
        self.process_command(text)
    
    def parse_fine_control(self, text):
        """解析精细控制指令，支持时间和程度控制"""
        # 时间模式：前进5秒、左转0.5秒
        time_patterns = [
            (r'前进\s*(\d+\.?\d*)\s*秒', 'forward'),
            (r'往前\s*(\d+\.?\d*)\s*秒', 'forward'),
            (r'后退\s*(\d+\.?\d*)\s*秒', 'back'),
            (r'往后\s*(\d+\.?\d*)\s*秒', 'back'),  
            (r'左转\s*(\d+\.?\d*)\s*秒', 'turn_left'),
            (r'右转\s*(\d+\.?\d*)\s*秒', 'turn_right')
        ]
        
        # 程度模式：前进一点、右转一点
        degree_patterns = [
            (r'前进\s*(一点|一些|少许)', 'forward', 0.3),
            (r'往前\s*(一点|一些|少许)', 'forward', 0.3),
            (r'后退\s*(一点|一些|少许)', 'back', 0.3),
            (r'往后\s*(一点|一些|少许)', 'turn_left', 0.5),
            (r'右转\s*(一点|一些|少许)', 'turn_right', 0.5),
            (r'前进\s*(很多|大量)', 'forward', 2.0),
            (r'后退\s*(很多|大量)', 'back', 2.0)
        ]

        # 先尝试匹配时间模式
        for pattern, action in time_patterns:
            match = re.search(pattern, text)
            if match:
                duration = float(match.group(1))
                return action, duration
        
        # 再尝试匹配程度模式
        for pattern, action, default_duration in degree_patterns:
            if re.search(pattern, text):
                return action, default_duration

        return None, None
    def run_listen(self):
        """用 Paraformer SDK 做流式识别"""
        while True:
            try:
                service = VocabularyService()
                vocabulary_id = None

                # 1. 定义热词数据和目标模型
                target_model = "paraformer-realtime-v2"
                my_vocabulary_data = [
                    {"text": "小影", "weight": 5, "lang": "zh"},
                #    {"text": "前进", "weight": 4, "lang": "zh"},
                #    {"text": "后退", "weight": 4, "lang": "zh"},
                #    {"text": "左转", "weight": 4, "lang": "zh"},
                #    {"text": "右转", "weight": 4, "lang": "zh"},
                #    {"text": "往前走", "weight": 4, "lang": "zh"},
                #    {"text": "往后走", "weight": 4, "lang": "zh"},
                    {"text": "趴下", "weight": 4, "lang": "zh"}
                ]


                # vocabulary_id = service.create_vocabulary(
                #     prefix="command",  # 自定义前缀，用于分类管理
                #     target_model=target_model,
                #     vocabulary=my_vocabulary_data
                # )

                # tmp1 = len(service.list_vocabularies("command"))-1
                # list1 = service.list_vocabularies("command")
                vocabulary_id = service.list_vocabularies("command")[0]["vocabulary_id"]
                # for i in range(0, tmp1):
                #     service.delete_vocabulary(list1[i]["vocabulary_id"])



                callback = VoiceAssistant._ASRCallback(self)
                recognition = Recognition(
                    model="paraformer-realtime-v2",
                    format="pcm",
                    sample_rate=self.rate,
                    callback=callback,
                    language_hints=["zh"],
                    semantic_punctuation_enabled=False,
                    vocabulary_id=vocabulary_id,
                    max_sentence_silence=500
                )
                recognition.start()

                pa = pyaudio.PyAudio()
                stream = pa.open(format=self.format, channels=self.channels,
                                rate=self.rate, input=True,
                                frames_per_buffer=self.chunk,
                                input_device_index=0)
                #print("开始实时识别(Paraformer)")
                logger.info("开始实时识别(Paraformer)")
                while self.running and self.listening:
                    # print(self.memory_manager.runSpeaking.value)
                    # print(self.memory_manager.runSpeaking.is_set())
                    try:
                        raw = stream.read(self.chunk)
                    except Exception as e:
                        #print("stream read出错:", e)
                        logger.error(f"stream read出错: {e}")

                        
                    if not self.memory_manager.runSpeaking.is_set():
                        recognition.send_audio_frame(raw) #发送
                        # time.sleep(0.1)
            except Exception as e:
                logger.error(f"注意语音识别线程出错，尝试重启，错误：{e}，可能是网络连接问题")


    ### ========== 命令处理 ========== ###
    def process_command(self, text):
        """处理语音命令"""
        text = text.replace(" ", "")  # 去除所有空格
        if len(text)<= 3:
            return  # 太少的字就忽略调了

        ### QA记忆的相关代码写在这里 ###
        # 走问答知识库
        answer, score, audio_path, command = self.qa_manager.query(text, top_k=1, threshold=0.65)
        if command=="No Answer":
            #print(answer)
            logger.info(answer)
            return
        #print(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer},音频文件：{audio_path},动作：{command}")
        logger.info(f"Q&A 匹配分数: {score:.2f}, 初步答案: {answer},音频文件：{audio_path},动作：{command}")

        #if score< 75:
        #    return  # 太少的字就忽略调了

        # 交给生成式模型润色
        final_answer = answer

        # final_answer = self.generate_answer(text, answer)
        duration=0.5
        if score<=0.8 and len(text)<=10:
            fine_action,fine_duration= self.parse_fine_control(text)
            if fine_action is not None:
                command=fine_action
                duration=fine_duration

        for c in self.commands_db["commands"]:
            if c["action"]==command:
                self.execute_action(command["action"], text)
                break

        # 触发语音事件，兼容 speak_text / audio_file

        if self.play_mode:
            tmp = self.execute_action(command, text, duration)
            if tmp:
                audio_path = tmp
            event_payload = {
                "speak_text": final_answer,
                "timestamp": time.time(),
                "audio_file": audio_path
            }
            self.memory_manager.trigger_event("speak_event", event_payload)
        if command == "play on":
            self.play_mode = True
        elif command == "play off":
            self.play_mode = False

        print(f"最终回答: {final_answer}")
        # self.speak(final_answer)

    def execute_action(self, action, text ,fine_duration):
        """执行对应动作"""
        #print("动作时间：",fine_duration)
        logger.info(f"文本:{text},动作:{action},动作时间：{fine_duration}")
        if(fine_duration is None):
            fine_duration=0.5
        if action == "add_reminder":
            self.handle_add_reminder(text)
        elif action == "add_question":
            self.handle_add_question(text)
        elif action == "stand up":
            print("动作：狗站起来")
        elif action == "go round":
             print("动作：狗转个圈")
        elif action == "help":
            print("我可以帮您添加提醒、设置问题、控制机器狗行动")
        elif action == "stand up":
            print("命令：站起来")
            self.robot_controller.stand_up()
            return "assets/voice/dog/dog_ok.wav"
        elif action == "lie down":
            print("命令：趴下")
            self.robot_controller.stand_up()
            return "assets/voice/dog/dog_ok.wav"
        elif action == "turn left":
            print("命令：左转")
            self.robot_controller.move_turn_left_90(fine_duration)
            return "assets/voice/dog/dog_ok.wav"
        elif action == "turn right":
            print("命令：右转")
            self.robot_controller.move_turn_right_90(fine_duration)
            return "assets/voice/dog/dog_ok.wav"
        elif action == "forward":
            print("命令：前进")
            self.robot_controller.forward(fine_duration)
            return "assets/voice/dog/dog_ok.wav"
        elif action == "back":
            print("命令：后退")
            self.robot_controller.back(fine_duration)
            return "assets/voice/dog/dog_ok.wav"
        elif action == "give hand":
            print("命令：握手")
            self.robot_controller.give_hand()
            return "assets/voice/dog/dog_give_hand.wav"
        elif action == "track on":
            print("开启跟随")
            self.follower_thread.start()
            return "assets/voice/dog/dog_ok.wav"
        elif action == "track on":
            print("关闭跟随(暂时关不掉)")

            return "assets/voice/dog/dog_ok.wav"
        else:
            print(f"警告：无命令: {action}")

    def call_online_model(self, user_text):
        """使用 OpenAI 兼容接口调用通义千问"""
        try:
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                             {"role": "system",
                              "content": "你是一个友好的语音助手,请用简洁、自然的语气回答问题,回答尽量使用一句话"},
                         ] + self.conversation_history,
                timeout=30,
            )
            content = completion.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            print(f"调用生成式模型错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return "抱歉,我暂时无法回答您的问题。"

    ### ========== 时间解析等工具方法 ========== ###
    def parse_time_from_text(self, text):
        pass

    def handle_add_reminder(self, text):
        pass

    def handle_add_question(self, text):
        pass

    ### ========== 启动/停止 ========== ###
    def start(self):
        print("启动语音助手(使用 Paraformer 实时识别)")
        listen_thread = threading.Thread(target=self.run_listen, daemon=True)
        listen_thread.start()

    def stop(self):
        self.running = False
        self.listening = False
        try:
            self.speech_queue.put(None, timeout=1)
        except:
            pass
        print("语音助手已停止")
        if self.tts_process.is_alive():
            self.tts_process.terminate()
            self.tts_process.join(timeout=2)

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
    # qa_manager = QAManager()
    #创建语音引擎


    # 创建语音助手实例
    assistant = VoiceAssistant(memory_manager,robot_controller)

    #tmp = assistant.execute_action("","前进5秒")
    assistant.start()
    speech_engine = SpeechEngine(memory_manager)
    try:
        while True:
           time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        print("语音助手已停止")
