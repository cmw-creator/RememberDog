#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版本地语音助手（单文件示例）
功能：
 - 保留并改进音频降噪/AGC
 - 增强 NLU：本地规则 + 可选 OpenAI 用于意图/实体识别
 - 多轮对话支持：会话历史、上下文、短期记忆（用 memory_manager 持久化）
 - 智能响应生成：本地模板 + OpenAI 回答作为回退或增强
 - 中文注释
"""

import os
import sys
import json
import re
import time
import threading
import numpy as np

# 音频与语音识别相关
import pyaudio
import wave
import pyttsx3
from vosk import Model, KaldiRecognizer

# 音频增强
import noisereduce as nr

# 可选：OpenAI 集成（仅当用户提供 API key）
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# --------------------------
# 简易内存管理（如果用户已有 memory_manager 可替换）
# --------------------------
class SimpleMemoryManager:
    """一个极简的内存管理器实现，用于示例和测试"""
    def __init__(self, filename="memory_store.json"):
        self.filename = filename
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.store = json.load(f)
        else:
            self.store = {"reminders": [], "qa": []}
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.store, f, ensure_ascii=False, indent=2)
    
    def add_reminder(self, time_str, event):
        self.store["reminders"].append({"time": time_str, "event": event})
        self.save()
    
    def list_reminders(self):
        return self.store["reminders"]
    
    def add_question(self, q, a):
        self.store["qa"].append({"q": q, "a": a})
        self.save()
    
    def query_question(self, q):
        # 简单匹配已存问题
        for item in self.store["qa"]:
            if item["q"] == q:
                return item["a"]
        return None

# --------------------------
# 对话管理器（多轮会话状态）
# --------------------------
class DialogueManager:
    """
    管理会话历史和上下文（短期上下文）。
    - 每个用户会话一个会话ID（此示例用单一会话）
    - 支持保存最近 N 条消息作为上下文
    """
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = []  # 存储 (role, text) tuples, role in {"user","assistant","system"}
        self.slots = {}    # 用于存储槽位（例如提醒时间、提醒内容）
    
    def add_user(self, text):
        self._add("user", text)
    
    def add_assistant(self, text):
        self._add("assistant", text)
    
    def add_system(self, text):
        self._add("system", text)
    
    def _add(self, role, text):
        self.history.append((role, text))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self):
        """返回拼接的上下文字符串（供本地规则或 API 使用）"""
        lines = []
        for role, text in self.history:
            prefix = "用户：" if role=="user" else ("系统：" if role=="system" else "助理：")
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)
    
    def reset(self):
        self.history = []
        self.slots = {}

# --------------------------
# NLU 模块：本地规则 + OpenAI（可选）
# --------------------------
class NLU:
    """
    任务：
     - 提取意图(intent) 和 实体(entities)
     - 本地实现为规则/正则；当 OpenAI 可用且配置为使用时，向 OpenAI 发送简短提示来更准确识别
    返回：
     dict: {"intent": "intent_name", "score": float, "entities": {k: v}}
    """
    def __init__(self, use_openai=False):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        if self.use_openai:
            # 需在环境变量中设置 OPENAI_API_KEY，示例如: export OPENAI_API_KEY="sk-..."
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        # 本地正则意图及实体
        self.intent_patterns = [
            ("add_reminder", r"(提醒|设置提醒|记得).*"),
            ("query_reminders", r"(查看提醒|有哪些提醒|提醒列表)"),
            ("add_qa", r"(问题|记住).*答案.*"),
            ("stop_listen", r"(停止|别听)"),
            ("start_listen", r"(开始|继续)"),
            ("greeting", r"^(你好|您好|嗨|早上好|下午好)"),
            ("ask_question", r"(什么是|怎么|如何|为什么|多少)"),
            # 更多意图可在这里添加
        ]
        # 时间识别正则（简化）
        self.time_patterns = [
            r'(?:(明天|后天|今天)\s*)?(\d{1,2})点(?:(\d{1,2})分)?',
            r'(\d{1,2}):(\d{2})'
        ]
    
    def parse_local(self, text):
        text_low = text.strip()
        # 匹配意图（按优先级）
        for intent, pat in self.intent_patterns:
            if re.search(pat, text_low):
                entities = {}
                # 提取时间
                for tp in self.time_patterns:
                    m = re.search(tp, text_low)
                    if m:
                        entities['time'] = m.groups()
                        break
                # 尝试提取提醒事件（中文"提醒"后的内容）
                m = re.search(r'提醒[我你]*([^，。！？]*)', text_low)
                if m:
                    entities['event'] = m.group(1).strip()
                return {"intent": intent, "score": 0.5, "entities": entities}
        # 未匹配到，默认返回 None 意图
        return {"intent": "unknown", "score": 0.0, "entities": {}}
    
    def parse_with_openai(self, text):
        """
        使用 OpenAI 进行 NLU（可选）：
        - 发送一个简短 prompt 请 API 返回 JSON: {"intent":"...","entities":{...}}
        注意：这部分是可选的，且受限于网络/API 配额。
        """
        if not self.use_openai:
            return None
        # 简短 Prompt，指导模型输出 JSON
        prompt = (
            "你是一个意图和实体解析器。"
            "将下列用户输入解析为 JSON，格式："
            '{"intent":"...", "score":float, "entities": {"time":"", "event":""}}。\n'
            "只返回 JSON，不要其他额外文本。\n\n"
            f"用户输入：{text}\n"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 如果不可用，可改为 gpt-3.5-turbo 或用户可修改
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=200
            )
            content = resp['choices'][0]['message']['content'].strip()
            # 尝试解析 JSON（容错）
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            print("OpenAI NLU 调用失败:", e)
            return None
    
    def parse(self, text, use_openai_for_this=False):
        """主解析函数：优先尝试 OpenAI（若允许），否则本地解析"""
        if use_openai_for_this and self.use_openai:
            parsed = self.parse_with_openai(text)
            if parsed:
                return parsed
        return self.parse_local(text)

# --------------------------
# 响应生成模块：本地模板 + OpenAI 回答（可选）
# --------------------------
class ResponseGenerator:
    """
    根据意图+实体生成响应。
    - 若可用，使用 OpenAI 作为生成器（带会话上下文）
    - 否则使用本地模板或 memory_manager 回答
    """
    def __init__(self, memory_manager=None, dialogue_manager=None, use_openai=False):
        self.memory = memory_manager
        self.dialogue = dialogue_manager
        self.use_openai = use_openai and OPENAI_AVAILABLE
        if self.use_openai:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
    
    def generate_local(self, intent, entities, raw_text):
        """本地生成策略（规则与模板）"""
        if intent == "greeting":
            return "你好！有什么我可以帮助您的？"
        if intent == "add_reminder":
            # 从实体提取时间和事件
            time_str = None
            if 'time' in entities and entities['time']:
                # entities['time'] 可能为 regex groups，做简单格式化
                g = entities['time']
                try:
                    if isinstance(g, tuple):
                        # 处理 ('明天','9','30') 或 ('','9','') 等
                        day_marker = g[0] if g[0] else ""
                        hour = g[1] if len(g) > 1 and g[1] else None
                        minute = g[2] if len(g) > 2 and g[2] else '00'
                        if hour:
                            time_str = f"{day_marker}{hour}:{minute}"
                except Exception:
                    time_str = None
            # 事件
            event = entities.get('event') or raw_text
            if time_str is None:
                # 未解析到时间，提示用户补充（开启多轮槽位对话）
                return {"followup":"ask_time", "text":"好的，你想什么时候提醒我？（例如：明天9点）", "save": {"event": event}}
            else:
                # 保存提醒
                if self.memory:
                    # 对 memory_manager 的 API 假设为 add_reminder(time_str,event)
                    try:
                        self.memory.add_reminder(time_str, event)
                        return f"已为你设置提醒：{event}，时间：{time_str}"
                    except Exception:
                        return "保存提醒时发生错误，请稍后重试。"
                else:
                    return f"（未保存）提醒：{event} -> {time_str}"
        if intent == "query_reminders":
            if self.memory:
                rems = self.memory.list_reminders()
                if not rems:
                    return "当前没有提醒。"
                lines = ["你当前的提醒有："]
                for r in rems:
                    lines.append(f"- {r.get('time')} ： {r.get('event')}")
                return "\n".join(lines)
            else:
                return "未找到提醒存储器。"
        if intent == "add_qa":
            # attempt to parse question/answer
            m_q = re.search(r'问题[是：:]*([^，。！？]*)', raw_text)
            m_a = re.search(r'答案[是：:]*([^，。！？]*)', raw_text)
            if m_q and m_a:
                q = m_q.group(1).strip()
                a = m_a.group(1).strip()
                if self.memory:
                    self.memory.add_question(q, a)
                    return f"已记住：问题“{q}”，答案是“{a}”。"
                else:
                    return "已记住（内存未启用）。"
            else:
                return "请按格式说：问题是...，答案是...。"
        if intent == "ask_question":
            # 判断是否 memory 中已有答案
            if self.memory:
                ans = self.memory.query_question(raw_text)
                if ans:
                    return ans
            # 默认通用回应
            return "这是个好问题。我现在还不能完全回答所有问题，可以试试给我更多信息，或者开启 OpenAI 回答来得到更智能的回复。"
        if intent == "unknown":
            return "抱歉，我没听清楚。你可以换一种说法，或者让我使用更强的理解能力（OpenAI）。"
        # 默认
        return "我听到了，但不确定如何处理。"
    
    def generate_with_openai(self, intent, entities, raw_text):
        """使用 OpenAI 生成回答（包含会话上下文）"""
        if not (self.use_openai and OPENAI_AVAILABLE):
            return None
        # 构建 messages，包含对话上下文
        context = self.dialogue.get_context() if self.dialogue else ""
        system_prompt = (
            "你是一个中文的智能语音助手。用户可能会设置提醒、提问题或进行对话。"
            "如果用户在请求保存信息，请返回友好的确认。"
        )
        messages = [
            {"role":"system", "content": system_prompt},
        ]
        # 加入最近上下文（尽量不要超过 token 限制）
        if context:
            messages.append({"role":"system", "content": "历史上下文：\n" + context})
        # 加入当前 user
        messages.append({"role":"user", "content": raw_text})
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.6,
                max_tokens=400
            )
            text = resp['choices'][0]['message']['content'].strip()
            return text
        except Exception as e:
            print("OpenAI 生成失败:", e)
            return None
    
    def generate(self, intent, entities, raw_text, prefer_openai=False):
        """主入口：根据配置决定是否尝试 OpenAI"""
        if prefer_openai and self.use_openai:
            res = self.generate_with_openai(intent, entities, raw_text)
            if res:
                return res
        # 本地生成（并支持返回 followup 指令用于多轮）
        return self.generate_local(intent, entities, raw_text)

# --------------------------
# Voice Assistant（改进）
# --------------------------
class VoiceAssistant:
    def __init__(self, memory_manager=None, use_openai_nlu=False, use_openai_generate=False):
        # 记忆管理器
        self.memory_manager = memory_manager or SimpleMemoryManager()
        # 对话管理器
        self.dialogue = DialogueManager(max_history=12)
        # NLU & 生成器
        self.nlu = NLU(use_openai=use_openai_nlu)
        self.generator = ResponseGenerator(memory_manager=self.memory_manager,
                                           dialogue_manager=self.dialogue,
                                           use_openai=use_openai_generate)
        
        # 音频/识别模型路径（保留你原来的默认）
        self.model_path = "assets/voice_models/vosk-model-small-cn-0.22"
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
        # 音频增强参数
        self.noise_profile = None
        self.noise_profile_length = 2.0
        self.noise_reduction_enabled = True
        self.agc_enabled = True
        self.noise_gate_threshold = 0.02
        
        # tts 引擎
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # 运行状态
        self.running = True
        self.listening = True
        
        # 加载本地命令数据库（若存在）
        self.commands_db = self.load_commands_db()
        
        # 初始化识别器
        self.setup_voice_recognition()
        # 收集初始噪声
        self.initialize_noise_profile()
        # 启动监控线程
        self.start_quality_monitor()
    
    # ----------------------
    # 初始化 / 资源加载
    # ----------------------
    def load_commands_db(self):
        """加载本地命令配置文件，若不存在则返回默认"""
        commands_path = "assets/voice_commands/commands.json"
        if os.path.exists(commands_path):
            try:
                with open(commands_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        # 默认结构（极简）
        return {
            "commands": [
                {"pattern": "提醒", "action": "add_reminder"},
                {"pattern": "查看提醒", "action": "query_reminders"},
                {"pattern": "问题", "action": "add_question"},
                {"pattern": "停止", "action": "stop"},
                {"pattern": "开始", "action": "start"},
            ],
            "responses": {
                "greeting": ["你好", "您好", "嗨"],
                "unknown": ["抱歉，我没有理解。"]
            }
        }
    
    def setup_voice_recognition(self):
        """初始化 VOSK 语音识别模型（本地）"""
        try:
            if os.path.exists(self.model_path):
                self.model = Model(self.model_path)
                self.recognizer = KaldiRecognizer(self.model, self.rate)
                print("本地语音识别模型加载成功")
            else:
                print("语音模型未找到，转为文本输入测试模式")
                self.model = None
                self.recognizer = None
        except Exception as e:
            print("语音识别初始化异常:", e)
            self.model = None
            self.recognizer = None
    
    # ----------------------
    # 音频采集与增强（保留原有逻辑并微调）
    # ----------------------
    def initialize_noise_profile(self):
        """尝试捕获一小段背景噪声用于降噪"""
        try:
            # 若无法采集音频（无设备或在测试模式），忽略
            noise = self.record_audio(duration=1.0)
            if noise:
                arr = np.frombuffer(noise, dtype=np.int16).astype(np.float32) / 32768.0
                self.noise_profile = arr
                print("已采集噪声样本")
        except Exception as e:
            print("噪声样本采集失败:", e)
            self.noise_profile = None
    
    def apply_audio_enhancement(self, audio_data_bytes):
        """执行降噪 + 简单 AGC；返回 bytes"""
        if not audio_data_bytes:
            return audio_data_bytes
        try:
            audio_np = np.frombuffer(audio_data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if self.noise_reduction_enabled and self.noise_profile is not None and len(audio_np) > 0:
                try:
                    audio_np = nr.reduce_noise(y=audio_np, y_noise=self.noise_profile, sr=self.rate, prop_decrease=0.8)
                except Exception as e:
                    print("降噪异常:", e)
            if self.agc_enabled and len(audio_np) > 0:
                rms = np.sqrt(np.mean(audio_np**2))
                if rms > 0:
                    target = 0.1
                    gain = min(5.0, target / rms)
                    audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            # 噪声门
            rms2 = np.sqrt(np.mean(audio_np**2))
            if rms2 < self.noise_gate_threshold:
                audio_np = np.zeros_like(audio_np)
            out = (audio_np * 32767).astype(np.int16).tobytes()
            return out
        except Exception as e:
            print("音频增强失败:", e)
            return audio_data_bytes
    
    def record_audio(self, duration=1.0):
        """录制音频（若没有设备则返回空 bytes）"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)
            frames = []
            frames_count = int(self.rate / self.chunk * duration)
            # frames_count 有可能为 0，请防护
            if frames_count < 1:
                frames_count = 1
            for _ in range(frames_count):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            return b''.join(frames)
        except Exception as e:
            # 发生错误（例如测试环境或没有麦克风），返回空
            # print("录音失败:", e)
            return b''
    
    # ----------------------
    # 音频质量监控（后台）
    # ----------------------
    def monitor_audio_quality(self):
        while self.running:
            try:
                time.sleep(5)
                if not self.listening:
                    continue
                s = self.record_audio(duration=0.5)
                if not s:
                    continue
                level = np.sqrt(np.mean(np.frombuffer(s, dtype=np.int16).astype(np.float32)**2))
                # 简单调整 AGC 参数（演示）
                if level < 100:
                    self.agc_enabled = True
                # 可扩展：根据环境自动调节其他参数
            except Exception as e:
                print("监控线程错误:", e)
                time.sleep(1)
    
    def start_quality_monitor(self):
        t = threading.Thread(target=self.monitor_audio_quality, daemon=True)
        t.start()
    
    # ----------------------
    # 识别与处理流程
    # ----------------------
    def recognize_speech_offline(self, audio_bytes):
        """如果存在 VOSK 模型则使用本地识别"""
        if not self.recognizer:
            return None
        try:
            self.recognizer.Reset()
            chunk_size = 4000
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))
                self.recognizer.AcceptWaveform(chunk)
            result = json.loads(self.recognizer.FinalResult())
            return result.get('text', '')
        except Exception as e:
            print("本地识别异常:", e)
            return None
    
    def tts(self, text):
        """语音合成（异步）"""
        def run_tts(s):
            try:
                self.engine.say(s)
                self.engine.runAndWait()
            except Exception as e:
                print("TTS 异常:", e)
        threading.Thread(target=run_tts, args=(text,), daemon=True).start()
    
    def process_text(self, text):
        """
        主流程：接收文本（来自识别或调试输入）
         - 将输入放入对话历史
         - NLU 解析意图与实体
         - 生成响应（本地或 OpenAI）
         - 处理 follow-up（多轮），保存槽位到 dialogue.slots
        """
        text = text.strip()
        if not text:
            return
        print("用户:", text)
        self.dialogue.add_user(text)
        
        # 如果之前有 followup slot（对话管理器保存）
        if self.dialogue.slots.get("expecting") == "time_for_reminder":
            # 处理补全的时间槽
            # 将时间写入 memory 并确认
            hour_match = re.search(r'(\d{1,2})[:点](\d{1,2})?', text)
            if hour_match:
                h = hour_match.group(1)
                m = hour_match.group(2) or "00"
                # 事件存储在 slots.event
                event = self.dialogue.slots.get("pending_event", "提醒事项")
                time_str = f"{h}:{m}"
                try:
                    self.memory_manager.add_reminder(time_str, event)
                    reply = f"好的，已为你设置提醒：{event}，时间：{time_str}"
                except Exception:
                    reply = "保存提醒失败，请稍后再试。"
                self.dialogue.add_assistant(reply)
                self.tts(reply)
                # 清除槽位
                self.dialogue.slots.pop("expecting", None)
                self.dialogue.slots.pop("pending_event", None)
                return
            else:
                reply = "没听清时间，请以类似“明天9点”或“9:30”这样的格式告诉我时间。"
                self.dialogue.add_assistant(reply)
                self.tts(reply)
                return
        
        # 先尝试本地 NLU（优先使用本地规则），若不确定可尝试 OpenAI NLU（视配置）
        nlu_result = self.nlu.parse(text, use_openai_for_this=False)
        intent = nlu_result.get("intent")
        entities = nlu_result.get("entities", {})
        score = nlu_result.get("score", 0.0)
        
        # 如果本地解析为 unknown 且 OpenAI 可用，则尝试 OpenAI NLU
        if intent == "unknown" and self.nlu.use_openai:
            oa = self.nlu.parse(text, use_openai_for_this=True)
            if oa and "intent" in oa:
                intent = oa.get("intent", intent)
                entities = oa.get("entities", entities)
                score = oa.get("score", score)
        
        # 交给生成模块
        # 如果是添加提醒且 generator 返回要求补充时间（followup），则进入多轮
        gen = self.generator.generate(intent, entities, text, prefer_openai=self.generator.use_openai)
        # generator 可能返回 dict 表示有 followup
        if isinstance(gen, dict) and gen.get("followup") == "ask_time":
            # 保存 pending 事件到槽位
            self.dialogue.slots["expecting"] = "time_for_reminder"
            self.dialogue.slots["pending_event"] = gen.get("save", {}).get("event", text)
            reply = gen.get("text", "请告诉我时间。")
            self.dialogue.add_assistant(reply)
            self.tts(reply)
            return
        # 普通文本响应
        reply_text = gen if isinstance(gen, str) else str(gen)
        self.dialogue.add_assistant(reply_text)
        print("助理:", reply_text)
        self.tts(reply_text)
    
    # ----------------------
    # 运行与输入接口
    # ----------------------
    def run_listen(self):
        """
        主监听循环。出于开发/测试便利，默认启用文本输入模式（你也可以启用实际麦克风）。
        在实际部署中可将 `TEXT_MODE` 设为 False 使用麦克风。
        """
        TEXT_MODE = True  # 调试时建议 True。上线时设为 False。
        if TEXT_MODE:
            print("文本测试模式：请输入要说的话（Ctrl-C 退出）")
            try:
                while self.running:
                    text = input("你：")
                    if not text:
                        continue
                    self.process_text(text)
            except KeyboardInterrupt:
                print("退出文本模式")
                self.stop()
            return
        
        # 真实麦克风模式（若启用）
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        buffer = bytes()
        buffer_duration = 1.0
        buffer_size = int(self.rate * buffer_duration * 2)
        while self.running:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                buffer += data
                if len(buffer) >= buffer_size:
                    enhanced = self.apply_audio_enhancement(buffer)
                    text = None
                    if self.recognizer:
                        text = self.recognize_speech_offline(enhanced)
                    if text and text.strip():
                        self.process_text(text)
                    buffer = bytes()
            except Exception as e:
                print("监听异常:", e)
                buffer = bytes()
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def start(self):
        print("启动语音助手（支持多轮对话与可选 OpenAI 集成）")
        t = threading.Thread(target=self.run_listen, daemon=True)
        t.start()
    
    def stop(self):
        self.running = False
        self.listening = False
        print("语音助手已停止")
        try:
            self.engine.stop()
        except:
            pass

# --------------------------
# 运行示例（如果直接执行此文件）
# --------------------------
if __name__ == "__main__":
    # 将项目根加入路径（如果需要导入其他模块）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 选择是否启用 OpenAI（环境变量 OPENAI_API_KEY）
    use_openai_nlu = bool(os.environ.get("USE_OPENAI_NLU", "")) and OPENAI_AVAILABLE
    use_openai_generate = bool(os.environ.get("USE_OPENAI_GENERATE", "")) and OPENAI_AVAILABLE
    if (os.environ.get("USE_OPENAI_NLU", "") or os.environ.get("USE_OPENAI_GENERATE", "")) and not OPENAI_AVAILABLE:
        print("警告：您的系统未安装 openai 包或无法导入；OpenAI 功能将不可用。")
    
    # 你可以替换为你自己的 MemoryManager（例如：from memory.memory_manager import MemoryManager）
    memory_manager = SimpleMemoryManager()
    
    assistant = VoiceAssistant(memory_manager=memory_manager,
                               use_openai_nlu=use_openai_nlu,
                               use_openai_generate=use_openai_generate)
    
    # 启动并进入文本输入测试模式（默认）
    assistant.start()
    try:
        while assistant.running:
            time.sleep(1)
    except KeyboardInterrupt:
        assistant.stop()
        print("程序退出")
