#!/usr/bin/env python3
# speech_event_handler_web_dynamic.py - 语音事件处理器 + 最近3句话动态网页显示
import time
from multiprocessing import Process, Queue as MPQueue
from flask import Flask, render_template_string, jsonify
import logging
logger = logging.getLogger(name='Log')
logger.info("开始加载语音引擎模块")


class SpeechEngine:
    def __init__(self, memory_manager, rate=150, volume=0.9):
        self.memory_manager = memory_manager
        # self.engine = pyttsx3.init()
        # self.engine.setProperty('rate', rate)
        # self.engine.setProperty('volume', volume)
        # self.engine.setProperty('voice', 'sit/cmn')


        self.is_speaking = False
        self.running = True

        # 最近说过的3句话
        self.history = []

        # 注册事件回调
        self.memory_manager.register_event_callback("speak_event", self.speak_event, "SpeechEventHandler")


        self.speech_queue = MPQueue()
        #self._process_sepech_queue(self.speech_queue, self.memory_manager.runSpeaking)
        # 启动独立 TTS 进程
        #self.tts_process = Process(
        #    target=self._process_speech_queue,
        #    args=(self.speech_queue, self.memory_manager.runSpeaking),
        #    daemon=True
        #)
        # self.tts_process = Process(target=self._process_speech_queue, args=(self.speech_queue, self.memory_manager.runSpeaking), daemon=True)
        #self.tts_process.start()


        # 启动Flask线程
        # self.flask_thread = threading.Thread(target=self._start_flask)
        # self.flask_thread.daemon = True
        # self.flask_thread.start()

        print("语音事件处理器初始化完成")

    # ----------------- 事件回调 -----------------
    def speak_event(self, event_data):
        """普通语音事件 - 低优先级"""
        if event_data:
            self.add_to_queue(event_data, priority=3)

    def handle_medicine_event(self, event_data):
        """药品检测事件 - 高优先级"""
        if event_data:
            self.add_to_queue(event_data, priority=1)

    def handle_face_event(self, event_data):
        """人脸检测事件 - 高优先级"""
        if event_data:
            self.add_to_queue(event_data, priority=1)

    def handle_photo_event(self, event_data):
        """照片检测事件 - 高优先级"""
        if event_data:
            self.add_to_queue(event_data, priority=1)

    # ----------------- 队列管理 -----------------
    def add_to_queue(self, event_data, priority=0):
        text = event_data.get("speak_text", "")
        # 播放
        # 保存历史，最多8条
        if text:
            self.history.append(text)
            print("加入历史")
            if len(self.history) > 8:
                self.history = self.history[-8:]
        audio_file = event_data.get("audio_file", None)
        print(f"添加到队列: text={text}, audio_file={audio_file}, priority={priority}")
        self.speech_queue.put((priority, text, audio_file))
    def _process_speech_queue(self ,running_flag=None):
        speech_queue = self.speech_queue
        runSpeaking = self.memory_manager.runSpeaking
        is_speaking_flag = self.is_speaking
        import pyttsx3
        import time
        import os
        import queue as _queue
        from playsound import playsound

        print("[TTS进程] 已启动, 等待任务...")
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        # self.engine.setProperty('rate', rate)
        # self.engine.setProperty('volume', volume)
        engine.setProperty('voice', 'sit/cmn')
        while True:
            # 如果外部提供了 running_flag（例如 multiprocessing.Event），可随时停止
            if running_flag is not None and not running_flag.is_set():
                print("[TTS进程] 收到停止信号 (running_flag)")
                break

            try:
                # 从队列获取第一个任务（带超时）
                try:
                    priority, text, audio_file = speech_queue.get(timeout=1)
                except _queue.Empty:
                    continue

                if text is None and audio_file is None:  # 停止信号
                    print("[TTS进程] 收到退出信号")
                    break

                # 标记正在播放
                if is_speaking_flag is not None:
                    try:
                        is_speaking_flag.put(True, block=False)
                    except Exception:
                        pass

                # 把队列里当前所有项都取出来，用于决定最高优先级
                items = [(priority, text, audio_file)]
                while True:
                    try:
                        items.append(speech_queue.get_nowait())
                    except _queue.Empty:
                        break

                # 选择优先级最高的一项
                highest_priority = None
                highest_text = None
                highest_audio = None
                for p, t, a in items:
                    if highest_priority is None or p > highest_priority:
                        highest_priority = p
                        highest_text = t
                        highest_audio = a

                # 把除最高优先项外的项目重新放回队列
                used = False
                for p, t, a in items:
                    if not used and p == highest_priority and t == highest_text and a == highest_audio:
                        used = True
                        continue
                    speech_queue.put((p, t, a))

                if highest_audio and os.path.exists(highest_audio):
                    print(f"播放音频文件: {highest_audio}")
                    runSpeaking.set()
                    playsound(highest_audio)
                    runSpeaking.clear()
                elif highest_text:
                    print(f"[TTS进程] TTS 发声: {highest_text}")

                    runSpeaking.set()
                    engine.say(highest_text)
                    engine.runAndWait()
                    print("说完了")
                    runSpeaking.clear()


                # 延迟，释放资源
                time.sleep(0.1)

                # 播放结束标志
                if is_speaking_flag is not None:
                    try:
                        is_speaking_flag.put(False, block=False)
                    except Exception:
                        pass

            except Exception as e:
                print(f"[TTS进程] 异常: {e}")
                import traceback
                traceback.print_exc()
                if is_speaking_flag is not None:
                    try:
                        is_speaking_flag.put(False, block=False)
                    except Exception:
                        pass
                time.sleep(1)
        print("[TTS进程] 已停止")

    def _monitor_speaking_status(self):
        """监控播放状态（从进程队列读取）"""
        self._is_speaking = False
        while self.running:
            try:
                # 非阻塞读取状态
                try:
                    self._is_speaking = self.is_speaking_flag.get_nowait()
                except:
                    pass
                time.sleep(0.1)
            except Exception as e:
                print(f"[状态监控] 异常: {e}")
                time.sleep(1)

    def stop(self):
        """停止语音引擎"""
        print("[停止] 正在关闭语音引擎...")
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)

    def is_busy(self):
        return self.is_speaking

    # ----------------- Flask Web -----------------
    def _start_flask(self):
        app = Flask(__name__)

        # 首页HTML，使用AJAX动态获取历史语音
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>最近语音记录</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h2 { color: #333; }
                ul { font-size: 18px; }
            </style>
        </head>
        <body>
            <h2>回复信息</h2>
            <ul id="history-list"></ul>

            <script>
            function updateHistory() {
                fetch('/history')
                    .then(response => response.json())
                    .then(data => {
                        const ul = document.getElementById('history-list');
                        ul.innerHTML = '';
                        data.history.forEach(item => {
                            const li = document.createElement('li');
                            li.textContent = item;
                            ul.appendChild(li);
                        });
                    })
                    .catch(err => console.error(err));
            }

            // 每秒刷新一次
            setInterval(updateHistory, 1000);
            // 页面加载时立即刷新
            updateHistory();
            </script>
        </body>
        </html>
        """

        @app.route("/")
        def index():
            return render_template_string(html_template)

        @app.route("/history")
        def get_history():
            return jsonify({"history": self.history})

        app.run(host="0.0.0.0", port=8080, debug=False)

