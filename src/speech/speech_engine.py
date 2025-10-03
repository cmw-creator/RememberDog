#!/usr/bin/env python3
# speech_event_handler_web_dynamic.py - 语音事件处理器 + 最近3句话动态网页显示

import pyttsx3
import threading
import time
import queue
import os
import pygame
from flask import Flask, render_template_string, jsonify
from multiprocessing import Process, Queue as MPQueue


class SpeechEngine:
    def __init__(self, memory_manager, rate=150, volume=0.9):
        self.memory_manager = memory_manager
        self.rate = rate
        self.volume = volume

        # 初始化 pygame mixer
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.set_volume(volume)
            print("[音频] pygame mixer 初始化完成")
        except Exception as e:
            print(f"[音频] pygame mixer 初始化失败: {e}")

        # === 修改：使用多进程队列 ===
        self.speech_queue = MPQueue()  # 进程间队列
        self.is_speaking_flag = MPQueue()  # 用于跨进程共享状态
        self.running = True

        # 最近说过的话(改为8条)
        self.history = []
        self.history_lock = threading.Lock()

        # 启动独立 TTS 进程
        self.tts_process = Process(
            target=self._tts_process_worker,
            args=(self.speech_queue, self.is_speaking_flag, self.rate, self.volume),
            daemon=True
        )
        self.tts_process.start()
        # ================================

        # 注册事件回调
        self.memory_manager.register_event_callback("speak_event", self.speak_event, "SpeechEventHandler")
        self.memory_manager.register_event_callback("medicine_detected", self.handle_medicine_event,
                                                    "SpeechEventHandler")
        self.memory_manager.register_event_callback("face_detected", self.handle_face_event, "FaceEventHandler")
        self.memory_manager.register_event_callback("photo_detected", self.handle_photo_event, "FaceEventHandler")

        # 启动状态监控线程（用于从进程队列读取状态）
        self.status_thread = threading.Thread(target=self._monitor_speaking_status, daemon=True)
        self.status_thread.start()

        # 启动Flask线程
        self.flask_thread = threading.Thread(target=self._start_flask, daemon=True)
        self.flask_thread.start()

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
    def add_to_queue(self, event_data, priority=3):
        """添加到队列"""
        text = event_data.get("speak_text", "")
        audio_file = event_data.get("audio_file", None)

        if not text and not audio_file:
            return

        # 添加到历史记录
        if text:
            with self.history_lock:
                self.history.append(text)
                if len(self.history) > 8:
                    self.history = self.history[-8:]
            print(f"[历史记录] 已添加: {text}")

        # 添加到队列: (优先级, 文本, 音频文件)
        try:
            self.speech_queue.put((priority, text, audio_file), timeout=1)
            print(f"[队列] 添加任务 - 优先级:{priority}, 文本:{text[:30] if text else 'None'}, 音频:{audio_file}")
        except Exception as e:
            print(f"[队列] 添加失败: {e}")

    # ----------------- TTS 进程工作函数 -----------------
    @staticmethod
    def _tts_process_worker(speech_queue, is_speaking_flag, rate, volume):
        """独立进程的 TTS 工作线程"""
        import pyttsx3
        import pygame
        import time
        import os

        print("[TTS进程] 已启动，等待任务...")
        play_count = 0

        # 初始化 pygame（进程内独立初始化）
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.set_volume(volume)
        except Exception as e:
            print(f"[TTS进程] pygame 初始化失败: {e}")

        while True:
            try:
                # 从队列获取任务
                try:
                    priority, text, audio_file = speech_queue.get(timeout=1)
                except:
                    continue

                if text is None and audio_file is None:  # 停止信号
                    print("[TTS进程] 收到退出信号")
                    break

                # 更新播放状态
                try:
                    is_speaking_flag.put(True, block=False)
                except:
                    pass

                play_count += 1
                print(f"[TTS进程 #{play_count}] 开始 - 优先级:{priority}, 文本:{text[:30] if text else 'None'}")

                # 尝试播放音频文件
                audio_played = False
                if audio_file and os.path.exists(audio_file):
                    try:
                        pygame.mixer.music.load(audio_file)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        print(f"[TTS进程 #{play_count}] 音频播放完成: {audio_file}")
                        audio_played = True
                    except Exception as e:
                        print(f"[TTS进程 #{play_count}] 音频播放失败: {e}")

                # 如果音频未播放且有文本，则使用 TTS
                if not audio_played and text:
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', rate)
                        engine.setProperty('volume', volume)
                        engine.say(text)
                        engine.runAndWait()
                        del engine
                        print(f"[TTS进程 #{play_count}] TTS 播放完成")
                    except Exception as e:
                        print(f"[TTS进程 #{play_count}] TTS 错误: {e}")

                # 更新播放状态
                try:
                    is_speaking_flag.put(False, block=False)
                except:
                    pass

                time.sleep(0.1)

            except Exception as e:
                print(f"[TTS进程] 异常: {e}")
                import traceback
                traceback.print_exc()
                try:
                    is_speaking_flag.put(False, block=False)
                except:
                    pass
                time.sleep(1)

        # 清理资源
        try:
            pygame.mixer.quit()
        except:
            pass
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

        # 发送停止信号到 TTS 进程
        try:
            self.speech_queue.put((0, None, None), timeout=1)
        except:
            pass

        # 等待 TTS 进程结束
        if self.tts_process.is_alive():
            self.tts_process.terminate()
            self.tts_process.join(timeout=2)

        # 清理 pygame（主进程）
        try:
            pygame.mixer.quit()
        except:
            pass

        print("[停止] 语音引擎已关闭")

    def is_busy(self):
        """检查是否正在播放"""
        return getattr(self, '_is_speaking', False)

    def get_queue_size(self):
        """获取队列大小"""
        return self.speech_queue.qsize()

    # ----------------- Flask Web -----------------
    def _start_flask(self):
        """启动Flask Web服务"""
        app = Flask(__name__)

        # 首页HTML,使用AJAX动态获取历史语音
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>语音记录 - RememberDog</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    padding: 30px;
                }
                h2 {
                    color: #667eea;
                    margin-bottom: 20px;
                    text-align: center;
                    font-size: 2em;
                }
                .status {
                    text-align: center;
                    margin-bottom: 20px;
                    padding: 10px;
                    background: #f0f0f0;
                    border-radius: 8px;
                    font-weight: bold;
                }
                .status.speaking { background: #ffebee; color: #c62828; }
                .status.idle { background: #e8f5e9; color: #2e7d32; }
                ul {
                    list-style: none;
                    padding: 0;
                }
                li {
                    background: #f5f5f5;
                    margin: 10px 0;
                    padding: 15px 20px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    animation: slideIn 0.3s ease-out;
                    font-size: 16px;
                    line-height: 1.6;
                }
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: translateX(-20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateX(0);
                    }
                }
                .empty {
                    text-align: center;
                    color: #999;
                    padding: 40px;
                    font-style: italic;
                }
                .footer {
                    text-align: center;
                    margin-top: 20px;
                    color: #999;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>RememberDog 语音记录</h2>
                <div id="status" class="status idle">系统空闲</div>
                <ul id="history-list"></ul>
                <div id="empty-msg" class="empty">暂无语音记录</div>
                <div class="footer">
                    队列任务: <span id="queue-size">0</span> | 
                    最后更新: <span id="last-update">--</span>
                </div>
            </div>

            <script>
            function updateHistory() {
                fetch('/history')
                    .then(response => response.json())
                    .then(data => {
                        const ul = document.getElementById('history-list');
                        const emptyMsg = document.getElementById('empty-msg');
                        const status = document.getElementById('status');
                        const queueSize = document.getElementById('queue-size');
                        const lastUpdate = document.getElementById('last-update');

                        // 更新状态
                        if (data.is_speaking) {
                            status.textContent = '正在播放...';
                            status.className = 'status speaking';
                        } else {
                            status.textContent = '系统空闲';
                            status.className = 'status idle';
                        }

                        // 更新队列大小
                        queueSize.textContent = data.queue_size || 0;

                        // 更新时间
                        const now = new Date();
                        lastUpdate.textContent = now.toLocaleTimeString('zh-CN');

                        // 更新历史记录
                        ul.innerHTML = '';
                        if (data.history && data.history.length > 0) {
                            emptyMsg.style.display = 'none';
                            // 倒序显示(最新的在上面)
                            data.history.slice().reverse().forEach(item => {
                                const li = document.createElement('li');
                                li.textContent = item;
                                ul.appendChild(li);
                            });
                        } else {
                            emptyMsg.style.display = 'block';
                        }
                    })
                    .catch(err => {
                        console.error('获取数据失败:', err);
                    });
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
            with self.history_lock:
                history_copy = self.history.copy()

            return jsonify({
                "history": history_copy,
                "is_speaking": self.is_busy(),
                "queue_size": self.get_queue_size()
            })

        # 禁用Flask的日志输出
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        try:
            app.run(host="0.0.0.0", port=80, debug=False, use_reloader=False)
        except Exception as e:
            print(f"[Flask] 启动失败: {e}")
            print("[Flask] 提示: 端口80可能需要管理员权限,或尝试使用其他端口(如5000)")
