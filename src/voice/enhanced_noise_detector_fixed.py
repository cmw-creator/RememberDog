
import numpy as np
import threading
import queue
import time
import os
import sys
import pyaudio
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
class EnhancedNoiseDetectorYamnet:
    """智能版本的YAMNet噪音检测器 - 区分正常和异常声音"""

    def __init__(self, memory_manager, sensitivity=0.3, model_path=None, device_index=None):
        self.memory_manager = memory_manager
        self.sensitivity = sensitivity  # 提高灵敏度，避免误报
        self.device_index = device_index
        self.running = False

        # 音频参数
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_queue = queue.Queue()

        # YAMNet要求至少0.975秒的音频（15600个样本）
        self.required_samples = 15600
        self.audio_buffer = np.array([], dtype=np.float32)

        # 模型组件
        self.model = None
        self.params = None
        self.class_names = None

        # 统计和状态
        self.chunk_count = 0
        self.last_buffer_size = 0
        self.normal_sound_count = 0
        self.abnormal_sound_count = 0
        self.last_event_time = 0
        self.event_cooldown = 5  # 事件冷却时间（秒）

        print("🔧 初始化智能YAMNet噪音检测器...")

        # 初始化YAMNet模型
        if self._init_yamnet_model(model_path):
            print("✅ 智能YAMNet噪音检测器初始化完成")
            self.model_available = True
        else:
            print("❌ YAMNet噪音检测器初始化失败")
            self.model_available = False

    def _init_yamnet_model(self, model_path):
        """初始化YAMNet模型"""
        try:
            print("🔄 步骤1: 准备YAMNet环境...")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, "yamnet"),
                os.path.join(current_dir, "..", "yamnet"),
                "E:/RememberDog/assets/voice_models/yamnet",
                "assets/voice_models/yamnet",
            ]

            yamnet_dir = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "yamnet.py")):
                    yamnet_dir = path
                    break

            if not yamnet_dir:
                print("❌ 未找到YAMNet模块目录")
                return False

            if yamnet_dir not in sys.path:
                sys.path.insert(0, yamnet_dir)

            if model_path is None:
                model_path = os.path.join(yamnet_dir, "yamnet.h5")

            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return False

            print("🔄 步骤2: 导入YAMNet模块...")
            from assets.voice_models.yamnet.params import Params
            import yamnet as yamnet_model

            print("🔄 步骤3: 初始化参数和模型...")
            self.params = Params()
            self.model = yamnet_model.yamnet_frames_model(self.params)
            self.model.load_weights(model_path)

            print("🔄 步骤4: 加载类别名称...")
            class_map_path = os.path.join(yamnet_dir, "yamnet_class_map.csv")
            if os.path.exists(class_map_path):
                self.class_names = yamnet_model.class_names(class_map_path)
            else:
                self.class_names = self._get_default_class_names()

            print(f"✅ YAMNet模型加载成功: {model_path}")
            print(f"   类别数量: {len(self.class_names)}")
            return True

        except Exception as e:
            print(f"❌ YAMNet模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_default_class_names(self):
        """获取完整的类别名称"""
        return np.array([
            'Speech', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
            'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell',
            'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter',
            'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing',
            'Baby cry, infant cry', 'Whimper', 'Wail, moan', 'Sigh', 'Singing',
            'Choir', 'Yodeling', 'Chant', 'Mantra', 'Child singing', 'Synthetic singing',
            'Rapping', 'Humming', 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze',
            'Snoring', 'Gasp', 'Pant', 'Snort', 'Cough', 'Throat clearing', 'Sneeze',
            'Sniff', 'Run', 'Shuffle', 'Walk, footsteps', 'Chewing, mastication',
            'Biting', 'Gargling', 'Stomach rumble', 'Burping, eructation', 'Hiccup',
            'Fart', 'Hands', 'Finger snapping', 'Clapping', 'Heart sounds, heartbeat',
            'Heart murmur', 'Cheering', 'Applause', 'Chatter', 'Crowd', 'Hubbub, speech noise, speech babble',
            'Children playing', 'Animal', 'Domestic animals, pets', 'Dog', 'Bark',
            'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper (dog)', 'Cat', 'Purr',
            'Meow', 'Hiss', 'Caterwaul', 'Livestock, farm animals, working animals',
            'Horse', 'Clip-clop', 'Neigh, whinny', 'Cattle, bovinae', 'Moo',
            'Cowbell', 'Pig', 'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl', 'Chicken, rooster',
            'Cluck', 'Crowing, cock-a-doodle-doo', 'Turkey', 'Gobble', 'Duck', 'Quack',
            'Goose', 'Honk', 'Wild animals', 'Roaring cats (lions, tigers)', 'Roar',
            'Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet',
            'Squawk', 'Pigeon, dove', 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot', 'Bird flight, flapping wings',
            'Canidae, dogs, wolves', 'Rodents, rats, mice', 'Mouse', 'Patter',
            'Insect', 'Cricket', 'Mosquito', 'Fly, housefly', 'Buzz', 'Bee, wasp, etc.',
            'Frog', 'Croak', 'Snake', 'Rattle', 'Whale vocalization', 'Music',
            'Musical instrument', 'Plucked string instrument', 'Guitar', 'Electric guitar',
            'Bass guitar', 'Acoustic guitar', 'Steel guitar, slide guitar', 'Tapping (guitar technique)',
            'Strum', 'Banjo', 'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Keyboard (musical)',
            'Piano', 'Electric piano', 'Organ', 'Electronic organ', 'Hammond organ',
            'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Drum kit',
            'Drum machine', 'Drum', 'Snare drum', 'Rimshot', 'Drum roll', 'Bass drum',
            'Timpani', 'Tabla', 'Cymbal', 'Hi-hat', 'Wood block', 'Tambourine',
            'Rattle (instrument)', 'Maraca', 'Gong', 'Triangle', 'Bell', 'Jingle bell',
            'Tuning fork', 'Chime', 'Wind instrument, woodwind instrument', 'Flute',
            'Saxophone', 'Clarinet', 'Harp', 'Bell ringing', 'Church bell', 'Jingle bell',
            'Bicycle bell', 'Tuning fork', 'Chime', 'Wind chime', 'Ringtone', 'Telephone',
            'Telephone bell ringing', 'Dial tone', 'Busy signal', 'Alarm clock',
            'Siren', 'Civil defense siren', 'Air horn', 'Foghorn', 'Whistle',
            'Steam whistle', 'Vehicle', 'Engine', 'Light engine (high frequency)',
            'Dental drill, dentist\'s drill', 'Lawn mower', 'Chainsaw', 'Medium engine (mid frequency)',
            'Heavy engine (low frequency)', 'Engine knocking', 'Engine starting',
            'Idling', 'Accelerating, revving, vroom', 'Door', 'Doorbell', 'Ding-dong',
            'Sliding door', 'Slam', 'Knock', 'Tap', 'Squeak', 'Cupboard open or close',
            'Drawer open or close', 'Dishes, pots, and pans', 'Cutlery, silverware',
            'Chopping (food)', 'Frying (food)', 'Microwave oven', 'Blender', 'Water tap, faucet',
            'Sink (filling or washing)', 'Bathtub (filling or washing)', 'Hair dryer',
            'Toilet flush', 'Toothbrush', 'Electric toothbrush', 'Vacuum cleaner',
            'Zipper (clothing)', 'Keys jangling', 'Coin (dropping)', 'Scissors',
            'Electric shaver, electric razor', 'Shuffling cards', 'Typing', 'Typewriter',
            'Computer keyboard', 'Writing', 'Pen', 'Pencil', 'Scratch', 'Whit noise',
            'Thunder', 'Wind', 'Rustling leaves', 'Rain', 'Raindrop', 'Rain on surface',
            'Stream', 'Waterfall', 'Ocean', 'Waves, surf', 'Crackle', 'Cricket',
            'Fire', 'Match', 'Smoke', 'Gunshot, gunfire', 'Machine gun', 'Fusillade',
            'Artillery fire', 'Cap gun', 'Fireworks', 'Firecracker', 'Burst, pop',
            'Eruption', 'Boom', 'Wood', 'Bamboo', 'Breaking', 'Crack', 'Snap',
            'Glass', 'Chink, clink', 'Splash, splatter', 'Slosh', 'Squish', 'Drip',
            'Pour', 'Trickle, dribble', 'Gush', 'Fill (with liquid)', 'Spray',
            'Pump (liquid)', 'Stir', 'Boiling', 'Sonar', 'Radar', 'Laser', 'Explosion',
            'Implosion', 'Rumble', 'Whir', 'Clatter', 'Sizzle', 'Click', 'Clang',
            'Beep', 'Ping', 'Ding', 'Tick', 'Tick-tock', 'Toot', 'Honk', 'Beep-beep',
            'Ring', 'Buzz', 'Hum', 'Whir', 'Screech', 'Rattle', 'Vibration', 'Silence'
        ])

    def is_normal_human_activity(self, yamnet_class):
        """判断是否为正常人类活动声音"""
        normal_activities = [
            'Speech', 'Child speech', 'Conversation', 'Narration', 'Babbling',
            'Whispering', 'Laughter', 'Baby laughter', 'Giggle', 'Snicker',
            'Belly laugh', 'Chuckle', 'Chortle', 'Singing', 'Choir', 'Yodeling',
            'Chant', 'Mantra', 'Child singing', 'Rapping', 'Humming',
            'Breathing', 'Snoring', 'Gasp', 'Pant', 'Snort', 'Cough',
            'Throat clearing', 'Sneeze', 'Sniff', 'Chatter', 'Crowd',
            'Hubbub', 'Cheering', 'Applause'
        ]

        yamnet_class_lower = yamnet_class.lower()
        for activity in normal_activities:
            if activity.lower() in yamnet_class_lower:
                return True
        return False

    def is_abnormal_noise(self, yamnet_class, energy):
        """判断是否为真正的异常噪音"""
        # 首先排除正常人类活动
        if self.is_normal_human_activity(yamnet_class):
            return False

        # 异常噪音类型
        abnormal_noises = [
            'Glass', 'Breaking', 'Crash', 'Explosion', 'Slam', 'Thump', 'Bang',
            'Alarm', 'Siren', 'Emergency vehicle', 'Screaming', 'Yell', 'Shout',
            'Baby cry', 'Crying', 'Sobbing', 'Whimper', 'Wail', 'Moan',
            'Gunshot', 'Fireworks', 'Firecracker', 'Burst', 'Eruption', 'Boom'
        ]

        yamnet_class_lower = yamnet_class.lower()
        for noise in abnormal_noises:
            if noise.lower() in yamnet_class_lower:
                # 对于某些噪音，需要足够的能量才认为是异常
                if noise.lower() in ['slam', 'thump', 'bang'] and energy < 50:
                    return False
                return True

        return False

    def process_audio_chunk(self, audio_chunk):
        """处理音频块 - 智能区分正常和异常声音"""
        try:
            # 转换为numpy数组
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

            # 计算原始能量
            raw_energy = np.sqrt(np.mean(np.square(audio_data.astype(np.float64))))

            # 转换为YAMNet需要的格式并归一化
            audio_data_float = audio_data.astype(np.float32) / 32768.0

            # 累积音频数据
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data_float])

            # 更新计数器
            self.chunk_count += 1

            # 显示累积进度
            buffer_length = len(self.audio_buffer)
            if buffer_length != self.last_buffer_size and self.chunk_count % 10 == 0:
                required_ratio = buffer_length / self.required_samples
                energy_level = "🔇" if raw_energy < 10 else "🔈" if raw_energy < 50 else "🔉" if raw_energy < 100 else "🔊"
                print(
                    f"{energy_level} 音频缓冲区: {buffer_length}/{self.required_samples} 样本 ({required_ratio:.1%}) - 能量: {raw_energy:.1f}")
                self.last_buffer_size = buffer_length

            # 只有当累积了足够长的音频时才进行分类
            if buffer_length >= self.required_samples:
                # 取出足够长度的音频进行分类
                classification_audio = self.audio_buffer[:self.required_samples]

                # 保留剩余音频在缓冲区中（滑动窗口）
                keep_samples = buffer_length - self.chunk_size
                if keep_samples > 0:
                    self.audio_buffer = self.audio_buffer[-keep_samples:]
                else:
                    self.audio_buffer = np.array([], dtype=np.float32)

                # 使用YAMNet分类
                yamnet_class, confidence = self.classify_audio(classification_audio)

                current_time = time.time()

                # 显示检测结果
                if yamnet_class and confidence > 0.1:
                    if yamnet_class == "Silence":
                        if self.chunk_count % 30 == 0:  # 减少静音显示频率
                            print(f"🔇 环境静音 - 置信度: {confidence:.3f}")
                    elif self.is_normal_human_activity(yamnet_class):
                        self.normal_sound_count += 1
                        if self.normal_sound_count % 5 == 0:  # 减少正常声音显示频率
                            print(f"💬 正常活动: {yamnet_class:<25} 置信度: {confidence:.3f}")
                    else:
                        print(f"🔊 环境声音: {yamnet_class:<25} 置信度: {confidence:.3f}")

                # 异常检测逻辑 - 只在检测到真正的异常噪音时触发
                if (confidence > self.sensitivity and
                        yamnet_class != "Silence" and
                        self.is_abnormal_noise(yamnet_class, raw_energy) and
                        (current_time - self.last_event_time) > self.event_cooldown):

                    noise_type, risk_level = self.map_to_noise_type(yamnet_class, raw_energy)
                    self.abnormal_sound_count += 1
                    self.last_event_time = current_time

                    print(f"🚨 检测到异常噪音: {yamnet_class} -> {noise_type}")
                    print(f"   置信度: {confidence:.3f}, 风险: {risk_level}, 能量: {raw_energy:.1f}")

                    # 触发事件
                    event_data = {
                        "noise_type": noise_type,
                        "risk_level": risk_level,
                        "confidence": float(confidence),
                        "yamnet_class": yamnet_class,
                        "energy": float(raw_energy),
                        "timestamp": current_time
                    }

                    if risk_level in ["high", "critical"]:
                        self.memory_manager.trigger_event("urgent_noise_alert", event_data)
                    else:
                        self.memory_manager.trigger_event("abnormal_noise_detected", event_data)

        except Exception as e:
            print(f"处理音频块错误: {e}")

    def classify_audio(self, audio_data):
        """使用YAMNet对音频进行分类"""
        if self.model is None or self.class_names is None:
            return None, 0

        try:
            # 确保音频数据是float32类型，范围[-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 确保音频长度合适
            current_length = len(audio_data)
            if current_length < self.required_samples:
                padding = self.required_samples - current_length
                audio_data = np.pad(audio_data, (0, padding), mode='constant')

            # 使用YAMNet进行预测
            scores, embeddings, spectrogram = self.model(audio_data)

            # 获取平均分数
            mean_scores = np.mean(scores, axis=0)

            # 获取最高分数的类别
            top_class_idx = np.argmax(mean_scores)
            top_score = mean_scores[top_class_idx]
            top_class_name = self.class_names[top_class_idx]

            return top_class_name, top_score

        except Exception as e:
            print(f"音频分类错误: {e}")
            return None, 0

    def map_to_noise_type(self, yamnet_class, energy):
        """将YAMNet类别映射到噪声类型和风险级别"""
        if yamnet_class is None:
            return "unknown", "low"

        yamnet_class_lower = yamnet_class.lower()

        # 噪声类型映射
        noise_mapping = {
            'Glass': 'glass_break',
            'Breaking': 'glass_break',
            'Crash': 'impact',
            'Explosion': 'impact',
            'Slam': 'impact',
            'Thump': 'impact',
            'Bang': 'impact',
            'Alarm': 'alarm_sound',
            'Siren': 'alarm_sound',
            'Emergency vehicle': 'alarm_sound',
            'Screaming': 'high_pitch',
            'Yell': 'high_pitch',
            'Shout': 'high_pitch',
            'Baby cry': 'baby_cry',
            'Crying': 'crying',
            'Sobbing': 'crying',
            'Whimper': 'crying',
            'Wail': 'crying',
            'Moan': 'moaning_crying',
            'Gunshot': 'gunshot',
            'Fireworks': 'explosion',
            'Firecracker': 'explosion'
        }

        # 基础风险级别
        base_risk_levels = {
            'glass_break': 'high',
            'impact': 'high',
            'alarm_sound': 'critical',
            'high_pitch': 'medium',
            'baby_cry': 'medium',
            'crying': 'medium',
            'moaning_crying': 'high',
            'gunshot': 'critical',
            'explosion': 'high'
        }

        # 检查映射
        for key, value in noise_mapping.items():
            if key.lower() in yamnet_class_lower:
                base_risk = base_risk_levels.get(value, "medium")
                # 根据能量调整风险级别
                if energy > 100 and base_risk != "critical":
                    return value, "high"
                return value, base_risk

        return "unknown", "low"

    def _processing_loop(self):
        """处理循环"""
        print("🔄 启动智能音频处理循环...")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                self.process_audio_chunk(audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理循环错误: {e}")
                time.sleep(0.1)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.running and in_data is not None and len(in_data) > 0:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        """启动检测器"""
        if self.model is None:
            print("警告: YAMNet模型未加载，无法启动检测器")
            return False

        self.running = True

        try:
            self.audio = pyaudio.PyAudio()

            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                input_device_index=self.device_index
            )

            self.stream.start_stream()
            print("✅ 音频流启动成功")

        except Exception as e:
            print(f"❌ 启动音频流失败: {e}")
            self.running = False
            return False

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("✅ 智能YAMNet噪声检测器已启动 - 只检测真正的异常噪音")
        return True

    def stop(self):
        """停止检测器"""
        print("🛑 停止噪声检测器...")
        self.running = False

        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if hasattr(self, 'audio') and self.audio:
            try:
                self.audio.terminate()
            except:
                pass

        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # 输出统计信息
        print(f"📊 检测统计:")
        print(f"   总音频块: {self.chunk_count}")
        print(f"   正常活动检测: {self.normal_sound_count}")
        print(f"   异常噪音检测: {self.abnormal_sound_count}")
        print("✅ 智能YAMNet噪声检测器已停止")