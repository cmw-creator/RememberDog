import numpy as np
import threading
import queue
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用 GPU
import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)
print("可用设备:", tf.config.list_physical_devices())

class EnhancedNoiseDetectorYamnet:
    """基于YAMNet的增强噪声检测器"""

    def __init__(self, memory_manager, sensitivity=0.3, model_path=None):
        self.memory_manager = memory_manager
        self.sensitivity = sensitivity
        self.running = False

        # 音频参数
        self.sample_rate = 16000
        self.chunk_size = 1024

        # 音频队列
        self.audio_queue = queue.Queue()

        # 延迟导入YAMNet相关模块
        self.model = None
        self.params = None
        self.class_names = None

        # 初始化YAMNet模型
        self._init_yamnet_model(model_path)

        # 噪声类型映射到我们的分类
        self.noise_mapping = {
            'Dog': 'dog_bark',
            'Bark': 'dog_bark',
            'Howl': 'dog_bark',
            'Baby cry, infant cry': 'baby_cry',
            'Crying, sobbing': 'crying',
            'Whimper': 'crying',
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
            'Groan': 'moaning_crying',
            'Moan': 'moaning_crying',
            'Whine': 'moaning_crying'
        }

        # 风险级别映射
        self.risk_levels = {
            'dog_bark': 'low',
            'baby_cry': 'medium',
            'crying': 'medium',
            'glass_break': 'high',
            'impact': 'high',
            'alarm_sound': 'critical',
            'high_pitch': 'medium',
            'moaning_crying': 'high'
        }

        print("YAMNet噪声检测器初始化完成")

    def _init_yamnet_model(self, model_path):
        """初始化YAMNet模型"""
        try:
            # 方法1：直接导入当前目录的YAMNet模块
            import sys
            import os

            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 尝试多种可能的YAMNet路径
            possible_paths = [
                os.path.join(current_dir, "yamnet"),  # 同级目录下的yamnet文件夹
                os.path.join(current_dir, "..", "yamnet"),  # 上级目录的yamnet文件夹
                "E:/RememberDog/assets/voice_models/yamnet",  # 绝对路径
                "assets/voice_models/yamnet",  # 相对路径
            ]

            yamnet_dir = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "yamnet.py")):
                    yamnet_dir = path
                    break

            if not yamnet_dir:
                print("❌ 未找到YAMNet模块目录")
                return False

            # 添加到Python路径
            if yamnet_dir not in sys.path:
                sys.path.insert(0, yamnet_dir)

            # 设置模型路径
            if model_path is None:
                model_path = os.path.join(yamnet_dir, "yamnet.h5")

            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                # 尝试其他可能的模型路径
                model_path = self._find_model_file(yamnet_dir)
                if not model_path:
                    return False

            # 动态导入YAMNet模块
            from params import Params
            import yamnet as yamnet_model

            # 初始化参数和模型
            self.params = Params()
            self.model = yamnet_model.yamnet_frames_model(self.params)
            self.model.load_weights(model_path)

            # 加载类别名称
            class_map_path = os.path.join(yamnet_dir, "yamnet_class_map.csv")
            if os.path.exists(class_map_path):
                self.class_names = yamnet_model.class_names(class_map_path)
            else:
                self.class_names = self._get_default_class_names()

            print(f"✅ YAMNet模型加载成功: {model_path}")
            return True

        except ImportError as e:
            print(f"❌ 导入YAMNet模块失败: {e}")
            print("Python路径:", sys.path)
            return False
        except Exception as e:
            print(f"❌ YAMNet模型初始化失败: {e}")
            return False

    def _find_model_file(self, base_dir):
        """查找模型文件"""
        possible_locations = [
            "yamnet.h5",
            "model.h5",
            "assets/voice_models/yamnet/yamnet.h5",
            "E:/RememberDog/assets/voice_models/yamnet/yamnet.h5",
        ]

        for location in possible_locations:
            full_path = os.path.join(base_dir, location) if not os.path.isabs(location) else location
            if os.path.exists(full_path):
                print(f"✅ 找到模型文件: {full_path}")
                return full_path

        print("❌ 未找到模型文件，请下载YAMNet模型")
        return None
    def _get_default_class_names(self):
        """获取默认的类别名称"""
        # 这里是YAMNet的521个类别的简化版本
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

    def classify_audio(self, audio_data):
        """使用YAMNet对音频进行分类"""
        if self.model is None or self.class_names is None:
            return None, 0

        try:
            # 确保音频数据是float32类型，范围[-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # 确保音频长度合适
            if len(audio_data) < self.sample_rate * 0.5:  # 至少0.5秒
                # 填充音频到最小长度
                target_length = int(self.sample_rate * 0.5)
                if len(audio_data) < target_length:
                    padding = target_length - len(audio_data)
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

    def map_to_noise_type(self, yamnet_class):
        """将YAMNet类别映射到我们的噪声类型"""
        if yamnet_class is None:
            return "unknown", "low"

        yamnet_class_lower = yamnet_class.lower()

        # 检查映射
        for key, value in self.noise_mapping.items():
            if key.lower() in yamnet_class_lower:
                risk_level = self.risk_levels.get(value, "low")
                return value, risk_level

        # 默认映射
        return "unknown", "low"

    def process_audio_chunk(self, audio_chunk):
        """处理音频块"""
        try:
            print(f"接收音频块，长度: {len(audio_chunk)}字节")  # 新增调试日志
            # 转换为numpy数组
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # 转换为[-1, 1]

            # 使用YAMNet分类
            yamnet_class, confidence = self.classify_audio(audio_data)

            if confidence > self.sensitivity and yamnet_class != "Silence":
                # 映射到我们的噪声类型
                noise_type, risk_level = self.map_to_noise_type(yamnet_class)

                print(f"检测到噪声: {yamnet_class} -> {noise_type}, 置信度: {confidence:.3f}, 风险: {risk_level}")

                # 触发事件
                event_data = {
                    "noise_type": noise_type,
                    "risk_level": risk_level,
                    "confidence": float(confidence),
                    "yamnet_class": yamnet_class,
                    "timestamp": time.time()
                }

                # 根据风险级别触发不同事件
                if risk_level in ["high", "critical"]:
                    self.memory_manager.trigger_event("urgent_noise_alert", event_data)
                else:
                    self.memory_manager.trigger_event("abnormal_noise_detected", event_data)

        except Exception as e:
            print(f"处理音频块错误: {e}")

    def _processing_loop(self):
        """处理循环"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    self.process_audio_chunk(audio_chunk)
                    self.audio_queue.task_done()
                else:
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"处理循环错误: {e}")
                time.sleep(0.1)

    def add_audio_data(self, audio_data):
        """添加音频数据到处理队列"""
        if self.running and self.model is not None:
            self.audio_queue.put(audio_data)
        """添加音频数据到处理队列"""
        if self.running and self.model is not None:
            self.audio_queue.put(audio_data)
            # 新增日志：验证数据是否入队
            print(f"音频数据入队，当前队列大小: {self.audio_queue.qsize()}")
        else:
            print("噪声检测器未运行或模型未加载，无法接收音频数据")
    def start(self):
        """启动检测器"""
        if self.model is None:
            print("警告: YAMNet模型未加载，无法启动检测器")
            return False

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("YAMNet噪声检测器已启动")
        return True

    def stop(self):
        """停止检测器"""
        self.running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        print("YAMNet噪声检测器已停止")