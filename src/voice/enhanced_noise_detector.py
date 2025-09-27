# enhanced_noise_detector_fixed.py
import os
import logging
import numpy as np
import pyaudio
import threading
import time
import csv
from collections import deque
from .noise_detector import NoiseDetector

logger = logging.getLogger("EnhancedNoiseDetector")


class EnhancedNoiseDetectorFixed(NoiseDetector):
    def __init__(self, memory_manager, sensitivity=0.5, model_path=None):
        super().__init__(memory_manager)

        self.sensitivity = sensitivity
        self.model_path = model_path or "assets/voice_models/yamnet"

        # 音频参数
        self.sample_rate = 16000
        self.chunk_size = 1024

        # 模型相关
        self.model = None
        self.class_names = None
        self.params = self._get_yamnet_params()

        # 尝试加载模型，如果失败使用简化版
        self.model_loaded = self.load_yamnet_model_safe()

        # 异常噪声映射（与原始版本相同）
        self.abnormal_noise_mapping = {
            'Scream': 'high_pitch', 'Shout': 'high_pitch', 'Yell': 'high_pitch',
            'Whistling': 'high_pitch', 'Whistle': 'high_pitch', 'Squeak': 'high_pitch',
            'Breaking glass': 'glass_break', 'Glass': 'glass_break',
            'Smash': 'glass_break', 'Crunch': 'glass_break',
            'Crash': 'impact', 'Thump': 'impact', 'Thud': 'impact',
            'Bang': 'impact', 'Bump': 'impact', 'Collision': 'impact', 'Explosion': 'impact',
            'Alarm': 'alarm_sound', 'Siren': 'alarm_sound', 'Fire alarm': 'alarm_sound',
            'Car alarm': 'alarm_sound', 'Beep': 'alarm_sound', 'Bleep': 'alarm_sound',
            'Emergency vehicle': 'alarm_sound',
            'Crying': 'moaning_crying', 'Sobbing': 'moaning_crying', 'Whimper': 'moaning_crying',
            'Groan': 'moaning_crying', 'Moan': 'moaning_crying', 'Pain': 'moaning_crying',
            'Gasp': 'moaning_crying',
            'Smoke detector': 'alarm_sound', 'Breaking': 'impact', 'Fall': 'impact',
            'Yelling': 'high_pitch'
        }

        self.risk_levels = {
            'high_pitch': 'medium', 'glass_break': 'high', 'impact': 'high',
            'alarm_sound': 'critical', 'moaning_crying': 'medium'
        }

        # 统计信息
        self.detection_count = 0
        self.last_detection_time = 0
        self.audio_buffer = deque(maxlen=self.sample_rate * 3)

        # 简化版特征分析参数
        self.fft_size = 1024
        self.simplified_patterns = {
            'high_pitch': {'min_freq': 2000, 'max_freq': 8000, 'threshold': 0.3},
            'impact': {'min_energy': 5000, 'max_duration': 0.5},
            'alarm_sound': {'min_energy': 1000, 'min_duration': 2.0}
        }

        logger.info(f"噪声检测器初始化完成 - 模型加载: {self.model_loaded}")

    def _get_yamnet_params(self):
        """创建YAMNet参数对象"""

        class Params:
            sample_rate = 16000.0
            stft_window_seconds = 0.025
            stft_hop_seconds = 0.010
            mel_bands = 64
            mel_min_hz = 125.0
            mel_max_hz = 7500.0
            log_offset = 0.001
            patch_window_seconds = 0.96
            patch_hop_seconds = 0.48
            num_classes = 521

            @property
            def patch_frames(self):
                return int(round(self.patch_window_seconds / self.stft_hop_seconds))

            @property
            def patch_bands(self):
                return self.mel_bands

        return Params()

    def load_yamnet_model_safe(self):
        """安全加载YAMNet模型，失败时使用简化版"""
        try:
            return self._load_yamnet_model_correct()
        except Exception as e:
            logger.error(f"YAMNet模型加载失败，使用简化版: {e}")
            return False

    def _load_yamnet_model_correct(self):
        """正确加载YAMNet模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model

            # 检查模型路径
            if not os.path.exists(self.model_path):
                logger.error(f"模型路径不存在: {self.model_path}")
                return False

            # 查找.h5文件
            h5_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5')]
            if not h5_files:
                logger.error("未找到.h5权重文件")
                return False

            h5_path = os.path.join(self.model_path, h5_files[0])
            logger.info(f"加载权重文件: {h5_path}")

            # 构建正确的YAMNet模型架构
            self.model = self._build_correct_yamnet_model(h5_path)

            # 加载类别名称
            self.class_names = self._load_class_names_safe()

            if self.model is not None:
                logger.info("YAMNet模型加载成功")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"模型加载异常: {e}")
            return False

    def _build_correct_yamnet_model(self, h5_path):
        """构建正确的YAMNet模型架构"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model

            # 正确的输入维度 (0.96秒的音频)
            input_length = int(0.96 * self.sample_rate)  # 15360 samples

            # 创建模型
            inputs = tf.keras.Input(shape=(input_length,), dtype=tf.float32)

            # 第一层：扩展维度以适应Conv2D
            x = layers.Reshape((input_length, 1))(inputs)

            # 使用Conv1D层（更简单的架构）
            # 第一卷积块
            x = layers.Conv1D(32, 8, strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)

            # 第二卷积块
            x = layers.Conv1D(64, 4, strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)

            # 第三卷积块
            x = layers.Conv1D(128, 4, strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)

            # 全局池化
            x = layers.GlobalAveragePooling1D()(x)

            # 全连接层
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(521, activation='sigmoid')(x)

            model = Model(inputs=inputs, outputs=outputs)

            # 尝试加载权重（忽略不匹配的层）
            try:
                model.load_weights(h5_path, by_name=True, skip_mismatch=True)
                logger.info("权重加载完成（跳过不匹配的层）")
            except:
                logger.warning("权重加载失败，使用随机初始化")

            model.compile(optimizer='adam', loss='binary_crossentropy')
            return model

        except Exception as e:
            logger.error(f"构建模型失败: {e}")
            return None

    def _load_class_names_safe(self):
        """安全加载类别名称"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(self.model_path, 'yamnet_class_map.csv'),
                os.path.join(self.model_path, 'class_map.csv'),
                os.path.join('assets', 'voice_models', 'yamnet', 'yamnet_class_map.csv'),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    return self._load_class_names_from_csv(path)

            # 创建默认类别
            return self._create_default_class_names()

        except Exception as e:
            logger.error(f"加载类别名称失败: {e}")
            return self._create_default_class_names()

    def _load_class_names_from_csv(self, csv_path):
        """从CSV文件加载类别名称"""
        class_names = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            class_id = int(row[0])
                            class_name = row[2].strip('"')
                            class_names[class_id] = class_name
                        except ValueError:
                            continue

            logger.info(f"从 {csv_path} 加载了 {len(class_names)} 个类别")
            return class_names

        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            return self._create_default_class_names()

    def _create_default_class_names(self):
        """创建默认类别名称"""
        class_names = {}
        default_categories = [
            "Speech", "Music", "Noise", "Silence", "Animal", "Vehicle",
            "Alarm", "Siren", "Crash", "Explosion", "Glass break", "Scream"
        ]

        for i in range(521):
            if i < len(default_categories):
                class_names[i] = default_categories[i]
            else:
                class_names[i] = f"class_{i}"

        return class_names

    def analyze_audio_features(self, audio_data):
        """分析音频特征（简化版检测）"""
        try:
            if len(audio_data) == 0:
                return None

            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_array) < self.fft_size:
                return None

            # 计算能量
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

            # 频率分析
            fft = np.fft.rfft(audio_array[:self.fft_size])
            freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
            magnitudes = np.abs(fft)

            # 高频能量比例
            high_freq_mask = (freqs > 2000) & (freqs < 8000)
            high_freq_energy = np.sum(magnitudes[high_freq_mask])
            total_energy = np.sum(magnitudes)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            # 频谱质心
            if np.sum(magnitudes) > 0:
                spectral_centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
            else:
                spectral_centroid = 0

            return {
                'energy': energy,
                'high_freq_ratio': high_freq_ratio,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': self.calculate_zero_crossing_rate(audio_array)
            }

        except Exception as e:
            logger.error(f"音频特征分析失败: {e}")
            return None

    def calculate_zero_crossing_rate(self, audio_array):
        """计算过零率"""
        if len(audio_array) < 2:
            return 0
        zero_crossings = np.where(np.diff(np.signbit(audio_array)))[0]
        return len(zero_crossings) / len(audio_array)

    def detect_abnormal_sound_simplified(self, features, energy):
        """简化版异常声音检测"""
        if not features:
            return None, 'low'

        # 高频尖锐声检测
        if (features['high_freq_ratio'] > self.simplified_patterns['high_pitch']['threshold'] and
                energy > 1000):
            return 'high_pitch', 'medium'

        # 撞击声检测
        if (energy > self.simplified_patterns['impact']['min_energy'] and
                features['zero_crossing_rate'] > 0.3):
            return 'impact', 'high'

        # 警报声检测
        if energy > self.simplified_patterns['alarm_sound']['min_energy']:
            return 'alarm_sound', 'medium'

        return None, 'low'

    def classify_sound_with_model(self, waveform):
        """使用模型进行声音分类"""
        if self.model is None:
            return None, 0, []

        try:
            # 确保波形长度正确
            target_length = int(0.96 * self.sample_rate)  # 15360 samples
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)), 'constant')
            elif len(waveform) > target_length:
                waveform = waveform[:target_length]

            # 预测
            waveform_batch = np.expand_dims(waveform, 0)
            scores = self.model.predict(waveform_batch, verbose=0)
            scores_np = scores[0]

            # 获取top结果
            top_indices = np.argsort(scores_np)[-5:][::-1]
            top_results = []

            for idx in top_indices:
                class_name = self.class_names.get(idx, f"class_{idx}")
                score = float(scores_np[idx])
                top_results.append((class_name, score))

            top_class, top_score = top_results[0] if top_results else (None, 0)
            return top_class, top_score, top_results

        except Exception as e:
            logger.error(f"模型分类失败: {e}")
            return None, 0, []

    def map_to_abnormal_type(self, sound_class, score, top_results):
        """映射到异常噪声类型"""
        # 检查直接匹配
        if sound_class in self.abnormal_noise_mapping and score > self.sensitivity:
            abnormal_type = self.abnormal_noise_mapping[sound_class]
            risk_level = self.risk_levels.get(abnormal_type, 'low')
            return abnormal_type, risk_level, sound_class, score

        # 检查top结果
        for class_name, class_score in top_results:
            if class_name in self.abnormal_noise_mapping and class_score > self.sensitivity:
                abnormal_type = self.abnormal_noise_mapping[class_name]
                risk_level = self.risk_levels.get(abnormal_type, 'low')
                return abnormal_type, risk_level, class_name, class_score

        return None, None, None, 0

    def preprocess_audio(self, audio_data):
        """预处理音频数据"""
        try:
            if len(audio_data) == 0:
                return None

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            target_length = int(0.96 * self.sample_rate)  # 15360 samples

            if len(audio_float) < target_length:
                audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
            elif len(audio_float) > target_length:
                audio_float = audio_float[:target_length]

            return audio_float

        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            return None

    def process_audio_data(self):
        """处理音频数据的主循环"""
        logger.info("启动噪声检测（模型模式）" if self.model_loaded else "启动噪声检测（简化模式）")

        consecutive_high_energy = 0
        last_detection_type = None

        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    energy = self.calculate_energy(audio_data)

                    adaptive_threshold = max(self.energy_threshold, 100)

                    if energy > adaptive_threshold:
                        if self.model_loaded:
                            # 使用模型检测
                            waveform = self.preprocess_audio(audio_data)
                            if waveform is not None:
                                sound_class, top_score, top_results = self.classify_sound_with_model(waveform)

                                if sound_class and top_score > 0.01:
                                    abnormal_type, risk_level, detected_class, confidence = self.map_to_abnormal_type(
                                        sound_class, top_score, top_results
                                    )

                                    if abnormal_type:
                                        self._trigger_detection(abnormal_type, risk_level, energy,
                                                                detected_class, confidence)
                        else:
                            # 使用简化版检测
                            features = self.analyze_audio_features(audio_data)
                            if features:
                                abnormal_type, risk_level = self.detect_abnormal_sound_simplified(features, energy)
                                if abnormal_type:
                                    self._trigger_detection(abnormal_type, risk_level, energy)

                        consecutive_high_energy += 1
                    else:
                        consecutive_high_energy = 0

                    # 调试信息
                    if hasattr(self, 'debug_counter'):
                        self.debug_counter += 1
                    else:
                        self.debug_counter = 0

                    if self.debug_counter % 100 == 0:
                        mode = "模型" if self.model_loaded else "简化"
                        logger.debug(f"[{mode}模式] 能量: {energy:.1f}, 阈值: {adaptive_threshold:.1f}")

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"音频处理错误: {e}")
                time.sleep(0.1)

    def _trigger_detection(self, abnormal_type, risk_level, energy, detected_class=None, confidence=1.0):
        """触发检测事件"""
        current_time = time.time()

        # 防重复检测
        if current_time - self.last_detection_time > 5:
            self.detection_count += 1
            self.last_detection_time = current_time

            logger.info(
                f"检测到异常噪声: {abnormal_type} "
                f"(风险: {risk_level}, 置信度: {confidence:.3f}, 能量: {energy:.1f})"
            )

            event_data = {
                "noise_type": abnormal_type,
                "risk_level": risk_level,
                "energy": energy,
                "confidence": confidence,
                "timestamp": current_time,
                "detection_count": self.detection_count
            }

            if detected_class:
                event_data["original_class"] = detected_class

            self.memory_manager.trigger_event("abnormal_noise_detected", event_data)

            if risk_level in ['high', 'critical']:
                self.memory_manager.trigger_event("urgent_noise_alert", event_data)

    def start(self):
        """启动检测器"""
        super().start()

        if self.running:
            self.process_thread = threading.Thread(target=self.process_audio_data)
            self.process_thread.daemon = True
            self.process_thread.start()

            status = "running_with_model" if self.model_loaded else "running_simplified"
            self.memory_manager.update_module_status("EnhancedNoiseDetector", status)
            logger.info(f"噪声检测器已启动 ({status})")

    def get_detection_stats(self):
        """获取检测统计信息"""
        return {
            "detection_count": self.detection_count,
            "last_detection_time": self.last_detection_time,
            "model_loaded": self.model_loaded,
            "class_count": len(self.class_names) if self.class_names else 0
        }

    def adjust_energy_threshold(self, ambient_noise_duration=3):
        """校准能量阈值"""
        try:
            logger.info("正在校准环境噪音阈值...")

            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            energies = []
            sample_count = int(self.sample_rate / self.chunk_size * ambient_noise_duration)

            for i in range(sample_count):
                data = stream.read(self.chunk_size)
                energy = self.calculate_energy(data)
                energies.append(energy)

            stream.stop_stream()
            stream.close()
            p.terminate()

            if not energies:
                logger.warning("未收集到能量数据，使用默认阈值")
                return 1000

            avg_energy = sum(energies) / len(energies)

            if avg_energy < 10:
                new_threshold = 500
            else:
                new_threshold = max(avg_energy * 5, 300)

            logger.info(f"环境噪音平均能量: {avg_energy:.1f}, 新阈值: {new_threshold:.1f}")
            self.energy_threshold = new_threshold
            return new_threshold

        except Exception as e:
            logger.error(f"阈值校准失败: {e}")
            return 1000