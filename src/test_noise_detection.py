# test_noise_detection_extended.py
import numpy as np
import wave
import time
import threading
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.voice.enhanced_noise_detector_fixed import  EnhancedNoiseDetectorYamnet
except ImportError:
    # 如果导入失败，尝试直接使用当前目录的模块
    from src.voice.enhanced_noise_detector_fixed import   EnhancedNoiseDetectorYamnet


class MockMemoryManager:
    """模拟记忆管理器用于测试"""

    def __init__(self):
        self.events = []
        self.module_status = {}

    def trigger_event(self, event_type, event_data):
        print(f"🔊 触发事件: {event_type}")
        print(f"   数据: {event_data}")
        self.events.append((event_type, event_data))

    def update_module_status(self, module_name, status):
        self.module_status[module_name] = status
        print(f"📊 模块状态更新: {module_name} -> {status}")


class ExtendedNoiseDetectionTester:
    """扩展噪声检测测试器 - 测试更多声音类型"""

    def __init__(self):
        self.memory_manager = MockMemoryManager()
        # 显式指定模型路径（确保正确）
        model_abs_path = "E:/RememberDog/assets/voice_models/yamnet/yamnet.h5"

        # 检查模型文件是否存在
        if not os.path.exists(model_abs_path):
            print(f"❌ 模型文件不存在: {model_abs_path}")
            print("请检查模型文件路径或下载所需的YAMNet模型")
            # 可以尝试使用默认路径或备用方案
            model_abs_path = self._find_alternative_model_path()

        self.detector =  EnhancedNoiseDetectorYamnet(
            memory_manager=self.memory_manager,
            sensitivity=0.3,
            model_path=model_abs_path  # 传递绝对路径
        )

        # 关键修复：定义sample_rate属性
        self.sample_rate = 16000  # 与检测器保持一致的采样率

    def _find_alternative_model_path(self):
        """尝试查找备用的模型路径"""
        possible_paths = [
            "assets/voice_models/yamnet/yamnet.h5",
            "E:/RememberDog/assets/voice_models/yamnet/yamnet.h5",
            "/home/RememberDog/assets/voice_models/yamnet/yamnet.h5",
            "./assets/voice_models/yamnet/yamnet.h5"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 找到备用模型文件: {path}")
                return path

        print("❌ 未找到任何可用的模型文件")
        return "assets/voice_models/yamnet/yamnet.h5"

    def generate_dog_bark(self, duration=1.0):
        """生成狗吠声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 狗吠特征：短促的爆发声，带有频率变化
        bark1 = np.exp(-40 * t) * np.sin(2 * np.pi * 800 * t)  # 低频部分
        bark2 = np.exp(-60 * t) * np.sin(2 * np.pi * 2000 * t) * 0.7  # 高频部分

        # 添加多个吠叫
        bark_sound = np.zeros_like(t)
        for i in range(3):  # 3次连续的吠叫
            start_idx = int(i * len(t) / 4)
            end_idx = int((i + 1) * len(t) / 4)
            if end_idx > len(t):
                end_idx = len(t)
            segment_len = end_idx - start_idx
            if segment_len > 0:
                segment_t = np.linspace(0, duration / 4, segment_len)
                segment_bark = np.exp(-40 * segment_t) * np.sin(2 * np.pi * (800 + i * 100) * segment_t)
                bark_sound[start_idx:end_idx] = segment_bark

        # 组合声音
        dog_bark = bark_sound + 0.5 * bark2[:len(bark_sound)]
        dog_bark = dog_bark / np.max(np.abs(dog_bark)) * 0.8

        return dog_bark

    def generate_baby_cry(self, duration=3.0):
        """生成婴儿哭声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 婴儿哭声特征：高频、有节奏的波动
        base_freq = 600  # 基频

        # 创建哭声音调变化
        cry_sound = np.zeros_like(t)
        cry_duration = 0.8  # 每次哭声持续时间
        pause_duration = 0.4  # 停顿时间

        current_time = 0
        while current_time < duration:
            # 哭声段
            cry_start = int(current_time * self.sample_rate)
            cry_end = int(min((current_time + cry_duration) * self.sample_rate, len(t)))

            if cry_start < len(t):
                cry_segment_len = cry_end - cry_start
                if cry_segment_len > 0:
                    cry_t = np.linspace(0, cry_duration, cry_segment_len)
                    # 频率在哭声中上升
                    freq_mod = 200 * np.sin(2 * np.pi * 2 * cry_t)  # 频率调制
                    cry_pitch = base_freq + freq_mod

                    # 幅度包络
                    envelope = np.minimum(cry_t * 10, 1.0) * np.exp(-2 * (cry_t - cry_duration / 2) ** 2)

                    cry_segment = envelope * np.sin(2 * np.pi * cry_pitch * cry_t)
                    cry_sound[cry_start:cry_end] = cry_segment

            current_time += cry_duration + pause_duration

        # 添加呼吸声
        breath_sound = 0.3 * np.random.normal(0, 0.1, len(t)) * np.exp(-0.5 * t)
        baby_cry = cry_sound + breath_sound
        baby_cry = baby_cry / np.max(np.abs(baby_cry)) * 0.7

        return baby_cry

    def generate_doorbell(self, duration=2.0):
        """生成门铃声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 门铃特征：清脆的铃声，有衰减
        # 第一声铃响
        bell1 = np.exp(-15 * t) * (np.sin(2 * np.pi * 800 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t))

        # 第二声铃响（稍后开始）
        bell2 = np.zeros_like(t)
        second_start = 0.5  # 第二声开始时间
        second_idx = int(second_start * self.sample_rate)
        if second_idx < len(t):
            second_t = t[second_idx:] - second_start
            bell2_segment = np.exp(-15 * second_t) * (
                        np.sin(2 * np.pi * 800 * second_t) + 0.5 * np.sin(2 * np.pi * 1200 * second_t))
            if len(bell2_segment) <= len(bell2[second_idx:]):
                bell2[second_idx:second_idx + len(bell2_segment)] = bell2_segment

        doorbell = bell1 + bell2
        doorbell = doorbell / np.max(np.abs(doorbell)) * 0.6

        return doorbell

    def generate_thunder(self, duration=4.0):
        """生成雷声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 雷声特征：低频轰鸣，逐渐增强然后衰减
        # 主要雷声（低频）
        main_thunder = 0.7 * np.exp(-2 * t) * np.sin(2 * np.pi * 80 * t)

        # 雷声滚动的中频成分
        roll_thunder = 0.4 * np.exp(-1.5 * t) * np.sin(2 * np.pi * 120 * (t + 0.5))

        # 初始雷击（高频成分）
        strike_start = 0.2
        strike_idx = int(strike_start * self.sample_rate)
        strike_t = t[strike_idx:] - strike_start
        strike_thunder = np.zeros_like(t)
        if len(strike_t) > 0:
            strike_segment = 0.5 * np.exp(-50 * strike_t) * np.sin(2 * np.pi * 1000 * strike_t)
            if len(strike_segment) <= len(strike_thunder[strike_idx:]):
                strike_thunder[strike_idx:strike_idx + len(strike_segment)] = strike_segment

        # 组合雷声
        thunder = main_thunder + roll_thunder + strike_thunder

        # 添加随机低频噪声模拟远处雷声
        background_thunder = 0.2 * np.random.normal(0, 0.1, len(t)) * np.exp(-0.3 * t)
        thunder += background_thunder

        thunder = thunder / np.max(np.abs(thunder)) * 0.9

        return thunder

    def generate_car_horn(self, duration=2.0):
        """生成汽车喇叭声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 汽车喇叭特征：稳定的中频声音
        horn_freq = 500  # 基频

        # 创建喇叭声（有轻微的频率波动）
        freq_mod = 20 * np.sin(2 * np.pi * 5 * t)  # 轻微的频率调制
        car_horn = np.sin(2 * np.pi * (horn_freq + freq_mod) * t)

        # 幅度包络（快速达到最大然后保持）
        envelope = np.minimum(t * 10, 1.0)
        car_horn = envelope * car_horn

        # 添加一些谐波
        harmonic = 0.3 * np.sin(2 * np.pi * 2 * horn_freq * t)
        car_horn += harmonic

        car_horn = car_horn / np.max(np.abs(car_horn)) * 0.8

        return car_horn

    def generate_construction_noise(self, duration=5.0):
        """生成建筑工地噪音"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 建筑噪音：多种声音的组合
        construction = np.zeros_like(t)

        # 电钻声（间歇性高频噪声）
        for i in range(4):
            drill_start = i * 1.2
            drill_duration = 0.8
            drill_idx_start = int(drill_start * self.sample_rate)
            drill_idx_end = int(min((drill_start + drill_duration) * self.sample_rate, len(t)))

            if drill_idx_start < len(t) and drill_idx_end > drill_idx_start:
                drill_t = t[drill_idx_start:drill_idx_end] - drill_start
                drill_sound = np.exp(-10 * drill_t) * np.sin(2 * np.pi * 2000 * drill_t)
                construction[drill_idx_start:drill_idx_end] += 0.6 * drill_sound

        # 锤击声（周期性冲击）
        for i in range(8):
            hammer_start = i * 0.6
            hammer_duration = 0.1
            hammer_idx_start = int(hammer_start * self.sample_rate)
            hammer_idx_end = int(min((hammer_start + hammer_duration) * self.sample_rate, len(t)))

            if hammer_idx_start < len(t) and hammer_idx_end > hammer_idx_start:
                hammer_t = t[hammer_idx_start:hammer_idx_end] - hammer_start
                hammer_sound = np.exp(-100 * hammer_t) * np.sin(2 * np.pi * 300 * hammer_t)
                construction[hammer_idx_start:hammer_idx_end] += 0.7 * hammer_sound

        # 背景机械噪声
        background_noise = 0.3 * np.random.normal(0, 0.2, len(t))
        construction += background_noise

        construction = construction / np.max(np.abs(construction)) * 0.7

        return construction

    def generate_cat_meow(self, duration=2.0):
        """生成猫叫声"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # 猫叫特征：高频、短促、有音调变化
        meow_sound = np.zeros_like(t)

        # 第一声喵叫
        meow1_start = 0.3
        meow1_duration = 0.5
        meow1_idx_start = int(meow1_start * self.sample_rate)
        meow1_idx_end = int(min((meow1_start + meow1_duration) * self.sample_rate, len(t)))

        if meow1_idx_start < len(t) and meow1_idx_end > meow1_idx_start:
            meow1_t = t[meow1_idx_start:meow1_idx_end] - meow1_start
            # 频率从高到低变化
            freq_sweep = 1500 - 800 * meow1_t / meow1_duration
            meow1 = np.exp(-8 * meow1_t) * np.sin(2 * np.pi * freq_sweep * meow1_t)
            meow_sound[meow1_idx_start:meow1_idx_end] = meow1

        # 第二声喵叫（更短）
        meow2_start = 1.2
        meow2_duration = 0.3
        meow2_idx_start = int(meow2_start * self.sample_rate)
        meow2_idx_end = int(min((meow2_start + meow2_duration) * self.sample_rate, len(t)))

        if meow2_idx_start < len(t) and meow2_idx_end > meow2_idx_start:
            meow2_t = t[meow2_idx_start:meow2_idx_end] - meow2_start
            freq_sweep = 1400 - 600 * meow2_t / meow2_duration
            meow2 = np.exp(-10 * meow2_t) * np.sin(2 * np.pi * freq_sweep * meow2_t)
            meow_sound[meow2_idx_start:meow2_idx_end] += 0.8 * meow2

        cat_meow = meow_sound / np.max(np.abs(meow_sound)) * 0.6 if np.max(np.abs(meow_sound)) > 0 else meow_sound

        return cat_meow

    def float_to_pcm16(self, audio_float):
        """将浮点音频转换为16位PCM"""
        audio_int16 = (audio_float * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def test_single_sound(self, sound_name, sound_generator, duration=3.0, expected_type=None):
        """测试单个声音类型"""
        print(f"\n🎯 测试 {sound_name} 识别...")

        # 生成声音
        audio_float = sound_generator(duration)
        audio_data = self.float_to_pcm16(audio_float)

        # 将音频数据分割成chunk并送入检测器
        chunk_size = 1024
        num_chunks = len(audio_data) // chunk_size

        print(f"   生成了 {duration}秒音频，分割为 {num_chunks} 个chunk")

        # 清空之前的事件
        self.memory_manager.events = []

        # 模拟实时音频流输入
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_data = audio_data[start_idx:end_idx]

            if len(chunk_data) == chunk_size:
                self.detector.audio_queue.put(chunk_data)

            # 稍微延迟模拟实时流
            time.sleep(0.01)

        # 等待处理完成
        time.sleep(1.0)

        # 检查检测结果
        detected_events = [e for e in self.memory_manager.events
                           if e[0] in ['abnormal_noise_detected', 'urgent_noise_alert']]

        if detected_events:
            print(f"✅ {sound_name} 检测成功！")
            for event_type, data in detected_events:
                noise_type = data.get('noise_type', '未知')
                risk_level = data.get('risk_level', '未知')
                confidence = data.get('confidence', 0)
                print(f"   检测类型: {noise_type}")
                print(f"   风险等级: {risk_level}")
                print(f"   置信度: {confidence:.3f}")

                # 检查是否符合预期类型
                if expected_type and noise_type == expected_type:
                    print(f"   ✅ 符合预期类型: {expected_type}")
                elif expected_type:
                    print(f"   ⚠️  预期类型: {expected_type}, 实际检测: {noise_type}")
        else:
            print(f"❌ {sound_name} 未检测到")

        return len(detected_events) > 0

    def test_all_sounds(self):
        """测试所有声音类型"""
        print("=" * 70)
        print("🧪 开始扩展噪声检测测试")
        print("=" * 70)

        # 启动检测器
        print("启动噪声检测器...")
        self.detector.start()
        time.sleep(2)  # 等待检测器初始化

        # 测试各种声音
        test_cases = [
            ("狗吠声", self.generate_dog_bark, "high_pitch"),
            ("婴儿哭声", self.generate_baby_cry, "moaning_crying"),
            ("门铃声", self.generate_doorbell, "alarm_sound"),
            ("雷声", self.generate_thunder, "impact"),
            ("汽车喇叭", self.generate_car_horn, "alarm_sound"),
            ("建筑噪音", self.generate_construction_noise, "impact"),
            ("猫叫声", self.generate_cat_meow, "high_pitch"),
        ]

        results = {}
        for sound_name, generator, expected_type in test_cases:
            success = self.test_single_sound(sound_name, generator, expected_type=expected_type)
            results[sound_name] = success
            time.sleep(1)  # 测试间隔

        # 显示统计结果
        print("\n" + "=" * 70)
        print("📊 扩展测试结果统计")
        print("=" * 70)

        total_tests = len(results)
        passed_tests = sum(results.values())

        for sound, success in results.items():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{sound}: {status}")

        print(f"\n总计: {passed_tests}/{total_tests} 项测试通过")

        # 停止检测器
        self.detector.stop()

        return passed_tests == total_tests

    def save_test_sounds(self):
        """保存测试声音为WAV文件用于调试"""
        import os
        os.makedirs("test_sounds_extended", exist_ok=True)

        sounds = [
            ("dog_bark", self.generate_dog_bark()),
            ("baby_cry", self.generate_baby_cry()),
            ("doorbell", self.generate_doorbell()),
            ("thunder", self.generate_thunder()),
            ("car_horn", self.generate_car_horn()),
            ("construction", self.generate_construction_noise()),
            ("cat_meow", self.generate_cat_meow()),
        ]

        for name, audio_float in sounds:
            filename = f"test_sounds_extended/{name}.wav"
            audio_int16 = (audio_float * 32767).astype(np.int16)

            with wave.open(filename, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())

            print(f"💾 保存测试声音: {filename}")


def main():
    """主测试函数"""
    # 首先检查模型文件是否存在
    model_path = "E:/RememberDog/assets/voice_models/yamnet/yamnet.h5"
    if not os.path.exists(model_path):
        print(f"❌ 主测试: 模型文件不存在: {model_path}")
        print("请确保YAMNet模型已下载并放置在正确位置")
        return False

    tester = ExtendedNoiseDetectionTester()

    # 可选：保存测试声音文件
    print("💾 生成扩展测试声音文件...")
    tester.save_test_sounds()

    # 运行测试
    success = tester.test_all_sounds()

    if success:
        print("\n🎉 所有扩展测试通过！噪声检测功能正常。")
    else:
        print("\n⚠️  部分扩展测试失败，可能需要调整检测参数。")

    return success


if __name__ == "__main__":
    main()