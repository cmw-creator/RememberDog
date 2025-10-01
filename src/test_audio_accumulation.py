# test_audio_detection_optimized.py
import sys

sys.path.insert(0, 'src')

from memory.memory_manager import MemoryManager
import time


def test_detection():
    print("🧪 测试音频检测功能 - 优化版本")
    print("=" * 50)
    print("优化内容:")
    print("- 降低灵敏度阈值 (0.1)")
    print("- 增加音频增益 (3.0x)")
    print("- 改进噪声类型映射")
    print("- 添加能量级别指示器")
    print("=" * 50)

    memory_manager = MemoryManager()

    try:
        # 使用优化版本的检测器
        from src.voice.enhanced_noise_detector_fixed import EnhancedNoiseDetectorYamnet
        detector = EnhancedNoiseDetectorYamnet(memory_manager, sensitivity=0.1)  # 降低灵敏度

        if detector.start():
            print("✅ 噪音检测器启动成功")
            print("🎯 请尝试以下声音测试:")
            print("   - 拍手 👏")
            print("   - 说话 🗣️")
            print("   - 敲击桌子 👊")
            print("   - 其他明显声音")
            print("\n测试30秒钟...")

            start_time = time.time()
            while time.time() - start_time < 30:
                time.sleep(1)

            detector.stop()
            print("✅ 测试完成")

        else:
            print("❌ 测试失败 - 检测器启动失败")

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_detection()