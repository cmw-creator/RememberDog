# main.py - 更新版本
import threading
import time
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    global noise_detector_manager, speech_engine, speech_service, cam_manager
    print("正在初始化系统...")

    try:
        # 第一步：初始化记忆管理器（核心）
        from memory.memory_manager import MemoryManager
        memory_manager = MemoryManager()
        print("✓ 记忆管理器初始化完成")

        # 第二步：初始化语音服务（基础服务）
        from speech.speech_service import speech_service
        print("✓ 语音服务初始化完成")

        # 第三步：初始化语音引擎
        from speech.speech_engine import SpeechEngine
        speech_engine = SpeechEngine(memory_manager)
        print("✓ 语音引擎初始化完成")

        # 第四步：初始化摄像头管理器
        from vision.camera_manager import CameraManager
        cam_manager = CameraManager()
        cam_manager.start()
        print("✓ 摄像头管理器初始化完成")

        # 第五步：初始化各个视觉检测器
        from vision.qr_code_detector import QRCodeDetector
        from vision.face_detector import FaceDetector
        from vision.photo_detector import PhotoDetector

        qr_detector = QRCodeDetector(cam_manager, memory_manager)
        face_detector = FaceDetector(cam_manager, memory_manager)
        photo_detector = PhotoDetector(cam_manager, memory_manager)
        print("✓ 视觉检测器初始化完成")

        # 第六步：初始化噪声检测器
        from voice.noise_detector_manager import NoiseDetectorManager
        noise_detector_manager = NoiseDetectorManager(memory_manager)
        if noise_detector_manager.initialize_detector():
            print(f"✓ 噪声检测器初始化完成 - 使用{noise_detector_manager.get_detector_type()}模式")
        else:
            print("✗ 噪声检测器初始化失败")
            noise_detector_manager = None

        # 第七步：初始化语音助手
        from speech.voice_assistant import VoiceAssistant
        voice_assistant = VoiceAssistant(memory_manager)
        print("✓ 语音助手初始化完成")

        # 注册模块状态
        modules = {
            "QRCodeDetector": qr_detector,
            "FaceDetector": face_detector,
            "PhotoDetector": photo_detector,
            "VoiceAssistant": voice_assistant,
            "MemoryManager": memory_manager,
            "SpeechEngine": speech_engine
        }

        if noise_detector_manager:
            modules["NoiseDetector"] = noise_detector_manager

        for name, module in modules.items():
            memory_manager.update_module_status(name, "initialized")

        # 按顺序启动线程
        print("正在启动各个模块...")

        # 1. 启动记忆管理器
        memory_manager.start()

        # 2. 启动噪声检测器
        if noise_detector_manager:
            noise_thread = threading.Thread(target=noise_detector_manager.start, name="NoiseDetector")
            noise_thread.daemon = True
            noise_thread.start()

        # 3. 启动视觉检测器
        qr_thread = threading.Thread(target=qr_detector.run_detection, name="QRDetector")
        face_thread = threading.Thread(target=face_detector.run_detection, name="FaceDetector")
        photo_thread = threading.Thread(target=photo_detector.run_detection, name="PhotoDetector")

        qr_thread.daemon = True
        face_thread.daemon = True
        photo_thread.daemon = True

        qr_thread.start()
        time.sleep(0.5)  # 稍微错开启动时间

        face_thread.start()
        time.sleep(0.5)

        photo_thread.start()

        # 4. 启动语音助手
        voice_assistant.start()

        # 更新状态为运行中
        for name in modules.keys():
            memory_manager.update_module_status(name, "running")

        print("所有模块启动完成，系统正常运行中...")
        print("噪声检测功能已启用，正在监听环境声音...")

        # 主循环
        try:
            while True:
                time.sleep(1)
                # 可以在这里添加系统状态监控

        except KeyboardInterrupt:
            print("\n正在停止系统...")

    except Exception as e:
        print(f"系统启动失败: {e}")
        traceback.print_exc()
        return

    finally:
        # 清理资源
        try:
            if noise_detector_manager:
                noise_detector_manager.stop()
            speech_engine.stop()
            speech_service.stop()
            cam_manager.stop()
            print("系统已安全停止")
        except:
            pass

if __name__ == '__main__':
    main()