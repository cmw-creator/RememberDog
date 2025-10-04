# main.py - 修复版本
import threading
import time
import traceback
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def main():
    global memory_manager
    print("正在初始化系统...")

    try:
        # 第一步：初始化记忆管理器（核心）
        try:
            from memory.memory_manager import MemoryManager
        except ImportError:
            from src.memory.memory_manager import MemoryManager

        memory_manager = MemoryManager()
        print("记忆管理器初始化完成")

        # 第二步：初始化语音服务（基础服务）
        try:
            from speech.speech_service import speech_service
        except ImportError:
            from src.speech.speech_service import speech_service
        print("语音服务初始化完成")

        # 第三步：初始化语音引擎
        try:
            from speech.speech_engine import SpeechEngine
        except ImportError:
            from src.speech.speech_engine import SpeechEngine
        speech_engine = SpeechEngine(memory_manager)
        print("语音引擎初始化完成")

        # 第四步：初始化摄像头管理器
        try:
            from vision.camera_manager import CameraManager
        except ImportError:
            from src.vision.camera_manager import CameraManager
        cam_manager = CameraManager()
        cam_manager.start()
        print("摄像头管理器初始化完成")

        # 第五步：初始化各个检测器
        try:
            from vision.qr_code_detector import QRCodeDetector
            from vision.face_detector import FaceDetector
            from vision.photo_detector import PhotoDetector
        except ImportError:
            from src.vision.qr_code_detector import QRCodeDetector
            from src.vision.face_detector import FaceDetector
            from src.vision.photo_detector import PhotoDetector

        qr_detector = QRCodeDetector(cam_manager, memory_manager)
        face_detector = FaceDetector(cam_manager, memory_manager)
        photo_detector = PhotoDetector(cam_manager, memory_manager)
        print("视觉检测器初始化完成")

        # 第六步：初始化噪声检测器（只使用YAMNet）
        print("初始化噪声检测器...")
        noise_detector = None
        try:
            from voice.enhanced_noise_detector_fixed import EnhancedNoiseDetectorYamnet
            noise_detector = EnhancedNoiseDetectorYamnet(memory_manager, sensitivity=0.1)

            if hasattr(noise_detector, 'model') and noise_detector.model is not None:
                print("YAMNet噪声检测器初始化完成")
            else:
                print("YAMNet模型未加载")
                noise_detector = None

        except ImportError as e:
            print(f"导入错误: {e}")
            try:
                from src.voice.enhanced_noise_detector_fixed import EnhancedNoiseDetectorYamnet
                noise_detector = EnhancedNoiseDetectorYamnet(memory_manager, sensitivity=0.3)

                if hasattr(noise_detector, 'model') and noise_detector.model is not None:
                    print("YAMNet噪声检测器初始化完成")
                else:
                    print("YAMNet模型未加载")
                    noise_detector = None

            except Exception as e:
                print(f"噪声检测器初始化失败: {e}")
                noise_detector = None
        except Exception as e:
            print(f"噪声检测器初始化失败: {e}")
            noise_detector = None

        # 第七步：初始化语音助手
        try:
            from speech.voice_assistant import VoiceAssistant
        except ImportError:
            from src.speech.voice_assistant import VoiceAssistant
        voice_assistant = VoiceAssistant(memory_manager)
        print("语音助手初始化完成")

        # 注册模块状态
        modules = {
            "QRCodeDetector": qr_detector,
            "FaceDetector": face_detector,
            "PhotoDetector": photo_detector,
            "VoiceAssistant": voice_assistant,
            "NoiseDetector": noise_detector,
            "MemoryManager": memory_manager,
            "SpeechEngine": speech_engine
        }

        for name, module in modules.items():
            memory_manager.update_module_status(name, "initialized")

        # 按顺序启动线程
        print("正在启动各个模块...")

        # 1. 启动记忆管理器
        memory_manager.start()

        # 2. 启动噪声检测器
        if noise_detector and hasattr(noise_detector, 'start'):
            noise_thread = threading.Thread(target=noise_detector.start, name="NoiseDetector")
            noise_thread.daemon = True
            noise_thread.start()
            print("噪声检测器启动完成")

        # 3. 启动视觉检测器
        qr_thread = threading.Thread(target=qr_detector.run_detection, name="QRDetector")
        face_thread = threading.Thread(target=face_detector.run_detection, name="FaceDetector")
        photo_thread = threading.Thread(target=photo_detector.run_detection, name="PhotoDetector")

        qr_thread.daemon = True
        face_thread.daemon = True
        photo_thread.daemon = True

        qr_thread.start()
        time.sleep(0.5)

        face_thread.start()
        time.sleep(0.5)

        photo_thread.start()
        print("视觉检测器启动完成")

        # 4. 启动语音助手
        voice_assistant.start()
        print("语音助手启动完成")

        # 更新状态为运行中
        for name in modules.keys():
            memory_manager.update_module_status(name, "running")

        print("所有模块启动完成，系统正常运行中...")
        print("噪声检测功能已启用，正在监听环境声音...")

        # 主循环
        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("正在停止系统...")

    except Exception as e:
        print(f"系统启动失败: {e}")
        traceback.print_exc()
        return

    finally:
        # 清理资源
        try:
            if 'noise_detector' in locals() and hasattr(noise_detector, 'stop'):
                noise_detector.stop()
            if 'speech_engine' in locals():
                speech_engine.stop()
            if 'speech_service' in locals():
                speech_service.stop()
            if 'cam_manager' in locals():
                cam_manager.stop()
            print("系统已安全停止")
        except:
            pass


if __name__ == '__main__':
    main()