# main.py
from vision.camera_manager import CameraManager
from vision.qr_code_detector import QRCodeDetector
from vision.face_detector import FaceDetector
from vision.photo_detector import PhotoDetector
from vision.hand_pose_estimator import HandPoseEstimator
from voice.noise_detector import NoiseDetector
from speech.voice_assistant import VoiceAssistant
from memory.memory_manager import MemoryManager
from speech.speech_engine import SpeechEngine
from speech.speech_service import speech_service
import threading
import time  # 添加这行导入


def main():
    # 创建记忆管理器
    memory_manager = MemoryManager()

    # 创建语音引擎
    speech_engine = SpeechEngine(memory_manager)

    # 创建摄像头管理器
    cam_manager = CameraManager()
    cam_manager.start()

    # 创建检测器
    qr_detector = QRCodeDetector(cam_manager, memory_manager)
    face_detector = FaceDetector(cam_manager, memory_manager)
    photo_detector = PhotoDetector(cam_manager, memory_manager)

    # 创建噪声检测器
    noise_detector = NoiseDetector(memory_manager, sensitivity=0.7)

    # 创建手部姿态检测器
    hand_estimator = HandPoseEstimator(
        cam_manager,
        memory_manager,
        model_path="./assets/handpose_x/weights/squeezenet1_1-size-256-loss-0.0732.pth",
        model='squeezenet1_1',
        num_classes=42,
        GPUS='0',
        img_size=(256, 256)
    )

    voice_assistant = VoiceAssistant(memory_manager)

    # 注册模块状态
    memory_manager.update_module_status("QRCodeDetector", "initialized")
    memory_manager.update_module_status("FaceDetector", "initialized")
    memory_manager.update_module_status("PhotoDetector", "initialized")
    memory_manager.update_module_status("HandPoseEstimator", "initialized")
    memory_manager.update_module_status("VoiceAssistant", "initialized")
    memory_manager.update_module_status("NoiseDetector", "initialized")

    # 为每个检测器创建单独的线程
    qr_thread = threading.Thread(target=qr_detector.run_detection, name="QR_Detector")
    face_thread = threading.Thread(target=face_detector.run_detection, name="Face_Detector")
    photo_thread = threading.Thread(target=photo_detector.run_detection, name="Photo_Detector")
    hand_thread = threading.Thread(target=hand_estimator.run_estimation, name="Hand_Pose_Estimator")
    noise_thread = threading.Thread(target=noise_detector.start, name="Noise_Detector")

    # 启动所有线程
    qr_thread.start()
    face_thread.start()
    photo_thread.start()
    noise_thread.start()
    # hand_thread.start()  # 可选启动手势识别

    memory_manager.start()  # 内部创造线程了
    voice_assistant.start()  # 内部创造线程了

    # 更新模块状态为运行中
    memory_manager.update_module_status("QRCodeDetector", "running")
    memory_manager.update_module_status("FaceDetector", "running")
    memory_manager.update_module_status("PhotoDetector", "running")
    memory_manager.update_module_status("HandPoseEstimator", "running")
    memory_manager.update_module_status("VoiceAssistant", "running")
    memory_manager.update_module_status("MemoryManager", "running")
    memory_manager.update_module_status("SpeechEventHandler", "running")
    memory_manager.update_module_status("NoiseDetector", "running")

    try:
        # 主循环
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止所有服务...")
        # 停止所有服务
        noise_detector.stop()
        speech_engine.stop()
        speech_service.stop()
        cam_manager.stop()
        print("服务已停止")


if __name__ == '__main__':
    main()