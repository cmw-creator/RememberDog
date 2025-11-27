# -*- coding: utf-8 -*-
import logging
import utils.logger
from src.track.person_follow_yolo import PersonFollower

logger_level=logging.INFO # 日志级别，DEBUG/INFO/WARNING/ERROR
utils.logger.get_logger(name='Log', log_level=logger_level)
logger = logging.getLogger(name='Log')
logger.setLevel(logger_level)
logger.debug("\n\n\n")
logger.debug("=====程序开始运行=====")

import time
from vision.camera_manager  import  CameraManager
from vision.qr_code_detector  import  QRCodeDetector
from vision.face_detector  import  FaceDetector
from vision.photo_detector  import  PhotoDetector
#from vision.pose           import FallDetector
#from vision.hand_pose_estimator import HandPoseEstimator
from speech.voice_assistant import VoiceAssistant
from memory.memory_manager import MemoryManager 
from speech.speech_engine import SpeechEngine
from control.control import RobotController
import threading
import time
import sys
logger.debug("main导入完成")

def main():
    #管理机器狗动作
    robot_controller=RobotController()
    # 创建摄像头管理器
    def get_camera_source():
        if sys.platform.startswith('win'): 
            # Windows 系统
            logger.info("在Windows系统上运行，使用默认摄像头0")
            return 0
        elif sys.platform.startswith('linux'):
            # Linux 系统
            logger.info("在Linux系统上运行，使用rtsp摄像头")
            return "rtsp://192.168.1.120:8554/test" 
        else:
            # 其他未知系统，可以返回一个默认值或抛出异常
            logger.warning(f"不支持的操作系统: {sys.platform}，使用默认值0")
            return 0
    cam_manager = CameraManager(get_camera_source())
    # 创建记忆管理器
    memory_manager = MemoryManager()
    # 创建语音引擎
    speech_engine = SpeechEngine(memory_manager)
    speech_engine_thread = threading.Thread(target=speech_engine._process_speech_queue, name="QR_Detector")
    speech_engine_thread.start()
    cam_manager.start() # 启动摄像头线程(延迟)，等待摄像头初始化完成
    # 创建检测器
    qr_detector = QRCodeDetector(cam_manager,memory_manager)
    face_detector=FaceDetector(cam_manager,memory_manager)
    photo_detector=PhotoDetector(cam_manager,memory_manager)
    #fall_detector = FallDetector(cam_manager,memory_manager)
    # 创建手部姿态检测器
    '''
    hand_estimator = HandPoseEstimator(
        cam_manager,
        memory_manager,
        model_path="./assets/handpose_x/weights/squeezenet1_1-size-256-loss-0.0732.pth",   #轻量模型
        model='squeezenet1_1',
        num_classes=42,
        GPUS='0',
        img_size=(256, 256)
    )
    '''
    voice_assistant = VoiceAssistant(memory_manager,robot_controller, cam_manager)
    

    # 为每个检测器创建单独的线程
    #qr_detector.run()
    qr_thread = threading.Thread(target=qr_detector.run_detection, name="QR_Detector")
    face_thread = threading.Thread(target=face_detector.run_detection, name="Face_Detector")
    photo_thread = threading.Thread(target=photo_detector.run_detection, name="Photo_Detector")
    #fall_thread = threading.Thread(target=fall_detector.run, name="Fall_Detector")
    #hand_thread = threading.Thread(target=hand_estimator.run_estimation, name="Hand_Pose_Estimator")
    # 启动所有线程
    qr_thread.start()
    face_thread.start()
    photo_thread.start()
    #fall_thread.start()
    #hand_thread.start()
    memory_manager.start()#内部创造线程了
    voice_assistant.start()#内部创造线程了

def test_cpu(start_time,times):
    logger.debug("测试CPU负载函数")
    times.append((time.time()-start_time)*1000)
    if len(times)>=10:
        times.pop(0)
    logger.debug(f"CPU负载测试耗时: {sum(times)/len(times):.5f}ms")

if __name__ == '__main__':
    main()
    start_time = time.time()
    times=[]
    while True:
        #test_cpu(start_time,times)
        time.sleep(1)
        #start_time=time.time()
    time.sleep(10000)
    logger.debug("程序超时结束运行")