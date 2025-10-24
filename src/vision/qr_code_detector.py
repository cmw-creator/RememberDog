#!/usr/bin/env python3
#识别二维码和条形码，二维码识别效果更好一点
#import rospy
import cv2
import numpy as np
import os
from datetime import datetime
import time
from pyzbar.pyzbar import decode
from PIL import Image
import pyttsx3  # 离线TTS引擎
import threading
import json

class QRCodeDetector:
    def __init__(self, camera_manager, memory_manager):
        # 硬件配置
        self.camera_manager = camera_manager #摄像头
        self.memory_manager = memory_manager  #记忆模块
        # # 语音引擎初始化
        # self.engine = pyttsx3.init()
        # self.engine.setProperty('rate', 150)  # 语速调节
        
        # 存储识别的编码数据（药瓶二维码->语音映射）
        with open('assets/qr_code_info.json', 'r', encoding='utf-8') as f:
            self.medical_db = json.load(f)
        
        # 异步语音线程控制
        self.speaking = False

        self.frame_count = 0
        self.frame_skip = 10  # 如果为2，则每3帧处理1帧 50ms一帧
        
        self.recently_processed = {}  # 记录最近处理的二维码和时间戳
        self.cooldown_period = 20  # 冷却时间（秒）
    
    def preprocess_image(self, img):
        """图像预处理：增强识别率"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自适应直方图均衡化（解决光照不均）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # 锐化边缘
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened

    def decode_codes(self, img):
        """识别二维码/条形码并返回结果列表"""
        # 预处理可能会影响识别，暂时不使用
        # processed_img = self.preprocess_image(img)
        # decoded_objects = decode(Image.fromarray(processed_img))
        decoded_objects = decode(Image.fromarray(img))
        return decoded_objects

    def should_process_qr(self, data):
        """检查是否应该处理这个二维码（防重复机制）"""
        current_time = time.time()
        
        # 如果这个二维码最近被处理过，并且在冷却期内，则跳过
        if data in self.recently_processed:
            last_processed_time = self.recently_processed[data]
            if current_time - last_processed_time < self.cooldown_period:
                return False
        
        # 更新处理时间
        self.recently_processed[data] = current_time
        return True

    def cleanup_old_entries(self):
        """清理过期的记录，防止内存无限增长"""
        current_time = time.time()
        keys_to_remove = []
        
        for data, timestamp in self.recently_processed.items():
            if current_time - timestamp > self.cooldown_period * 2:  # 两倍冷却时间后清理
                keys_to_remove.append(data)
        
        for key in keys_to_remove:
            del self.recently_processed[key]

    def handle_medical_qr(self, data):
        """药瓶二维码处理逻辑"""
        if data in self.medical_db:
            medicine_info = self.medical_db[data]
            print(f"发送识别到药品二维码，{medicine_info}")

            # 通过记忆管理器共享信息
            self.memory_manager.set_shared_data(
                "last_detected_medicine",
                {"name": data, "info": medicine_info},
                "QRCodeDetector"
            )

            # 触发语音事件，兼容 speak_text / audio_file
            event_payload = {
                "medicine": data,
                "timestamp": time.time()
            }
            if isinstance(medicine_info, dict):
                # 新版JSON: 包含 speak_text 和 audio_file
                if "description" in medicine_info:
                    event_payload["speak_text"] = medicine_info["description"]
                if "audio_file" in medicine_info:
                    event_payload["audio_file"] = medicine_info["audio_file"]
            else:
                # 兼容旧版: 值就是字符串
                event_payload["speak_text"] = str(medicine_info)

            self.memory_manager.trigger_event("speak_event", event_payload)

        else:
            print(f"未找到该药品信息，请联系家人更新数据库,{data}")


    def detection(self):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            time.sleep(0.05)
            return  # 跳过部分帧以降低计算负载

        frame = self.camera_manager.get_frame()
        if frame is None:
            time.sleep(0.1)
            return
        
        #print("self.frame_count:",self.frame_count)    
        

        # 识别二维码/条形码
        decoded_objects = self.decode_codes(frame)
        
        # 处理所有识别到的二维码
        processed_codes = []
        for obj in decoded_objects:
            try:
                data = obj.data.decode('utf-8')
                print("识别结果：", data)
                
                # 检查是否应该处理这个二维码（防重复机制）
                if self.should_process_qr(data):
                    processed_codes.append(data)
                    # 逻辑处理
                    if data.startswith("med_"):  # 药瓶二维码
                        self.handle_medical_qr(data)
                    elif data.startswith("photo_"):  # 照片条码（示例）
                        self.speak("识别到老照片，正在加载回忆...")
                        
            except Exception as e:
                print(f"处理二维码时出错: {e}")
        
        # 清理过期的记录
        if self.frame_count % 60 == 0:  # 每30帧清理一次
            self.cleanup_old_entries()
            
        # 显示实时画面（调试用）
        # cv2.imshow('QR/Barcode Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
            
        time.sleep(0.1)  # 减少识别频次
            
    def run_detection(self):
        """主循环：实时识别并处理结果"""
        print("启动条形码和二维码识别")
        while True:
            self.detection()
        # 释放资源
        cv2.destroyAllWindows()
    def run(self):
        qr_thread = threading.Thread(target=self.run_detection, name="QR_Detector")
        qr_thread.start()


if __name__ == '__main__':
    print("二维码识别测试")
    import sys
    import os
    
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    
    from memory.memory_manager import MemoryManager
    from vision.camera_manager import CameraManager
    from speech.speech_engine import SpeechEngine

    
    memory_manager = MemoryManager()
    speech_engine = SpeechEngine(memory_manager)
    cam_manager = CameraManager()
    cam_manager.start()
    time.sleep(0.5)
    
    qr_code_detector = QRCodeDetector(cam_manager, memory_manager)
    qr_code_detector.run_detection()