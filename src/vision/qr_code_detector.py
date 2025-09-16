#!/usr/bin/env python3
#识别二维码和条形码，二维码识别效果更好一点
#import rospy
import cv2
import numpy as np
import os
from datetime import datetime
from pyzbar.pyzbar import decode
from PIL import Image
import pyttsx3  # 离线TTS引擎
import threading

class QRCodeDetector:
    def __init__(self, camera_manager):
        # 初始化ROS节点
        #rospy.init_node('qr_barcode_detector', anonymous=True)
        
        # 硬件配置
        #self.camera = cv2.VideoCapture(0)  # 使用默认摄像头（广角相机）
        #self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # 设置分辨率
        #self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.camera_manager = camera_manager
        # 语音引擎初始化
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 语速调节
        
        # 存储识别的编码数据（药瓶二维码->语音映射）
        self.medical_db = {
            "med_12345": "降压药，每日早饭后服用1粒",
            "med_67890": "维生素D，每日中饭后服用2粒"
        }
        
        # 异步语音线程控制
        self.speaking = False
    
    #预处理有问题，现在不用
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
        processed_img = self.preprocess_image(img) #预处理
        #decoded_objects = decode(Image.fromarray(processed_img)) #预处理有问题，现在不用
        decoded_objects = decode(Image.fromarray(img))
        return decoded_objects

    def speak(self, text):
        """异步语音播报（避免阻塞主线程）"""
        def run():
            self.speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.speaking = False
        threading.Thread(target=run).start()

    def handle_medical_qr(self, data):
        """药瓶二维码处理逻辑"""
        if data in self.medical_db:
            medicine_info = self.medical_db[data]
            self.speak(f"识别到药品二维码，{medicine_info}")
        else:
            self.speak("未找到该药品信息，请联系家人更新数据库")

    def run_detection(self):
        """主循环：实时识别并处理结果"""
        #while not rospy.is_shutdown() and self.camera.isOpened():
        while True:
            frame = self.camera_manager.get_frame()
            
            # 识别二维码/条形码
            decoded_objects = self.decode_codes(frame)
            
            # 在图像上绘制结果
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                print("识别结果：",data)
                # 绘制边框和文字
                points = obj.polygon
                if len(points) >= 4:
                    hull = cv2.convexHull(np.array(points, dtype=np.int32))
                    cv2.polylines(frame, [hull], True, (0, 255, 0), 3)
                    cv2.putText(frame, data, (obj.rect.left, obj.rect.top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 逻辑处理
                if data.startswith("med_"):  # 药瓶二维码
                    self.handle_medical_qr(data)
                elif data.startswith("photo_"):  # 照片条码（示例）
                    self.speak("识别到老照片，正在加载回忆...")
            
            # 显示实时画面（调试用）
            cv2.imshow('QR/Barcode Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # 释放资源
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pass