#!/usr/bin/env python3
#人脸识别
#import rospy   #ros相关的先不管
import cv2
import dlib
import numpy as np
import os
import time
import json
#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

class FaceDetector:
    def __init__(self, camera_manager,memory_manager):
        print("开始加载人脸识别")
        # ROS节点初始化
        #rospy.init_node('face_recognition_node', anonymous=True)
        #self.bridge = CvBridge()
        
        # 加载Dlib模型（路径需根据实际位置调整）
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("assets/face_detector/shape_predictor_68_face_landmarks.dat")
        self.face_recognizer = dlib.face_recognition_model_v1("assets/face_detector/dlib_face_recognition_resnet_model_v1.dat")
        
        # 已知人脸数据库
        self.known_faces = self.load_known_faces("assets/face_detector/known_faces")  # 存储人物照片的目录
        
        # ROS话题定义
        #self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        #self.result_pub = rospy.Publisher("/face_recognition/result", String, queue_size=10)
        self.camera_manager = camera_manager #使用的摄像头
        self.memory_manager = memory_manager # 添加 memory_manager 引用
        # 性能优化
        self.frame_skip = 50  # 如果为2，则每3帧处理1帧 50ms/帧
        self.frame_count = 0

        self.recently_processed = {}  # 记录最近处理的二维码和时间戳
        self.cooldown_period = 20  # 冷却时间（秒）


        # 存储识别的编码数据（人脸ID->语音映射）
        with open('assets/face_info.json', 'r', encoding='utf-8') as f:
            self.face_db = json.load(f)

    def load_known_faces(self, dir_path):
        """加载已知人脸的特征描述符"""
        known_faces = {}
        for file in os.listdir(dir_path):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(dir_path, file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = self.detector(img_rgb, 1)[0]  # 取第一张检测到的人脸
                shape = self.predictor(img_rgb, face)
                descriptor = np.array(self.face_recognizer.compute_face_descriptor(img_rgb, shape))
                known_faces[file.split(".")[0]] = descriptor  # 文件名作为人名（如"女儿.jpg"）
        return known_faces
    
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
    
    def run_detection(self):# + msg
        """处理摄像头图像"""
        print("启动人脸识别")
        while True:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                time.sleep(0.05)
                continue  # 跳过部分帧以降低计算负载
            
            
            try:
                # 转换ROS图像消息为OpenCV格式
                #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv_image=self.camera_manager.get_frame()

                cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                #print("人脸识别")
                # 人脸检测
                faces = self.detector(cv_image_rgb, 1)  # 不进行上采样（速度优先）
                
                for face in faces:
                    # 关键点检测与特征提取
                    shape = self.predictor(cv_image_rgb, face)
                    descriptor = np.array(self.face_recognizer.compute_face_descriptor(cv_image_rgb, shape))
                    
                    # 与已知人脸比对
                    match_name = "unknown"
                    min_distance = 0.6  # 相似度阈值（<0.6可视为同一人）
                    
                    for name, known_descriptor in self.known_faces.items():
                        distance = np.linalg.norm(descriptor - known_descriptor)
                        if distance < min_distance:
                            min_distance = distance
                            match_name = name
                    
                    # 发布识别结果
                    result_msg = f"识别结果:{match_name} (可信度: {1 - min_distance:.2f})"
                    print(result_msg)

                    if (1 - min_distance) <=0.6:
                        print("忽略人脸")
                        continue

                    
                    
                    # 在图像上绘制结果（调试用）
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if match_name != "unknown":
                        if self.should_process_qr(match_name):
                            speak_text = "数据库里没有该人的故事信息"
                            audio_file = None

                            if match_name in self.face_db:
                                face_info = self.face_db[match_name]
                                if isinstance(face_info, dict):  
                                    # 新格式：有可能包含 speak_text 和 audio_file
                                    speak_text = face_info.get("description", speak_text)
                                    audio_file = face_info.get("audio_file", None)
                                else:
                                    # 兼容旧格式：直接是文本
                                    speak_text = str(face_info)

                            print(f"发送识别到: {match_name} (可信度: {1 - min_distance:.2f})")
                            self.memory_manager.set_shared_data(
                                "last_recognized_face", 
                                {"name": match_name, "confidence": 1 - min_distance},
                                "FaceDetector"
                            )

                            # 发送给语音事件处理器
                            event_payload = {
                                "name": match_name,
                                "confidence": 1 - min_distance,
                                "speak_text": speak_text,
                                "timestamp": time.time()
                            }
                            if audio_file:   # 如果有音频文件则一并发过去
                                event_payload["audio_file"] = audio_file

                            self.memory_manager.trigger_event("speak_event", event_payload)
                    time.sleep(1)#防止一直识别成功

                # 清理过期的记录
                if self.frame_count % 60 == 0:  # 每60帧清理一次
                    self.cleanup_old_entries()
                
                time.sleep(0.1)#减少识别频次
                # 显示实时画面（可选）
                #cv2.imshow("Face Recognition", cv_image)
                #cv2.waitKey(1)
                
            except Exception as e:
                #rospy.logerr(f"处理图像失败: {str(e)}")
                print(f"处理图像失败: {str(e)}")

if __name__ == '__main__':
    print("人脸识别测试")
    import sys
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from camera_manager  import  CameraManager
    from memory.memory_manager import MemoryManager
    from speech.speech_engine import SpeechEngine
    
    cam_manager=CameraManager(0)
    cam_manager.start()

    memory_manager = MemoryManager()
    speech_engine = SpeechEngine(memory_manager)
    
    face_detector=FaceDetector(cam_manager,memory_manager)
    face_detector.run_detection()