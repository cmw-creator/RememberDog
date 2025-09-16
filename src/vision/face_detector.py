#!/usr/bin/env python3
#人脸识别
#import rospy   #ros相关的先不管
import cv2
import dlib
import numpy as np
import os
import time
#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

class FaceDetector:
    def __init__(self, camera_manager):
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
        # 性能优化
        self.frame_skip = 2  # 如果为2，则每3帧处理1帧
        self.frame_count = 0

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

    def run(self):# + msg
        """处理摄像头图像"""
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return  # 跳过部分帧以降低计算负载
        
        
        try:
            # 转换ROS图像消息为OpenCV格式
            #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image=self.camera_manager.get_frame()

            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 人脸检测
            faces = self.detector(cv_image_rgb, 0)  # 不进行上采样（速度优先）
            
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
                result_msg = f"识别到: {match_name} (可信度: {1 - min_distance:.2f})"
                print(result_msg)
                #self.result_pub.publish(result_msg)
                
                
                # 在图像上绘制结果（调试用）
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if match_name!="unknown":
                    time.sleep(1)#防止一直识别成功
            
            time.sleep(0.1)#减少识别频次
            # 显示实时画面（可选）
            #cv2.imshow("Face Recognition", cv_image)
            #cv2.waitKey(1)
            
        except Exception as e:
            #rospy.logerr(f"处理图像失败: {str(e)}")
            print(f"处理图像失败: {str(e)}")
    def run_detection(self):
        print("启动人脸识别")
        while True:
            self.run()

if __name__ == '__main__':
    print("人脸识别测试")
    from camera_manager  import  CameraManager
    cam_manager=CameraManager()
    cam_manager.start()
    
    face_detector=FaceDetector(cam_manager)
    face_detector.run_detection()