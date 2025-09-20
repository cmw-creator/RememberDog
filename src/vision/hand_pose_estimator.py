# -*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Real-time Hand Pose Inference from Camera
import time
import os
import sys
sys.path.append("assets/handpose_x")
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.squeezenet import squeezenet1_1, squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5, shufflenet_v2_x1_0, shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1
from hand_data_iter.datasets import draw_bd_handpose

class HandPoseEstimator:
    def __init__(self, camera_manager,memory_manager, model_path='./weights/ReXNetV1-size-256-wingloss102-0.122.pth', 
                 model='ReXNetV1', num_classes=42, GPUS='0', img_size=(256, 256)):
        self.camera_manager = camera_manager
        self.memory_manager = memory_manager 
        self.model_path = model_path
        self.model_type = model
        self.num_classes = num_classes
        self.GPUS = GPUS
        self.img_size = img_size
        
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPUS
        
        # 构建模型
        print('use model : %s' % (self.model_type))
        if self.model_type == 'resnet_50':
            self.model = resnet50(num_classes=self.num_classes, img_size=self.img_size[0])
        elif self.model_type == 'resnet_18':
            self.model = resnet18(num_classes=self.num_classes, img_size=self.img_size[0])
        elif self.model_type == 'resnet_34':
            self.model = resnet34(num_classes=self.num_classes, img_size=self.img_size[0])
        elif self.model_type == 'resnet_101':
            self.model = resnet101(num_classes=self.num_classes, img_size=self.img_size[0])
        elif self.model_type == "squeezenet1_0":
            self.model = squeezenet1_0(num_classes=self.num_classes)
        elif self.model_type == "squeezenet1_1":
            self.model = squeezenet1_1(num_classes=self.num_classes)
        elif self.model_type == "shufflenetv2":
            self.model = ShuffleNetV2(ratio=1., num_classes=self.num_classes)
        elif self.model_type == "shufflenet_v2_x1_5":
            self.model = shufflenet_v2_x1_5(pretrained=False, num_classes=self.num_classes)
        elif self.model_type == "shufflenet_v2_x1_0":
            self.model = shufflenet_v2_x1_0(pretrained=False, num_classes=self.num_classes)
        elif self.model_type == "shufflenet_v2_x2_0":
            self.model = shufflenet_v2_x2_0(pretrained=False, num_classes=self.num_classes)
        elif self.model_type == "shufflenet":
            self.model = ShuffleNet(num_blocks=[2, 4, 2], num_classes=self.num_classes, groups=3)
        elif self.model_type == "mobilenetv2":
            self.model = MobileNetV2(num_classes=self.num_classes)
        elif self.model_type == "ReXNetV1":
            self.model = ReXNetV1(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        if os.access(self.model_path, os.F_OK):
            chkpt = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(chkpt)
            print('load test model : {}'.format(self.model_path))
        else:
            print(f"Warning: Model path {self.model_path} does not exist")


    def recognize_gesture(self, pts_hand):
        """
        根据手部关键点识别手势
        返回: 手势名称字符串
        """
        # 获取关键点坐标
        wrist = np.array([pts_hand['0']['x'], pts_hand['0']['y']])  # 手腕
        
        # 指尖关键点索引 (拇指、食指、中指、无名指、小指)
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertips = [np.array([pts_hand[str(i)]['x'], pts_hand[str(i)]['y']]) 
                     for i in fingertip_indices]
        
        # 1. 首先识别手掌张开状态 [6](@ref)
        # 计算所有指尖与手腕的距离
        dist_threshold = 100  # 距离阈值，可能需要根据实际情况调整
        is_open = True
        for fingertip in fingertips:
            distance = np.linalg.norm(fingertip - wrist)
            if distance < dist_threshold:
                is_open = False
                break
        
        if is_open:
            return "手掌张开"
        
        # 2. 识别数字手势 (基于手指伸直状态)
        # 获取各手指的中间关节点（用于判断手指弯曲）
        finger_joints = {
            'thumb': [1, 2, 3, 4],      # 拇指关节
            'index': [5, 6, 7, 8],      # 食指关节
            'middle': [9, 10, 11, 12],  # 中指关节
            'ring': [13, 14, 15, 16],   # 无名指关节
            'pinky': [17, 18, 19, 20]   # 小指关节
        }
        
        # 判断每个手指是否伸直
        extended_fingers = []
        for finger, joints in finger_joints.items():
            # 获取手指上的点
            points = [wrist] + [np.array([pts_hand[str(i)]['x'], pts_hand[str(i)]['y']]) 
                             for i in joints]
            
            # 计算手指方向向量
            vec1 = points[1] - points[0]  # 手腕到第一关节
            vec2 = points[2] - points[1]  # 第一到第二关节
            vec3 = points[3] - points[2]  # 第二到第三关节
            vec4 = points[4] - points[3]  # 第三到指尖
            
            # 计算相邻向量的角度余弦值
            def angle_cos(vec_a, vec_b):
                return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-6)
            
            cos1 = angle_cos(vec1, vec2)
            cos2 = angle_cos(vec2, vec3)
            cos3 = angle_cos(vec3, vec4)
            
            # 如果角度较小（余弦值较小），说明手指较直
            if cos1 < 0.5 and cos2 < 0.5 and cos3 < 0.5:
                extended_fingers.append(finger)
        
        # 根据伸直的手指判断手势
        extended_count = len(extended_fingers)
        
        if extended_count == 1 and 'index' in extended_fingers:
            return "数字一"
        elif extended_count == 2 and 'index' in extended_fingers and 'middle' in extended_fingers:
            return "数字二"
        elif extended_count == 3 and 'index' in extended_fingers and 'middle' in extended_fingers and 'ring' in extended_fingers:
            return "数字三"
        
        return "未知手势"

    def run_estimation(self):
        # 初始化摄像头
        print("启动手势识别")
        with torch.no_grad():
            frame_count = 0
            while True:
                frame = self.camera_manager.get_frame()

                frame_count += 1
                img = frame
                img_width = img.shape[1]
                img_height = img.shape[0]

                # 输入图片预处理
                img_ = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)
                img_ = img_.astype(np.float32)
                img_ = (img_ - 128.) / 256.

                img_ = img_.transpose(2, 0, 1)
                img_ = torch.from_numpy(img_)
                img_ = img_.unsqueeze_(0)

                if self.use_cuda:
                    img_ = img_.cuda()
                    
                pre_ = self.model(img_.float())
                output = pre_.cpu().detach().numpy()
                output = np.squeeze(output)

                pts_hand = {}
                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))

                    pts_hand[str(i)] = {
                        "x": x,
                        "y": y,
                    }
                #识别手势，包括手掌张开，一 二 三的手势
                gesture = self.recognize_gesture(pts_hand)
                print(gesture)
                # 通过记忆管理器共享手势信息
                if self.memory_manager and gesture != "未知手势":
                    self.memory_manager.set_shared_data(
                        "last_detected_gesture",
                        {"gesture": gesture, "details": pts_hand},
                        "HandPoseEstimator"
                    )
                    self.memory_manager.trigger_event("gesture_detected", {
                        "gesture": gesture,
                        "timestamp": time.time()
                    })

                # 绘制手部关键点和连线
                draw_bd_handpose(img, pts_hand, 0, 0)

                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))

                    cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                    cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

                cv2.imshow('Hand Pose Estimation', img)
                    
                key = cv2.waitKey(1) & 0xFF
                #if key == ord('q'):
                #    break
                #elif key == ord('s'):
                    # 保存当前帧
                    #cv2.imwrite(f"saved_frame_{frame_count}.jpg", img)
                    #print(f"Frame saved as saved_frame_{frame_count}.jpg")
                time.sleep(1)#减少识别频次

        cap.release()
        cv2.destroyAllWindows()
        print('Real-time hand pose estimation completed')
        return True



if __name__ == '__main__':
    print("手势识别测试")
    from camera_manager  import  CameraManager
    cam_manager=CameraManager()
    cam_manager.start()
    time.sleep(2)#等待摄像头启动
    hand_estimator = HandPoseEstimator(
        cam_manager,
#        model_path="./assets/handpose_x/weights/ReXNetV1-size-256-wingloss102-0.122.pth", #模型更大
#        model='ReXNetV1',
        model_path="./assets/handpose_x/weights/squeezenet1_1-size-256-loss-0.0732.pth",   #轻量模型
        model='squeezenet1_1',
        num_classes=42,
        GPUS='0',
        img_size=(256, 256)
    )
    hand_estimator.run_estimation()