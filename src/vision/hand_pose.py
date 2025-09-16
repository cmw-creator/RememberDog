# -*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Real-time Hand Pose Inference from Camera

import os
import sys
sys.path.append(r'C:\Users\wcm\Desktop\ICAN\RememberDog\assets\handpose_x')
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
    def __init__(self, camera_manager, model_path='./weights/ReXNetV1-size-256-wingloss102-0.122.pth', 
                 model='ReXNetV1', num_classes=42, GPUS='0', img_size=(256, 256)):
        self.camera_manager = camera_manager
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

    def run_estimation(self, camera_id=0, vis=True):
        # 初始化摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return False
            
        print("Press 'q' to quit, 's' to save current frame")

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
                    
                # 绘制手部关键点和连线
                draw_bd_handpose(img, pts_hand, 0, 0)

                for i in range(int(output.shape[0] / 2)):
                    x = (output[i * 2 + 0] * float(img_width))
                    y = (output[i * 2 + 1] * float(img_height))

                    cv2.circle(img, (int(x), int(y)), 3, (255, 50, 60), -1)
                    cv2.circle(img, (int(x), int(y)), 1, (255, 150, 180), -1)

                # 显示FPS
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if vis:
                    cv2.imshow('Hand Pose Estimation', img)
                    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    cv2.imwrite(f"saved_frame_{frame_count}.jpg", img)
                    print(f"Frame saved as saved_frame_{frame_count}.jpg")

        cap.release()
        cv2.destroyAllWindows()
        print('Real-time hand pose estimation completed')
        return True

# 保留原有命令行接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default='./weights/ReXNetV1-size-256-wingloss102-0.122.pth',
                        help='model_path')
    parser.add_argument('--model', type=str, default='ReXNetV1',
                        help='model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2,shufflenet_v2_x1_5,shufflenet_v2_x1_0,shufflenet_v2_x2_0,ReXNetV1')
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID (usually 0 for built-in camera)')
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')

    ops = parser.parse_args()
    
    estimator = HandPoseEstimator(
        model_path=ops.model_path,
        model=ops.model,
        num_classes=ops.num_classes,
        GPUS=ops.GPUS,
        img_size=ops.img_size
    )
    
    estimator.run_estimation(camera_id=ops.camera_id, vis=ops.vis)