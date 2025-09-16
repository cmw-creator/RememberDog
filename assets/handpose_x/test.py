# -*-coding:utf-8-*-
# date:2021-04-5
# Author: Eric.Lee
# function: Real-time Hand Pose Inference from Camera

import os
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default='./weights/ReXNetV1-size-256-wingloss102-0.122.pth',
                        help='model_path')  # 模型路径
    parser.add_argument('--model', type=str, default='ReXNetV1',
                        help='''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
                            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''')  # 模型类型
    parser.add_argument('--num_classes', type=int, default=42,
                        help='num_classes')  # 手部21关键点，(x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')  # GPU选择
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID (usually 0 for built-in camera)')  # 摄像头ID
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')  # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    ops = parser.parse_args()
    print('----------------------------------')

    unparsed = vars(ops)
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    # 构建模型
    print('use model : %s' % (ops.model))
    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False, num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks=[2, 4, 2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)
    elif ops.model == "ReXNetV1":
        model_ = ReXNetV1(num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()

    if os.access(ops.model_path, os.F_OK):
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    # 初始化摄像头 [6](@ref)
    cap = cv2.VideoCapture(ops.camera_id)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
        
    print("Press 'q' to quit, 's' to save current frame")

    with torch.no_grad():
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame_count += 1
            img = frame
            img_width = img.shape[1]
            img_height = img.shape[0]

            # 输入图片预处理
            img_ = cv2.resize(img, (ops.img_size[1], ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()
                
            pre_ = model_(img_.float())
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

            # 显示FPS [3](@ref)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ops.vis:
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