#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
机器狗基于 YOLO 的人物跟随脚本（使用 CameraManager 取帧）

放在项目: src/track/person_follow_yolo.py

依赖:
    pip install ultralytics opencv-python
"""

import time
import cv2
from ultralytics import YOLO
import logging

class PersonFollower:
    def __init__(self,
                 robot,
                 camera_manager,
                 model_path: str = "yolov8n.pt"):
        """
        :param robot: 机器人控制对象，比如 control.control.RobotController 实例
        :param camera_manager: 摄像头管理对象，比如 vision.frames.camera_manager.CameraManager 实例
        :param model_path: YOLO 模型权重路径
        """
        self.robot = robot
        self.camera_manager = camera_manager

        # 先尝试从 CameraManager 拿一帧，确定分辨率
        frame = None
        for _ in range(50):   # 最多等待 ~2.5 秒
            frame = self.camera_manager.get_frame()

            if frame is not None:
                break
            time.sleep(0.05)

        if frame is not None:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            # 拿不到就用 CameraManager 默认的分辨率兜底
            self.frame_width = 800
            self.frame_height = 600

        # 加载 YOLO 模型
        self.model = YOLO(model_path)

        # 一些可调参数
        self.center_tolerance = 0.35        # 人的中心点左右偏移 10% 以内算居中
        self.target_min_height_ratio = 0.50   # 人框高度 / 画面高度 的最小值（太小 → 前进）
        self.target_max_height_ratio = 0.90   # 人框高度 / 画面高度 的最大值（太大 → 后退）
        self.no_person_search_interval = 50000   # 连续多少帧没看到人 → 原地转一圈搜寻

        self.no_person_count = 0

    # ---------------- YOLO 检测相关 ----------------
    def _find_main_person_box(self, results):
        """
        从 YOLO 结果中找面积最大的人（class=0）
        :return: (x1, y1, x2, y2) or None
        """
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        max_area = 0
        target_box = None

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO: 0 对应 person
            if cls_id != 0:
                continue
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                target_box = (x1, y1, x2, y2)

        return target_box

    # ---------------- 机器人控制相关 ----------------
    def _control_robot_by_box(self, box):
        """
        根据检测框位置控制机器人：
        - offset_x 决定 左/右 转动
        - height_ratio 决定 前进/后退
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        box_h = y2 - y1

        offset_x = (cx - self.frame_width / 2) / self.frame_width  # -0.5 ~ 0.5
        height_ratio = box_h / self.frame_height                  # 0 ~ 1

        print(f"[DEBUG] offset_x={offset_x:.3f}, height_ratio={height_ratio:.3f}")

        # 1) 先调整朝向：尽量保证人出现在中间
        if offset_x > self.center_tolerance:
            print("→ 目标在偏右，右转一点")
            # 这里用你已有的“移动时右转”接口，时间缩短成小角度调整
            self.robot.move_turn_right_90(duration=0.1, value=12000)
            time.sleep(0.1)

        elif offset_x < -self.center_tolerance:
            print("→ 目标在偏左，左转一点")
            self.robot.move_turn_left_90(duration=0.1, value=-12000)
            time.sleep(0.1)

        else:
            # 2) 人已经基本居中，再根据距离调整前后
            if height_ratio < self.target_min_height_ratio:
                print("→ 目标太远，前进一点")
                self.robot.forward(duration=0.4, value=13000)

            elif height_ratio > self.target_max_height_ratio:
                print("→ 目标太近，后退一点")
                self.robot.back(duration=0.4, value=-13000)

            else:
                print("→ 距离合适，保持原地")

    # ---------------- 主循环 ----------------
    def run(self):
        """
        主循环：从 CameraManager 取帧 + YOLO 检测 + 控制机器狗
        按下 'q' 退出
        """
        print("启动人物跟随 ... 按下 'q' 退出程序")

        # 起立 + 切换平地慢速（如果有这个模式的话）
        self.robot.stand_up()
        time.sleep(2)
        try:
            self.robot.change_flat_slow_mode()
        except AttributeError:
            # 没有该函数就忽略
            pass

        try:
            while True:
                frame = self.camera_manager.get_frame()
                # if frame is not None:
                #     save_dir = "output_frames"
                #     os.makedirs(save_dir, exist_ok=True)
                #     filename = os.path.join(save_dir, f"{int(time.time() * 1000)}.jpg")
                #     cv2.imwrite(filename, frame)
                #     print("保存到：", filename)
                # else:
                #     print("没有拿到帧，无法保存")
                if frame is None:
                    # 还没取到帧就稍等一下

                    time.sleep(0.02)
                    continue

                # YOLO 推理
                results = self.model(frame, verbose=False)
                main_box = self._find_main_person_box(results)

                if main_box is None:
                    self.no_person_count += 1
                    print(f"未检测到人 ({self.no_person_count})")

                    # 长时间没看到人，原地慢慢转圈找人
                    if self.no_person_count > self.no_person_search_interval:
                        print("长时间未发现目标，原地转一小圈搜索")
                        self.robot.move_turn_left_90(duration=0.5, value=-8000)
                        self.no_person_count = 0

                else:
                    self.no_person_count = 0
                    self._control_robot_by_box(main_box)

                    # 调试可视化
                    x1, y1, x2, y2 = main_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "TARGET", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Person Follow (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("检测到按键 q ，准备退出")
        finally:
            cv2.destroyAllWindows()
            print("已退出人物跟随模块")


# ---------------- 独立运行示例 ----------------
if __name__ == "__main__":
    import os
    import sys

    # 把项目根目录加入 sys.path，方便从 src 直接运行该脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    # 按你项目里的包结构导入
    from control.control import RobotController
    from vision.camera_manager import CameraManager

    logger = logging.getLogger(name='Log')
    # 创建摄像头管理器（你也可以改成 RTSP 地址等）
    # cam_manager = CameraManager(0)
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
    cam_manager.start()

    # 创建机器人控制对象（里面会自动启动 HeartBeat）
    robot = RobotController()

    follower = PersonFollower(
        robot=robot,
        camera_manager=cam_manager,
        model_path="yolov8n.pt"   # 确保本地能加载，或让 ultralytics 自动下载
    )

    try:
        follower.run()
    finally:
        cam_manager.stop()
        print("摄像头已停止")
