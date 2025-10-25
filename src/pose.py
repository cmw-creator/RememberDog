# pose.py
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from collections import deque

class FallDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 buffer_size=5, fall_drop_threshold_ratio=0.25):
        """
        buffer_size: 连续帧投票长度
        fall_drop_threshold_ratio: 头部在几帧内下降超过图像高度比例，判定快速下落
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)

        self.head_history = deque(maxlen=buffer_size)  # 记录头部 y 坐标
        self.fall_drop_threshold_ratio = fall_drop_threshold_ratio

    def detect_fall(self, landmarks, image_height, image_width):
        """
        跌倒判别规则：
        1. 身体外接矩形纵横比
        2. 头部与髋关节高度差
        3. 躯干倾斜角度
        4. 快速下落检测
        """
        try:
            # 关键点像素坐标
            points = [(lm.x * image_width, lm.y * image_height) for lm in landmarks]

            # ---- 1. 身体bounding box纵横比 ----
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            aspect_ratio = width / (height + 1e-5)

            # ---- 2. 头部与髋关节高度 ----
            nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * image_height
            l_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
            r_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
            hip_y = (l_hip_y + r_hip_y) / 2
            head_vs_hip = nose_y - hip_y  # 头部比髋关节低 → 躺下

            # ---- 3. 躯干倾斜角度 ----
            l_sh = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

            mid_shoulder = ((l_sh.x + r_sh.x) / 2 * image_width,
                            (l_sh.y + r_sh.y) / 2 * image_height)
            mid_hip = ((l_hip.x + r_hip.x) / 2 * image_width,
                       (l_hip.y + r_hip.y) / 2 * image_height)

            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_shoulder[1] - mid_hip[1]
            angle = np.degrees(np.arctan2(dy, dx))  # 躯干倾角

            # ---- 4. 快速下落检测 ----
            self.head_history.append(nose_y)
            fast_drop = False
            if len(self.head_history) == self.head_history.maxlen:
                dy_drop = self.head_history[-1] - self.head_history[0]
                if dy_drop > image_height * self.fall_drop_threshold_ratio:
                    fast_drop = True

            # ---- 综合判定 ----
            fall_flag = False
            if aspect_ratio > 1.0:  # 横着
                fall_flag = True
            elif head_vs_hip > -image_height * 0.05:  # 头部接近髋关节高度
                if fast_drop:  # 头部快速下降才算跌倒
                    fall_flag = True
            elif abs(angle) < 30 or abs(angle) > 150:  # 躯干水平
                if fast_drop:
                    fall_flag = True

            return "跌倒" if fall_flag else "未跌倒"

        except Exception:
            return "未跌倒"

    def smooth_status(self, status):
        self.history.append(status)
        if self.history.count("跌倒") > self.buffer_size // 2:
            return "跌倒"
        return "未跌倒"

    def run(self, source=0):
        """
        source:
            - int: 摄像头编号 (0 默认摄像头)
            - str: 视频文件路径
        """
        cap = cv2.VideoCapture(source)
        frame_id = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_id += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            status = "未检测到人"
            if results.pose_landmarks:
                h, w, _ = frame.shape
                raw_status = self.detect_fall(results.pose_landmarks.landmark, h, w)
                status = self.smooth_status(raw_status)

                # 绘制骨架（仅窗口可视化）
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

            # 终端打印：时间戳 + 帧号 + 状态
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Frame {frame_id}: {status}")

            cv2.imshow('Fall Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC 退出
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FallDetector()
    # 摄像头
    # detector.run(0)
    # 本地视频
    detector.run(0)
    #"E:/learning materials/ican/diedao/ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-d3edb5d/Subject 1/FALL/01.mp4"
