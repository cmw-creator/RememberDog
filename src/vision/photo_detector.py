#!/usr/bin/env python3
# 场景匹配器对象

import cv2
import numpy as np
import os
import time
import threading

class PhotoDetector:
    def __init__(self, camera_manager):
        #场景匹配器初始化
        self.reference_folder = "assets/photo_detector"

        self.camera_manager = camera_manager
        # 特征检测器和匹配器
        self.sift = cv2.SIFT_create()
        self.flann = self._create_flann_matcher()
        
        # 匹配参数
        self.min_match_count = 50  # 最小匹配点数
        self.ratio_threshold = 0.7  # Lowe's ratio测试阈值
        
        # 参考图像数据
        self.reference_images = []
        self._load_reference_images()
        
        # 运行状态
        self.running = False
        self.thread = None
        
    def _create_flann_matcher(self):
        """创建FLANN匹配器"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 指定递归遍历的次数
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def _load_reference_images(self):
        """加载参考图像并提取特征"""
        # 检查参考图像目录是否存在
        if not os.path.exists(self.reference_folder):
            raise FileNotFoundError(f"参考图像目录 '{self.reference_folder}' 不存在")
        
        # 获取目录下所有图片文件
        image_files = [f for f in os.listdir(self.reference_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            raise ValueError(f"在目录 '{self.reference_folder}' 中未找到任何图片文件")
        
        print(f"在目录 '{self.reference_folder}' 中找到 {len(image_files)} 个图片文件。开始加载...")
        
        for img_file in image_files:
            img_path = os.path.join(self.reference_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"警告：无法加载图像 {img_path}，跳过此文件")
                continue
            
            # 提取关键点和描述符
            kp, des = self.sift.detectAndCompute(img, None)
            
            if des is None:
                print(f"警告：无法从图像 {img_file} 中提取特征描述符，跳过此文件")
                continue
            
            # 存储参考图像信息
            self.reference_images.append({
                'name': img_file,
                'keypoints': kp,
                'descriptors': des,
                'image': img
            })
            print(f"成功加载并处理图像: {img_file}")
        
        if not self.reference_images:
            raise RuntimeError("没有成功加载任何有效的参考图像")
        
        print(f"成功加载 {len(self.reference_images)} 张参考图像")
    
    def match_scene(self, frame):
        """
        在给定帧中匹配场景
        
        参数:
            frame: 输入图像帧 (BGR格式)
        
        返回:
            match_result: 匹配结果字典，包含匹配的参考图像名称和匹配点数
            debug_img: 调试用图像 (可选)
        """
        # 转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 提取当前帧特征
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        if des_frame is None:
            return None, None
        
        # 初始化最佳匹配变量
        best_match = None
        max_good_matches = 0
        best_matches = None
        
        # 遍历所有参考图像进行匹配
        for ref in self.reference_images:
            if ref['descriptors'] is None:
                continue
            
            try:
                # 使用KNN匹配
                matches = self.flann.knnMatch(ref['descriptors'], des_frame, k=2)
            except Exception:
                continue
            
            # 应用比率测试筛选好的匹配点
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:  # 确保每个配对都有两个匹配点
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # 检查是否找到足够多的匹配点
            if len(good_matches) > self.min_match_count:
                # 如果当前参考图像的匹配点数最多，则记录它
                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = ref
                    best_matches = good_matches
        
        # 准备返回结果
        match_result = None
        debug_img = None
        
        if best_match is not None:
            match_result = {
                'name': best_match['name'],
                'match_count': max_good_matches
            }
            
            # 创建调试图像
            ref_img_color = cv2.cvtColor(best_match['image'], cv2.COLOR_GRAY2BGR)
            debug_img = cv2.drawMatches(
                ref_img_color, best_match['keypoints'],
                frame, kp_frame,
                best_matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0), singlePointColor=None
            )
        
        return match_result, debug_img
    
    def run(self):
        """主循环：实时匹配场景"""
        self.running = True
        
        while self.running:
            frame = self.camera_manager.get_frame()
            
            # 进行场景匹配
            match_result, debug_img = self.match_scene(frame)
            
            # 在图像上显示结果
            if match_result:
                match_text = f"Matched: {match_result['name']} ({match_result['match_count']} points)"
                cv2.putText(frame, match_text, (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示匹配点图像
                if debug_img is not None:
                    cv2.imshow('Good Matches', debug_img)
            else:
                cv2.putText(frame, "No match found", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 关闭可能存在的匹配点窗口
                if cv2.getWindowProperty('Good Matches', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Good Matches')
            
            # 显示摄像头画面
            cv2.imshow('Camera Feed', frame)
            


if __name__ == '__main__':
    try:
        # 创建场景匹配器
        matcher = PhotoDetector(reference_folder)
        
        # 启动匹配
        matcher.run()
    except Exception as e:
        print(f"错误: {e}")