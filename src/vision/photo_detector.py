#!/usr/bin/env python3
# 改进的场景匹配器对象

import cv2
import numpy as np
import os
import time
import threading
import json

class PhotoDetector:
    def __init__(self, camera_manager, memory_manager):
        # 场景匹配器初始化
        self.reference_folder = "assets/photo_detector"

        self.camera_manager = camera_manager # 摄像头
        self.memory_manager = memory_manager  # 记忆模块
        
        # 特征检测器和匹配器
        self.sift = cv2.SIFT_create()
        self.flann = self._create_flann_matcher()
        
        # 改进的匹配参数
        self.min_match_count = 80  # 降低最小匹配点数，但结合更严格的验证
        self.ratio_threshold = 0.4  # 更严格的Lowe's ratio测试阈值[5](@ref)
        self.ransac_threshold = 3.0  # RANSAC重投影误差阈值
        
        # 参考图像数据
        self.reference_images = []
        self._load_reference_images()
        
        # 运行状态
        self.running = False
        self.thread = None

        # 性能优化
        self.frame_skip = 20
        self.frame_count = 0

        self.recently_processed = {}
        self.cooldown_period = 20

        # 存储识别的编码数据
        with open('assets/photo_info.json', 'r', encoding='utf-8') as f:
            self.photo_db = json.load(f)
        
    def _create_flann_matcher(self):
        """创建FLANN匹配器"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 指定递归遍历的次数
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def _load_reference_images(self):
        """加载参考图像并提取特征"""
        if not os.path.exists(self.reference_folder):
            raise FileNotFoundError(f"参考图像目录 '{self.reference_folder}' 不存在")
        
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
            
            kp, des = self.sift.detectAndCompute(img, None)
            
            if des is None:
                print(f"警告：无法从图像 {img_file} 中提取特征描述符，跳过此文件")
                continue
            
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
    
    def _apply_geometric_verification(self, ref_kp, frame_kp, matches, min_inliers=10):
        """
        应用几何验证（RANSAC）来剔除误匹配点[3,5](@ref)
        
        参数:
            ref_kp: 参考图像的关键点
            frame_kp: 当前帧的关键点
            matches: 初步匹配结果
            min_inliers: 最小内点数
            
        返回:
            inlier_matches: 通过几何验证的匹配点
            inlier_count: 内点数量
            homography: 单应性矩阵
        """
        if len(matches) < min_inliers:
            return [], 0, None
            
        try:
            # 提取匹配点的坐标
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 使用RANSAC计算单应性矩阵[3](@ref)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                                self.ransac_threshold)
            
            if homography is None or mask is None:
                return [], 0, None
                
            # 提取内点
            inlier_count = np.sum(mask)
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
            
            return inlier_matches, inlier_count, homography
            
        except Exception as e:
            print(f"几何验证错误: {e}")
            return [], 0, None
    
    def _calculate_matching_quality(self, matches, inlier_matches, homography):
        """
        计算匹配质量分数，综合考虑多个因素[1,2](@ref)
        """
        if len(matches) == 0 or len(inlier_matches) == 0:
            return 0.0
            
        # 1. 内点比例
        inlier_ratio = len(inlier_matches) / len(matches)
        
        # 2. 匹配点分布均匀性（避免局部集中）
        if homography is not None and len(inlier_matches) > 4:
            try:
                src_pts = np.float32([inlier_matches[i].queryIdx for i in range(len(inlier_matches))])
                if len(src_pts) > 0:
                    # 计算关键点分布的方差
                    distribution_score = 1.0  # 简化处理，实际可计算空间分布
                else:
                    distribution_score = 0.5
            except:
                distribution_score = 0.5
        else:
            distribution_score = 0.5
            
        # 3. 匹配点距离质量
        avg_distance = np.mean([m.distance for m in inlier_matches]) if inlier_matches else 1.0
        distance_score = 1.0 / (1.0 + avg_distance)  # 距离越小分数越高
        
        # 综合质量分数
        quality_score = (inlier_ratio * 0.5 + distribution_score * 0.2 + distance_score * 0.3)
        
        return quality_score
    
    def match_scene(self, frame):
        """
        改进的场景匹配方法，增加几何验证和质量评估
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
        kp_frame, des_frame = self.sift.detectAndCompute(gray_frame, None)
        
        if des_frame is None or len(des_frame) < 10:
            return None, None
        
        best_match = None
        best_quality = 0.0
        best_inlier_count = 0
        best_matches = None
        best_homography = None
        
        for ref in self.reference_images:
            if ref['descriptors'] is None or len(ref['descriptors']) < 10:
                continue
            
            try:
                matches = self.flann.knnMatch(ref['descriptors'], des_frame, k=2)
            except Exception as e:
                continue
            
            # 应用比率测试[5](@ref)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.min_match_count:
                continue
            
            # 应用几何验证（RANSAC）[3](@ref)
            inlier_matches, inlier_count, homography = self._apply_geometric_verification(
                ref['keypoints'], kp_frame, good_matches)
            
            if inlier_count < max(self.min_match_count // 2, 10):
                continue
            
            # 计算匹配质量
            quality = self._calculate_matching_quality(good_matches, inlier_matches, homography)
            
            # 综合考虑内点数量和质量分数
            combined_score = inlier_count * quality
            
            if combined_score > best_quality:
                best_quality = combined_score
                best_match = ref
                best_inlier_count = inlier_count
                best_matches = inlier_matches  # 使用内点进行后续处理
                best_homography = homography
        
        match_result = None
        debug_img = None
        
        if best_match is not None and best_quality > 50:  # 质量阈值
            match_result = {
                'name': best_match['name'],
                'match_count': best_inlier_count,
                'quality': best_quality
            }
            
            # 创建调试图像（只显示内点）
            ref_img_color = cv2.cvtColor(best_match['image'], cv2.COLOR_GRAY2BGR)
            debug_img = cv2.drawMatches(
                ref_img_color, best_match['keypoints'],
                frame, kp_frame,
                best_matches[:50], None,  # 只显示前50个内点
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0), singlePointColor=None
            )
        
        return match_result, debug_img
    
    def should_process_qr(self, data):
        """检查是否应该处理这个二维码（防重复机制）"""
        current_time = time.time()
        
        if data in self.recently_processed:
            last_processed_time = self.recently_processed[data]
            if current_time - last_processed_time < self.cooldown_period:
                return False
        
        self.recently_processed[data] = current_time
        return True

    def cleanup_old_entries(self):
        """清理过期的记录，防止内存无限增长"""
        current_time = time.time()
        keys_to_remove = []
        
        for data, timestamp in self.recently_processed.items():
            if current_time - timestamp > self.cooldown_period * 2:
                keys_to_remove.append(data)
        
        for key in keys_to_remove:
            del self.recently_processed[key]

    def run_detection(self):
        """主循环：实时匹配场景"""
        print("启动改进的图片识别")
        self.running = True
        
        while self.running:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                time.sleep(0.05)
                continue
                
            frame = self.camera_manager.get_frame()
            
            match_result, debug_img = self.match_scene(frame)
            
            if match_result:
                photo_name = match_result['name'].split(".")[0]
                score = match_result['match_count']
                quality = match_result['quality']
                
                print(f"高质量匹配: {match_result['name']} (内点: {score}, 质量: {quality:.2f})")
                
                if self.should_process_qr(photo_name):
                    print(f"发送匹配信息: {match_result['name']}")
                    
                    if photo_name in self.photo_db:
                        entry = self.photo_db[photo_name]
                        if isinstance(entry, str):
                            speak_text = entry
                            audio_file = None
                        elif isinstance(entry, dict):
                            speak_text = entry.get("description", "数据库里没有该图片的故事信息哦")
                            audio_file = entry.get("audio_file", None)
                        else:
                            speak_text = "数据库数据格式错误"
                            audio_file = None
                    else:
                        speak_text = "数据库里没有该图片的故事信息"
                        audio_file = None

                    self.memory_manager.set_shared_data(
                        "last_matched_scene",
                        {"photo_name": photo_name, "score": score, "quality": quality},
                        "PhotoDetector"
                    )

                    self.memory_manager.trigger_event("speak_event", {
                        "photo_name": photo_name,
                        "score": score,
                        "speak_text": speak_text,
                        "audio_file": audio_file,
                        "timestamp": time.time()
                    })
                
                time.sleep(1)
            else:
                pass
            
            if self.frame_count % 60 == 0:
                self.cleanup_old_entries()

            time.sleep(0.1)

if __name__ == '__main__':
    print("改进的图片识别测试")
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from camera_manager import CameraManager
    from memory.memory_manager import MemoryManager
    from speech.speech_engine import SpeechEngine
    
    cam_manager = CameraManager(0)
    cam_manager.start()

    memory_manager = MemoryManager()
    
    
    photo_detector = PhotoDetector(cam_manager, memory_manager)
    photo_detector.run_detection()
    speech_engine = SpeechEngine(memory_manager)