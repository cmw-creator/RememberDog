#!/usr/bin/env python3
# 场景匹配器对象

import cv2
import numpy as np
import os
import time
import threading
import json

class PhotoDetector:
    def __init__(self, camera_manager, memory_manager):
        #场景匹配器初始化
        self.reference_folder = "assets/photo_detector"

        self.camera_manager = camera_manager #摄像头
        self.memory_manager = memory_manager  #记忆模块
        # 特征检测器和匹配器
        self.sift = cv2.SIFT_create()
        self.flann = self._create_flann_matcher()
        
        # 匹配参数
        self.min_match_count = 75  # 最小匹配点数
        self.ratio_threshold = 0.7  # Lowe's ratio测试阈值
        
        # 参考图像数据
        self.reference_images = []
        self._load_reference_images()
        
        # 运行状态
        self.running = False
        self.thread = None

        # 性能优化
        self.frame_skip = 60  # 如果为2，则每3帧处理1帧
        self.frame_count = 0

        self.recently_processed = {}  # 记录最近处理的二维码和时间戳
        self.cooldown_period = 20  # 冷却时间（秒）

        # 存储识别的编码数据（照片ID->语音映射）
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

    def run_detection(self):
        """主循环：实时匹配场景"""
        print("启动图片识别")
        self.running = True
        
        while self.running:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                time.sleep(0.01)
                continue  # 跳过部分帧以降低计算负载
            frame = self.camera_manager.get_frame()
            
            # 进行场景匹配
            match_result, debug_img = self.match_scene(frame)
            # 在图像上显示结果
            if match_result:
                photo_name = match_result['name'].split(".")[0]#去除文件后缀
                score = match_result['match_count']
                print(f"匹配到: {match_result['name']} ({match_result['match_count']} points)")
                if self.should_process_qr(photo_name):
                    print(f"发送匹配信息: {match_result['name']} ({match_result['match_count']} points)")
                    
                    if photo_name in self.photo_db:
                        entry = self.photo_db[photo_name]
                        # 兼容旧格式（字符串）
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

                    # 通过记忆管理器共享信息
                    self.memory_manager.set_shared_data(
                        "last_matched_scene",
                        {"photo_name": photo_name, "score": score},
                        "PhotoDetector"
                    )

                    # 触发语音事件（带 audio_file）
                    self.memory_manager.trigger_event("speak_event", {
                        "photo_name": photo_name,
                        "score": score,
                        "speak_text": speak_text,
                        "audio_file": audio_file,
                        "timestamp": time.time()
                    })
                # 显示匹配点图像
                #if debug_img is not None:
                #    cv2.imshow('Good Matches', debug_img)
                time.sleep(1)#防止一直识别成功
            else:
                # 关闭可能存在的匹配点窗口
                #if cv2.getWindowProperty('Good Matches', cv2.WND_PROP_VISIBLE) >= 1:
                #    cv2.destroyWindow('Good Matches')
                pass
            
            # 清理过期的记录
            if self.frame_count % 60 == 0:  # 每60帧清理一次
                self.cleanup_old_entries()

            time.sleep(0.1)#减少识别频次
            # 显示摄像头画面
            #cv2.imshow('Camera Feed', frame)
            #cv2.waitKey(1)
            
if __name__ == '__main__':
    print("图片识别测试")
    import sys
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from camera_manager  import  CameraManager
    from memory.memory_manager import MemoryManager
    from speech.speech_engine import SpeechEngine
    
    cam_manager=CameraManager()
    cam_manager.start()

    memory_manager = MemoryManager()
    speech_engine = SpeechEngine(memory_manager)
    
    photo_detector=PhotoDetector(cam_manager,memory_manager)
    photo_detector.run_detection()