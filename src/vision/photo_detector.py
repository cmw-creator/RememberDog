#!/usr/bin/env python3
# 场景匹配器对象

import cv2
import numpy as np
import os
import time
import threading

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
        self.min_match_count = 50  # 最小匹配点数
        self.ratio_threshold = 0.7  # Lowe's ratio测试阈值
        
        # 参考图像数据
        self.reference_images = []
        self._load_reference_images()
        
        # 运行状态
        self.running = False
        self.thread = None

        # 性能优化
        self.frame_skip = 20  # 如果为2，则每3帧处理1帧
        self.frame_count = 0

        self.recently_processed = {}  # 记录最近处理的二维码和时间戳
        self.cooldown_period = 20  # 冷却时间（秒）

        # 存储识别的编码数据（照片ID->语音映射），现在先写在这，以后读取文件获取
        self.photo_db = {
            "photo_00001": "这是您去年在人民英雄纪念碑前的照片。那天是国庆节，您穿着那件深蓝色的外套，戴着您最喜欢的帽子。天气特别好，阳光明媚，您还记得吗？您在那里站了很久，说想起了年轻时候的事情。回来后您还特意把这张照片洗出来放在相册的第一页。",
            "photo_00002": "这是您七十岁生日时的全家福照片。所有的孩子和孙子们都回来了，大家围着您唱生日歌。您看，大女儿特意从上海赶回来，还带了您最爱吃的蛋糕。小孙子当时才五岁，正调皮地想要抓蛋糕上的奶油呢。这张照片记录了全家团聚的幸福时刻。",
            "photo_00003": "这是您和老伴金婚纪念日时在西湖边拍的照片。您穿着中山装，老伴穿着红色的旗袍，两人手牵着手站在断桥上。那天湖面上的荷花正好开了，微风拂过，花瓣轻轻摇曳。您说那是你们第五次去西湖，每次去都有不同的回忆。",
            "photo_00004": "这是您退休时学校同事们为您举办的欢送会照片。您站在讲台前，手里拿着学生们送的鲜花和纪念册。您在教育岗位上工作了整整四十年，培养了一代又一代的学生。很多学生现在都成了各行各业的精英，他们经常回来看望您。",
            "photo_00005": "这是您第一次抱孙子的照片。您小心翼翼地抱着刚满月的小孙子，脸上洋溢着幸福的笑容。您还记得吗？那天您特别紧张，手都有些发抖，但还是不肯让别人帮忙。小孙子现在都已经上小学了，时间过得真快啊。",
            "photo_00006": "这是您参加社区书法比赛获得一等奖的作品和照片。您的楷书写得工整有力，评委们都说很有颜真卿的风骨。您从六十岁开始学习书法，每天坚持练习两个小时，这份坚持和毅力真是令人敬佩。",
            "photo_00007": "这是您和老朋友们在公园下棋的照片。每周二和周四下午，您都会去公园和老张、老李他们下几盘象棋。您最拿手的是用马后炮，经常杀得他们措手不及。这张照片是去年春天拍的，旁边的樱花树开得正盛。",
            "photo_00008": "这是您全家第一次去北京旅游的照片。站在天安门广场上，您显得特别激动。您说这是您从小的心愿，终于实现了。还记得那天风很大，您的帽子差点被吹走，是小孙子眼疾手快帮您抓住了。",
            "photo_00009": "这是您六十大寿时拍的正式肖像照。您穿着女儿特意为您定做的唐装，坐在红木椅子上，神态庄重而慈祥。这张照片后来被放大挂在了客厅的正中央，每个来家里的客人都会称赞您的气质好。",
            "photo_00010": "这是您和大学同学们毕业五十周年聚会的合影。虽然大家都已经白发苍苍，但聚在一起时仿佛又回到了青春年华。您们一起唱起了当年的校歌，很多人都感动得流下了眼泪。这份同窗情谊已经持续了半个多世纪。",
            "photo_00011": "这是您在家里的阳台上养的花开的照片。您最喜欢那盆君子兰，已经养了十多年了，每年都会开出鲜艳的花朵。您每天早晨起床第一件事就是给它们浇水，说这是您的小花园，看着它们成长就像看着自己的孩子一样。",
            "photo_00012": "这是您最后一次和母亲一起过春节的照片。那时她已经九十高龄了，但精神很好，还能给您包饺子。您依偎在她身边，就像小时候一样。这张照片特别珍贵，记录了您们母子间最温馨的时刻。"
        }
        
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
                        speak_text=self.photo_db[photo_name]
                    else:
                        speak_text="数据库里没有该图片的故事信息"
                    # 通过记忆管理器共享信息
                    self.memory_manager.set_shared_data(
                        "last_matched_scene",
                        {"photo_name": photo_name, "score": score},
                        "PhotoDetector"
                    )
                    self.memory_manager.trigger_event("photo_detected", {
                        "photo_name": photo_name,
                        "score": score,
                        "speak_text": speak_text,
                        "timestamp": time.time()
                    })
                # 显示匹配点图像
                #if debug_img is not None:
                #    cv2.imshow('Good Matches', debug_img)
                time.sleep(1)#防止一直识别成功
            else:
                # 关闭可能存在的匹配点窗口
                if cv2.getWindowProperty('Good Matches', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Good Matches')
            
            # 清理过期的记录
            if self.frame_count % 60 == 0:  # 每60帧清理一次
                self.cleanup_old_entries()

            time.sleep(0.1)#减少识别频次
            # 显示摄像头画面
            #cv2.imshow('Camera Feed', frame)
            cv2.waitKey(1)
            
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