import cv2
import threading
import time
from collections import deque
import os
import logging
logger = logging.getLogger(name='Log')
logger.info("初始化摄像头管理器")

class CameraManager:
    def __init__(self, camera_id="rtsp://192.168.1.120:8554/test", width=800, height=600, save_dir="frames", max_frames=10):
        """初始化摄像头管理器
        
        Args:
            camera_id (int): 摄像头ID (默认0)
            width (int): 图像宽度
            height (int): 图像高度
            save_dir (str): 保存帧图片的文件夹
            max_frames (int): 缓存最近帧数量
        """
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FPS, 10)
        
        # 当前帧和锁
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 帧缓存
        self.max_frames = max_frames
        self.frame_buffer = deque(maxlen=max_frames)
        
        # 保存路径
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 运行状态
        self.running = False
        self.thread = None

    def start(self):
        """启动摄像头捕获线程"""
        #print("启动摄像头")
        logger.info("启动摄像头捕获线程")
        if self.running or not self.camera.isOpened():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        return True

    def _capture_frames(self):
        """内部方法：持续捕获帧"""
        frame_count = 0
        while self.running:
            for _ in range(30):
                self.camera.grab()  # 清空缓冲区
            
            ret, frame = self.camera.retrieve()
            if ret:
                with self.frame_lock:
                    frame_count += 1
                    self.current_frame = frame.copy()
                    # 添加到缓存
                    #self.frame_buffer.append(frame.copy())
                    if False:
                        # 保存到磁盘（可选）
                        filename = os.path.join(self.save_dir, f"{int(time.time()*1000)}_frame_debug.jpg")
                        cv2.imwrite(filename, frame)
                        logger.debug(f"保存调试帧: {filename}")
            else:
                logger.warning(f"未获取到有效帧，跳过")
            if frame_count % 10 == 0:
                logger.debug(f"已获取 {frame_count} 帧")
            time.sleep(0.02)

    def get_frame(self):
        """获取当前帧的副本"""
        with self.frame_lock:
            if self.current_frame is not None:
                # 只有当确实需要时才复制
                return self.current_frame.copy()
        return None

    def get_recent_frames(self):
        """获取缓存的最近 N 帧"""
        with self.frame_lock:
            return list(self.frame_buffer)

    def stop(self):
        """停止摄像头并释放资源"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.camera.isOpened():
            self.camera.release()
        
        with self.frame_lock:
            self.current_frame = None
            self.frame_buffer.clear()

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    print("摄像头测试")
    camera_manager = CameraManager(0)
    camera_manager.start()
    try:
        while True:
            time.sleep(1)
            recent_frames = camera_manager.get_recent_frames()
            print(f"缓存帧数量: {len(recent_frames)}")
    except KeyboardInterrupt:
        camera_manager.stop()
