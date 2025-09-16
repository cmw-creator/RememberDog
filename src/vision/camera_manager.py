import cv2
import threading

class CameraManager:
    def __init__(self, camera_id=0, width=800, height=600):
        """初始化摄像头管理器
        
        Args:
            camera_id (int): 摄像头ID (默认0)
            width (int): 图像宽度
            height (int): 图像高度
        """
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 当前帧和锁
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 运行状态
        self.running = False
        self.thread = None

    def start(self):
        """启动摄像头捕获线程"""
        print("启动摄像头")
        if self.running or not self.camera.isOpened():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        return True

    def _capture_frames(self):
        """内部方法：持续捕获帧"""
        while self.running:
            ret, frame = self.camera.read()
            cv2.imshow('Camera:', frame)
            cv2.waitKey(1)
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
    
    def get_frame(self):
        """获取当前帧的副本"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        """停止摄像头并释放资源"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.camera.isOpened():
            self.camera.release()
        
        with self.frame_lock:
            self.current_frame = None

    def __del__(self):
        self.stop()

if __name__ == "__main__":
    print("摄像头测试")
    camera_manager=CameraManager()
    camera_manager.start()
    while True:
        pass