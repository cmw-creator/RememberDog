# voice/noise_detector_manager.py
import logging
import threading

logger = logging.getLogger(__name__)


class NoiseDetectorManager:
    """噪声检测器管理器"""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.detector = None
        self.detector_type = None

    def initialize_detector(self, detector_type="auto", sensitivity=0.2):
        """初始化噪声检测器，允许设置灵敏度"""
        try:

            if detector_type == "yamnet" or detector_type == "auto":
                # 尝试加载YAMNet检测器
                from .enhanced_noise_detector_fixed import EnhancedNoiseDetectorYamnet
                self.detector = EnhancedNoiseDetectorYamnet(
                    self.memory_manager,
                    sensitivity=self.sensitivity,
                    model_path=self.model_path
                )
                # 关键：将检测器注册到内存管理器，供其他模块获取
                self.memory_manager.register_module("NoiseDetector", self)
                if self.detector.model is not None:
                    self.detector_type = "yamnet"
                    logger.info(f"使用YAMNet噪声检测器，灵敏度: {sensitivity}")
                    return True



        except Exception as e:
            logger.error(f"噪声检测器初始化失败: {e}")
            return False

    def start(self):
        """启动噪声检测"""
        if self.detector is None:
            if not self.initialize_detector():
                return False
        if self.model is None:
            print("警告: YAMNet模型未加载，无法启动检测器")
            return False
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        # 新增日志：确认线程启动
        print(f"噪声检测器线程启动状态: {self.processing_thread.is_alive()}")
        return self.detector.start()


    def stop(self):
        """停止噪声检测"""
        if self.detector:
            self.detector.stop()

    def get_detector_type(self):
        """获取检测器类型"""
        return self.detector_type