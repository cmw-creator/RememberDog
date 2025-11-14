import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    """
    一个简单的日志类，支持输出到控制台和文件，并自动按日期滚动日志文件。
    """
    def __init__(self):
        self.logger=None
    def create_logger(self, name='Log', log_level=logging.INFO, log_dir='logs'):
        # 创建logger实例
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 避免重复添加Handler（防止Jupyter等多环境下的日志重复问题）
        if not self.logger.handlers:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 设置日志格式
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 1. 控制台处理器 (StreamHandler)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 2. 文件处理器 (RotatingFileHandler)，按日期和大小滚动
            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,           # 保留5个备份文件
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    # 提供便捷的日志记录方法
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)


def get_logger(name='Log', log_level=logging.DEBUG, log_dir='logs'):
    logger = Logger()
    logger.create_logger(name=name, log_level=log_level, log_dir=log_dir)
    return logger
logger = get_logger(name='Log', log_level=logging.DEBUG)



# 使用示例
if __name__ == "__main__":
    logger.info("这是一个信息级别的日志")
    logger.error("这是一个错误级别的日志")