import json
import random
import time
import threading
import os
from datetime import datetime, timedelta
from multiprocessing import Process, Queue as MPQueue, Manager
import pyttsx3
from memory.qa_manager import QAManager

class MemoryManager:
    def __init__(self, family_info_path="assets/family_info.json", reminder_path="assets/reminders.json"):
        # 加载家庭信息库
        self.family_info = self.load_json(family_info_path)
        # 加载提醒数据库
        self.reminders = self.load_json(reminder_path)
        # 语音引擎
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        # 运行标志
        self.running = True
        # 问答线程
        self.quiz_thread = None
        # 提醒线程
        self.reminder_thread = None
        ### 共享数据初始化 ###
        self.shared_memory = {}  # 共享内存字典
        self.event_callbacks = {}  # 事件回调函数注册表
        self.lock = threading.Lock()  # 线程安全锁
        # ✅ 使用 Manager().Value 实现进程间共享布尔值
        manager = Manager()
        self.runSpeaking = manager.Value('b', False)
        #QA_Manager
        self.qa_manager=QAManager()
        
    def load_json(self, file_path):
        """加载JSON文件"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print(f"警告：无法加载文件 {file_path}")
        return {"questions": [], "reminders": []}
    
    def save_json(self, data, file_path):
        """保存JSON文件"""
        print("准备保存文件")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_question(self, question, answer):
        """添加问答对到家庭信息库"""
        self.family_info["questions"].append({
            "question": question,
            "answer": answer
        })
        self.save_json(self.family_info, "assets/family_info.json")
    
    def add_reminder(self, time_str, event):
        """添加新提醒"""
        self.reminders["reminders"].append({
            "time": time_str,
            "event": event,
            "completed": False
        })
        self.save_json(self.reminders, "assets/reminders.json")
        print(f"已经添加新的提醒{event}并保存到文件")
    
    def get_random_question(self):
        """随机获取一个问题"""
        if not self.family_info["questions"]:
            return None
            
        question_data = random.choice(self.family_info["questions"])
        return question_data
    
    def speak(self, text):
        """语音播报"""
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run).start()
    
    def ask_question(self):
        """提问随机问题"""
        question_data = self.get_random_question()
        if question_data:
            self.speak(question_data["question"])
            # 等待10秒后给出答案
            time.sleep(10)
            self.speak(f"答案是: {question_data['answer']}")
    
    def check_reminders(self):
        """检查并执行提醒"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        for reminder in self.reminders["reminders"]:
            # 跳过已完成的提醒
            if reminder["completed"]:
                continue
                
            # 检查时间是否匹配
            if reminder["time"] == current_time:
                self.speak(f"提醒: {reminder['event']}")
                print(f"假装在说:提醒: {reminder['event']}")
                reminder["completed"] = True
                self.save_json(self.reminders, "assets/reminders.json")
    
    def run_quiz_scheduler(self):
        """随机问答调度器"""
        while self.running:
            # 在9:00-21:00之间随机提问
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 21:
                # 随机等待时间 (30-120分钟)
                wait_time = random.randint(1800, 7200)
                time.sleep(wait_time)
                self.ask_question()
            else:
                # 夜间等待时间更长
                time.sleep(3600)  # 1小时
    
    def run_reminder_scheduler(self):
        """定时提醒调度器"""
        while self.running:
            self.check_reminders()
            time.sleep(60)  # 每分钟检查一次
    
    def start(self):
        """启动记忆管理器"""
        print("启动记忆管理器")
        # 启动问答线程
        self.quiz_thread = threading.Thread(target=self.run_quiz_scheduler)
        self.quiz_thread.daemon = True
        self.quiz_thread.start()
        
        # 启动提醒线程
        self.reminder_thread = threading.Thread(target=self.run_reminder_scheduler)
        self.reminder_thread.daemon = True
        self.reminder_thread.start()
    
    def stop(self):
        """停止记忆管理器"""
        self.running = False
        if self.quiz_thread:
            self.quiz_thread.join(timeout=1.0)
        if self.reminder_thread:
            self.reminder_thread.join(timeout=1.0)


    ### 共享数据方法 ###
    def set_shared_data(self, key, value, module_name=""):
        """设置共享数据"""
        with self.lock:
            self.shared_memory[key] = {
                "value": value,
                "timestamp": time.time(),
                "source": module_name
            }
    
    def get_shared_data(self, key, default=None):
        """获取共享数据"""
        with self.lock:
            return self.shared_memory.get(key, {}).get("value", default)
    
    def register_event_callback(self, event_type, callback_func, module_name=""):
        """注册事件回调函数"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append({
            "callback": callback_func,
            "module": module_name
        })
    
    def trigger_event(self, event_type, event_data=None):
        """触发事件，通知所有注册的模块"""
        if event_type in self.event_callbacks:
            for callback_info in self.event_callbacks[event_type]:
                try:
                    callback_info["callback"](event_data)
                except Exception as e:
                    print(f"事件回调执行失败: {str(e)}")
    
    # 添加模块状态跟踪
    def update_module_status(self, module_name, status, info=None):
        """更新模块状态"""
        status_key = f"module_status_{module_name}"
        self.set_shared_data(status_key, {
            "status": status,
            "info": info,
            "timestamp": time.time()
        }, "memory_manager")
    
    def get_module_status(self, module_name):
        """获取模块状态"""
        return self.get_shared_data(f"module_status_{module_name}")
# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    family_info = {
        "questions": [
            {"question": "您女儿的名字是？", "answer": "张晓"},
            {"question": "您最喜欢的家乡菜是什么？", "answer": "红烧肉"},
            {"question": "您出生在哪个城市？", "answer": "北京"}
        ]
    }
    
    reminders = {
        "reminders": [
            {"time": "08:00", "event": "吃降压药", "completed": False},
            {"time": "12:00", "event": "吃维生素D", "completed": False},
            {"time": "16:40", "event": "提醒测试", "completed": False}
        ]
    }
    
    # 保存示例数据
    with open("assets/family_info.json", "w", encoding="utf-8") as f:
        json.dump(family_info, f, ensure_ascii=False, indent=2)
    
    with open("assets/reminders.json", "w", encoding="utf-8") as f:
        json.dump(reminders, f, ensure_ascii=False, indent=2)
    
    # 启动记忆管理器
    manager = MemoryManager()
    manager.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()