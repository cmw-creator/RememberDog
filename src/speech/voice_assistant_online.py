# 文档7增强版
import threading
import json
import time
import speech_recognition as sr
import pyttsx3
# from std_msgs.msg import String

class VoiceAssistant:
    def __init__(self, memory_manager):
        # 初始化记忆管理器
        self.memory_manager = memory_manager
        
        # 硬件配置
        self.mic = sr.Microphone(device_index=1)
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # 运行状态
        self.running = True
        self.listening = False
        
        # 命令映射
        self.commands = {
            "过来": "come", 
            "走开": "leave",
            "吃药时间": "medicine_remind",
            "添加提醒": "add_reminder",
            "添加问题": "add_question"
        }
        
    def listen_command(self):
        """持续监听语音指令"""
        while self.running:
            if not self.listening:
                time.sleep(0.5)
                continue
                
            try:
                with self.mic as source:
                    print("请说话...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio, language='zh-CN')
                print(f"识别结果: {text}")
                
                # 处理特殊命令
                if any(cmd in text for cmd in ["添加提醒", "设置提醒"]):
                    self.handle_add_reminder(text)
                elif any(cmd in text for cmd in ["添加问题", "设置问题"]):
                    self.handle_add_question(text)
                else:
                    # 处理普通命令
                    self.process_command(text)
                    
            except sr.UnknownValueError:
                print("未能识别语音")
            except sr.RequestError:
                print("网络连接失败")
            except Exception as e:
                print(f"错误: {str(e)}")
    
    def process_command(self, text):
        """处理语音命令"""
        for cmd, action in self.commands.items():
            if cmd in text:
                self.execute_action(action)
                return
                
        # 默认回复
        self.speak("我没有听懂，请再说一次")
    
    def execute_action(self, action):
        """执行对应动作"""
        if action == "add_reminder":
            self.speak("请告诉我提醒的内容和时间")
        elif action == "add_question":
            self.speak("请告诉我问题和答案")
        else:
            # 其他动作处理
            pass
    
    def handle_add_reminder(self, text):
        """处理添加提醒命令"""
        try:
            # 简单解析时间 (示例: "明天下午3点提醒我吃药")
            if "明天" in text:
                time_base = datetime.now() + timedelta(days=1)
            else:
                time_base = datetime.now()
                
            if "上午" in text:
                period = "AM"
            elif "下午" in text:
                period = "PM"
            else:
                period = ""
                
            # 提取时间数字
            time_str = ""
            for word in text.split():
                if word.isdigit():
                    hour = int(word)
                    time_str = f"{hour}:00" if period == "AM" else f"{hour+12}:00"
                    break
                    
            if time_str:
                # 提取事件描述
                event = text.split("提醒我")[-1].strip()
                
                # 添加到提醒系统
                self.memory_manager.add_reminder(time_str, event)
                self.speak(f"已添加提醒: {event} 在 {time_str}")
            else:
                self.speak("未识别到时间信息")
                
        except Exception as e:
            self.speak("添加提醒失败，请重试")
            print(f"添加提醒错误: {str(e)}")
    
    def handle_add_question(self, text):
        """处理添加问题命令"""
        try:
            # 简单解析 (示例: "添加问题:我的女儿叫什么?答案:张晓")
            if "问题" in text and "答案" in text:
                parts = text.split("答案")
                question = parts[0].replace("问题", "").strip()
                answer = parts[1].strip()
                
                # 添加到家庭信息库
                self.memory_manager.add_question(question, answer)
                self.speak(f"已添加问题: {question}")
            else:
                self.speak("请按照'问题...答案...'的格式说明")
                
        except Exception as e:
            self.speak("添加问题失败，请重试")
            print(f"添加问题错误: {str(e)}")
    
    def speak(self, text):
        """语音合成"""
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run).start()
    
    def start_listening(self):
        """开始监听"""
        self.listening = True
    
    def stop_listening(self):
        """停止监听"""
        self.listening = False
    
    def start(self):
        """启动语音助手"""
        print("启动语音助手")
        self.listening = True
        listen_thread = threading.Thread(target=self.listen_command)
        listen_thread.daemon = True
        listen_thread.start()
    
    def stop(self):
        """停止语音助手"""
        self.running = False
        self.listening = False

# 修改主程序集成语音助手
def main():
    import sys
    import os
    
    # 添加项目根目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    from memory.memory_manager import MemoryManager

    # 创建记忆管理器
    memory_manager = MemoryManager()
    memory_manager.start()
    
    # 创建语音助手 (新增)
    voice_assistant = VoiceAssistant(memory_manager)
    voice_assistant.start()
    
    time.sleep(1000)

if __name__ == '__main__':
    main()
