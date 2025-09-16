#!/usr/bin/env python3
#目前功能不正常，windows和ubuntu部分模块不通用

#import rospy
import threading
import numpy as np
import speech_recognition as sr
import snowboydecoder
import pyttsx3
#from std_msgs.msg import String

class VoiceAssistant:
    def __init__(self):
        # ROS节点初始化
        rospy.init_node('voice_assistant', anonymous=True)
        
        # 硬件配置
        self.mic = sr.Microphone(device_index=0)  # 默认麦克风
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # 语速设置
        
        # 唤醒参数
        self.wakeup_model = "resources/snowboy.umdl"  # 唤醒词模型路径
        self.sensitivity = 0.45  # 唤醒灵敏度（值越低越灵敏）
        self.detector = snowboydecoder.HotwordDetector(
            self.wakeup_model, 
            sensitivity=self.sensitivity
        )
        
        # 对话管理
        self.chat_mode = False  # 是否处于对话状态
        self.commands = {
            "过来": "come", 
            "走开": "leave",
            "吃药时间": "medicine_remind"
        }
        
        # ROS话题设置
        self.wakeup_pub = rospy.Publisher("wakeup_status", String, queue_size=10)
        self.chat_pub = rospy.Publisher("voice_command", String, queue_size=10)
        
    def _wakeup_callback(self):
        """唤醒回调函数"""
        rospy.loginfo("唤醒词检测成功！")
        self.wakeup_pub.publish("activated")
        self.speak("我在听，请说指令")
        self.chat_mode = True  # 进入对话模式
        
    def _audio_preprocess(self, audio_data):
        """音频预处理（降噪+增益）"""
        # 使用SpeechRecognition内置降噪
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # 音频数据归一化
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0
        return audio_np

    def _listen_command(self):
        """持续监听语音指令"""
        while not rospy.is_shutdown() and self.chat_mode:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio, language='zh-CN')
                rospy.loginfo(f"识别结果: {text}")
                
                # 指令匹配逻辑
                for cmd, action in self.commands.items():
                    if cmd in text:
                        self.chat_pub.publish(action)
                        self.speak(f"执行{cmd}指令")
                        break
                else:  # 未匹配时进入闲聊模式
                    self._handle_chat(text)
                    
            except sr.UnknownValueError:
                rospy.logwarn("未能识别语音")
            except sr.RequestError:
                self.speak("网络连接失败，切换离线模式")
                self._offline_asr(audio)  # 离线识别备用方案

    def _offline_asr(self, audio):
        """离线语音识别（PocketSphinx备用）"""
        try:
            text = self.recognizer.recognize_sphinx(audio, language='zh-CN')
            rospy.loginfo(f"离线识别: {text}")
        except Exception as e:
            rospy.logerr(f"离线识别失败: {str(e)}")

    def _handle_chat(self, text):
        """简单对话处理（可替换为Rasa/NLU引擎）"""
        if "你好" in text:
            self.speak("您好，我是您的陪伴机器狗")
        elif "今天天气" in text:
            self.speak("当前室外温度25度，适合散步")
        elif "谁是我的女儿" in text:
            self.speak("您的女儿叫张晓，上次视频是昨天下午")
        else:
            self.speak("我没有听懂，请再说一次")

    def speak(self, text):
        """语音合成（离线）"""
        def _speak_thread():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=_speak_thread).start()

    def start(self):
        """启动语音助手"""
        # 唤醒检测线程
        wake_thread = threading.Thread(
            target=self.detector.start, 
            kwargs={"detected_callback": self._wakeup_callback}
        )
        wake_thread.daemon = True
        wake_thread.start()
        
        # 主循环
        rospy.loginfo("等待唤醒词...")
        while not rospy.is_shutdown():
            if self.chat_mode:
                self._listen_command()
                self.chat_mode = False  # 单次对话后退出
            rospy.sleep(0.1)
        
        self.detector.terminate()

if __name__ == '__main__':
    assistant = VoiceAssistant()
    assistant.start()