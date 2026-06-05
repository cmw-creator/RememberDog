# RememberDog 🐕

AD Companion Robot Dog with RAG Memory + SLAM Navigation + LLM Dialogue

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ROS](https://img.shields.io/badge/ROS-Noetic-blue)

> **基于绝影Lite3机器狗的阿尔茨海默症陪伴机器人** | 智慧社区国赛一等奖 | ICAN创新创业大赛三等奖

## 🎯 项目概述

面向阿尔茨海默症患者的具身智能陪伴机器狗，集成多模态交互能力：
- 🧠 **记忆系统** — Sentence-BERT向量记忆库（RAG架构）
- 🗣️ **语音交互** — GPT-SoVITS克隆亲人声音
- 🤖 **LLM对话** — 实时感知+记忆检索→情景化自然对话
- 🚶 **自主导航** — ORB-SLAM3室内定位与自适应跟随
- 📱 **远程控制** — Flutter App（WebRTC实时图传）

## 🏆 获奖
- ICAN创新创业大赛 全国三等奖

## 🧠 技术架构

```mermaid
┌─────────────────────────────────────────────────┐
│                   用户交互层                       │
│  Flutter App (WebRTC)  ←→ 语音输入/输出          │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                   决策层                          │
│  大语言模型（LLM）←→ Sentence-BERT 记忆库(RAG)   │
│  GPT-SoVITS 语音合成(TTS)                        │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│                   感知+运动层                      │
│  ORB-SLAM3定位导航  ←→  自适应跟随                 │
│  SFace人脸识别  +  pyzbar二维码识别                │
└─────────────────────────────────────────────────┘
```

## 🛠️ 核心功能

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| 记忆库 | Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2) + FAISS | 384维语义向量，支持模糊记忆检索 |
| 语音合成 | GPT-SoVITS | 克隆亲人声音（MOS~3.5），个性化语音提醒 |
| 对话引擎 | LLM + RAG | 位置+时间+记忆检索→情景化应答 |
| 视觉SLAM | ORB-SLAM3 | 室内定位误差~5cm |
| 人脸识别 | SFace | 家庭成员识别 |
| 远程控制 | Flutter + WebRTC | 实时图传与控制 |

## 🚀 快速开始

参见 [安装文档](docs/setup.md)

## 📂 项目结构

```
src/
├── main.py          # 主入口
├── memory/          # 记忆模块（RAG）
├── voice/           # 语音合成模块
├── navigation/      # SLAM导航模块
├── vision/          # 视觉识别模块
└── app/             # 远程控制接口
```

## 📧 联系
王承孟 | wcm@njust.edu.cn | [GitHub](https://github.com/cmw-creator)


