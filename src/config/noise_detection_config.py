NOISE_DETECTION_CONFIG = {
    "model_path": "assets/voice_models/yamnet",
    "sample_rate": 16000,
    "chunk_size": 1024,
    "sensitivity": 0.7,
    "energy_threshold": 1000,

    # 运行模式配置
    "mode": "yamnet_only",  # 只使用YAMNet模型模式

    # 模型要求配置
    "model_required": True,  # 必须加载模型才能运行
    "min_confidence": 0.1,  # 最小置信度阈值

    # 异常噪声类型配置
    "abnormal_noise_types": {
        "high_pitch": {
            "description": "尖锐的高分贝声响",
            "risk_level": "medium",
            "response_priority": 3
        },
        "glass_break": {
            "description": "玻璃破碎声",
            "risk_level": "high",
            "response_priority": 4
        },
        "impact": {
            "description": "物体跌落/撞击",
            "risk_level": "high",
            "response_priority": 5
        },
        "alarm_sound": {
            "description": "持续异常声",
            "risk_level": "critical",
            "response_priority": 5
        },
        "moaning_crying": {
            "description": "痛苦呻吟/哭泣",
            "risk_level": "medium",
            "response_priority": 3
        }
    }
}