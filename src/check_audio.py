# check_audio.py
import pyaudio
import numpy as np


def check_audio_devices():
    p = pyaudio.PyAudio()

    print("可用的音频输入设备:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"设备 {i}: {info['name']}")
            print(f"  最大输入通道: {info['maxInputChannels']}")
            print(f"  默认采样率: {info['defaultSampleRate']}")
            print()

    # 测试默认设备
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)

        print("测试音频输入...")
        data = stream.read(1024)
        audio_array = np.frombuffer(data, dtype=np.int16)
        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

        print(f"音频数据范围: [{np.min(audio_array)}, {np.max(audio_array)}]")
        print(f"能量值: {energy}")

        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"音频测试失败: {e}")

    p.terminate()


if __name__ == "__main__":
    check_audio_devices()