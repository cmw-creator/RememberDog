import speech_recognition as sr

#从系统麦克风拾取音频数据，采样率为 16000
def rec(rate=16000):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=rate) as source:
        print('正在获取声音中...')
        audio = r.listen(source)

    with open("recording.wav", "wb") as f:
        f.write(audio.get_wav_data())
        print('声音获取完成.')

    return 1

from speechbrain.pretrained import EncoderDecoderASR
import torch
import torchaudio

def voice_into_word():
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell",
                                               savedir="pretrained_models/asr-transformer-aishell")

    audio_1 = r"./test.wav"
    ddd = torchaudio.list_audio_backends()

    print('start...')

    snt_1, fs = torchaudio.load(audio_1)
    wav_lens = torch.tensor([1.0])
    res = asr_model.transcribe_batch(snt_1, wav_lens)

    word = res[0][0].replace(' ', '')
    print(word)

    return word

if __name__ == '__main__':
    rec()
    voice_into_word()

