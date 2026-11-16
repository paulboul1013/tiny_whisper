import numpy as np
import wave

sample_rate = 48000
duration = 1
frequency = 880
amplitude = 0.5

# 1. 建立 1 秒的時間軸
t = np.arange(sample_rate * duration) / sample_rate

# 2. 產生 440 Hz 正弦波
audio = amplitude * np.sin(2 * np.pi * frequency * t)

# 3. 將浮點數轉成 16-bit PCM 整數
# PCM（Pulse Code Modulation）：用一串離散的數值來表示數位音訊訊號
pcm = (audio * 32767).astype(np.int16)

# 4. 寫入 WAV 檔案
with wave.open("tone.wav", "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(pcm.tobytes())

print("完成:tone.wav")
print("Samples:", len(audio))
print("Shape:", audio.shape)
