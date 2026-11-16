import numpy as np
import matplotlib.pyplot as plt

# 1. 建立測試音訊
sr = 16000
t = np.arange(sr) / sr

audio = np.where(
    t < 0.5,
    0.5 * np.sin(2 * np.pi * 440 * t),
    0.5 * np.sin(2 * np.pi * 880 * t)
)

# 2. STFT 參數
n_fft = 400
hop_length = 160

# 3. 建立週期型 Hann Window
n = np.arange(n_fft)
window = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)

# 4. 逐個視窗執行 FFT
frames = []
times = []

for start in range(0, len(audio) - n_fft + 1, hop_length):
    segment = audio[start:start + n_fft]

    # 讓片段邊界平滑
    windowed = segment * window

    # 分析目前這個時間片段的頻率
    spectrum = np.fft.rfft(windowed)

    frames.append(spectrum)
    times.append((start + n_fft / 2) / sr)

# 5. 將所有時間片段排列成矩陣
stft = np.stack(frames, axis=1)

# 6. 建立頻率軸
freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

print("STFT shape:", stft.shape)

# 7. 取幅度，轉成相對 dB 方便繪圖
magnitude = np.abs(stft)
relative = magnitude / max(magnitude.max(), 1e-12)
db = 20 * np.log10(np.maximum(relative, 1e-8))

# 8. 顯示時間－頻率圖
plt.figure(figsize=(10, 5))
plt.pcolormesh(times, freqs, db, shading="auto",
               vmin=-80, vmax=0)
plt.ylim(0, 1500)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title("STFT Spectrogram")
plt.colorbar(label="Relative magnitude (dB)")
plt.show()
