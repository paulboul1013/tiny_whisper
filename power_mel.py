import numpy as np
import matplotlib.pyplot as plt
import librosa

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

# 6. STFT 複數 → 功率頻譜
power = np.abs(stft) ** 2

# 7. 建立 80 個 Mel Filter
mel_filters = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=80
)

# 8. 將 201 個頻率 Bin 轉成 80 個 Mel 頻帶
mel_spectrogram = mel_filters @ power

# 9. 轉 dB，方便顯示
power_db = 10 * np.log10(np.maximum(power, 1e-10))
mel_db = 10 * np.log10(np.maximum(mel_spectrogram, 1e-10))

# 10. 建立頻率軸
freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

# 11. 顯示資料形狀
print("STFT:", stft.shape)
print("Power:", power.shape)
print("Mel Filters:", mel_filters.shape)
print("Mel Spectrogram:", mel_spectrogram.shape)

# -------------------------------------------------
# 繪圖：Power Spectrogram / Mel Filter Weights / Mel Spectrogram
# -------------------------------------------------
plt.figure(figsize=(12, 12))

# (1) Power Spectrogram
plt.subplot(3, 1, 1)
plt.imshow(
    power_db,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    cmap="magma"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title("Power Spectrogram")
plt.colorbar(label="Power (dB)")

# (2) Mel Filter Weights
plt.subplot(3, 1, 2)

# 為了不要 80 條線太擠，這裡只畫部分 filter 當示意
show_filters = [0, 5, 10, 20, 30, 40, 50, 60, 70, 79]

for i in show_filters:
    plt.plot(freqs, mel_filters[i], label=f"Mel {i}")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Weight")
plt.title("Mel Filter Weights")
plt.legend(loc="upper right", fontsize=8, ncol=2)
plt.grid(True)

# (3) Mel Spectrogram
plt.subplot(3, 1, 3)
plt.imshow(
    mel_db,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], 0, mel_filters.shape[0]],
    cmap="magma"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Mel Band Index")
plt.title("Mel Spectrogram")
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.show()