
import numpy as np
import matplotlib.pyplot as plt
import librosa

# ========================================
# 1. 建立測試音訊
# ========================================
sr = 16000
t = np.arange(sr) / sr

audio = np.where(
    t < 0.5,
    0.5 * np.sin(2 * np.pi * 440 * t),
    0.5 * np.sin(2 * np.pi * 880 * t)
)

# ========================================
# 2. STFT
# ========================================
n_fft = 400
hop_length = 160

n = np.arange(n_fft)
window = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)

frames = []
times = []

for start in range(0, len(audio) - n_fft + 1, hop_length):
    segment = audio[start:start + n_fft]
    windowed = segment * window

    spectrum = np.fft.rfft(windowed)

    frames.append(spectrum)
    times.append((start + n_fft / 2) / sr)

stft = np.stack(frames, axis=1)
times = np.array(times)

# ========================================
# 3. Power Spectrum
# ========================================
power = np.abs(stft) ** 2

power_db = 10 * np.log10(
    np.maximum(power, 1e-10)
)

# ========================================
# 4. Mel Filterbank
# ========================================
mel_filters = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=80
)

mel_spectrogram = mel_filters @ power

# ========================================
# 5. Whisper Log-Mel
# ========================================
log_mel = np.log10(
    np.maximum(mel_spectrogram, 1e-10)
)

log_mel = np.maximum(
    log_mel,
    log_mel.max() - 8.0
)

log_mel = (log_mel + 4.0) / 4.0

# ========================================
# 6. 檢查 Shape
# ========================================
freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

print("STFT:", stft.shape)
print("Power:", power.shape)
print("Mel Filters:", mel_filters.shape)
print("Mel Spectrogram:", mel_spectrogram.shape)
print("Whisper Log-Mel:", log_mel.shape)

# ========================================
# 圖片 1：Power Spectrogram
# 暗色熱力圖
# ========================================
fig1, ax1 = plt.subplots(figsize=(11, 4))

img1 = ax1.imshow(
    power_db,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], 0, sr / 2],
    cmap="magma"
)

ax1.set_ylim(0, 1500)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("1. Power Spectrogram")

fig1.colorbar(img1, ax=ax1, label="Power (dB)")
fig1.tight_layout()
fig1.savefig("01_power.png", dpi=200)

# ========================================
# 圖片 2：Mel Filter Weights
# 多色折線圖
# ========================================
fig2, ax2 = plt.subplots(figsize=(11, 4))

show_filters = [0, 5, 10, 20, 30, 40, 50, 60, 70, 79]
colors = plt.cm.tab10(np.linspace(0, 1, len(show_filters)))

for i, color in zip(show_filters, colors):
    ax2.plot(
        freqs,
        mel_filters[i],
        label=f"Mel {i}",
        color=color,
        linewidth=2
    )

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Filter Weight")
ax2.set_title("2. Mel Filter Weights")
ax2.grid(True, alpha=0.25)
ax2.legend(fontsize=8, ncol=2)

fig2.tight_layout()
fig2.savefig("02_mel_filters.png", dpi=200)

# ========================================
# 圖片 3：Whisper Log-Mel
# 藍綠色熱力圖
# ========================================
fig3, ax3 = plt.subplots(figsize=(11, 4))

img3 = ax3.imshow(
    log_mel,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], 0, 80],
    cmap="viridis"
)

ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("Mel Band Index")
ax3.set_title("3. Whisper Log-Mel Spectrogram")

fig3.colorbar(
    img3,
    ax=ax3,
    label="Normalized Log-Mel"
)

fig3.tight_layout()
fig3.savefig("03_log_mel.png", dpi=200)

# 一次開啟三張獨立圖片視窗
plt.show()
