import torch
import torch.nn as nn
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



class TinyAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            80, 16,
            kernel_size=3,
            padding=1
        )
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            16, 16,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.gelu2 = nn.GELU()

    def forward(self, x):
        print("Input:", x.shape)

        # 第一層輸出
        out1 = self.gelu1(self.conv1(x))
        print("Conv1 + GELU:", out1.shape)

        # 第二層輸出
        out2 = self.gelu2(self.conv2(out1))
        print("Conv2 + GELU:", out2.shape)

        # 同時回傳兩層的結果
        return out1, out2


torch.manual_seed(7)

model = TinyAudioEncoder()
model.eval()

x = torch.from_numpy(log_mel).float().unsqueeze(0)

with torch.no_grad():
    out1, out2 = model(x)



# 將 PyTorch Tensor 轉成 NumPy Array
input_img = x[0].cpu().numpy()
conv1_img = out1[0].cpu().numpy()
conv2_img = out2[0].cpu().numpy()

# 建立三張上下排列的圖片
fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

duration = len(audio) / sr

data_list = [
    (input_img, "1. Input Log-Mel",
     "Mel Band Index", "magma"),

    (conv1_img, "2. Conv1D + GELU",
     "Feature Channel", "coolwarm"),

    (conv2_img, "3. Conv1D (stride=2) + GELU",
     "Feature Channel", "coolwarm")
]

for ax, (data, title, ylabel, cmap) in zip(
    axes, data_list
):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, data.shape[0]],
        cmap=cmap,
        interpolation="nearest"
    )

    ax.set_title(f"{title} | Shape: {data.shape}")
    ax.set_ylabel(ylabel)

    # 標示聲音從 440 Hz 切換為 880 Hz 的時間
    ax.axvline(
        0.5,
        color="white" if cmap == "magma" else "black",
        linestyle="--",
        alpha=0.7
    )

    fig.colorbar(im, ax=ax)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Tiny Whisper Encoder - Feature Comparison",
    fontsize=15
)

plt.savefig("conv_comparison.png", dpi=200)
plt.show()
