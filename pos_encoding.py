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



def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0

    # 不同維度使用不同的振盪速度
    log_increment = np.log(max_timescale) / (channels // 2 - 1)

    inv_timescales = torch.exp(
        -log_increment * torch.arange(channels // 2)
    )

    # 每個時間位置乘上不同的頻率尺度
    scaled_time = (
        torch.arange(length)[:, None]
        * inv_timescales[None, :]
    )

    # 前半維度使用 Sin，後半維度使用 Cos
    return torch.cat([
        torch.sin(scaled_time),
        torch.cos(scaled_time)
    ], dim=1)

torch.manual_seed(7)

model = TinyAudioEncoder()
model.eval()

x = torch.from_numpy(log_mel).float().unsqueeze(0)

with torch.no_grad():
    out1, out2 = model(x)

print(out2.shape) #torch.Size([1, 16, 49])

features = out2.permute(0, 2, 1)

# 49 個位置、每個位置 16 維
pos = sinusoids(49, 16)

# 把位置資訊加到聲音特徵
features_with_pos = features + pos.to(
    device=features.device,
    dtype=features.dtype
)

print("Features:", features.shape)
print("Position:", pos.shape)
print("Output:", features_with_pos.shape)


# 轉成 NumPy，並轉為 [Channels, Time] 方便繪圖
before = features[0].detach().cpu().numpy().T
position = pos.detach().cpu().numpy().T
after = features_with_pos[0].detach().cpu().numpy().T

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

images = [
    (before, "1. Conv1D + GELU"),
    (position, "2. Positional Encoding"),
    (after, "3. Audio Features + Position")
]

for ax, (data, title) in zip(axes, images):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[0, 0.98, 0, 16]
    )

    ax.set_title(title)
    ax.set_ylabel("Feature Channel")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)

    fig.colorbar(im, ax=ax)

axes[-1].set_xlabel("Approximate Time (seconds)")

plt.savefig("positional_encoding.png", dpi=200)
plt.show()
