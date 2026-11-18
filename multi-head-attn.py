
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn

# =========================================================
# 1. 建立測試音訊：前半秒 440 Hz，後半秒 880 Hz
# =========================================================
torch.manual_seed(7)

sr = 16000
t = np.arange(sr) / sr

audio = np.where(
    t < 0.5,
    0.5 * np.sin(2 * np.pi * 440 * t),
    0.5 * np.sin(2 * np.pi * 880 * t),
)

# =========================================================
# 2. 手動實作 STFT
# =========================================================
n_fft = 400
hop_length = 160

n = np.arange(n_fft)

# 週期型 Hann Window
window = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)

frames = []
times = []

for start in range(0, len(audio) - n_fft + 1, hop_length):

    segment = audio[start:start + n_fft]

    # Hann Window
    windowed = segment * window

    # FFT
    spectrum = np.fft.rfft(windowed)

    frames.append(spectrum)

    times.append(
        (start + n_fft / 2) / sr
    )

stft = np.stack(frames, axis=1)
times = np.array(times)

# =========================================================
# 3. Power Spectrum
# =========================================================
power = np.abs(stft) ** 2

# =========================================================
# 4. Mel Filterbank
# =========================================================
mel_filters = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=80
)

mel = mel_filters @ power

# =========================================================
# 5. Whisper-style Log-Mel
# =========================================================
log_mel = np.log10(
    np.maximum(mel, 1e-10)
)

log_mel = np.maximum(
    log_mel,
    log_mel.max() - 8.0
)

log_mel = (log_mel + 4.0) / 4.0

# =========================================================
# 6. Conv1D + GELU
# =========================================================
class TinyAudioStem(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=80,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.gelu(self.conv1(x))

        x = self.gelu(self.conv2(x))

        # [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        return x


# =========================================================
# 7. Sinusoidal Positional Encoding
# =========================================================
def sinusoids(length, channels, max_timescale=10000):

    assert channels > 2
    assert channels % 2 == 0

    log_increment = (
        np.log(max_timescale)
        / (channels // 2 - 1)
    )

    inv_timescales = torch.exp(
        -log_increment * torch.arange(channels // 2)
    )

    scaled_time = (
        torch.arange(length)[:, None]
        * inv_timescales[None, :]
    )

    return torch.cat(
        [
            torch.sin(scaled_time),
            torch.cos(scaled_time)
        ],
        dim=1
    )


# =========================================================
# 8. Multi-Head Self-Attention
# =========================================================
class TinyMultiHeadAttention(nn.Module):

    def __init__(self, dim=16, heads=4):
        super().__init__()

        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        # Q, K, V Linear Projection
        self.q_proj = nn.Linear(dim, dim)

        self.k_proj = nn.Linear(
            dim, dim, bias=False
        )

        self.v_proj = nn.Linear(dim, dim)

        # Output Projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):

        B, T, D = x.shape

        # -----------------------------------------
        # 1. Q, K, V
        # [B,T,D] -> [B,H,T,D/H]
        # -----------------------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(
            B, T, self.heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            B, T, self.heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            B, T, self.heads, self.head_dim
        ).transpose(1, 2)

        # -----------------------------------------
        # 2. Attention Scores
        # [B,H,T,T]
        # -----------------------------------------
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # -----------------------------------------
        # 3. Softmax
        # -----------------------------------------
        weights = torch.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------
        # 4. Attention × V
        # [B,H,T,D/H]
        # -----------------------------------------
        context = weights @ v

        # -----------------------------------------
        # 5. Concat Heads
        # [B,H,T,D/H] -> [B,T,D]
        # -----------------------------------------
        context = context.transpose(1, 2)

        context = context.contiguous().reshape(
            B, T, D
        )

        # -----------------------------------------
        # 6. Output Projection
        # -----------------------------------------
        output = self.out_proj(context)

        return output, weights


# =========================================================
# 9. Transformer Encoder Block
# =========================================================
class TinyEncoderBlock(nn.Module):

    def __init__(self, dim=16, heads=4):
        super().__init__()

        # Attention
        self.attn_ln = nn.LayerNorm(dim)

        self.attn = TinyMultiHeadAttention(
            dim=dim,
            heads=heads
        )

        # MLP
        self.mlp_ln = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        # -----------------------------------------
        # 1. Pre-LayerNorm + Multi-Head Attention
        # -----------------------------------------
        attn_result, weights = self.attn(
            self.attn_ln(x)
        )

        # -----------------------------------------
        # 2. Residual Connection
        # -----------------------------------------
        after_attention = x + attn_result

        # -----------------------------------------
        # 3. Pre-LayerNorm + MLP
        # -----------------------------------------
        mlp_result = self.mlp(
            self.mlp_ln(after_attention)
        )

        # -----------------------------------------
        # 4. Residual Connection
        # -----------------------------------------
        after_mlp = after_attention + mlp_result

        return after_mlp, after_attention, weights


# =========================================================
# 10. 組裝 Tiny Whisper Encoder
# =========================================================
stem = TinyAudioStem().eval()

block = TinyEncoderBlock(
    dim=16,
    heads=4
).eval()

ln_post = nn.LayerNorm(16).eval()

# NumPy -> PyTorch Tensor
# [80,98] -> [1,80,98]
x = torch.from_numpy(
    log_mel
).float().unsqueeze(0)

with torch.no_grad():

    # Conv1D + GELU
    local_features = stem(x)

    # Positional Encoding
    position = sinusoids(
        local_features.shape[1],
        local_features.shape[2]
    )

    input_features = (
        local_features
        + position.to(local_features)
    )

    # Transformer Encoder Block
    after_mlp, after_attention, weights = block(
        input_features
    )

    # Final LayerNorm
    encoder_output = ln_post(after_mlp)


# =========================================================
# 11. 檢查 Tensor Shape
# =========================================================
print("\n========== Tensor Shapes ==========")

print("Log-Mel:               ", tuple(x.shape))

print(
    "After Conv/Position:   ",
    tuple(input_features.shape)
)

print(
    "Per-head attention:    ",
    tuple(weights.shape)
)

print(
    "After attention:       ",
    tuple(after_attention.shape)
)

print(
    "After MLP:             ",
    tuple(after_mlp.shape)
)

print(
    "Post-Encoder LayerNorm:",
    tuple(encoder_output.shape)
)

print("MLP width: 16 -> 64 -> 16")

print(
    "Attention row sum per head:",
    weights[0, :, 20].sum(dim=-1).numpy()
)

print(
    "Block trainable parameters:",
    sum(p.numel() for p in block.parameters())
)


# =========================================================
# 12. Matplotlib 圖片 A：四個 Attention Heads
# =========================================================
attention_np = weights[0].cpu().numpy()

fig1, axes1 = plt.subplots(
    2, 2,
    figsize=(11, 9),
    constrained_layout=True
)

vmax = attention_np.max()

for i, ax in enumerate(axes1.flat):

    im = ax.imshow(
        attention_np[i],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax
    )

    # 標示 Query Frame 20
    ax.axhline(
        20,
        color="white",
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

    ax.set_title(
        f"Head {i+1} | Attention [49 x 49]"
    )

    ax.set_xlabel("Key Frame")
    ax.set_ylabel("Query Frame")

fig1.colorbar(
    im,
    ax=axes1,
    shrink=0.8,
    label="Attention Weight"
)

fig1.suptitle(
    "Four Multi-Head Self-Attention Maps",
    fontsize=15
)


# =========================================================
# 13. Matplotlib 圖片 B：Encoder Block 特徵比較
# =========================================================
arrays = [
    input_features[0].cpu().numpy().T,
    after_attention[0].cpu().numpy().T,
    after_mlp[0].cpu().numpy().T
]

names = [
    "1. Input: Conv + Positional Encoding",
    "2. Multi-Head Attention + Residual",
    "3. MLP + Residual (Encoder Block Output)"
]

# 原始 STFT 每個 Frame 相隔 10ms
# Conv2 stride=2 後約相隔 20ms
feature_times = times[::2]

left = feature_times[0] - 0.01
right = feature_times[-1] + 0.01

# 使用相同顏色範圍方便比較
vlim = max(
    np.max(np.abs(a))
    for a in arrays
)

fig2, axes2 = plt.subplots(
    3, 1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

for ax, arr, title in zip(
    axes2,
    arrays,
    names
):

    im2 = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        extent=[
            left,
            right,
            0,
            arr.shape[0]
        ],
        interpolation="nearest"
    )

    # 標示 440Hz -> 880Hz 切換時間
    ax.axvline(
        0.5,
        color="black",
        linestyle="--",
        alpha=0.5
    )

    ax.set_title(title)
    ax.set_ylabel("Feature Channel")

    fig2.colorbar(
        im2,
        ax=ax,
        label="Feature Value"
    )

axes2[-1].set_xlabel(
    "Audio Time (seconds)"
)

fig2.suptitle(
    "Tiny Whisper Encoder Block - Feature Stages",
    fontsize=15
)


# =========================================================
# 14. Matplotlib 圖片 C：Query Frame 20
# =========================================================
query_index = 20

fig3, ax3 = plt.subplots(
    figsize=(11, 4),
    constrained_layout=True
)

for i in range(4):

    ax3.plot(
        np.arange(attention_np.shape[-1]),
        attention_np[i, query_index],
        label=f"Head {i+1}",
        linewidth=1.7
    )

ax3.set_xlabel("Key Frame")

ax3.set_ylabel("Attention Weight")

ax3.set_title(
    "Query Frame 20: Attention Distribution"
)

ax3.grid(alpha=0.25)

ax3.legend()


# =========================================================
# 15. 只顯示圖表，不儲存任何檔案
# =========================================================
print("\nDisplaying three Matplotlib figures...")

plt.show()
