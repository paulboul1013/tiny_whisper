
import numpy as np
import matplotlib.pyplot as plt

# 每秒記錄 8 個取樣點
sample_rate = 8

# 聲音每秒振動 1 次
frequency = 1

# 聲音長度為 1 秒
duration = 1

# 建立取樣時間
t = np.arange(sample_rate * duration) / sample_rate

# 產生正弦波
audio = np.sin(2 * np.pi * frequency * t)

print("時間:", t)
print("聲音:", np.round(audio, 3))


plt.stem(t, audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("1 Hz Sound / 8 Hz Sample Rate")
plt.ylim(-1.2, 1.2)
plt.grid(True)
plt.show()