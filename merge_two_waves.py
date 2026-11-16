import numpy as np
import matplotlib.pyplot as plt

# 每秒取樣 8000 次
sample_rate = 8000

# 錄製 1 秒
duration = 1.0

# 產生 8000 個時間點
t = np.arange(int(sample_rate * duration)) / sample_rate

# 聲音 A：440 Hz
sound_a = 0.3 * np.sin(2 * np.pi * 440 * t)

# 聲音 B：880 Hz
sound_b = 0.3 * np.sin(2 * np.pi * 880 * t)

# 將兩個聲音逐點相加
mixed = sound_a + sound_b

# 對混合音訊做 FFT
fft_result = np.fft.rfft(mixed)

# 取得每個頻率位置對應的 Hz
frequencies = np.fft.rfftfreq(
    len(mixed),
    d=1/sample_rate
)

# 計算各頻率的振幅
magnitudes = np.abs(fft_result) * 2 / len(mixed)

# 找出振幅最大的兩個頻率
indices = np.argsort(magnitudes)[-2:]

for i in indices:
    print(f"{frequencies[i]:.0f} Hz: {magnitudes[i]:.3f}")

print("A:", sound_a.shape)
print("B:", sound_b.shape)
print("Mixed:", mixed.shape)

# 只顯示前 20 毫秒，才能看清波形
n = int(0.02 * sample_rate)

# 顯示時域
plt.figure(figsize=(10, 5))
plt.plot(t[:n], sound_a[:n], label="440 Hz")
plt.plot(t[:n], sound_b[:n], label="880 Hz")
plt.plot(t[:n], mixed[:n], label="Mixed", linewidth=2)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()


# 顯示頻域
plt.figure(figsize=(10, 4))
plt.plot(frequencies, magnitudes)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT Frequency Spectrum")
plt.xlim(0, 1500)
plt.grid(True)

plt.show()
