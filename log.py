import numpy as np

mel_energy = np.array([1, 10, 100, 1000])

log_mel = np.log10(mel_energy)

print("原始能量:", mel_energy)
print("Log 後:", log_mel)