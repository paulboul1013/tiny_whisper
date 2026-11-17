import numpy as np

power = np.array([4, 9, 1])
weights = np.array([0.2, 0.6, 0.2])

mel_energy = np.sum(power * weights)

print(mel_energy)  # 6.4