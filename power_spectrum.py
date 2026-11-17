import numpy as np

stft_value = 3 + 4j

magnitude = np.abs(stft_value)
power = magnitude ** 2

print(magnitude)  # 5.0
print(power)      # 25.0