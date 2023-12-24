from __future__ import print_function
import numpy as np
import time
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pydub import AudioSegment

start_time = time.time()

# Tải file âm thanh
y, sr = librosa.load('./input/initial_input.wav', duration=200)

# Tính toán Spectrogram
S_full, phase = librosa.magphase(librosa.stft(y))

# Giảm tiếng ồn
S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

S_filter = np.minimum(S_full, S_filter)

# Áp dụng Mask
width = int(librosa.time_to_frames(5, sr=sr)) 
margin_i, margin_v = 1, 10
power = 1

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Tách foreground và background
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# Nghịch đảo STFT và Tái tạo
y_foreground = librosa.istft(S_foreground * phase)

end_time = time.time()
processing_time = end_time - start_time

sf.write('./output/lib_output.wav', y_foreground, samplerate=sr)

# Tính tỉ lệ sạch của file âm thanh
# Tải file âm thanh ban đầu và file giảm ồn
original_audio, sr_original = librosa.load('./input/initial_input.wav', duration=200)
reduced_noise_audio, sr_reduced = librosa.load('./output/lib_output.wav', duration=200)

# Đảm bảo rằng cả hai tín hiệu có cùng độ dài
min_length = min(len(original_audio), len(reduced_noise_audio))
original_audio = original_audio[:min_length]
reduced_noise_audio = reduced_noise_audio[:min_length]

# Tính tỉ lệ năng lượng của tín hiệu giảm ồn và tín hiệu nền
energy_foreground = np.sum(reduced_noise_audio**2)
energy_background = np.sum((original_audio - reduced_noise_audio)**2)

# Tính tỉ lệ tương phản
cleanliness_ratio_librosa = energy_foreground / (energy_foreground + energy_background)

print(f"Tỉ lệ sạch (Librosa): {cleanliness_ratio_librosa:.2%}")
print(f"Thời gian xử lý: {processing_time:.2f} giây")