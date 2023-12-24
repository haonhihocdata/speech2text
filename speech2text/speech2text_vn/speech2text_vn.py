# import
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor

# load the tokenizer and model for Vietnamese
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# load the audio data (use your own wav file here!)
input_audio, sr = librosa.load('noise_reduction.wav', sr=16000)  # assuming 'test_vi.wav' is your Vietnamese audio file

# process the audio
input_values = processor(input_audio, return_tensors="pt", padding="longest").input_values

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

# print the output
print(transcription)

'''@misc{Thai_Binh_Nguyen_wav2vec2_vi_2021,
  author = {Thai Binh Nguyen},
  doi = {10.5281/zenodo.5356039},
  month = {09},
  title = {{Vietnamese end-to-end speech recognition using wav2vec 2.0}},
  url = {https://github.com/vietai/ASR},
  year = {2021}
}'''