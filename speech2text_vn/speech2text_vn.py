from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import warnings

warnings.simplefilter("ignore", UserWarning)

# Load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# Define function to read in sound file
def map_to_array(batch):
    speech, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch

# Load dummy dataset and read sound files
ds = map_to_array({
    "file": 'test_vn.wav'
})

# Tokenize with the correct sampling rate
input_values = processor(ds["speech"], sampling_rate=ds["sampling_rate"], return_tensors="pt", padding="longest").input_values

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Retrieve logits
logits = model(input_values).logits

# Decode the predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(transcription)
