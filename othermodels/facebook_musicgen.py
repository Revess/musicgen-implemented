import torch
import soundfile as sf
from transformers import pipeline

synthesiser = pipeline("text-to-audio", "facebook/musicgen-stereo-small", device="cuda:0", torch_dtype=torch.float16)

music = synthesiser("lo-fi music with a soothing melody", forward_params={"max_new_tokens": 256})

sf.write("musicgen_out.wav", music["audio"][0].T, music["sampling_rate"])

from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small").to("cuda")

inputs = processor(
    text=["Samba based Drum and Bass"],
    padding=True,
    return_tensors="pt",
).to("cuda")

audio_values = model.generate(**inputs, max_new_tokens=256)

import soundfile as sf

sampling_rate = model.config.audio_encoder.sampling_rate
audio_values = audio_values.cpu().numpy()
sf.write("musicgen_out2.wav", audio_values[0].T, sampling_rate)

