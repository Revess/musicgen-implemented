from diffusers import MusicLDMPipeline
import torch
import scipy

repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Birds chirping"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=8.0)

audio = audio.audios[0]

# save the audio sample as a .wav file
scipy.io.wavfile.write("techno.wav", rate=44100, data=audio)