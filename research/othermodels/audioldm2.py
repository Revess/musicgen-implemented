from diffusers import AudioLDM2Pipeline
import torch

repo_id = "cvssp/audioldm2-large"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0).audios[0]

import scipy

scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)
