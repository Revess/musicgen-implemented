import glob, tqdm, fire
from models import *

from aac_datasets import Clotho
from audiolm_pytorch import SemanticTransformerTrainer, SoundStreamTrainer, CoarseTransformerTrainer, FineTransformerTrainer

import torch
from torch.amp import autocast, GradScaler
import torchaudio.transforms as T

from accelerate import Accelerator
import wandb

#TODO: Load Wandb in the trainers.
def load_dataset(subset="dev"):
    if len(glob.glob(f"./datasets/CLOTHO_v2.1/clotho_audio_files/{subset}*")) == 0:
        print(f"Downloading subset: {subset} of Clotho")
        return Clotho(root="./datasets/", subset=subset, download=True)
    return Clotho(root="./datasets/", subset=subset, download=False)

def train_mulan():
    dataset = load_dataset()
    val_set = load_dataset(subset="val")
    wandb.init(
        # set the wandb project where this run will be logged
        project="musiclm",
        entity="bjmaat", 
        name="mulan", 
    )

    mulan, audio_transformer, text_transformer = build_mulan()
    mulan = mulan.cuda()
    wandb.watch(mulan)

    # Setting up the learning
    audio_lr = 4e-5 
    text_lr = 5e-5
    optimizer = torch.optim.Adam([
        {'params': audio_transformer.parameters(), 'lr': audio_lr},
        {'params': text_transformer.parameters(), 'lr': text_lr}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.9)
    scaler = GradScaler()  

    downsample_rate = 22050//4
    resample = T.Resample(orig_freq=44100, new_freq=downsample_rate) 
    accumulation_steps = 64

    for epoch in range(25):
        mulan.train()
        with tqdm.tqdm(total=len(dataset)//64, desc=f"Epoch {epoch}") as pbar:
            optimizer.zero_grad()
            for i, data in enumerate(dataset):
                grad_accum = False
                audio, captions = data["audio"], data["captions"]
                audio = resample(audio)
                audio = audio.cuda()
                with autocast(device_type='cuda'):
                    loss = mulan(audio, raw_texts=captions)
                    loss = loss / accumulation_steps
                    wandb.log({'losses/train_loss': loss.item()})
                scaler.scale(loss).backward()

                if i % accumulation_steps == 0 and i != 0:    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)
                    grad_accum = True
            if not grad_accum:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
                grad_accum = True

        mulan.eval()
        with tqdm.tqdm(total=len(val_set), desc=f"Epoch {epoch}") as pbar:
            with torch.no_grad():
                tot_loss = 0
                for i, data in enumerate(dataset):
                    grad_accum = False
                    audio, captions = data["audio"], data["captions"]
                    audio = resample(audio)
                    audio = audio.cuda()
                    loss = mulan(audio, raw_texts=captions)
                    tot_loss += loss.item()
                    pbar.update(1)
                wandb.log({'losses/val_loss': tot_loss / len(val_set)})

    torch.save(mulan.state_dict(), './models/mulan/ckpt.pt')
    print("done training saved file in: ./models/mulan/ckpt.pt")

def train_semantic_transformer():
    mulan, _, _ = build_mulan()
    mulan.load_state_dict(torch.load('./models/mulan/ckpt.pt'))
    wav2vec = build_wav2vec()
    quantizer = build_quantizer(mulan)
    semantic_transformer = build_semantic_transformer(quantizer, wav2vec)

    trainer = SemanticTransformerTrainer(
        transformer = semantic_transformer,
        wav2vec = wav2vec,
        audio_conditioner = quantizer,
        folder ='./datasets/CLOTHO_v2.1/clotho_audio_files',
        results_folder='./models/SemanticTransformer',
        batch_size = 1,
        data_max_length = 320 * 32,
        num_train_steps = 1_000_000,
        accelerate_kwargs={'cpu': True},
        force_clear_prev_results = True
    )

    trainer.train()
    torch.save(mulan.state_dict(), './mulan/SemanticTransformer')
    print("done training saved file in: ./models/SemanticTransformer/")

def train_sound_stream():
    soundstream = build_sound_stream()

    trainer = SoundStreamTrainer(
        soundstream,
        folder ='./datasets/CLOTHO_v2.1/clotho_audio_files',
        batch_size = 4,
        grad_accum_every = 8,         # effective batch size of 32
        data_max_length_seconds = 2,  # train on 2 second audio
        num_train_steps = 1_000_000,
        results_folder='./models/SoundStream',
        accelerator=Accelerator(cpu=True),  #Remove when training on GPU
        force_clear_prev_results = True
    )

    trainer.train()
    print("done training saved file in: ./models/SoundStream/")

def train_coarse_transformer():
    soundstream = SoundStream.init_and_load_from('./models/SoundStream/soundstream.0.pt') # Do set this to the latest path
    mulan, _, _ = build_mulan()
    mulan.load_state_dict(torch.load('./models/mulan/ckpt.pt'))
    quantizer = build_quantizer(mulan)

    wav2vec = build_wav2vec()
    coarse_transformer = build_coarse_transformer(wav2vec)

    trainer = CoarseTransformerTrainer(
        transformer = coarse_transformer,
        wav2vec = wav2vec,
        codec = soundstream,
        audio_conditioner = quantizer,
        folder ='./datasets/CLOTHO_v2.1/clotho_audio_files',
        batch_size = 4,
        grad_accum_every = 8,         # effective batch size of 32
        data_max_length = 320 * 32,
        num_train_steps = 1,
        results_folder='./models/CoarseTransformer/',
        accelerate_kwargs={'cpu': True},
        force_clear_prev_results = True
    )

    trainer.train()
    print("done training saved file in: ./models/CoarseTransformer/")

def train_fine_transformer():
    soundstream = SoundStream.init_and_load_from('./models/SoundStream/soundstream.0.pt') # Do set this to the latest path
    mulan, _, _ = build_mulan()
    mulan.load_state_dict(torch.load('./models/mulan/ckpt.pt'))
    quantizer = build_quantizer(mulan)

    fine_transformer = build_fine_transformer()

    trainer = FineTransformerTrainer(
        transformer = fine_transformer,
        codec = soundstream,
        audio_conditioner = quantizer,
        folder ='./datasets/CLOTHO_v2.1/clotho_audio_files',
        batch_size = 4,
        grad_accum_every = 8,         # effective batch size of 32
        data_max_length = 320 * 32,
        num_train_steps = 1,
        results_folder='./models/FineTransformer/',
        accelerate_kwargs={'cpu': True},
        force_clear_prev_results = True
    )

    trainer.train()
    print("done training saved file in: ./models/FineTransformer/")

if __name__ == "__main__":
    fire.Fire()