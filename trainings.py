import glob, tqdm, fire, wandb, random
from models import *
from collections import defaultdict
from torch.utils.data import Dataset
from aac_datasets.utils.collate import BasicCollate

from aac_datasets import Clotho
from audiolm_pytorch import SemanticTransformerTrainer, SoundStreamTrainer, CoarseTransformerTrainer, FineTransformerTrainer

import torch
from torch.amp import autocast, GradScaler
import torchaudio.transforms as T

from accelerate import Accelerator

class DS(Dataset):
    '''wrapper for the dataset fn'''
    def __init__(self, subset='dev'):
        self.data = load_dataset(subset=subset)

    def __getitem__(self, index):
        x, y = self.data[index]['audio'][0, :441000], self.data[index]['captions']
        return x
    
    def __len__(self):
        return len(self.data)

#TODO: Load Wandb in the trainers.
def load_dataset(subset="dev"):
    if len(glob.glob(f"./datasets/CLOTHO_v2.1/clotho_audio_files/{subset}*")) == 0:
        print(f"Downloading subset: {subset} of Clotho")
        return Clotho(root="./datasets/", subset=subset, download=True)
    return Clotho(root="./datasets/", subset=subset, download=False)

def train_mulan(device='cuda'):
    sec=10
    dataset = load_dataset()
    val_set = load_dataset(subset="val")
    wandb.init(
        # set the wandb project where this run will be logged
        project="musiclm",
        entity="bjmaat", 
        name="mulan"
    )

    mulan, audio_transformer, text_transformer = build_mulan()
    if device == 'cuda':
        mulan = mulan.cuda()
    wandb.watch(mulan)

    # Setting up the learning
    audio_lr = 4e-5 
    text_lr = 5e-5
    optimizer = torch.optim.Adam([
        {'params': mulan.audio.parameters(), 'lr': audio_lr},
        {'params': mulan.text.parameters(), 'lr': text_lr},
        {'params': mulan.contrast.parameters(), 'lr': text_lr},
        {'params': mulan.text_to_latents.parameters(), 'lr': text_lr},
        {'params': mulan.audio_to_latents.parameters(), 'lr': audio_lr},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.9)
    scaler = GradScaler()  

    accumulation_steps = 64

    for epoch in range(25):
        mulan.train()
        with tqdm.tqdm(total=len(dataset)//accumulation_steps, desc=f"Epoch {epoch}") as pbar:
            optimizer.zero_grad()
            for i, data in enumerate(dataset):
                grad_accum = False
                audio, captions = data["audio"][0][None, :sec*44100], data["captions"]
                if device == 'cuda': 
                    audio = audio.cuda()
                with autocast(device_type=device):
                    loss = mulan(audio, raw_texts=captions)
                    loss = loss / accumulation_steps
                    wandb.log({
                        'losses/train_loss': loss.item(), 
                        'loss_parts/denom_i': mulan.contrast.denominator_i.mean().item(), 
                        'loss_parts/denom_j': mulan.contrast.denominator_j.mean().item(), 
                        'loss_parts/numer': mulan.contrast.numerator.mean().item(), 
                        'loss_parts/sims': mulan.contrast.sims.mean().item(), 
                        'loss_parts/temperatures': mulan.contrast.temperatures.mean().item(),
                        'loss_logged/denom_i': torch.log(mulan.contrast.denominator_i.mean().clamp(min = 1e-20)).item(),
                        'loss_logged/denom_j': torch.log(mulan.contrast.denominator_j.mean().clamp(min = 1e-20)).item(),
                        'loss_logged/numer': -torch.log(mulan.contrast.numerator.mean().clamp(min = 1e-20)).item()
                    })
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
                tot_losses = defaultdict(float)
                for i, data in enumerate(val_set):
                    grad_accum = False
                    audio, captions = data["audio"][0][None, :sec*44100], data["captions"]
                    if device == 'cuda':
                        audio = audio.cuda()
                    loss = mulan(audio, raw_texts=captions)
                    tot_losses['losses/val_loss'] += loss.item()
                    tot_losses['val_loss_parts/denom_i'] += mulan.contrast.denominator_i.mean().item()
                    tot_losses['val_loss_parts/denom_j'] += mulan.contrast.denominator_j.mean().item()
                    tot_losses['val_loss_parts/numer'] += mulan.contrast.numerator.mean().item()
                    tot_losses['val_loss_logged/denom_i'] += torch.log(mulan.contrast.denominator_i.mean().clamp(min = 1e-20)).item()
                    tot_losses['val_loss_logged/denom_j'] += torch.log(mulan.contrast.denominator_j.mean().clamp(min = 1e-20)).item()
                    tot_losses['val_loss_logged/numer'] += -torch.log(mulan.contrast.numerator.mean().clamp(min = 1e-20)).item()
                    pbar.update(1)

                tot_losses['losses/val_loss'] /= len(val_set)
                tot_losses['val_loss_parts/denom_i'] /= len(val_set)
                tot_losses['val_loss_parts/denom_j'] /= len(val_set)
                tot_losses['val_loss_parts/numer'] /= len(val_set)
                tot_losses['val_loss_logged/denom_i'] /= len(val_set)
                tot_losses['val_loss_logged/denom_j'] /= len(val_set)
                tot_losses['val_loss_logged/numer'] /= len(val_set)

                wandb.log(tot_losses) 

    torch.save(mulan.state_dict(), './models/mulan/ckpt.pt')
    print("done training saved file in: ./models/mulan/ckpt.pt")

def train_semantic_transformer():
    dataset = DS()
    val_set = DS(subset="val")

    mulan, _, _ = build_mulan()
    mulan.load_state_dict(torch.load('./models/mulan/ckpt.pt', map_location='cpu'))
    print('done loading mulan')
    wav2vec = build_wav2vec()
    quantizer = build_quantizer(mulan)
    semantic_transformer = build_semantic_transformer(quantizer, wav2vec)

    trainer = SemanticTransformerTrainer(
        transformer = semantic_transformer,
        wav2vec = wav2vec,
        dataset=dataset,
        valid_dataset=val_set,
        audio_conditioner = quantizer,
        results_folder='./models/SemanticTransformer',
        batch_size = 1,
        data_max_length = 320 * 32,
        num_train_steps = 1_000_000,
        accelerate_kwargs={'cpu': True},
        force_clear_prev_results = True,
        use_wandb_tracking=True,
        grad_accum_every=64
    )

    trainer.wandb_tracker(project='musiclm', run='semantic_transformer')

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