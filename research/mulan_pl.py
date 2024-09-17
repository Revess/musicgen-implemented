import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities import rank_zero_only

import fire, random, wandb, random, numpy, os, natsort, glob, math, random
from tqdm import tqdm
from collections import defaultdict

import torch.distributed as dist

from trainings import load_dataset
from models import *
from aac_datasets.utils.collate import BasicCollate

class LitMuLaN(L.LightningModule):
    def __init__(self, strategy, val_samples, batch_size):
        super().__init__()
        self.strategy = strategy
        self.save_hyperparameters()
        self.val_log_dict = defaultdict(float)
        self.val_samples = val_samples
        self.batch_size = batch_size
        self.i = 0

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        audio, captions = batch['audio'][0][0][None, :441000], [random.choice(batch['captions'][0])] #Assert mono for now by taking the 0 channel and expanding it as a fake batch
        loss = self.model(audio, raw_texts=captions)
        log_dict = {
            'losses/train_loss': loss.item(), 
            'loss_parts/denom_i': self.model.contrast.denominator_i.mean().item(), 
            'loss_parts/denom_j': self.model.contrast.denominator_j.mean().item(), 
            'loss_parts/numer': self.model.contrast.numerator.mean().item(), 
            'loss_parts/sims': self.model.contrast.sims.mean().item(), 
            'loss_parts/temperatures': self.model.contrast.temperatures.mean().item(),
            'loss_logged/denom_i': torch.log(self.model.contrast.denominator_i.mean().clamp(min = 1e-20)).item(),
            'loss_logged/denom_j': torch.log(self.model.contrast.denominator_j.mean().clamp(min = 1e-20)).item(),
            'loss_logged/numer': -torch.log(self.model.contrast.numerator.mean().clamp(min = 1e-20)).item()
        }
        self.log_dict(log_dict, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, captions = batch['audio'][0][0][None, :441000], [random.choice(batch['captions'][0])]
        print(audio.device)
        loss = self.model(audio, raw_texts=captions)

        self.val_log_dict['losses/val_loss'] += loss.item()
        self.val_log_dict['val_loss_parts/denom_i'] += self.model.contrast.denominator_i.mean().item()
        self.val_log_dict['val_loss_parts/denom_j'] += self.model.contrast.denominator_j.mean().item()
        self.val_log_dict['val_loss_parts/numer'] += self.model.contrast.numerator.mean().item()
        self.val_log_dict['val_loss_logged/denom_i'] += torch.log(self.model.contrast.denominator_i.mean().clamp(min = 1e-20)).item()
        self.val_log_dict['val_loss_logged/denom_j'] += torch.log(self.model.contrast.denominator_j.mean().clamp(min = 1e-20)).item()
        self.val_log_dict['val_loss_logged/numer'] += -torch.log(self.model.contrast.numerator.mean().clamp(min = 1e-20)).item()

    def on_validation_epoch_end(self):
        print('val_epoch_end')
        self.val_log_dict['losses/val_loss'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_parts/denom_i'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_parts/denom_j'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_parts/numer'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_logged/denom_i'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_logged/denom_j'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.val_log_dict['val_loss_logged/numer'] /= (self.val_samples / self.batch_size) / torch.cuda.device_count()
        self.log_dict(self.val_log_dict, sync_dist=True)
        self.val_log_dict = defaultdict(float)
    
    def configure_optimizers(self):
        self.model, self.audio_transformer, self.text_transformer = build_mulan()

        audio_lr = 4e-5 
        text_lr = 5e-5
        optimizer = torch.optim.Adam([
            {'params': self.audio_transformer.parameters(), 'lr': audio_lr, 'foreach': False if self.strategy in ['fsdp', 'FSDP'] else None},
            {'params': self.text_transformer.parameters(), 'lr': text_lr, 'foreach': False if self.strategy in ['fsdp', 'FSDP'] else None},
            {'params': self.model.contrast.parameters(), 'lr': text_lr, 'foreach': False if self.strategy in ['fsdp', 'FSDP'] else None},
            {'params': self.model.text_to_latents.parameters(), 'lr': text_lr, 'foreach': False if self.strategy in ['fsdp', 'FSDP'] else None},
            {'params': self.model.audio_to_latents.parameters(), 'lr': audio_lr, 'foreach': False if self.strategy in ['fsdp', 'FSDP'] else None},
        ])
            
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.9)

        return (
            [optimizer], 
            [{
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }]
        )

class FSDPWeightsLogger(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # Check if it's the end of an accumulation cycle
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0 and trainer.global_rank == 0:

            @rank_zero_only  # Ensure logging happens only on rank 0
            def log_weights():
                all_params = []
                for param in pl_module.parameters():
                    gathered_param = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
                    dist.gather(param.data, gathered_param, dst=0)
                    all_params.append(torch.cat(gathered_param))
                wandb.log({"model_weights_after_step": all_params}, step=trainer.global_step)

            log_weights()

def train_mulan(
        seed=42, 
        batch_size = 1,
        devel=False,
        resume = None,
        strategy= 'auto', #'deepspeed',
        nodes=1,
        grad_accum=64,
        max_epochs=1_000_000,
        cp=False,
        run_name = "mulan"
    ):
    '''
    Trainer for training the scales dataset
    resume must be None or id of the run you want to continue
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    print(locals())
    
    wandb_logger = WandbLogger(
        project = "musiclm",
        entity = "bjmaat", 
        name = "mulan"
    )

    if resume:
        run_name += f'_{resume}'
    else:
        run_name += f'_{wandb_logger.experiment.id}' if not devel and isinstance(wandb_logger.experiment.id, str) else ''

    torch.set_float32_matmul_precision(precision='medium')

    train_set = load_dataset()
    val_set = load_dataset(subset="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=18, pin_memory=True, collate_fn=BasicCollate())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=18, pin_memory=True, collate_fn=BasicCollate())

    model = LitMuLaN(
        strategy=strategy, 
        val_samples=len(val_loader.dataset),
        batch_size = batch_size
    ).cuda()
    model.configure_optimizers()

    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    wandb_logger.watch(model)

    if strategy in ['FSDP', 'fsdp']: 
        strategy = FSDPStrategy(
            sharding_strategy='FULL_SHARD',
            cpu_offload=False,
            auto_wrap_policy=None,
            state_dict_type="full",
            # activation_checkpointing_policy= {
            #     TransformerBlock
            # }
        )

    callbacks = [
        # LearningRateMonitor(logging_interval='step'),
        # GradientAccumulationScheduler(scheduling={0: 4, 1: 8, 2: 12, 5: 16, 8: 32, 12: 64})
    ]
    if not devel:
        if cp:
            callbacks.append(ModelCheckpoint(
                monitor='losses/val_loss',
                dirpath=f'./checkpoints/{run_name}/',
                save_last=True,
                filename='epoch{epoch:02d}-step{step:04d}',
                save_top_k=1,
                every_n_epochs=1
            ))
    if strategy in ['FSDP', 'fsdp']:
        callbacks.append(FSDPWeightsLogger())
        
    print('Num avail GPUS:', torch.cuda.device_count())

    trainer = L.Trainer(
        default_root_dir=f'./models/mulan/' if not devel else '',
        strategy=strategy if not devel or torch.cuda.device_count() > 1 else 'auto',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_nodes=nodes,
        callbacks=callbacks,
        # enable_checkpointing=(not devel),
        enable_checkpointing=cp,
        logger=[wandb_logger] if not devel else [],
        log_every_n_steps=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
        max_epochs=max_epochs,
        accumulate_grad_batches=grad_accum,
        precision = "bf16-mixed"
    )

    os.system('nvidia-smi') #Print current GPU information

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=natsort.natsorted(glob.glob(f'./models/mulan/*'))[-1] if resume is not None else None
    )

if __name__ == "__main__":
    fire.Fire(train_mulan)