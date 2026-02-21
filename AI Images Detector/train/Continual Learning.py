import torch
import torch.nn as nn
import torch.optim as optim
from timm.optim import create_optimizer_v2
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms import train_transforms, test_transforms, dda_transforms, continual_transforms
from model.convnext import build_model
from engine import train, test
from test.test import my_test


# This Continual Learning cycle to finetune the model on latest generative AI images to keep it up to date. 
# we'll be using rehearsal buffer (replay), 1:8 worked soo much better than 1:2, 1:1, 1:4... in what i've tested.
# so we'll do New dataset 20k samples + Replay 2.5K images (all old datasets stratfied) 

# phase1 best checkpoint
CKPT_PATH = "checkpoints/checkpoint_phase1.pth"

# Replay budget 
REPLAY_P1 = 2500 
REPLAY_SEED = 70

# ====== PATHS 
# New (20k ish)
SUPER_GENAI = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\Super_GenAI_Dataset"
JULIEN_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\midjourney-dalle-sd-nanobananapro-dataset\train"
# validation
JULIEN_VAL  = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\midjourney-dalle-sd-nanobananapro-dataset\test" 
DEFACTIFY_VAL = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Defactify\val" # old val
# Old Datasets 
DDA_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\DDA-Training-Set\train"
DEFACTIFY_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Defactify\train" 
#VISCOUNTER_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\VisCounter_COCOAI\train"  Removed it from this phase cuz its very imbalanced for the Replay
GENIMAGETINY_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\genimage_tiny\train"
ARTAI_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\art_artai\train"
MIDJRN_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Midjourney_small\train"
DF40_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\DF40\train"
GRAVEX_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\gravex200k\train" 
STYLEGAN2_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\StyleGan2\train"
HASS_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\human_faces_hass\train" 
MONK_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\dfk_oldmonk\train" 

EVALGEN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\GenEval"
PRSN_DSNT_EXIST = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\ThisPersonDoesNotExist"

PHASE1_PATHS = [
    DDA_TRAIN, DEFACTIFY_TRAIN,
    GENIMAGETINY_TRAIN,
    ARTAI_TRAIN, MIDJRN_TRAIN,
    DF40_TRAIN, GRAVEX_TRAIN,
    STYLEGAN2_TRAIN, HASS_TRAIN, MONK_TRAIN
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = Continual_dataloaders(0) # we'll be loading the train_loader in each epoch too, to load different replays each epoch.
    
    model = build_model().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    # this worked best out of more than 30 combinations (i was experimenting), 1:8 replay with these settings
    loss = nn.CrossEntropyLoss(label_smoothing=0.05) # kept label smoothing, worked better
    num_epochs = 5
    optimizer = create_optimizer_v2( 
        model,
        opt='adamw',
        lr=7.5e-5,       
        weight_decay=0.02, 
        layer_decay=0.85, 
        filter_bias_and_bn=True  
    )
    steps_per_epoch = len(train_loader) 
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.07) #  7% warmup

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps) # the warmup makes our LRs start from *0.01, gradually increasing to the normal
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=1e-7) # after the warmup we make our LRs gradually decrease like an arc (cosine) to lr=1e-7 (our earliest layer with the LLRD is bigger than that) 
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]) # (scheduler batch steps for smoothness)

    if device.type == "cuda": # only use AMP on gpus with tensor cores
        try:
            major, _ = torch.cuda.get_device_capability(device)
            use_amp = major >= 7 
        except:
            use_amp = False
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        torch.backends.cudnn.benchmark = True
    else:
        scaler = None

    for epoch in range(num_epochs):
        train_loader, _= Continual_dataloaders(epoch)
        train_loss = train(model, train_loader, optimizer, loss, scaler, device, scheduler) # ,scheduler

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None
        }, f"train/checkpoint{epoch+1}_phase2.pth")
        print("model saved")

        val_loss, val_f1, val_auc = test(model, val_loader, loss, device)

        print(f"Phase2 Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | val_auc={val_auc:.4f}")

        # test
        if (epoch+1) % 2 == 0:
            my_test(model, device)

        print(time.strftime("%H:%M:%S"))



_DS_CACHE = None # cache so ImageFolder doesnt rescan disk every epoch

def Continual_dataloaders(epoch, batch_size=28, num_workers=8):
    global _DS_CACHE

    if _DS_CACHE is None:
        new_super = torchvision.datasets.ImageFolder(SUPER_GENAI, continual_transforms())
        new_julien = torchvision.datasets.ImageFolder(JULIEN_TRAIN, continual_transforms())
        new_train = ConcatDataset([new_super, new_julien])

        new_julien_val = torchvision.datasets.ImageFolder(JULIEN_VAL, test_transforms())
        defactify_val = torchvision.datasets.ImageFolder(DEFACTIFY_VAL, test_transforms())
        val_data = ConcatDataset([new_julien_val, defactify_val])

        #=====
        # phase1 equal replay
        ds_list = []
        for i, path in enumerate(PHASE1_PATHS):
            if path == DDA_TRAIN:
                ds = torchvision.datasets.ImageFolder(path, dda_transforms())
            else:
                ds = torchvision.datasets.ImageFolder(path, continual_transforms())
            ds_list.append((i, ds))

        _DS_CACHE = (new_train, val_data, ds_list)

    new_train, val_data, ds_list = _DS_CACHE

    #=====
    # phase1 equal replay
    per_p1 = REPLAY_P1 // len(PHASE1_PATHS)

    ks = []
    total = 0
    for i, ds in ds_list:
        k = min(per_p1, len(ds))
        ks.append(k)
        total += k

    # if a dataset doesnt have enough samples to keep up with the replay, fill with other datasets that has leftover
    left = REPLAY_P1 - total
    if left > 0:
        spare = [len(ds_list[j][1]) - ks[j] for j in range(len(ds_list))] # leftover of each dataset
        j = 0
        while left > 0:
            if spare[j] > 0:  # find next dataset that has leftovers
                add = min(spare[j], left)
                ks[j] += add
                spare[j] -= add
                left -= add
            j += 1
            if j >= len(ds_list):
                j = 0
            if left > 0 and sum(spare) == 0: # if no left over across all datasets, break
                break

    p1_parts = []
    for (i, ds), k in zip(ds_list, ks):
        p1_parts.append(random_k_subset(ds, k, REPLAY_SEED + epoch + i))

    replay = ConcatDataset(p1_parts)
    #========
    train_data = ConcatDataset([new_train, replay])

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False, # cuz we're recreating our data loader in each epoch, it wont make sense if we try to persist the threads
        prefetch_factor=1
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    return train_loader, val_loader

def random_k_subset(ds, k, seed=101):
    n = len(ds)
    if k >= n:
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(ds, idx)

if __name__ == "__main__":
    main()
