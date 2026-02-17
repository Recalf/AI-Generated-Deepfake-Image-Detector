

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from timm.optim import create_optimizer_v2
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transforms import continual_transforms, test_transforms, dda_transforms
from model.convnext import build_model
from engine import train, test


# This Continual Learning cycle to finetune the model on latest generative AI images to keep it up to date. 
# we'll be using rehearsal buffer (replay 1:1)
# so New dataset 20k samples + Replay 20k images (all old datasets stratfied) 


# phase2 best checkpoint
CKPT_PATH = r"checkpoints\checkpoint_phase2.pth"

# Replay budget 
REPLAY_P1 = 20000
REPLAY_SEED = 101


# ====== PATHS 

# New (20k ish)
SUPER_GENAI = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\Super_GenAI_Dataset"
JULIEN_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\midjourney-dalle-sd-nanobananapro-dataset\train"
# validation
JULIEN_VAL  = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\New-Gen\midjourney-dalle-sd-nanobananapro-dataset\test" 
DEFACTIFY_VAL = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Defactify\val" # old val
DF40_VAL = r'D:\Pytorch\data\GEN_IMAGE_DATA_V2\DeepFake\DF40\val' # old val

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

    train_loader, val_loader = Continual_dataloaders(0) # we'll be loading the train_loader later in the epochs loop, to load in each epoch different replay

    model = build_model().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    loss = nn.CrossEntropyLoss(label_smoothing=0.05) # kept label smoothing, worked better than 0 even in this phase

    # i tried and researched lots of different combinations and this worked out best, the model depends on this fine tune a lot for latest gen performance, so we need that high lr low wd.
    num_epochs = 8   # the 6th epoch worked best for me
    optimizer = create_optimizer_v2( 
        model,
        opt='adamw',
        lr=1e-4,            
        weight_decay=0.01, 
        layer_decay=0.8,      
        filter_bias_and_bn=True  
    )

    steps_per_epoch = len(train_loader) 
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(steps_per_epoch * 0.1) # 10% of first epoch warmup

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps) # the warmup makes our LRs start from *0.01, gradually increasing to the normal
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=0) # after the warmup we make our LRs gradually decrease like an arc (cosine). (scheduler batch steps for smoothness)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        train_loader, _= Continual_dataloaders(epoch)
        train_loss = train(model, train_loader, optimizer, loss, scaler, device, scheduler)

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
        print(time.strftime("%H:%M:%S"))



def Continual_dataloaders(epoch, batch_size=28, num_workers=8):
    new_super = torchvision.datasets.ImageFolder(SUPER_GENAI, continual_transforms())
    new_julien = torchvision.datasets.ImageFolder(JULIEN_TRAIN, continual_transforms())
    new_train = ConcatDataset([new_super, new_julien])

    new_julien_val = torchvision.datasets.ImageFolder(JULIEN_VAL, test_transforms())
    defactify_val = torchvision.datasets.ImageFolder(DEFACTIFY_VAL, test_transforms())
    df40_val = torchvision.datasets.ImageFolder(DF40_VAL, test_transforms())
    val_data = ConcatDataset([new_julien_val, df40_val, defactify_val])

    #=====
    # phase1 equal replay
    p1_parts = []
    per_p1 = REPLAY_P1 // len(PHASE1_PATHS)

    for i, path in enumerate(PHASE1_PATHS):
        if path == DDA_TRAIN:
            ds = torchvision.datasets.ImageFolder(path, dda_transforms())
        else:
            ds = torchvision.datasets.ImageFolder(path, continual_transforms())

        k = min(per_p1, len(ds))
        p1_parts.append(random_k_subset(ds, k, REPLAY_SEED + epoch + i)) # "+epoch" for different seed each epoch, "+i" extra safety

    replay = ConcatDataset(p1_parts)
    #========
    train_data = ConcatDataset([new_train, replay])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,        
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
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
