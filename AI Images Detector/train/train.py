import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from timm.optim import create_optimizer_v2
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders import dataloaders
from model.convnext import build_model
from engine import train, test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = dataloaders()
    model = build_model().to(device)
    loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    # we'll be using LLRD + AdamW + wd 0.02 (lower wd cuz we're finetuning); a famous combination backed by research papers for ViTs (compatible with ConvNeXt)
    num_epochs = 8
    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=2e-4,               # the head lr
        weight_decay=0.02,
        layer_decay=0.8,       # LLRD with *0.8 lr decay, each layer (block) gets current_lr*0.8 starting from head lr
        filter_bias_and_bn=True  # no weight decay on normalizations and bias (it degrades performance (pytorch documentation and in some researches))
    )

    # LinearLR (5% of total for warmup, batch steps (cuz smoother)) + Cosine Annealing scheduler (batch steps, no restarts)
    steps_per_epoch = len(train_loader) 
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.05) # 5% warmup

    warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps) # the warmup makes our LRs start from *0.01, gradually increasing to the normal
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=0) # after the warmup we make our LRs gradually decrease like an arc (cosine)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    if device.type == "cuda": # only use AMP on gpus with tensor cores (compute capability >= 7.0)
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
        train_loss = train(model, train_loader, optimizer, loss, scaler, device, scheduler=scheduler)
        # save checkpoint
        torch.save({
            "epoch": epoch+1, 
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None
        }, f"train/checkpoint{epoch+1}_phase1.pth")
        print("model saved")
        
        val_loss, val_f1, val_auc = test(model, val_loader, loss, device) 
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}| Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        print(time.strftime("%H:%M:%S"))
        
    # model on test split
    test_loss, test_f1, test_auc = test(model, test_loader, loss, device)
    print(f"Final Test: Loss: {test_loss:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()

