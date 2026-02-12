import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from data_loaders import custom_test_dataloaders


def train(model, train_loader, optimizer, loss, scaler, device, scheduler=None):
    running_loss = 0.0

    for images, labels in train_loader:
        model.train() # inside loop just to be more sure
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # non_blocking=True is needed for the pin_memory=True in the dataloader
        
        optimizer.zero_grad()
        # forward pass with tensor cores
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            batch_loss = loss(outputs, labels)
        
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += batch_loss.item() * images.size(0) # multiplied by batch size, to get Per-sample average (robust against last batch size variance)
        
        if scheduler: # batch steps (cosine ann & linearLr)
            scheduler.step()

    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def test(model, loader, loss, device): # F1 & AUC scores, 2 classes test
    all_logits = []
    all_labels = []
    loss_sum = 0.0

    with torch.no_grad():
        model.eval()
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            
            loss_sum += loss(outputs, labels).item() * images.size(0)

            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    preds = all_logits.argmax(1)
    # F1 (whole data, not per batch)
    f1 = f1_score(all_labels.numpy(), preds.numpy(), average='binary')
    # AUC
    auc = roc_auc_score(all_labels.numpy(), all_logits[:,1].numpy())

    avg_loss = loss_sum / len(loader.dataset)
    return avg_loss, f1, auc


def test_1class(model, loader, device, threshold=0.5): # accuracy with confidence, 1 class test. 
    scores = []
    with torch.no_grad():
        model.eval()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            fake_probs = probs[:, 1]
            scores.extend(fake_probs.cpu().numpy())

    scores = np.array(scores)

    avg_confidence = np.mean(scores)
    percent_fake = (scores > threshold).mean() * 100
    return avg_confidence, percent_fake

def finetune_test(model,device): # EvalGen + ThisPersonDoesNotExist shortcut test
    evalgen_loader = custom_test_dataloaders(path=r"D:\Pytorch\GEN_IMAGE_DATA\GenEval")
    prsn_doesnt_exist_loader = custom_test_dataloaders(path=r'D:\Pytorch\GEN_IMAGE_DATA\DeepFake\ThisPersonDoesNotExist')

    avg_conf, pct_fake = test_1class(model, evalgen_loader, device)
    print(f"EvalGen (1-class) | avg_fake_conf={avg_conf:.4f} | %pred_fake={pct_fake:.2f}%")

    avg_conf, pct_fake = test_1class(model, prsn_doesnt_exist_loader, device)
    print(f"ThisPersonDoesNotExist (1-class) | avg_fake_conf={avg_conf:.4f} | %pred_fake={pct_fake:.2f}%")

