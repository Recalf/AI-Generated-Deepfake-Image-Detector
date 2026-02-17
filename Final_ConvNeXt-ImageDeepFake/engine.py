import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from data_loaders import custom_test_dataloaders


def train(model, train_loader, optimizer, loss, scaler, device, scheduler=None): 
    model.train()
    running_loss = 0.0

    for (images, labels) in (train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # we need non_blocking=True for the pin_memory=True in our dataloader
    
        # with tensor cores
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            batch_loss = loss(outputs, labels) 
        
        optimizer.zero_grad(set_to_none=True) # set_to_none is better and faster, it sets grad tensors to none instead of zero when emptying
        scaler.scale(batch_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler: # scheduler steps per-batch, smoother than per-epoch (for our cosine annealing & linearLr)
            scheduler.step()

        running_loss += batch_loss.item() * images.size(0) # multiplied by batch size to get per-sample average (robust against last batch size variance)
        

    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


def test(model, loader, loss, device, threshold=0.5): # F1 & AUC scores, 2 classes test
    model.eval()
    all_logits = []
    all_labels = []
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            
            loss_sum += loss(outputs, labels).item() * images.size(0)

            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.softmax(all_logits, dim=1)
    fake_probs = probs[:, 1]
    preds = (fake_probs > threshold).long()

    f1 = f1_score(all_labels.numpy(), preds.numpy(), average='binary')
    auc = roc_auc_score(all_labels.numpy(), fake_probs.numpy())

    avg_loss = loss_sum / len(loader.dataset)
    return avg_loss, f1, auc


def test_1_class(model, loader, device, threshold=0.5): # accuracy with confidence, 1 class test. 
    model.eval()
    scores = []
    with torch.no_grad():
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


def my_test(model, device): # shortcut test for my EvalGen + ThisPersonDoesNotExist 
    evalgen_loader = custom_test_dataloaders(path=r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\GenEval")
    prsn_doesnt_exist_loader = custom_test_dataloaders(path=r'D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\ThisPersonDoesNotExist')

    avg_conf, pct_fake = test_1_class(model, evalgen_loader, device)
    print(f"EvalGen (1-class) | avg_fake_conf={avg_conf:.4f} | %pred_fake={pct_fake:.2f}%")

    avg_conf, pct_fake = test_1_class(model, prsn_doesnt_exist_loader, device)
    print(f"ThisPersonDoesNotExist (1-class) | avg_fake_conf={avg_conf:.4f} | %pred_fake={pct_fake:.2f}%")

