import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from data_loaders import custom_test_dataloaders
from torch.nn.utils import clip_grad_norm_


def _supports_tensor_cores(device): # check if gpu supports tensor cores (compute capability >= 7.0)
    if device.type != "cuda":
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 7 
    except:
        return False


def train(model, train_loader, optimizer, loss, scaler, device, scheduler=None): 
    model.train()
    running_loss = 0.0
    use_cuda = device.type == "cuda"
    use_amp = _supports_tensor_cores(device)  # only use AMP on gpus with tensor cores

    for (images, labels) in (train_loader):
        images, labels = images.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda)
        
        optimizer.zero_grad(set_to_none=True) # set_to_none is better and faster, it sets grad tensors to none instead of zero when emptying

        if use_amp:
            with torch.amp.autocast(device_type=device.type): # mixed precision training (AMP)
                outputs = model(images)
                batch_loss = loss(outputs, labels)
        else:
            outputs = model(images)
            batch_loss = loss(outputs, labels) 
        
        if scaler is not None:
            scaler.scale(batch_loss).backward()

            scaler.unscale_(optimizer) # gradient clipping
            clip_grad_norm_(model.parameters(), 1.0) 
            
            scaler.step(optimizer)
            scaler.update()

        else:
            batch_loss.backward()
            clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
        
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
    use_cuda = device.type == "cuda"
    use_amp = _supports_tensor_cores(device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda)
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
            else:
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
    use_cuda = device.type == "cuda"
    use_amp = _supports_tensor_cores(device)
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=use_cuda)
            
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
            else:
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            fake_probs = probs[:, 1]
            scores.extend(fake_probs.cpu().numpy())

    scores = np.array(scores)

    avg_confidence = np.mean(scores)
    percent_fake = (scores > threshold).mean() * 100
    return avg_confidence, percent_fake
