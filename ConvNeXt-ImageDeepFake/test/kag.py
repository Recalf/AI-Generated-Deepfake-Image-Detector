# run_diag1.py
import torch, os, shutil
import numpy as np
from torch.utils.data import DataLoader
import torchvision

import sys
import os
import kagglehub
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.convnext import build_model
from transforms import test_transforms
from engine import test  # or reuse your test function
import sklearn.metrics as skm
from tqdm import tqdm



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = kagglehub.dataset_download(
            "cashbowman/ai-generated-images-vs-real-images"
        )

    transform = test_transforms()
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    dataset.class_to_idx = {'real': 0, 'fake': 1}
    dataset.classes = ['real', 'fake']

    print("Class mapping:", dataset.class_to_idx)
    print("Class mapping:", dataset.classes)


    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load("train/phase2.pth", map_location=device))
    model.eval()

    y_true, y_pred, probs = [], [], []
    misdir = "kaggle_misclassified"
    shutil.rmtree(misdir, ignore_errors=True)
    os.makedirs(misdir, exist_ok=True)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader)):
            imgs = imgs.to(device)
            out = model(imgs)
            p = torch.softmax(out, dim=1).cpu().numpy()
            preds = p.argmax(1)
            for j in range(len(preds)):
                y_true.append(int(labels[j].item()))
                y_pred.append(int(preds[j]))
                probs.append(p[j])
                if preds[j] != labels[j].item():
                    # save misclassified image for inspection
                    idx = i*loader.batch_size + j
                    src = dataset.imgs[idx][0]
                    dst = os.path.join(misdir, f"{idx}_pred{preds[j]}_gt{labels[j].item()}_{os.path.basename(src)}")
                    shutil.copy(src, dst)

    # metrics
    cm = skm.confusion_matrix(y_true, y_pred)
    report = skm.classification_report(y_true, y_pred, target_names=dataset.classes)
    acc = skm.accuracy_score(y_true, y_pred)

    print("Classes:", dataset.class_to_idx, dataset.classes)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Report:\n", report)
    print("Misclassified samples saved to:", misdir)

if __name__ == "__main__":
    main()