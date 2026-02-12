import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import kagglehub
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.convnext import build_model
from transforms import test_transforms
from engine import test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download dataset
    """path = kagglehub.dataset_download(
        "cashbowman/ai-generated-images-vs-real-images"
    )
    print("Dataset path:", path)
     """
    path = r'D:\Pytorch\data\art_artai'

    transform = test_transforms()

    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = build_model().to(device)
    model.load_state_dict(torch.load("train/phase2.pth", map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_acc = test(model, loader, loss_fn, device)

    print(f"KAGGLE TEST → Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
