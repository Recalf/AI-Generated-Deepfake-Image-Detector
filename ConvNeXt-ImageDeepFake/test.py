import torch
import torch.nn as nn
import numpy as np


from data_loaders import dataloaders, custom_test_dataloaders
from model.convnext import build_model
from engine import test, test_1class, finetune_test
from train.finetune2_newgen import phase3_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    ckpt = torch.load("train/checkpoint1_phase3.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # custom test data
    #path = r'C:\Users\touto\Desktop\test\cifake\test'
    #loader = custom_test_dataloaders(path=path)


    """
    num_unique = len(loader.dataset.classes)
    print("classes:", num_unique)
    if num_unique == 2: # 2 classes test:
        loss = nn.CrossEntropyLoss()
        test_loss, test_f1, test_auc = test(model, loader, loss, device)
        print(f"Final Test: Loss: {test_loss:.4f}, F1: {test_f1*100:.4f}, AUC: {test_auc*100:.4f}")

    elif num_unique == 1: # 1 classe test   
        avg_confidence, percent_fake = test_1class(model, loader, device)

        print(f"Average Confidence (Fake): {avg_confidence:.4f}")
        print(f"% Predicted as Fake: {percent_fake:.2f}%")  
    """

    # EVALGEN test data && this person doesnt exist. both OOD.
    finetune_test(model, device)

if __name__ == "__main__":
    main()