import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders import dataloaders, custom_test_dataloaders
from model.convnext import build_model
from engine import test, test_1_class, my_test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    ckpt = torch.load("train/checkpoint8_phase2.pth", map_location=device)

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()


    # EVALGEN test data && this person doesnt exist. both OOD.
    my_test(model, device)


    # custom test data
    #path = r'C:\Users\touto\Desktop\test\CelebDF_V2\Celeb_V2\Test'
    #loader = custom_test_dataloaders(path=path)
    #custom_test(path, model, loader, device)



def custom_test(path, model, loader, device):
    num_unique = len(loader.dataset.classes)
    print("classes:", num_unique)

    if num_unique == 2: # 2 classes test:
        loss = nn.CrossEntropyLoss()
        test_loss, test_f1, test_auc = test(model, loader, loss, device, threshold=0.5)  # fake threshold, lower = classifies more as fake
        print(f"Final Test: Loss: {test_loss:.4f}, F1: {test_f1*100:.4f}, AUC: {test_auc*100:.4f}")

    elif num_unique == 1: # 1 classe test   
        avg_confidence, percent_fake = test_1_class(model, loader, device, threshold=0.5) 

        print(f"% Predicted as Fake: {percent_fake:.2f}%")  
        print(f"Average Confidence (Fake): {avg_confidence:.4f}")


if __name__ == "__main__":
    main()