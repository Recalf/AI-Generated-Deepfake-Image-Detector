import torch.nn as nn
import torchvision

def build_model():
    model = torchvision.models.convnext_small(weights="DEFAULT")
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, out_features=2)

    return model