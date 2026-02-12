from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image, ImageFile

from transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def custom_test_dataloaders(path, BATCH_SIZE = 32, NUM_WORKERS = 10):
    test_data = torchvision.datasets.ImageFolder(path, test_transforms())
    
    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True,
    )
    return test_loader

def dataloaders(BATCH_SIZE = 32, NUM_WORKERS = 10):

    # DIR CONFIG
    DDA_BASE = r"D:\Pytorch\GEN_IMAGE_DATA\DDA-Training-Set"
    DEFACTIFY_BASE = r"D:\Pytorch\GEN_IMAGE_DATA\Defactify" 
    GENIMAGETINY_BASE = r"D:\Pytorch\GEN_IMAGE_DATA\genimage_tiny"
    ARTAI_BASE = r"D:\Pytorch\GEN_IMAGE_DATA\art_artai"
    MIDJRN_BASE = r"D:\Pytorch\GEN_IMAGE_DATA\Midjourney_small"
    EVALGEN = r"D:\Pytorch\GEN_IMAGE_DATA\GenEval"

    # Training Datasets:

    # First dataset: DDA.
    # 110k coco, + 110k ai (SD2) similar image to each coco. (also this needs its own transforms cuz we dont want recompressing or heavy augs...)
    dda_train = torchvision.datasets.ImageFolder(DDA_BASE, dda_transforms())

    # Second dataset: Defactify (MS COCOAI).
    # 12k real vs random 70k ai, (SD21, SDXL, SD3, DALLE3, and MidjourneyV6)
    defactify_train = torchvision.datasets.ImageFolder(os.path.join(DEFACTIFY_BASE, "train"), train_transforms())
    
    # Third dataset: genimage_tiny.
    # 40k overall (midjourney, biggan, vqdm, sdv5, wukong, adm, glide)
    genimagetiny_train = torchvision.datasets.ImageFolder(GENIMAGETINY_BASE, train_transforms())

    # Fourth dataset: art_artai.
    # real art vs ai art, 1k overall
    artai_train = torchvision.datasets.ImageFolder(ARTAI_BASE, train_transforms())

    # Fifth dataset: Midjourney_small.
    # midjourney with a tiny imagnet, 1k overall
    midjrn_train = torchvision.datasets.ImageFolder(MIDJRN_BASE, train_transforms())


    # VALIDATION Dataset:
    # Defactify (MSCOCOAI), 3k real vs 10k ai. (we'll be using f1 score)
    val_data = torchvision.datasets.ImageFolder(os.path.join(DEFACTIFY_BASE, "val"), test_transforms())

    # TEST Dataset:
    # EVALGen, made by SOTA gen models (FLUX, GoT, Infinity, OmniGen, Nova), 11k each, 55k overall.
    test_data = torchvision.datasets.ImageFolder(EVALGEN, test_transforms())


    # Full Train Data Concate:
    train_data = ConcatDataset([dda_train, defactify_train, genimagetiny_train, artai_train, midjrn_train])


    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,          
        prefetch_factor=2, # preloads 2 batches 
        persistent_workers=True,              
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True,
    )


    return train_loader, val_loader, test_loader
