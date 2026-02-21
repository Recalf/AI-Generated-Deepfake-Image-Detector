from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image, ImageFile

from transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataloaders(BATCH_SIZE = 30, NUM_WORKERS = 8): 

    # DIRS CONFIG   
    DDA_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\DDA-Training-Set\train"
    DEFACTIFY_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Defactify\train" 
    VISCOUNTER_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\VisCounter_COCOAI\train"  
    GENIMAGETINY_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\genimage_tiny\train"
    ARTAI_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\art_artai\train"
    MIDJRN_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Midjourney_small\train"
    DF40_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\DF40\train"
    GRAVEX_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\gravex200k\train" 
    STYLEGAN2_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\StyleGan2\train"
    HASS_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\human_faces_hass\train" 
    MONK_TRAIN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\dfk_oldmonk\train" 
     
    DEFACTIFY_VAL = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Defactify\val" 
    DF40_VAL = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\Deepfake\DF40\val"

    EVALGEN = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\GenEval"
    PRSN_DSNT_EXIST = r"D:\Pytorch\data\GEN_IMAGE_DATA_V2\TEST\ThisPersonDoesNotExist"


    # TRAIN DATASETS: 
    # (ive lowered the samples in the big datasets to half, now we have a total around 400k training samples)
    # First dataset: DDA (arxiv:2505.14359). 55k coco, + 55k ai (sd2 gen) similar image to each coco. (also this needs its own transforms it needs light augs...)
    dda_train = torchvision.datasets.ImageFolder(DDA_TRAIN, dda_transforms())

    # Second: Defactify (MS COCOAI). 14k real vs 40k ai, (SD21, SDXL, SD3, DALLE3, and MidjourneyV6). (hugging face: Rajarshi-Roy-research/Defactify_Image_Dataset)
    defactify_train = torchvision.datasets.ImageFolder(DEFACTIFY_TRAIN, train_transforms())
    
    # Third: COCO AI (different than 2nd dataset). 1 real (dummy), 40k ai (SD21, SDXL, SD3, DALLE3, and MidjourneyV6). paper: arXiv:2411.16754.
    viscounter_train = torchvision.datasets.ImageFolder(VISCOUNTER_TRAIN, train_transforms())
    
    # Fourth: genimage_tiny. 40k overall (midjourney, biggan, vqdm, sdv5, wukong, adm, glide)
    genimagetiny_train = torchvision.datasets.ImageFolder(GENIMAGETINY_TRAIN, train_transforms())

    # Fifth real art vs ai art, 1k overall
    artai_train = torchvision.datasets.ImageFolder(ARTAI_TRAIN, train_transforms())

    # Sixth: midjourney with a tiny imagnet, 1k overall
    midjrn_train = torchvision.datasets.ImageFolder(MIDJRN_TRAIN, train_transforms())
    
    # DEEPFAKE Train:
    # First: DF40. real faces vs deepfake, 30k overall
    df40_train = torchvision.datasets.ImageFolder(DF40_TRAIN, train_transforms())

    # Second: Gravex200k, real vs deepfake, 100k overall.
    gravex_train = torchvision.datasets.ImageFolder(GRAVEX_TRAIN, train_transforms())

    # Third: StyleGan2. around 13K overall
    stylegan2_train = torchvision.datasets.ImageFolder(STYLEGAN2_TRAIN, train_transforms())

    # Fourth: around 10k overall
    hass_train = torchvision.datasets.ImageFolder(HASS_TRAIN, train_transforms())

    # Fifth: around 1k overall
    monk_train = torchvision.datasets.ImageFolder(MONK_TRAIN, train_transforms())

    train_data = ConcatDataset([dda_train, defactify_train, viscounter_train, genimagetiny_train, artai_train, midjrn_train, df40_train, gravex_train, stylegan2_train, hass_train, monk_train])


    # VALIDATION Datasets: 
    # (we'll be using f1 score, class imbalance)
    # Defactify val set, 3k real vs 10k ai.
    defactify_val = torchvision.datasets.ImageFolder(DEFACTIFY_VAL, test_transforms())

    # DF40 val set, 1.5k real vs 1.5k ai.
    df40_val = torchvision.datasets.ImageFolder(DF40_VAL, test_transforms())

    val_data = ConcatDataset([defactify_val, df40_val])

    # TEST OOD Datasets:
    # EVALGen, 55k ai images, made by SOTA gen models (FLUX, GoT, Infinity, OmniGen, Nova) 11k each
    evalgen = torchvision.datasets.ImageFolder(EVALGEN, test_transforms())

    # This_Person_Doesnt_Exist. DeepFake, 7k ai faces
    tpde = torchvision.datasets.ImageFolder(PRSN_DSNT_EXIST, test_transforms())

    test_data = ConcatDataset([evalgen, tpde])

    pin_memory = torch.cuda.is_available() # True if we have cuda, this makes cpu to gpu transfer faster
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,          
        prefetch_factor=1, # preloads 1 batche
        persistent_workers=True,              
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=pin_memory, 
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=pin_memory, 
        persistent_workers=True,
    )


    return train_loader, val_loader, test_loader


def custom_test_dataloaders(path, BATCH_SIZE = 28, NUM_WORKERS = 8): 
    test_data = torchvision.datasets.ImageFolder(path, test_transforms())
    pin_memory = torch.cuda.is_available()
    
    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=pin_memory, 
        persistent_workers=True,
    )
    return test_loader
