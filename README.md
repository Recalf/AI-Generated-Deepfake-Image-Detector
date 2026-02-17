# AI-Generated-Deepfake-Image-Detector
Detects if an image is Real or AI-Generated / DeepFake, using ConvNeXtV2_Base (pretrained on ImageNet) on 13 different datasets (400k images, with data augmentations and multiple epochs), And Continual Learning cycle to adapt on latest gen AI.<br>
It detects many SOTA generative models too like Nano Banana Pro, DALL-E3, Flux, Diffusion Models (SDXL, SD3...), MidjourneyV6...
<br>
I used 2 phases for training, 
phase 1: training my model on 400k from 11 datasets, using AdamW(wd 0.02)/ LLRD(headlr: 2e-4, lr_decay:0.8)/cosine annealing with warmup(linearLR)<br>
phase 2: Continual Learning, using Rehearsal Buffer (Replay 1:1) and AdamW(wd 0.01)/ LLRD(headlr: 1e-4, lr_decay:0.8)/cosine annealing with warmup(linearLR)  (i tried countless settings and this worked out best for my new small (20k images) latest gen dataset)<br>


OOD Test Scores:



Phase 1 Datasets: (the big datasets actually were 2x samples, i used half)
DDA: 55k coco, + 55Kk ai (SD2) similar image to each coco.

Defactify (MS COCOAI): 12k real vs random 70k ai, (SD21, SDXL, SD3, DALLE3, and MidjourneyV6)

genimage_tiny: 40k overall (midjourney, biggan, vqdm, sdv5, wukong, adm, glide)

art_artai: real art vs ai art, 1k overall

Midjourney_small: Midjourney with a tiny imagnet, 1k overall


VALIDATION Dataset:
Defactify (MSCOCOAI), 3k real vs 10k ai. (f1 score)

TEST Dataset:
EVALGen, made by SOTA gen models (FLUX, GoT, Infinity, OmniGen, Nova), 11k each, 55k overall.

Phase 2 Datasets (mostly for deepfake):

Phase 3 Datasets:
train:
SUPER_GENAI == 10k
JULIEN_TRAIN == 10k

val:
JULIEN_val == 3k
defactify_val == 13k




