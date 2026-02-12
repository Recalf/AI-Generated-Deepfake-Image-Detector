# AI-Generated-Deepfake-Image-Detector
Detects if an image is Real or AI-Generated / DeepFake, using ConvNeXt (pretrained on ImageNet) on 11 different datasets (700k images, with data augmentations and multiple epochs) 
It detects Deepfake AI faces images too, and many new and SOTA generative models like Flux, Diffusion Models (SDXL, SD3...), MidjourneyV6, DALL‑E3, Nano Banana Pro...

I fine tuned it in my training phase 2 on deepfake ai faces
And fine tuned it in training phase 3 on the latest Geneative AI images

I used techniques like: 
Differential Learning Rates, Sequential LR Scheduler with 1 epoch LinearLR warmup and rest with CosineAnnealing Scheduler
I used Continual Learning with rehearsal buffer for the phase 3 so that we dont get the catastrophic forgetting, and for phase 2 just percentage of phase 1 dataset without buffer


OOD Test Scores:
EvalGen: 97.53% accuracy
ThisPersonDoesNotExist: 99% accuracy
Ai vs Human Generated Images Kaggle Competition: 80% F1 score

Overall in practical, in random hard images or in ai vs real games, the model gets the label around 80% times right
Its not trained for partial ai gen images (real image with small part ai) so it could perform worse in it.


Phase 1 Datasets:
DDA: 110k coco, + 110k ai (SD2) similar image to each coco.

Defactify (MS COCOAI): 12k real vs random 70k ai, (SD21, SDXL, SD3, DALLE3, and MidjourneyV6)

genimage_tiny: 40k overall (midjourney, biggan, vqdm, sdv5, wukong, adm, glide)

art_artai: real art vs ai art, 1k overall

Midjourney_small: Midjourney with a tiny imagnet, 1k overall


VALIDATION Dataset:
Defactify (MSCOCOAI), 3k real vs 10k ai. (f1 score)

TEST Dataset:
EVALGen, made by SOTA gen models (FLUX, GoT, Infinity, OmniGen, Nova), 11k each, 55k overall.

Phase 2 Datasets (mostly for deepfake):
train:
AI vs. Human === 85k images
gravex === 200k images
df40 === 30k
style == 13k
hass == 10k
monk == 1k

val:
df40 == 3k
defactify_val == 13k

test:
prsn_dexist == 7k
evalgen == 55k

Phase 3 Datasets:
train:
SUPER_GENAI == 10k
JULIEN_TRAIN == 10k

val:
JULIEN_val == 3k
defactify_val == 13k




