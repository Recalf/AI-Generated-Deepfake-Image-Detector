# AI-Generated-Deepfake-Image-Detector
Detects if an image is Real or AI-Generated / DeepFake, using ConvNeXt (pretrained on ImageNet) on 11 different datasets (700k images, with data augmentations and multiple epochs) 
It detects Deepfake AI faces images too, and many new and SOTA generative models like Flux, Diffusion Models (SDXL, SD3...), MidjourneyV6, DALL‑E3, Nano Banana Pro...



I fine tuned it in training phase 2 on deepfake ai faces.
And fine tuned it in training phase 3 on latest Geneative AI images.

I used techniques like: 
Differential Learning Rates, Sequential LR Scheduler with 1 epoch LinearLR warmup and rest with CosineAnnealing Scheduler.
I used 

OOD Test Scores:
EvalGen: 97.53% accuracy
ThisPersonDoesNotExist: 99% accuracy
Ai vs Human Generated Images Kaggle Competition: 80% F1 score

Overall in practical, in random hard images or in ai vs real games, the model gets the label around 80% times right
Its not trained for partial ai gen images (real image with small part ai) so it could perform worse in it.
