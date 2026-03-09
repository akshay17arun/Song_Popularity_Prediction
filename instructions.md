## Project Overview

You will complete one semester-long project (vision, audio, or multi-modal). You will be evaluated on reasoning, evaluation, and insight.

This project has four phases:

* Check-in 1 — Problem framing (early): choose a problem, gather/prepare data, and do exploratory analysis so you understand what you’re working with.
* Check-in 2 — Fundamentals (mid-semester): build solid baselines using early-course methods (classical CV, feature engineering, standard CNNs, segmentation, detection, etc.).
* Check-in 3 — Advanced extension (late semester): add one advanced topic we cover later (tracking, generative models, ViTs/transformers, multi-modal, etc.).
Final deliverables (end of semester): polish the story and deliverables (final report + presentation).
You do not need to do every topic in the course. You will build one coherent project and extend it once.

## Example project ideas

### Vision
* Object detection: find traffic signs, pests, defects, etc. in images.
    * Fundamentals milestone: build a baseline detector and analyze failures (small objects, blur, low light).
    * Advanced extension ideas: add tracking, try a transformer-based model, or use generative augmentation.
* Segmentation: outline roads/buildings (satellite) or lesions/polyps (medical).
    * Fundamentals milestone: train a classic segmentation model like U-Net and analyze failures (thin structures, rare cases).
    * Advanced extension ideas: add tracking (video), try a transformer-based model, or use generative augmentation.
### Audio
* Detect events over time (e.g., siren/dog bark) using spectrograms.
    * Fundamentals milestone: spectrogram + CNN baseline + failure cases.
    * Advanced extension ideas: try a transformer, test robustness to noise, or use a generative/noise model.
## Multi-modal
* Match images and text: “given a caption, retrieve the right image (and vice versa).”
    * Fundamentals milestone (mid-semester): build the dataset + evaluation pipeline and a simple baseline that does not require advanced models. Examples:
        * Text features: TF‑IDF / bag-of-words over captions.
        * Image features: simple classical features (color histograms, edges) or off-the-shelf CNN features from a pre-trained model (no training required).
        * Then do nearest-neighbor retrieval + show successes/failures.
    * Advanced extension ideas: use a transformer-based vision backbone (e.g., ViT) and/or a vision–language model for embeddings; analyze sensitivity to wording/prompt choices.
### Generative
* Restore corrupted data: denoise images, fill missing regions (inpainting), or increase resolution.
    * Fundamentals milestone: do a classical baseline + CNN baseline; show before/after and one simple quantitative measure.
    * Advanced extension ideas: use a diffusion/GAN model, or use generation as augmentation and test downstream impact.

## Individual project

This is an individual project. You are free to collaborate with other students in the class, but each student will submit their own unique project.

## What you submit (each checkpoint)

* Repo containing the code for the project and dataset or instructions on how to access the dataset. The repo should contain the entire project from the beginning to the end.
* README.md file that documents the project including the current check-in or final report depending on the phase of the project. You could also include separate .md files for each phase (i.e. check-in-1.md, check-in-2.md, check-in-3.md, final.md, demo.md) so long as the README.md clearly directs the reader to the appropriate file. It should be clear and obvious for visitors to your project to navigate the project and find the appropriate files.
* Demo that demonstrates the progress of the project up to the check-in or final product. This could be a rendered Jupyter Notebook, Shiny App, website, or other type of demo depending on the project.

