# Dimensionality Reduction and Denoising of Spatial Transcriptomics Data Using Dual-channel Masked Graph Auto-encoder
## Introduction
In this paper, we propose an efficient self-supervised learning method utilizing a **dual-channel masked graph autoencoder**  for the analysis and research of spatial transcriptomics data (namely **STMask**). The gene representation learning channel cleverly integrates the masking mechanism into the graph autoencoder, reducing feature redundancy and enhancing model robustness. Additionally, we employ the scaled cosine loss as the training objective to further stabilize the embedding representation. Simultaneously, the gene relationship learning channel disrupts the spatial graph structure to generate multiple perspectives and combines contrastive learning principles to maximize mutual information, thereby strengthening the model's discriminative capability. It is worth noting that the two channels share the same encoder, which helps to enhance the model's ability to learn gene expressions and facilitates the acquisition of discriminative feature embeddings. 

![STMask.jpg](STMask.jpg)

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 3090)
-   `pip install -r requiremnts.txt`

## Get Started
- For a tutorial on recognition tasks in the spatial domain, please see `Identification_Domains.ipynb`
- For a tutorial on detecting spatially variable genes, please see `SVGs_Detection.ipynb`
- For a tutorial on differential gene detection and GOBP analysis, please see `GO_DEG.ipynb`
