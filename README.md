# DeepProfile Replication: Unsupervised Deep Learning for Genomic Analysis

## Overview
This repository contains a TensorFlow 2 implementation of the **DeepProfile** framework, originally proposed in the study *"A deep profile of gene expression across 18 human cancers"* (Nature Biomedical Engineering).

The primary objective of this project is to replicate the core dimensionality reduction methodology described by the authors. By implementing a **Variational Autoencoder (VAE)**, we aim to compress high-dimensional gene expression data (RNA-Seq) into a biologically meaningful, low-dimensional latent space. This approach overcomes the limitations of linear methods like PCA by capturing non-linear interactions between genes.

## Repository Structure

The project is structured to ensure modularity and reproducibility:

```text
.
├── data/                   # Directory for input datasets (not included in repo)
├── results/                # Directory for generated embeddings and visualization plots
├── src/
│   ├── vae_model.py        # VAE architecture definition (Encoder/Decoder class)
│   ├── train.py            # Training script: loads data, trains model, and saves embeddings
│   └── visualize.py        # Analysis script: performs PCA on embeddings for visualization
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

## Prerequisites

The codebase is implemented in Python 3.x. To set up the environment, install the required dependencies using the following command:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

## Data Acquisition

Due to the size of genomic datasets, the input files are not hosted in this repository. This project utilizes the pre-processed data provided by the original authors.

1. Access the official Figshare Repository: https://doi.org/10.6084/m9.figshare.25414765.v2.

2. Download the gene expression file for the desired cancer type. For example, for Ovarian Cancer (OV), download:
   - `OV_DATA_TOP2_JOINED_BATCH_CORRECTED_CLEANED.tsv`

3. Place the downloaded file directly into the `data/` directory.

## Usage

### 1. Training the VAE Model

The training script loads the gene expression data, normalizes it, and trains the Variational Autoencoder. It utilizes a KL Divergence warm-up strategy to prevent posterior collapse during the early epochs.

**Command:**
```bash
python src/train.py <CANCER_TYPE>
```

**Example:**
```bash
python src/train.py OV
```

**Expected Output:**
- Console output showing the loss progression over 50 epochs.
- Reconstruction R² Score: A metric indicating the fidelity of the reconstructed data (values closer to 1.0 indicate high fidelity).
- Output File: A tab-separated file containing the latent feature embeddings is saved to `results/OV_embeddings.tsv`.

### 2. Visualization of the Latent Space

To validate the biological relevance of the learned features, the `visualize.py` script applies Principal Component Analysis (PCA) to the 50-dimensional latent embeddings and generates a 2D scatter plot.

**Command:**
```bash
python src/visualize.py <CANCER_TYPE>
```

**Example:**
```bash
python src/visualize.py OV
```

**Output:**
- A visualization plot saved as `results/OV_latent_space.png`.

## Methodology

### Variational Autoencoder (VAE)

The model consists of two main components:

- **Encoder**: Maps the high-dimensional input x (thousands of genes) to a probabilistic latent distribution P(z∣x), parameterized by mean μ and variance σ.

- **Decoder**: Reconstructs the original input x′ from the sampled latent vector z.

The loss function is defined as:

$$\mathcal{L} = \mathcal{L}_{Reconstruction} + \beta \cdot D_{KL}(q(z|x) || p(z))$$

Where β is a dynamic hyperparameter that increases linearly during training (warm-up) to balance reconstruction accuracy with the regularization of the latent space.

## References

- **Original Publication**: Qiu, W., Dincer, A.B., et al. "Deep profiling of gene expression across 18 human cancers." Nature Biomedical Engineering (2024).

- **Original Codebase**: suinleelab/deepprofile-study

---

*University of Málaga - Machine Learning Course (Lab 4)*
*Author: Hugo Salas Calderón*
