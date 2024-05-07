# Hemingway AI Writer

## Overview
The Hemingway AI Writer is an innovative project developed to emulate the distinctive writing style of Ernest Hemingway using modern transformer-based models. This project leverages a GPT-like architecture, trained exclusively on the collected works of Hemingway to generate text that mirrors his concise and impactful style.

## Features
- Utilizes a custom GPT-like model architecture tailored to replicate Hemingway's writing style.
- Trained exclusively on the collected works of Hemingway to ensure stylistic accuracy.
- Easy-to-use text generation capabilities, allowing users to create novel text passages that mirror Hemingway’s style.


## Dependencies and Installation

To set up the experimental environment for this project, we have utilized `Google Colab` which provides access to a GPU. The necessary text sample for our analysis can be directly downloaded using the provided `gdown` link in the code. 

**Python Packages**
- `torch (1.8.1)`: Neural network library used for building and training the GPT-like models.
- `numpy (1.19.5)`: Provides support for efficient numerical computation.
- `nltk (3.5)`: Utilized for natural language processing tasks like text tokenization.
- `rouge (1.0.0)`: Used for evaluating text summaries.
- `scikit-learn (0.24.1)`: Employed for machine learning and statistical modeling including classification, regression, clustering, and dimensionality reduction

## Hardware Requirements
The model training and execution are optimized for CUDA-enabled GPUs. It has been specifically fine-tuned on a T4 GPU available through `Google Colab`, which significantly enhances performance due to accelerated computation capabilities. Users with access to other CUDA-compatible GPUs can also run the model with similar performance benefits.

Configure CUDA (if available): The model automatically detects and utilizes CUDA if it is available on the system. To check if CUDA is recognized by PyTorch and to confirm the GPU in use, you can add the following Python code snippet to your setup script.

```python
import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")
```
  
### Prerequisites
- Python 3.8+
- pip
- git (for cloning the repository)


