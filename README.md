
---

# PyTorch CNN vs MLP Comparison (CSCI3485)

A comparative study and implementation of Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs) using PyTorch.  
This project was developed for the CSCI3485 course to illustrate key differences in architecture, performance, and application on image classification tasks.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

The purpose of this project is to compare the effectiveness of CNNs and MLPs on image classification problems.  
Both models are implemented from scratch using PyTorch and evaluated on a standard dataset (e.g., MNIST or CIFAR-10).

**Key Features:**
- Modular model definitions for CNN and MLP
- Configurable training and evaluation scripts
- Experiment logging and result visualization

---

## Project Structure

```
.
├── data/                   # Scripts or instructions for downloading datasets
├── models/                 # Model definitions for CNN and MLP
├── utils/                  # Utility scripts (e.g., data loaders, visualization)
├── main.py                 # Main training and evaluation script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Getting Started

1. **Clone the repo**
    ```bash
    git clone https://github.com/rnovitski24/PyTorch-CNN-MLP-Comparison-CSCI3485.git
    cd PyTorch-CNN-MLP-Comparison-CSCI3485
    ```

2. **(Recommended) Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Train and evaluate a model by running the main script:

```bash
python main.py --model cnn  # For CNN
python main.py --model mlp  # For MLP
```

**Available arguments:**
- `--model [cnn|mlp]` &nbsp; Model type to train/evaluate
- `--epochs <int>` &nbsp; Number of training epochs
- `--batch-size <int>` &nbsp; Batch size for training
- `--lr <float>` &nbsp; Learning rate

Example:
```bash
python main.py --model cnn --epochs 10 --batch-size 64 --lr 0.001
```

---

## Experiments & Results

| Model | Test Accuracy | Notes        |
|-------|---------------|--------------|
| CNN   | xx.xx%        | ...          |
| MLP   | xx.xx%        | ...          |

*Fill in with your results and observations.*

---

## Requirements

- Python 3.8+
- PyTorch
- numpy
- matplotlib

Install with:
```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CSCI3485 Course Materials]()

---

## License

[MIT License](LICENSE)

---

Let me know if you want help filling in specific sections, adding badges, or customizing for your exact project structure!
