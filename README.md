# Cyclic Continuous Learning Framework

## Course Project: Mathematics of Deep Learning
**Columbia University - EECS 6694**  
**Spring 2025**

## Project Overview
This repository implements a cyclic continuous learning framework with a focus on Class Incremental Learning (CIL). The framework is evaluated on the MNIST-1D dataset.

## Dataset
The project utilizes the MNIST-1D dataset, a simplified version of the classic MNIST dataset that maps the 2D images to 1D sequences while preserving their key spatial characteristics. For more information about the dataset, refer to the [MNIST-1D paper](https://arxiv.org/abs/2011.14439).

## Features
- Implementation of a cyclic continuous learning framework
- Class Incremental Learning (CIL) methodology
- Performance evaluation on MNIST-1D dataset
- Analysis of learning patterns and model behavior

## Project Structure
```
.
├── model_weights
│   └── ten_class_best_cyclic_weights.pt
├── Plotting
│   ├── loss_landscape.pdf
│   ├── PlottingForgetting.ipynb
│   ├── Plotting.ipynb
│   ├── ToyExample.ipynb
│   ├── ZoomedInPlot.ipynb
│   └── zoomed_in_plot.pdf
├── __pycache__
├── README.md
├── Results ( Classwise training accuracies for various configurations )
│   ├── 1024
│   ├── 128
│   ├── 256
│   ├── 32
│   ├── 512
│   └── 64
├── training_scripts
│   ├── baseline.py
│   ├── class_incremental_model.py
│   ├── experiments.py
│   ├── training_loop.py
│   └── trainin_loop_baseline.py
└── verify_performance.ipynb
```