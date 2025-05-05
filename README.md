# Cognitive Modeling for Gaze Control in Visual Reasoning

This project implements a cognitive model that simulates human-like gaze control in visual reasoning tasks, specifically focused on odd-one-out detection. The model combines a limited field-of-view with reinforcement learning to learn efficient visual scanning strategies.

## Project Overview

The model mimics human visual attention by using:
- A limited field-of-view that can only see portions of an image at once
- A memory module (GRU) that integrates information over time
- A reinforcement learning agent that learns where to look and when to decide
- An image reconstruction component for validation and visualization

## Repository Structure

- **Model.py**: Neural network architecture (CNN feature extractor, GRU memory, actor-critic RL components)
- **train.py**: Training pipeline including dataset loading, episode generation, and RL optimization
- **Data/**: Contains the CVR dataset with odd-one-out visual reasoning tasks

## Installation

```bash
# Clone this repository
git clone <repository-url>
cd CognitiveModeling

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

You can modify the configuration parameters in `train.py` to adjust:
- Field-of-view size
- Learning rate and optimization parameters
- Reward structure
- Network architecture dimensions

## Implementation Details

The model operates by:

1. Looking at a limited portion of the image (field-of-view)
2. Extracting features from this view using a CNN
3. Integrating observations into memory using a GRU
4. Deciding whether to move the gaze or make a classification
5. Using reinforcement learning (actor-critic) to optimize this process

The model is trained to identify the odd-one-out image by moving its gaze across the image in a strategic manner, similar to how humans scan images when solving reasoning problems.

## Future Work

- Implementing finer-grained movement and smaller FOV sizes
- Adding different constraints to model performance
- Comparing model gaze patterns with human data
- Exploring different network architectures and memory mechanisms

## License

This project is part of a bachelor's thesis in Computational Neuroscience.