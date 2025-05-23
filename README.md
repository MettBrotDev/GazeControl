# GazeControl: Cognitive Modeling for Visual Reasoning

> A reinforcement learning approach to human-like visual attention in odd-one-out detection tasks

## 🎯 Overview

This project implements a cognitive model that simulates human-like gaze control in visual reasoning tasks. Unlike traditional computer vision systems that process entire images uniformly, our model mimics human foveated vision by sequentially attending to relevant image regions through a limited field-of-view, integrating observations over time to make efficient decisions.

## 🔬 Research Motivation

Humans solve complex visual reasoning tasks with remarkable speed and accuracy by strategically directing their attention. Our model addresses the research question: **Can artificial agents learn to sequentially "look" at parts of an image using a limited foveal window and integrate observations over time to solve visual reasoning tasks more efficiently?**

### Key Innovation
- **Foveated Vision**: Variable-focus sampling mechanism mimicking human visual attention
- **Sequential Processing**: Memory-based integration of visual information over time
- **Interpretable Decisions**: Gaze patterns reveal the model's reasoning process
- **Efficient Learning**: Reinforcement learning optimizes attention strategies

## 🏗️ Architecture

The model combines four core components:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Foveated Sampler│ -> │ Memory (GRU) │ -> │ Actor-Critic RL │
└─────────────────┘    └──────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌──────────────┐
                       │   Decoder    │
                       │ (Visualization)
                       └──────────────┘
```

### Components

1. **Foveated Sampler**
   - Extracts high-resolution features at current fixation point
   - Implements human-like limited field-of-view constraints

2. **Memory Module (GRU)**
   - Integrates sequential visual observations
   - Maintains spatial and temporal context across fixations

3. **Actor-Critic Agent**
   - **Actor**: Decides between gaze shifts and final classification
   - **Critic**: Evaluates state values for policy optimization

4. **Decoder Network**
   - Visualizes internal memory representations
   - Provides interpretability and auxiliary training signal

## 📊 Dataset

### Compositional Visual Relations (CVR)
- **103 compositional odd-one-out tasks**
- Evaluates sample efficiency and generalization
- Tests systematic visual reasoning capabilities

### Task Generation
```bash
# Generate dataset for specific task
python ./Dataset/generate_dataset.py --task_idx 0 --data_dir ./Data

# Process and label data
python ./Data/task_shape/MixNLabel.py
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- NumPy

### Installation
```bash
# Clone repository
git clone <repository-url>
cd GazeControl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Generate training data
python ./Dataset/generate_dataset.py --task_idx 0 --data_dir ./Data
python ./Data/task_shape/MixNLabel.py

# Train the model
python train.py
```

## ⚙️ Configuration

Modify training parameters in `train.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fov_size` | Field-of-view dimensions | `(64, 64)` |
| `learning_rate` | Optimizer learning rate | `1e-4` |
| `reward_correct` | Reward for correct decisions | `+1.0` |
| `step_penalty` | Penalty per time step | `-0.01` |
| `memory_dim` | GRU hidden dimensions | `512` |

## 🧠 Methodology

### Training Strategy
The model learns through a combination of:

1. **Reinforcement Learning Loss**
   - Rewards correct classifications
   - Penalizes incorrect decisions
   - Small step penalty encourages efficiency

2. **Auxiliary Decoder Loss**
   - Guides learning through reconstruction
   - Masked to visited regions only
   - Feature-level comparison for meaningful learning

### Learning Process
1. **Sample** limited field-of-view region
2. **Extract** CNN features from current glimpse
3. **Integrate** with memory using GRU
4. **Decide** between gaze shift or classification
5. **Update** policy using actor-critic RL

## 📈 Evaluation Metrics

- **Task Accuracy**: Correct odd-one-out identification
- **Sample Efficiency**: Performance vs. number of fixations
- **Gaze Patterns**: Attention trajectory analysis
- **Generalization**: Cross-task performance

## 🔍 Related Work

- **Recurrent Attention Model (RAM)** - Mnih et al., 2014
- **Active Vision RL** - Shang and Ryoo, 2023  
- **Horizontal GRU (hGRU)** - Linsley et al., 2018

## 📁 Project Structure

```
GazeControl/
├── Model.py              # Neural network architecture
├── train.py              # Training pipeline
├── Data/                 # CVR dataset
├── Dataset/              # Dataset generation code
├── WrittenStuff/         # Documentation and thesis
└── requirements.txt      # Dependencies
```

## 🔮 Future Directions

- **Enhanced Resolution**: Finer-grained movement patterns
- **Biological Constraints**: More realistic visual limitations  
- **Human Comparison**: Gaze pattern validation with eye-tracking data
- **Architecture Exploration**: Alternative memory and attention mechanisms

## 📄 Academic Context

This project is part of a Bachelor's Thesis in Cognitive Modeling, supervised by Tomáš Daniš and advised by Prof. Dr. Martin V. Butz.

---

*For detailed methodology and theoretical background, see the accompanying exposé in `WrittenStuff/GazeControlExpose.tex`*
