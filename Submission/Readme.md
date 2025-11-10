IIIT Bangalore Quantum Machine Learning Project – README
1. Project Overview

This project implements and compares three machine learning approaches for MNIST digit classification:

Classical CNN (Convolutional Neural Network)

Quantum Hybrid Model (Variational Quantum Algorithm integrated with CNN)

Classical SVM (Support Vector Machine)

The objective is to demonstrate the practical implementation of Quantum Machine Learning (QML) using PennyLane and to analyze how quantum-classical hybrid models perform against traditional classical techniques.

2. Objectives

Implement a state-of-the-art classical deep learning model for MNIST classification.

Build a hybrid quantum-classical neural network using variational quantum circuits.

Compare model performance, training efficiency, and limitations across classical and quantum approaches.

Explore the feasibility and scalability of quantum machine learning using current simulation technology.

3. Architecture Summary
Model 1: Classical CNN

Structure: 5 convolutional blocks with filter sizes increasing from 16 → 256.

Activation Functions: LeakyReLU, ELU, Swish, GELU.

Regularization: Dropout (0.4, 0.3) with Early Stopping.

Parameters: 500,490 (~1.91 MB).

Accuracy: 99.12%.

Model 2: Quantum Hybrid VQA

Classical Preprocessing: CNN-based feature extraction (Conv2D + flattening).

Quantum Circuit: 6 qubits, 3 variational layers.

Encoding: Amplitude embedding (64-dimensional → 2⁶ Hilbert space).

Framework: PyTorch + PennyLane.

Accuracy: 97.12%.

Model 3: SVM

Kernel: RBF with C = 10.

Feature Space: 784-dimensional flattened MNIST images.

Results: Accuracy not captured in output due to notebook display limitations.

4. Requirements
Software Dependencies
tensorflow>=2.19.0
pennylane==0.43.1
pennylane-lightning==0.43.0
torch
torchvision
scikit-learn
numpy
matplotlib
seaborn
tqdm

Hardware Requirements

Runtime: Google Colab (CPU/GPU).

RAM: 12 GB recommended.

Storage: 500 MB for MNIST dataset.

5. Quick Start
Option 1: Run All Cells

Open the notebook in Google Colab.

Navigate to Runtime → Restart and run all.

Wait approximately 45 minutes for full execution.

View final results and visualizations.

Option 2: Run Individual Models
# For CNN
Run cells [1]–[16]

# For Quantum Hybrid
Run cells [17]–[20]

# For SVM
Run cells in SVM section

6. Results Summary
Model	Test Accuracy	Training Time	Parameters
Classical CNN	99.12%	~13 min	500,490
Quantum Hybrid	97.12%	~7 min	~60K quantum
SVM	N/A	~21 min	N/A
Key Findings

Classical CNN achieved the highest accuracy (99.12%).

Quantum Hybrid demonstrated competitive 97% accuracy despite fewer parameters.

Early stopping improved both CNN and Quantum models.

Quantum model did not outperform classical CNN on MNIST due to data structure limitations.

7. Quantum Implementation Details
Quantum Circuit Specifications
N_QUBITS = 6
N_LAYERS = 3
ENCODING = "Amplitude Embedding"
DEVICE = "default.qubit"  # PennyLane simulator
VARIATIONAL_FORM = "Strongly Entangling Layers"

Circuit Architecture

Input: 64 classical features (from CNN).

Encoding: Amplitude embedding into a 6-qubit quantum state.

Parameterized Gates: 3 layers of rotation and entanglement.

Measurement: Pauli-Z expectation values (6 outputs).

Output: Linear classifier mapping 6 → 10 classes.

8. Visualizations Generated
CNN Model

Confusion Matrix (10×10).

Training & Validation Accuracy curves.

Training & Validation Loss curves.

Quantum Hybrid Model

Confusion Matrix (10×10).

Training & Validation Loss curves.

Validation Accuracy progression.

9. Notebook Structure
├── IIIT BANGALORE PLANK'D QUANTUM MACHINE LEARNING
│
├── CNN MODEL
│   ├── [1] Install TensorFlow
│   ├── [2] Import Libraries
│   ├── [3–9] Load Data & Train Model
│   ├── [10–16] Evaluate and Visualize Results
│
├── QUANTUM HYBRID MODEL
│   ├── [17] Install PennyLane
│   ├── [18] Build & Train Quantum Model
│   ├── [19–22] Visualization & Metrics
│
└── SVM
    ├── Data Loading & Preprocessing
    ├── SVM Training with RBF Kernel
    └── Evaluation (outputs not captured)

10. Key Features
Innovation Highlights

Multiple activation functions for diverse layer performance.

Quantum amplitude encoding for efficient 64→6 qubit transformation.

Hybrid CNN–Quantum architecture integrating classical and quantum layers.

Comprehensive evaluation using precision, recall, F1-score, and confusion matrices.

Reproducibility

Fixed random seeds (torch.manual_seed(42), np.random_seed(42)).

Defined hyperparameters and model checkpoints (best_vqa_mnist.pth).

Automatic dependency installation.

11. Known Limitations
Quantum Model

No quantum advantage observed on MNIST.

Limited by 6-qubit simulation capacity.

Shallow circuits (3 layers) reduce model expressivity.

Simulation on classical hardware causes execution delays.

Technical Issues

Missing SVM output due to Colab rendering.

Inconsistent test set sizes (CNN: 10K, Quantum: 800).

High computational demand during simulation.

12. Usage Examples
Modifying CNN Architecture
# Modify architecture in cell [6]
model.add(Conv2D(filters, kernel_size, ...))

# Adjust hyperparameters in cell [8]
epochs = 30
batch_size = 64

Configuring Quantum Circuit
# Modify in cell [18]
N_QUBITS = 8      # Increases quantum capacity
N_LAYERS = 5      # Deeper circuit, higher expressivity
TRAIN_SIZE = 20000

13. Evaluation Metrics

All models were evaluated on:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

14. Contributing

This project was developed as part of the IIIT Bangalore Quantum Machine Learning Track under the Planck’d 2025 Hackathon initiative.

Institution: IIIT Bangalore

Platform: Google Colaboratory

Date: November 2025

15. References & Resources

Frameworks: TensorFlow, Keras, PyTorch, PennyLane.
Key Papers: Variational Quantum Algorithms, Quantum Machine Learning, Hybrid Quantum-Classical Neural Networks.

16. Troubleshooting
Issue	Solution
PennyLane compatibility warning	Downgrade JAX to 0.6.2 or ignore
CUDA out of memory	Reduce batch size or use CPU
SVM output not visible	Re-run last Colab cell
Quantum training slow	Reduce qubits, training size, or epochs
17. Performance Benchmarks

Google Colab CPU Runtime:

TensorFlow installation: ~5s

PennyLane installation: ~12s

CNN Training: ~13 min (7 epochs)

Quantum Hybrid Training: ~7 min (25 epochs)

SVM Training: ~21 min

Total Runtime: ~45 minutes

18. Learning Outcomes

By completing this project, participants will:

Understand how to implement production-grade CNNs using TensorFlow.

Learn quantum circuit design and hybrid model integration using PennyLane.

Recognize when quantum ML can provide advantages over classical approaches.

Understand the current limitations of quantum machine learning technology.


19. Support

For help with:

Colab environment: Refer to Google Colab documentation.

Quantum modules: Visit PennyLane official documentation.

Model architectures: Review code comments and markdown instructions.

20. Highlights

Best Performing Model: Classical CNN (99.12% Accuracy).

Most Innovative Model: Quantum Hybrid VQA (97.12% Accuracy, 6 Qubits).

Training Efficiency: Quantum model trained faster with less data.

Deployment Ready: CNN model suitable for real-world deployment scenarios.
