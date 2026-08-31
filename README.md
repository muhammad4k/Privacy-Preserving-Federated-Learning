# A Modular and Reproducible Framework for Privacy-Preserving Federated Learning

**Source code and reproducibility materials for an MSc Artificial Intelligence dissertation investigating the privacy–utility trade-off in federated learning using differential privacy.**

---

## Overview

Federated Learning (FL) enables multiple clients to collaboratively train a shared machine-learning model without directly centralising their local training data. However, keeping data local does not by itself provide a formal privacy guarantee, since model updates may still reveal information about individual training examples.

This project investigates **Differential Privacy (DP)** as a privacy-preserving mechanism within federated learning and evaluates how different privacy settings affect model utility.

The framework provides a controlled and reproducible environment for studying the interaction between:

- Federated learning
- Differential privacy
- Gradient clipping
- Privacy budgets
- Client data heterogeneity
- Dataset characteristics
- Model utility

The project uses **FedAvg with sample-weighted aggregation** and evaluates both IID and non-IID client distributions.

> **Scope:** This repository implements a controlled federated-learning simulation for research and evaluation. It is not intended to represent a production-scale federated-learning deployment.

---

## Experimental Questions

The experimental evaluation focuses on four main questions:

1. **What utility is obtained from federated learning without differential privacy?**
2. **How does model utility change as the privacy budget becomes more restrictive?**
3. **How does gradient clipping independently affect model performance?**
4. **How do client heterogeneity and dataset characteristics influence the privacy–utility trade-off?**

---

## Experimental Setup

The principal experiments use:

| Parameter | Configuration |
|---|---|
| Number of clients | 5 |
| Communication rounds | 15 |
| Local epochs | 1 |
| Batch size | 64 |
| Optimiser | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Random seed | 42 |
| Aggregation | FedAvg |
| Aggregation weighting | Number of local training examples |

### Differential Privacy Configuration

| Parameter | Configuration |
|---|---|
| Privacy mechanism | Differentially private local training |
| Library | Opacus |
| Target ε | 1, 3, 10 |
| δ | `1e-5` |
| Main clipping norms | 1.0 and 5.0 |
| Privacy accounting | Opacus |
| Noise calibration | Target ε over the configured training horizon |

The optimiser's momentum state is reset between federated communication rounds for the privacy-preserving and clipping-only training configurations.

---

## Datasets

The experiments evaluate two image-classification datasets.

### MNIST

The standard MNIST handwritten-digit classification dataset is used as the primary benchmark for evaluating privacy–utility behaviour.

### Fashion-MNIST

Fashion-MNIST provides a second grayscale image-classification task with greater visual complexity, allowing the observed behaviour to be examined beyond handwritten digits.

The datasets are downloaded automatically when required and are not stored in the repository.

---

## Client Data Distributions

Two client-distribution settings are evaluated.

### IID

Training examples are distributed across clients using the project's IID partitioning configuration.

### Non-IID

The non-IID experiments use a **label-sorted shard strategy**, with two shards assigned to each client. This produces substantial differences in the class distributions observed by individual clients.

The non-IID experiments are intended to examine the additional effect of client heterogeneity rather than treating all performance degradation as a consequence of differential privacy.

---

## Model Architecture

The experiments use a compact convolutional neural network implemented in:

```text
models/cnn.py
```

The architecture is:

```text
Input
  │
  ├── Conv2D: 32 filters, 3×3
  ├── ReLU
  ├── MaxPool: 2×2
  │
  ├── Conv2D: 64 filters, 3×3
  ├── ReLU
  ├── MaxPool: 2×2
  │
  ├── Adaptive Average Pooling: 1×1
  ├── Flatten
  │
  ├── Linear: 64 → 128
  ├── ReLU
  ├── Dropout: 0.25
  │
  └── Linear: 128 → 10 classes
```

The architecture is deliberately compact so that repeated federated experiments can be performed efficiently while maintaining sufficient capacity for the selected benchmark datasets.

The same architecture is maintained across the principal experimental comparisons to reduce unnecessary sources of variation.

---

## Key Results

The principal results are consolidated in:

```text
results/summary/master_results.csv
```

Selected final accuracies are shown below.

| Experiment | Final Accuracy |
|---|---:|
| MNIST IID — Non-private | **94.31%** |
| MNIST IID — DP, C=5, ε=10 | **75.68%** |
| MNIST IID — DP, C=5, ε=3 | **75.53%** |
| MNIST IID — DP, C=5, ε=1 | **70.17%** |
| MNIST non-IID — Non-private | **47.91%** |
| MNIST non-IID — DP, C=5, ε=3 | **36.76%** |
| MNIST non-IID — DP, C=5, ε=1 | **34.54%** |
| Fashion-MNIST IID — Non-private | **78.22%** |
| Fashion-MNIST IID — DP, C=5, ε=3 | **68.35%** |
| Fashion-MNIST IID — DP, C=5, ε=1 | **65.43%** |

### Realised Privacy Budgets

The principal DP experiments produce approximately:

| Target ε | Realised ε |
|---:|---:|
| 10 | **9.9975** |
| 3 | **2.9991** |
| 1 | **0.9910** |

Small differences between the target and realised values result from the privacy-accounting and noise-calibration process.

---

## Clipping Sensitivity Ablation

A separate clipping-only experiment evaluates:

```text
C = 0.5
C = 1.0
C = 2.0
C = 5.0
```

This experiment uses **gradient clipping without Gaussian noise**.

The purpose is to isolate the effect of clipping from the effect of differential-privacy noise.

> **Important:** The clipping-only experiment does **not** provide a differential-privacy guarantee because Gaussian noise is not applied. It should therefore be interpreted strictly as an ablation experiment.

---

## Privacy Interpretation

Federated learning and differential privacy address different aspects of privacy.

Federated learning keeps the original training data on individual clients rather than sending the raw data to a central server. However, model updates can still potentially contain information about individual examples.

Differential privacy addresses this by limiting the influence of individual training examples through gradient clipping and calibrated Gaussian noise.

The DP experiments in this repository therefore provide **formal `(ε, δ)` privacy accounting for the specified training configuration**.

They should **not** be interpreted as demonstrating immunity against every possible privacy attack.

This project does not include a dedicated empirical membership-inference, reconstruction, or model-inversion attack evaluation.

---

## Repository Structure

```text
Privacy-Preserving-Federated-Learning/
│
├── configs/
│   └── YAML experiment configurations
│
├── datasets/
│   └── Dataset loading and partitioning
│
├── models/
│   └── CNN model definitions
│
├── training/
│   └── Local training and evaluation
│
├── federated/
│   └── Federated clients, server, simulation, and FedAvg
│
├── privacy/
│   └── Differential-privacy configuration and accounting
│
├── experiments/
│   └── Experiment runners and result consolidation
│
├── evaluation/
│   └── Metrics and result writers
│
├── visualisation/
│   └── Plot and figure generation
│
├── utils/
│   └── Configuration and reproducibility utilities
│
├── results/
│   ├── summary/
│   │   └── Consolidated experiment results
│   └── processed/
│       └── Processed analysis outputs
│
├── figures/
│   └── Generated figures
│
├── tests/
│   └── Automated tests
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/muhammad4k/Privacy-Preserving-Federated-Learning.git
cd Privacy-Preserving-Federated-Learning
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The principal dependencies include:

- PyTorch
- torchvision
- Opacus
- NumPy
- Flower
- pandas
- matplotlib
- pytest
- PyYAML

Exact versions are specified in `requirements.txt`.

---

## Running the Experiments

The project uses YAML configuration files to define experimental parameters.

The general experiment runner can be invoked with:

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_15rounds.yaml')"
```

> **Note:** `configs/default.yaml` is a development configuration and should not be used to reproduce the principal dissertation results.

### MNIST IID Baseline

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_15rounds.yaml')"
```

### MNIST IID — Differential Privacy, C=5

#### ε = 10

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps10.yaml')"
```

#### ε = 3

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps3.yaml')"
```

#### ε = 1

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps1.yaml')"
```

### MNIST IID — Differential Privacy, C=1

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps10.yaml')"

python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps3.yaml')"

python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps1.yaml')"
```

### MNIST Non-IID

#### Non-private baseline

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_non_iid_15rounds.yaml')"
```

#### DP, ε = 3

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_mnist_non_iid_c5_eps3.yaml')"
```

#### DP, ε = 1

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_mnist_non_iid_c5_eps1.yaml')"
```

### Fashion-MNIST

#### Non-private baseline

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_fashion_mnist_15rounds.yaml')"
```

#### DP, ε = 3

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_fashion_mnist_c5_eps3.yaml')"
```

#### DP, ε = 1

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_fashion_mnist_c5_eps1.yaml')"
```

### Clipping-Only Ablation

```bash
python -m experiments.run_clipping_sensitivity
```

This evaluates clipping thresholds of:

```text
0.5
1.0
2.0
5.0
```

without Gaussian noise.

---

## Testing

Run the automated test suite with:

```bash
pytest -q
```

The tests cover core components such as:

- Federated aggregation
- Dataset partitioning
- Local training
- Privacy configuration
- Federated simulation
- Evaluation metrics
- Reproducibility utilities

Successful tests validate implementation behaviour but do not replace reproduction of the complete dissertation experiments.

---

## Reproducibility

The principal experiments use:

```text
Random seed:           42
Clients:               5
Rounds:                15
Local epochs:          1
Batch size:            64
Learning rate:         0.01
Momentum:              0.9
Delta:                 1e-5
```

Experiment parameters are stored in YAML configuration files under:

```text
configs/
```

The consolidated results used for the dissertation are retained in:

```text
results/summary/master_results.csv
```

This allows the reported numerical findings to be inspected without rerunning the complete experiment suite.

---

## Principal Experiment Matrix

| Dataset | Distribution | Privacy | C | ε |
|---|---|---|---:|---:|
| MNIST | IID | None | — | — |
| MNIST | IID | DP | 1 | 10 |
| MNIST | IID | DP | 1 | 3 |
| MNIST | IID | DP | 1 | 1 |
| MNIST | IID | DP | 5 | 10 |
| MNIST | IID | DP | 5 | 3 |
| MNIST | IID | DP | 5 | 1 |
| MNIST | Non-IID | None | — | — |
| MNIST | Non-IID | DP | 5 | 3 |
| MNIST | Non-IID | DP | 5 | 1 |
| Fashion-MNIST | IID | None | — | — |
| Fashion-MNIST | IID | DP | 5 | 3 |
| Fashion-MNIST | IID | DP | 5 | 1 |
| MNIST | IID | Clipping-only | 0.5–5 | — |

---

## Limitations

The results should be interpreted within the controlled experimental setting used by this project.

Important limitations include:

- Five simulated clients
- Fifteen communication rounds
- One local epoch per round
- One principal CNN architecture
- MNIST and Fashion-MNIST
- One principal random seed
- FedAvg aggregation
- A specific label-shard non-IID construction
- Limited hyperparameter exploration
- No dedicated empirical privacy-attack evaluation
- No production or cross-device federated deployment

The reported results are therefore **point estimates under the specified experimental configuration** rather than estimates of variability across multiple independent random seeds.

---

## Research Context

This repository accompanies the MSc Artificial Intelligence dissertation **“A Modular and Reproducible Framework for Privacy-Preserving Federated Learning.”**

The project is structured around reproducible experimentation rather than proposing a new differential-privacy mechanism or a production federated-learning platform.

The central experimental focus is how privacy–utility behaviour changes with:

- Privacy budget
- Gradient clipping
- Client data heterogeneity
- Dataset difficulty
- Federated optimisation behaviour

---

## Author

**Muhammad Khalid**

GitHub: [@muhammad4k](https://github.com/muhammad4k)

Repository: [Privacy-Preserving-Federated-Learning](https://github.com/muhammad4k/Privacy-Preserving-Federated-Learning)

---

## Acknowledgements

This project makes use of open-source machine-learning and privacy-preserving technologies, including:

- PyTorch
- Torchvision
- Opacus
- Flower
- NumPy
- Pandas
- Matplotlib

Please refer to the respective projects and their documentation for licensing and implementation details.
