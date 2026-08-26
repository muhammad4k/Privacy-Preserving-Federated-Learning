# Privacy-Preserving Federated Learning Framework

Source code and reproducibility materials for an MSc Artificial Intelligence dissertation investigating the privacy–utility trade-off in federated learning using differential privacy.

## Overview

The framework simulates federated learning across five clients and evaluates how differential privacy affects predictive utility under different privacy budgets, clipping thresholds, data distributions, and datasets. The implementation supports:

- FedAvg with sample-weighted aggregation
- IID and label-shard non-IID client partitioning
- MNIST and Fashion-MNIST
- Differentially private local training with Opacus
- Target privacy budgets (epsilon) calibrated across the complete federated training horizon
- Clipping-only ablation experiments
- Accuracy, loss, macro-F1, realised epsilon, and noise-multiplier recording
- YAML-driven experiments and deterministic random seeds

## Environment

The experiments were developed with Python and PyTorch. A fresh virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The principal pinned dependencies are PyTorch 2.2.2, torchvision 0.17.2, Opacus 1.4.0, NumPy 1.26.4, and Flower 1.23.0. Dataset files are downloaded automatically on first use and are not stored in the repository.

## Running an experiment

The configuration-driven runner accepts a YAML configuration path:

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_15rounds.yaml')"
```

Do not use `configs/default.yaml` to reproduce the dissertation results: it is a short three-round development configuration.

### Principal dissertation experiments

MNIST IID baseline:

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_15rounds.yaml')"
```

MNIST IID differential privacy, relaxed clipping (`C=5`):

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps10.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps3.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c5_0_eps1.yaml')"
```

MNIST IID differential privacy, restrictive clipping (`C=1`):

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps10.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps3.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/final_dp_mnist_c1_0_eps1.yaml')"
```

MNIST non-IID experiments:

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_mnist_non_iid_15rounds.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_mnist_non_iid_c5_eps3.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_mnist_non_iid_c5_eps1.yaml')"
```

Fashion-MNIST experiments:

```bash
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/baseline_fashion_mnist_15rounds.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_fashion_mnist_c5_eps3.yaml')"
python -c "from experiments.run_experiment import run_experiment; run_experiment('configs/dp_fashion_mnist_c5_eps1.yaml')"
```

Clipping-only ablation:

```bash
python -m experiments.run_clipping_sensitivity
```

This ablation applies gradient clipping without Gaussian noise and therefore is **not** a differentially private experiment. It is used to isolate the utility effect of clipping.

## Reproducing the reported results

The main experiment settings are:

| Setting | Value |
|---|---|
| Clients | 5 |
| Communication rounds | 15 |
| Local epochs | 1 |
| Batch size | 64 |
| Optimiser | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Random seed | 42 |
| Delta | 1e-5 |
| Main target epsilons | 10, 3, 1 |
| Main clipping norms | 1.0, 5.0 |

The consolidated final outcomes used for analysis are provided in `results/summary/master_results.csv`. Selected final accuracies include:

| Experiment | Final accuracy |
|---|---:|
| MNIST IID baseline | 94.31% |
| MNIST IID DP, C=5, epsilon=10 | 75.68% |
| MNIST IID DP, C=5, epsilon=3 | 75.53% |
| MNIST IID DP, C=5, epsilon=1 | 70.17% |
| MNIST non-IID baseline | 47.91% |
| MNIST non-IID DP, C=5, epsilon=3 | 36.76% |
| MNIST non-IID DP, C=5, epsilon=1 | 34.54% |
| Fashion-MNIST IID baseline | 78.22% |
| Fashion-MNIST IID DP, C=5, epsilon=3 | 68.35% |
| Fashion-MNIST IID DP, C=5, epsilon=1 | 65.43% |

For the DP runs, the stored realised privacy expenditure is approximately 9.9975, 2.9991, and 0.9910 for target epsilon values 10, 3, and 1 respectively.

## Repository structure

```text
configs/          YAML experiment configurations
datasets/         Dataset loading and IID/non-IID partitioning
models/           CNN model definition
training/         Local training and evaluation
federated/        Client, server, simulator, and FedAvg aggregation
privacy/          Opacus differential-privacy configuration
experiments/      Experiment runners and result consolidation
evaluation/       Metrics and result writers
visualisation/    Plot generation
results/summary/  Consolidated dissertation result table
results/processed/ Processed analysis outputs
figures/          Generated dissertation figures
tests/            Automated tests
utils/            Configuration and reproducibility utilities
```

## Output files

Experiment runs can generate CSV and JSON records and accuracy/loss/macro-F1 figures. Raw run outputs are intentionally excluded from the repository to keep it compact; the consolidated final result table is retained under `results/summary/`.

## Notes on reproducibility

- All principal experiments use random seed 42.
- DP noise is calibrated with Opacus for the configured target epsilon and delta across the complete local-training horizon.
- In federated DP runs, the privacy engine/accountant is preserved across communication rounds for each client.
- The non-IID setting uses label-sorted shards with two shards assigned to each client.
- FedAvg weights client model parameters by the number of local training examples.

## Tests

After installing the dependencies, run:

```bash
pytest -q
```

The test suite covers core components including aggregation, partitioning, privacy configuration, simulation, training, metrics, and reproducibility utilities.

## Dissertation

This repository accompanies the MSc Artificial Intelligence dissertation on privacy-preserving machine learning in distributed/federated learning systems. The dissertation contains the full methodology, experimental design, results, analysis, limitations, and discussion.
