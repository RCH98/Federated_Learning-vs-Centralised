## MLDL24 Federated Learning Project: Impact of Data and System Heterogeneity on Federated Learning

The project evaluates the impact of Federated Learning on model accuracy and loss across different epochs, providing a comprehensive analysis of its feasibility and limitations. By addressing these challenges, this project aims to contribute to the understanding and application of Federated Learning in real-world machine learning scenarios.

## Datasets

1. CIFAR-100

  * **Overview:** consists of 60,000 color images, each of size 32×32 pixels
  * **Details:** 100 fine-grained classes
  * **Task:** Computer Vision Task

2. Shakespeare

  * **Overview:** Text Dataset of Shakespeare Dialogues
  * **Details:** 1129 users (reduced to 660 with our choice of sequence length.
  * **Task:** Next-Character Prediction

## Parameters:

- `K` number of clients
- `C` portion of clients selected in each round
- `B` local batch size
- `N` total number of communication rounds
- `J` number of local epochs
- `lr` learning rate
- `test_freq` testing frequency
- `participation` mode of client participation: `uniform` or `skewed`
- `gamma` scale of skewness in the case of skewed participation, otherwise ignored
- `d_value` parameter for dynamic pow-d client selector

Additional parameters concerning data splitting for CIFAR-100 are stored in `data_split_params`:
- `shard_type` method of splitting data: `iid` or `non-iid`
- `n_labels` number of labels per client in the case of non-IID data distribution

## Structure of the Project

- **`requirements.txt`**: A list of required libraries for setting up the environment.
- **`src/`**: Contains various scripts for different datasets and tasks.
  - `cifar/`: Scripts and resources specific to the CIFAR dataset applying fedavg.
    - `CIFAR_100_CENTRALIZED__TRAINING.ipynb`: Centralized baseline
    - `Client_participation.ipynb`: Definition of client participation
    - `data_sharding.ipynb`: Data sharding process
    - `Federated_baseline.ipynb`: First federated baseline
    - `Skewed_Uniform....ipynb`: Federated training on uniform and skewed client participation
    - `FEDERATED_HETERO....ipynb`: Hyperparameter tuning for federated training
    - `pow_d.py` - Dynamic PoW-d client selection 
  - `shakespeare/`: Scripts and resources related to the Shakespeare dataset applying fedavg.
    - `centralized_final.py`: Centralized baseline
    - `data_distribution_shakespeare.ipynb`: Data sharding process
    - `first_baseline.ipynb`: First federated baseline
    - `skewed_shakespeare.py`: Federated training on uniform and skewed client participation
    - `shakespeare_heteregeneous_federated.py`: Hyperparameter tuning for federated training
    - `pow_d_shakespeare.py`: Dynamic Pow-d client selection
- **`LICENSE`**: Licensing information for the project.
- **`README.md`**: Documentation for the project.