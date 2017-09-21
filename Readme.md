
# Readme

## Requirements:
1. python 2.7

2. tensorflow 1.1.0

3. numpy 1.12.1

4. matplotlib 2.0.2

5. pandas 0.20.2

6. scikit-learn 0.18.1

Please also install other packages if necessary (will be noticed if a package is not installed).

## Folders:

1. env: the environments used for generating states in reinforcement learning;

2. exp: the folders to maintain experiment outputs (checkpoint, summaries, etc.);

3. test: the unit tests for basic functions

## Files:

1. Anomaly Detection with Q-learning (RNN n-n, boosted binary tree train).ipynb: main file for training the model

2. Anomaly Detection with Q-learning (RNN n-n, boosted binary tree test).ipynb: main file for testing the model

These two files are roughly the same except the last section in each of them. One is for training using Yahoo benchmark dataset, and the other is for testing using Numenta datasets.

## Notes:

1. Please open "Anomaly Detection with Q-learning (RNN n-n, boosted binary tree test).ipynb" for the testing results of using the model for anomaly detection (Numenta datasets).

2. Please open "Anomaly Detection with Q-learning (RNN n-n, boosted binary tree training).ipynb" for the training of the model. Before running the training file, please check the file "time_series_repo_ext.py" in env to correct the environment to load the training datasets (Yahoo Benchmark A1-A4).

3. Practically, the training process could be done incrementally (by incrementally augmenting the datasets). But this functionality is currently not supported.

4. You could change the strings "exp_relative_dir" or "dataset_dir" to change the pointed locations of experiment folders and dataset folders respectively.
