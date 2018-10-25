# h-detach

This repository contains the code used for the paper [h-detach: Modifying the LSTM Gradient Towards Better Optimization](https://arxiv.org/abs/1810.03023) 

## Software Requirements

Python 3, Pytorch 0.4, tensor board, tqdm

## Copying Task

First change the directory path for saving results on `line 27` of `copying.py`. 

To run copying task with time delay of `300` steps using h-detach with `0.5` probability of blocking gradinets through the h-state using seed value `3`, execute the following command:

`python copying.py --save-dir=enter_experiment_dir_name_here --p-detach=0.5 --T=300 --seed=3`

To run copying task without h-detach, execute the following command:

`python copying.py --save-dir=enter_experiment_dir_name_here --T=300 --seed=3`

