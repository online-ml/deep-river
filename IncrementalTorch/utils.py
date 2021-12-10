from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from river.evaluate import Track
from tqdm import tqdm
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F

ACTIVATION_FNS = {
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

LOSS_FNS = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "smooth_mae": F.smooth_l1_loss,
    "bce": F.binary_cross_entropy,
    "kld": F.kl_div,
    "huber": F.huber_loss,
}

OPTIMIZER_FNS = {
    "adam": optim.Adam,
    "adam_w": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def get_activation_fn(activation_fn):
    return (
        ACTIVATION_FNS.get(activation_fn)
        if isinstance(activation_fn, str)
        else activation_fn
    )


def get_loss_fn(loss_fn):
    return loss_fn if callable(loss_fn) else LOSS_FNS.get(loss_fn)


def get_optimizer_fn(optimizer_fn):
    return (
        OPTIMIZER_FNS.get(optimizer_fn)
        if isinstance(optimizer_fn, str)
        else optimizer_fn
    )


def prep_input(x, device):
    if isinstance(x, dict):
        x = torch.Tensor([list(x.values())], device=device)
    elif isinstance(x, pd.DataFrame):
        x = torch.Tensor(x.values, device=device)
    return x



class ScoreStandardizer:
    def __init__(self, momentum=0.99, with_std=True):
        self.with_std = with_std
        self.momentum = momentum
        self.mean = None
        self.var = 0

    def learn_one(self, x):
        if self.mean is None:
            self.mean = x
        else:
            last_diff = x - self.mean
            self.mean += (1 - self.momentum) * last_diff
            if self.with_std:
                self.var = self.momentum * (self.var + (1 - self.momentum) * last_diff)

    def transform_one(self, x):
        x_centered = x - self.mean
        if self.with_std:
            x_standardized = np.divide(x_centered, self.var ** 0.5, where=self.var > 0)
        else:
            x_standardized = x_centered
        return x_standardized

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)
        

"""
from IncrementalDL.tracks.evaluate_tracks import Torch2RiverTrack


def evaluate_flexible_classes_track(
               track: Torch2RiverTrack,
               metric_name,
               models,
               n_samples,
               seed,
               n_checkpoints,
               learning_rate = None,
               result_path: Path = None,
               n_classes = None,
               class_tracker = False,
               verbose=1
    ):
    print(f'Track name: {track(n_samples=1, seed=seed).name}')
    result_data = {
        'step': [],
        'model': [],
        'errors': [],
        'r_times': [],
        'memories': [],
        'seed': [],
        'new_classes': [],
        'lr': []
    }

    for model_name, model in models.items():
        if verbose > 1:
            print(f'Evaluating {model_name}')
        step = []
        error = []
        r_time = []
        memory = []
        new_classes = []

        if verbose < 1:
            disable = True
        else:
            disable = False
        for checkpoint in tqdm(track(n_samples=n_samples, seed=seed, n_classes=n_classes).run(model, n_checkpoints),
                               disable=disable):
            # for checkpoint in tqdm(track(n_samples=n_samples, seed=seed).run(model, n_checkpoints), disable=disable):
            step.append(checkpoint["Step"])
            error.append(checkpoint[metric_name])
            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
            memory.append(raw_memory * 2 ** -10 if unit == 'KB' else raw_memory)
            new_classes.append(checkpoint["NewClasses"])
            # in case you want to track new classes
            #if class_tracker:
               # if len(classes) != checkpoint['classes']:
                #    pass
               # classes = checkpoint['classes']

        result_data['step'].extend(step)
        result_data['model'].extend(len(step) * [model_name])
        result_data['errors'].extend(error)
        result_data['r_times'].extend(r_time)
        result_data['memories'].extend(memory)
        result_data['new_classes'].extend(new_classes)
        result_data['seed'].extend(len(step) * [seed])
        result_data['lr'].extend(len(step) * [learning_rate])

    df = pd.DataFrame(result_data)
    if result_path is not None:
        result_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(result_path / f'{track().name}.pdf'))
        df.to_csv(str(result_path / f'{track().name}.csv'))

    return df


def plot_track(track: Torch2RiverTrack,
               metric_name,
               models,
               n_samples,
               seed,
               n_checkpoints,
               learning_rate = None,
               result_path: Path = None,
               n_classes = None,
               class_tracker = False,
               verbose=1):
    plt.clf()
    if class_tracker:
        nrows = 4
    else: nrows = 3
    fig, ax = plt.subplots(figsize=(5, 5), nrows=nrows, dpi=300)

    print(f'Track name: {track(n_samples=1, seed=seed).name}')
    result_data = {
        'step': [],
        'model': [],
        'errors': [],
        'r_times': [],
        'memories': [],
        'seed': [],
        'lr': []
    }

    for model_name, model in models.items():
        if verbose > 1:
            print(f'Evaluating {model_name}')
        step = []
        error = []
        r_time = []
        memory = []
        classes = []

        if verbose < 1:
            disable = True
        else:
            disable = False
        if n_classes == None:
            for checkpoint in tqdm(track(n_samples=n_samples, seed=seed).run(model, n_checkpoints), disable=disable):
                step.append(checkpoint["Step"])
                error.append(checkpoint[metric_name])
                # Convert timedelta object into seconds
                r_time.append(checkpoint["Time"].total_seconds())
                # Make sure the memory measurements are in MB
                raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
                memory.append(raw_memory * 2 ** -10 if unit == 'KB' else raw_memory)
        else:
            for checkpoint in tqdm(track(n_samples=n_samples, seed=seed, n_classes=n_classes).run(model, n_checkpoints), disable=disable):
                step.append(checkpoint["Step"])
                error.append(checkpoint[metric_name])
                # Convert timedelta object into seconds
                r_time.append(checkpoint["Time"].total_seconds())
                # Make sure the memory measurements are in MB
                raw_memory, unit = float(checkpoint["Memory"][:-3]), checkpoint["Memory"][-2:]
                memory.append(raw_memory * 2 ** -10 if unit == 'KB' else raw_memory)

        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)
        ax_numb = 0

        ax[ax_numb].plot(step, error, label=model_name +' - lr: '+ str(learning_rate), linewidth=.6)
        ax[ax_numb].set_ylabel(metric_name + " Accuracy")
        # ax[0].set_ylabel('Rolling 100\n Accuracy')
        ax_numb += 1

        # #in case you want to track new classes
        # if class_tracker:
        #     ax[3].grid(True)
        #     ax[ax_numb].plot(checkpoint['classes'], np.zeros_like(checkpoint['classes']) + 0, 'x', label=model_name + ' - lr: ' + str(learning_rate), color='black')
        #     #ax[ax_numb].get_yaxis().set_visible(False)
        #     ax[ax_numb].set_yticklabels([])
        #     ax[ax_numb].set_ylabel('Class Tracker')
        #     ax_numb += 1

        ax[ax_numb].plot(step, r_time, label=model_name +' - lr: '+ str(learning_rate), linewidth=.6)
        ax[ax_numb].set_ylabel('Time (seconds)')
        ax_numb += 1

        ax[ax_numb].plot(step, memory, label=model_name +' - lr: '+ str(learning_rate), linewidth=.6)
        ax[ax_numb].set_ylabel('Memory (MB)')
        ax[ax_numb].set_xlabel('Instances')

        ax_numb += 1
        # in case you want to track new classes
        if class_tracker:
            ax[3].grid(True)
            ax[ax_numb].plot(checkpoint['classes'], np.zeros_like(checkpoint['classes']) + 0, 'x',
                             label=model_name + ' - lr: ' + str(learning_rate), color='black')
            # ax[ax_numb].get_yaxis().set_visible(False)
            ax[ax_numb].set_yticklabels([])
            ax[ax_numb].set_ylabel('Class Tracker')


        result_data['step'].extend(step)
        result_data['model'].extend(len(step) * [model_name])
        result_data['errors'].extend(error)
        result_data['r_times'].extend(r_time)
        result_data['memories'].extend(memory)
        result_data['seed'].extend(len(step) * [seed])
        result_data['lr'].extend(len(step) * [learning_rate])

    plt.setp(ax, xlim=(0, max(step)))
    plt.legend()
    plt.tight_layout()
    # plt.show()
    df = pd.DataFrame(result_data)
    if result_path is not None:
        result_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(result_path / f'{track().name}.pdf'))
        df.to_csv(str(result_path / f'{track().name}.csv'))

    return df
 """
