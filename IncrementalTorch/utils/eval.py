from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from river.evaluate import Track
from tqdm import tqdm


def evaluate_flexible_classes_track(
        track,
        metric_name,
        models,
        n_samples,
        seed,
        n_checkpoints,
        learning_rate=None,
        result_path: Path = None,
        n_classes=None,
        class_tracker=False,
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
            # if class_tracker:
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


def plot_track(track,
               metric_name,
               models,
               n_samples,
               seed,
               n_checkpoints,
               learning_rate=None,
               result_path: Path = None,
               n_classes=None,
               class_tracker=False,
               verbose=1):
    plt.clf()
    if class_tracker:
        nrows = 4
    else:
        nrows = 3
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
            for checkpoint in tqdm(track(n_samples=n_samples, seed=seed, n_classes=n_classes).run(model, n_checkpoints),
                                   disable=disable):
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

        ax[ax_numb].plot(step, error, label=model_name + ' - lr: ' + str(learning_rate), linewidth=.6)
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

        ax[ax_numb].plot(step, r_time, label=model_name + ' - lr: ' + str(learning_rate), linewidth=.6)
        ax[ax_numb].set_ylabel('Time (seconds)')
        ax_numb += 1

        ax[ax_numb].plot(step, memory, label=model_name + ' - lr: ' + str(learning_rate), linewidth=.6)
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
