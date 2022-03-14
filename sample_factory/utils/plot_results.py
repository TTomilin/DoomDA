import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec
from scipy.ndimage import gaussian_filter1d


class Statistic:
    def __init__(self, timestamp: float, time_step: int, value: float):
        self.timestamp = timestamp
        self.date_time = datetime.fromtimestamp(timestamp / 1000.0)
        self.time_step = time_step
        self.value = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

    def arg(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    arg('--env_family', type=str, default=None, required=True, choices=['doom', 'box2d'],
        help='Name of the environment family')
    arg('--env', type=str, default=None, required=True, help='Name of the environment')
    arg('--tasks', type=str, default=[], required=True, nargs='+', help='List of the task names')
    arg('--metrics', type=str, default=['reward'], nargs='+', help='List of all the metrics to plot')
    arg('--title', type=str, default=None, help='Title of the plot')
    arg('--n_ticks', type=int, default=sys.maxsize, help='Limit the number of values to plot')
    arg('--style', type=str, default='seaborn', help='Matplotlib built-in style to be used for plotting')

    return parser.parse_args()


def env_to_title(env: str) -> str:
    return env.replace('_', ' ').title()


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title(f'{title}\n')
    row.set_frame_on(False)
    row.axis('off')


def plot_data(axis, file_name, task, ticks):
    time_steps, values = load_data(file_name.replace('*', task), ticks, sigma=0.8)
    axis.plot(time_steps, values, label=task)
    axis.legend()


def load_data(file_name, ticks, sigma=0.5):
    with open(file_name, 'r') as file:
        data = list(map(lambda x: Statistic(*x), json.load(file)))
        data = data[:ticks]  # Even out the number of time steps between different tasks
        time_steps = list(map(lambda x: x.time_step / 1e6, data))
        values = list(map(lambda x: x.value, data))
        values = gaussian_filter1d(values, sigma=sigma)
        return time_steps, values


def plot_results_gen(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    metrics = args.metrics
    test_task = '_'.join(args.tasks)
    test_task_title = env_to_title(args.tasks[0]) + ' + ' + env_to_title(args.tasks[1])
    plt.style.use(args.style)
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()

    for metric in metrics:

        fig, ax = plt.subplots(1, len(args.tasks), figsize=(12, 4))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize() if metric == 'reward' else 'Score', fontsize=20, labelpad=30)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=16, labelpad=30)
        main_ax.set_title(f'{env_to_title(args.env)}. Target Domain: {test_task_title}', fontsize=16, pad=10)

        task_data = np.empty((len(args.tasks), 1000))
        task_data[:] = np.nan

        file_name = f'{root_dir}/statistics/{args.env_family}/{args.env}/{test_task}/{metric}_*.json'
        for i, task in enumerate(args.tasks):
            plot_data(ax[i], file_name, 'default', args.n_ticks)
            plot_data(ax[i], file_name, task, args.n_ticks)

        fig.subplots_adjust(top=0.875, bottom=0.2)
        save_dir = f'{root_dir}/plots/{args.env_family}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{args.env}/{metric}_generalization.png', bbox_inches='tight')
        plt.show()


def plot_results_gen_individual(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    metrics = args.metrics
    test_task = '_'.join(args.tasks)
    test_task_title = env_to_title(args.tasks[0]) + ' + ' + env_to_title(args.tasks[1])
    plt.style.use(args.style)
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()

    for metric in metrics:
        for i, task in enumerate(args.tasks):

            task_data = np.empty((len(args.tasks), 1000))
            task_data[:] = np.nan

            file_name = f'{root_dir}/statistics/{args.env_family}/{args.env}/{test_task}/{metric}_*.json'
            time_steps, values = load_data(file_name.replace('*', 'default'), args.n_ticks, sigma=0.8)
            plt.plot(time_steps, values, label='default')
            time_steps, values = load_data(file_name.replace('*', task), args.n_ticks, sigma=0.8)
            plt.plot(time_steps, values, label=task)
            plt.legend()
            plt.ylabel(metric.capitalize() if metric == 'reward' else 'Score', fontsize=20, labelpad=10)
            plt.xlabel('Environment frames (M), skip=4', fontsize=16, labelpad=8)
            # plt.title(f'{env_to_title(args.env)}. Target Domain: {test_task_title}', fontsize=16, pad=10)

            save_dir = f'{root_dir}/plots/{args.env_family}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{save_dir}/{args.env}/{metric}_generalization_{task}.png', bbox_inches='tight')
            plt.show()


def plot_results_levels(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    metrics = args.metrics
    plt.style.use(args.style)
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    scenarios = {'defend_the_center': ('default', 'stone_wall', 'flying_enemies'),
                 'health_gathering': ('default', 'slime', 'obstacles'),
                 'dodge_projectiles': ('default', 'flames', 'cacodemons')}

    for metric in metrics:

        fig, ax = plt.subplots(1, len(scenarios), figsize=(16, 4))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize(), fontsize=20)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=16)
        main_ax.xaxis.labelpad = 25
        main_ax.yaxis.labelpad = 30
        for i, scenario in enumerate(list(scenarios)):
            tasks = scenarios[scenario]
            # test_task = '_'.join(tasks)
            test_task = f'{tasks[1]}_{tasks[2]}'

            task_data = np.empty((len(tasks), 1000))
            task_data[:] = np.nan

            for task in tasks:
                file_name = f'{root_dir}/statistics/{args.env_family}/{scenario}/{test_task}/{metric}_{task}.json'
                time_steps, values = load_data(file_name, args.n_ticks)

                # Plot the values according to time steps
                ax[i].plot(time_steps, values, label=task)
                # ax[i].legend()
                ax[i].set_title(scenario.replace('_', ' ').title(), fontsize=16)
                ax[i].spines['bottom'].set_linewidth(4)

        handles, labels = ax[0].get_legend_handles_labels()
        handles[0]._label = 'Different Textures & Entities'
        handles[1]._label = 'Different Textures, Same Entities'
        handles[2]._label = 'Same Textures, Different Entities'
        main_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.3),
                       ncol=3, fancybox=True, shadow=True)

        fig.subplots_adjust(top=0.80, bottom=0.2)
        save_dir = f'{root_dir}/plots/{args.env_family}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{metric}_generalization.png')
        plt.show()


def plot_results_env_all(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    metrics = args.metrics
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    rnns = ['', 'gru', 'lstm']
    scenarios = {'defend_the_center': ('default', 'stone_wall', 'flying_enemies', 'stone_wall_flying_enemies'),
                 'health_gathering': ('default', 'slime', 'obstacles', 'slime_obstacles'),
                 'dodge_projectiles': ('default', 'flames', 'cacodemons', 'flames_cacodemons')}

    for metric in metrics:
        grid = plt.GridSpec(3, 4)
        fig, ax = plt.subplots(len(scenarios), 4, figsize=(18, 11))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize() if metric == 'reward' else 'Score', fontsize=26, labelpad=45)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=20, labelpad=25)

        for i, scenario in enumerate(scenarios):
            tasks = scenarios[scenario]
            create_subtitle(fig, grid[i, ::], env_to_title(scenario))
            for j, task in enumerate(tasks):

                task_data = np.empty((len(tasks), 1000))
                task_data[:] = np.nan

                for rnn in rnns:
                    file_name_addition_with_rnn = '_' + rnn if rnn else ''
                    file_name = f'{root_dir}/statistics/{args.env_family}/{scenario}/{metric}{file_name_addition_with_rnn}_{task}.json'
                    time_steps, values = load_data(file_name, args.n_ticks, sigma=1)

                    # Plot the values according to time steps
                    ax[i, j].plot(time_steps, values, label='APPO' + (' + ' + rnn.upper() if rnn else ''))
                    ax[i, j].set_title(task, fontsize=12)
                    ax[i, j].spines['bottom'].set_linewidth(4)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        main_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3, fancybox=True,
                       shadow=True)

        plt.subplots_adjust(hspace=0.6, top=0.85, bottom=0.075)
        save_dir = f'{root_dir}/plots/{args.env_family}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/rnn_all.png', bbox_inches='tight')
        plt.show()


def plot_results_env(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    tasks = args.tasks
    metrics = args.metrics
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    rnns = ['', 'gru', 'lstm']

    for metric in metrics:

        fig, ax = plt.subplots(1, len(tasks), figsize=(18, 3))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize() if metric == 'reward' else 'Score', fontsize=20)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=16)
        main_ax.xaxis.labelpad = 25
        main_ax.yaxis.labelpad = 30
        for i, task in enumerate(tasks):

            task_data = np.empty((len(tasks), 1000))
            task_data[:] = np.nan

            for rnn in rnns:
                file_name_addition_with_rnn = '_' + rnn if rnn else ''
                file_name = f'{root_dir}/statistics/{args.env_family}/{args.env}/{metric}{file_name_addition_with_rnn}_{task}.json'
                time_steps, values = load_data(file_name, args.n_ticks)

                # Plot the values according to time steps
                ax[i].plot(time_steps, values, label='APPO' + (' + ' + rnn.upper() if rnn else ''))
                # ax[i].legend()
                ax[i].set_title(task, fontsize=12)
                ax[i].spines['bottom'].set_linewidth(4)

        handles, labels = ax[0].get_legend_handles_labels()
        main_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.475), ncol=3, fancybox=True,
                       shadow=True)
        # plt.title(args.title, fontsize=22, pad=25)
        fig.subplots_adjust(top=0.75, bottom=0.1)
        save_dir = f'{root_dir}/plots/{args.env_family}/{args.env}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{metric}.png')
        plt.show()


def plot_results_rnn_all(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    metrics = args.metrics
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    rnns = ['', 'gru', 'lstm']
    scenarios = {'defend_the_center': ('default', 'stone_wall', 'flying_enemies', 'stone_wall_flying_enemies'),
                 'health_gathering': ('default', 'slime', 'obstacles', 'slime_obstacles'),
                 'dodge_projectiles': ('default', 'flames', 'cacodemons', 'flames_cacodemons')}

    for metric in metrics:

        grid = plt.GridSpec(3, 3)
        fig, ax = plt.subplots(len(rnns), 3, figsize=(18, 11))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize(), fontsize=26)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=20)
        main_ax.xaxis.labelpad = 25
        main_ax.yaxis.labelpad = 30

        for i, rnn in enumerate(rnns):

            file_name_addition_with_rnn = '_' + rnn if rnn else ''

            for j, scenario in enumerate(scenarios):
                tasks = scenarios[scenario]
                create_subtitle(fig, grid[i, ::], rnn.upper() if rnn else 'No RNN')
                for task in tasks:

                    task_data = np.empty((len(tasks), 1000))
                    task_data[:] = np.nan

                    file_name = f'{root_dir}/statistics/{args.env_family}/{scenario}/{metric}{file_name_addition_with_rnn}_{task}.json'
                    time_steps, values = load_data(file_name, args.n_ticks, sigma=1)

                    # Plot the values according to time steps
                    ax[i, j].plot(time_steps, values, label=scenario)
                    ax[i, j].set_title(env_to_title(scenario), fontsize=12)
                    ax[i, j].spines['bottom'].set_linewidth(4)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        handles[0]._label = 'Default'
        handles[1]._label = 'New Textures'
        handles[2]._label = 'New Entities'
        handles[3]._label = 'New Textures & Entities'
        main_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=4, fancybox=True,
                       shadow=True)

        plt.subplots_adjust(hspace=0.6, top=0.85, bottom=0.075)
        save_dir = f'{root_dir}/plots/{args.env_family}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/env_all.png')
        plt.show()


def plot_results_rnn(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    tasks = args.tasks
    metrics = args.metrics
    plt.style.use(args.style)
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    rnns = ['', 'gru', 'lstm']

    for metric in metrics:

        fig, ax = plt.subplots(1, len(rnns), figsize=(16, 4))
        main_ax = fig.add_subplot(1, 1, 1, frameon=False)
        main_ax.get_xaxis().set_ticks([])
        main_ax.get_yaxis().set_ticks([])
        main_ax.set_ylabel(metric.capitalize(), fontsize=20)
        main_ax.set_xlabel('Environment frames (M), skip=4', fontsize=16)
        main_ax.xaxis.labelpad = 25
        main_ax.yaxis.labelpad = 30
        for i, rnn in enumerate(rnns):

            file_name_addition_with_rnn = '_' + rnn if rnn else ''
            task_data = np.empty((len(tasks), 1000))
            task_data[:] = np.nan

            for task in tasks:
                file_name = f'{root_dir}/statistics/{args.env_family}/{args.env}/{metric}{file_name_addition_with_rnn}_{task}.json'
                time_steps, values = load_data(file_name, args.n_ticks)

                # Plot the values according to time steps
                ax[i].plot(time_steps, values, label=task)
                # ax[i].legend()
                ax[i].set_title('APPO' + (' + ' + rnn.upper() if rnn else ''), fontsize=16)
                ax[i].spines['bottom'].set_linewidth(4)

        handles, labels = ax[0].get_legend_handles_labels()
        main_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, fancybox=True,
                       shadow=True)
        # plt.title(args.title, fontsize=22, pad=25)
        fig.subplots_adjust(top=0.825, bottom=0.2)
        save_dir = f'{root_dir}/plots/{args.env_family}/{args.env}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{metric}.png')
        plt.show()


def plot_results_with_clocktime(args: argparse.Namespace):
    root_dir = os.path.dirname(os.getcwd())
    tasks = args.tasks
    metrics = args.metrics
    plt.style.use(args.style)

    for metric in metrics:
        task_data = np.empty((len(tasks), 1000))
        task_data[:] = np.nan

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        col_ax = fig.add_subplot(1, 1, 1, frameon=False)
        col_ax.get_xaxis().set_ticks([])
        col_ax.get_yaxis().set_ticks([])
        for i, task in enumerate(tasks):
            with open(f'{root_dir}/statistics/{args.env_family}/{args.env}/{metric}_{task}.json', 'r') as f:
                try:
                    data = list(map(lambda x: Statistic(*x), json.load(f)))
                except Exception as e:
                    logging.error(f'Unable to plot metric {metric} for task {task}. Reason: {e}')
                    continue

                data = data[:args.n_ticks]  # Even out the number of time steps between different tasks

                time_steps = list(map(lambda x: x.time_step / 1e6, data))
                values = list(map(lambda x: x.value, data))
                values = gaussian_filter1d(values, sigma=0.5)

                # Plot the values according to time steps
                ax[0].set_ylabel(metric.capitalize(), fontsize=20)
                ax[0].set_xlabel('Timesteps (M)', fontsize=16)
                ax[0].plot(time_steps, values, label=task)
                ax[0].legend()

                # Plot the values according to clock time
                start_time = data[0].timestamp
                timestamps = list(map(lambda x: x.timestamp - start_time, data))
                ax[1].set_xlabel('Clocktime (s)', fontsize=16)
                ax[1].plot(timestamps, values, label=task)
                ax[1].legend()

                # Add std areas for tasks with multiple seeds
                # upper = values - 1.96 * np.nanstd(data, axis=0) / np.sqrt(30)
                # lower = values + 1.96 * np.nanstd(data, axis=0) / np.sqrt(30)
                # plt.fill_between(time_steps, upper, lower, alpha=0.5)

        plt.title(args.title, fontsize=20)
        plt.savefig(f'{root_dir}/plots/{args.env_family}/{args.env}/{metric}.png')
        plt.show()


if __name__ == '__main__':
    # plot_results_gen(parse_args())
    plot_results_gen_individual(parse_args())
    # plot_results_levels(parse_args())
    # plot_results_env(parse_args())
    # plot_results_env_all(parse_args())
    # plot_results_rnn(parse_args())
    # plot_results_rnn_all(parse_args())
